/**
 * FHE-GBDT TypeScript SDK
 *
 * Privacy-preserving machine learning inference using Fully Homomorphic Encryption
 * for Gradient Boosted Decision Tree models.
 */

import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import { EventEmitter } from 'events';
import * as path from 'path';

// Re-export types
export * from './types';

import {
  FHEGBDTConfig,
  Model,
  CompiledModel,
  PredictRequest,
  PredictResponse,
  KeyPair,
  Subscription,
  Usage,
  Plan,
} from './types';

/**
 * Default configuration values
 */
const DEFAULT_CONFIG: Partial<FHEGBDTConfig> = {
  endpoint: 'localhost:8080',
  timeout: 30000,
  retries: 3,
  retryDelay: 1000,
};

/**
 * Main client for FHE-GBDT Serving API
 */
export class FHEGBDTClient extends EventEmitter {
  private config: FHEGBDTConfig;
  private gatewayClient: any;
  private controlClient: any;
  private billingClient: any;
  private credentials: grpc.ChannelCredentials;
  private metadata: grpc.Metadata;

  constructor(config: Partial<FHEGBDTConfig> & { apiKey: string }) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config } as FHEGBDTConfig;
    this.metadata = new grpc.Metadata();
    this.metadata.set('x-api-key', this.config.apiKey);

    if (this.config.tenantId) {
      this.metadata.set('x-tenant-id', this.config.tenantId);
    }

    // Setup credentials
    if (this.config.useTLS) {
      this.credentials = grpc.credentials.createSsl();
    } else {
      this.credentials = grpc.credentials.createInsecure();
    }

    this.initializeClients();
  }

  private initializeClients(): void {
    const protoPath = path.resolve(__dirname, '../../../proto');

    // Load proto files
    const gatewayProto = protoLoader.loadSync(
      path.join(protoPath, 'gateway.proto'),
      {
        keepCase: true,
        longs: String,
        enums: String,
        defaults: true,
        oneofs: true,
      }
    );

    const controlProto = protoLoader.loadSync(
      path.join(protoPath, 'control.proto'),
      {
        keepCase: true,
        longs: String,
        enums: String,
        defaults: true,
        oneofs: true,
      }
    );

    const billingProto = protoLoader.loadSync(
      path.join(protoPath, 'billing.proto'),
      {
        keepCase: true,
        longs: String,
        enums: String,
        defaults: true,
        oneofs: true,
      }
    );

    // Create gRPC package definitions
    const gatewayPkg = grpc.loadPackageDefinition(gatewayProto) as any;
    const controlPkg = grpc.loadPackageDefinition(controlProto) as any;
    const billingPkg = grpc.loadPackageDefinition(billingProto) as any;

    // Create clients
    this.gatewayClient = new gatewayPkg.gateway.GatewayService(
      this.config.endpoint,
      this.credentials
    );

    this.controlClient = new controlPkg.control.ControlService(
      this.config.controlEndpoint || this.config.endpoint.replace(':8080', ':8081'),
      this.credentials
    );

    this.billingClient = new billingPkg.billing.BillingService(
      this.config.billingEndpoint || this.config.endpoint.replace(':8080', ':8084'),
      this.credentials
    );
  }

  /**
   * Register a new GBDT model
   */
  async registerModel(
    modelName: string,
    modelContent: Buffer,
    libraryType: 'xgboost' | 'lightgbm' | 'catboost'
  ): Promise<Model> {
    return this.withRetry(async () => {
      return new Promise((resolve, reject) => {
        this.controlClient.RegisterModel(
          {
            model_name: modelName,
            model_content: modelContent,
            library_type: libraryType,
            tenant_id: this.config.tenantId,
          },
          this.metadata,
          { deadline: this.getDeadline() },
          (err: grpc.ServiceError | null, response: any) => {
            if (err) {
              reject(this.handleError(err));
            } else {
              this.emit('model:registered', response.model_id);
              resolve({
                id: response.model_id,
                name: modelName,
                libraryType,
                status: 'registered',
                createdAt: new Date(),
              });
            }
          }
        );
      });
    });
  }

  /**
   * Compile a registered model for FHE inference
   */
  async compileModel(
    modelId: string,
    profile: 'fast' | 'balanced' | 'accurate' = 'balanced'
  ): Promise<CompiledModel> {
    return this.withRetry(async () => {
      return new Promise((resolve, reject) => {
        this.controlClient.CompileModel(
          {
            model_id: modelId,
            profile,
          },
          this.metadata,
          { deadline: this.getDeadline() },
          (err: grpc.ServiceError | null, response: any) => {
            if (err) {
              reject(this.handleError(err));
            } else {
              this.emit('model:compiled', response.compiled_model_id);
              resolve({
                id: response.compiled_model_id,
                modelId,
                profile,
                status: 'compiling',
                createdAt: new Date(),
              });
            }
          }
        );
      });
    });
  }

  /**
   * Get compilation status
   */
  async getCompileStatus(compiledModelId: string): Promise<CompiledModel> {
    return this.withRetry(async () => {
      return new Promise((resolve, reject) => {
        this.controlClient.GetCompileStatus(
          { compiled_model_id: compiledModelId },
          this.metadata,
          { deadline: this.getDeadline() },
          (err: grpc.ServiceError | null, response: any) => {
            if (err) {
              reject(this.handleError(err));
            } else {
              resolve({
                id: compiledModelId,
                modelId: '',
                profile: 'balanced',
                status: response.status,
                planId: response.plan_id,
                createdAt: new Date(),
              });
            }
          }
        );
      });
    });
  }

  /**
   * Wait for compilation to complete
   */
  async waitForCompilation(
    compiledModelId: string,
    pollInterval: number = 2000,
    maxWait: number = 300000
  ): Promise<CompiledModel> {
    const startTime = Date.now();

    while (Date.now() - startTime < maxWait) {
      const status = await this.getCompileStatus(compiledModelId);

      if (status.status === 'successful') {
        return status;
      }

      if (status.status === 'failed') {
        throw new Error(`Compilation failed for model ${compiledModelId}`);
      }

      await this.sleep(pollInterval);
    }

    throw new Error(`Compilation timeout for model ${compiledModelId}`);
  }

  /**
   * Make an encrypted prediction
   */
  async predict(request: PredictRequest): Promise<PredictResponse> {
    return this.withRetry(async () => {
      return new Promise((resolve, reject) => {
        const startTime = Date.now();

        this.gatewayClient.Predict(
          {
            tenant_id: this.config.tenantId,
            compiled_model_id: request.compiledModelId,
            batch: {
              payload: request.encryptedPayload,
            },
          },
          this.metadata,
          { deadline: this.getDeadline() },
          (err: grpc.ServiceError | null, response: any) => {
            const latency = Date.now() - startTime;

            if (err) {
              this.emit('predict:error', { error: err, latency });
              reject(this.handleError(err));
            } else {
              this.emit('predict:success', { latency });
              resolve({
                encryptedResult: response.result.payload,
                latencyMs: latency,
                requestId: response.request_id,
              });
            }
          }
        );
      });
    });
  }

  /**
   * Upload evaluation keys to the keystore
   */
  async uploadEvalKeys(evalKeys: Buffer): Promise<string> {
    return this.withRetry(async () => {
      return new Promise((resolve, reject) => {
        this.gatewayClient.UploadEvalKeys(
          {
            tenant_id: this.config.tenantId,
            eval_keys: evalKeys,
          },
          this.metadata,
          { deadline: this.getDeadline() },
          (err: grpc.ServiceError | null, response: any) => {
            if (err) {
              reject(this.handleError(err));
            } else {
              this.emit('keys:uploaded');
              resolve(response.key_id);
            }
          }
        );
      });
    });
  }

  // ============================================================================
  // Billing & Subscription Methods
  // ============================================================================

  /**
   * Get available subscription plans
   */
  async listPlans(): Promise<Plan[]> {
    return this.withRetry(async () => {
      return new Promise((resolve, reject) => {
        this.billingClient.ListPlans(
          { include_inactive: false },
          this.metadata,
          { deadline: this.getDeadline() },
          (err: grpc.ServiceError | null, response: any) => {
            if (err) {
              reject(this.handleError(err));
            } else {
              resolve(
                response.plans.map((p: any) => ({
                  id: p.id,
                  name: p.name,
                  description: p.description,
                  priceCents: parseInt(p.price_cents, 10),
                  currency: p.currency,
                  predictionLimit: parseInt(p.prediction_limit, 10),
                  features: p.features,
                }))
              );
            }
          }
        );
      });
    });
  }

  /**
   * Get current subscription
   */
  async getSubscription(): Promise<Subscription | null> {
    return this.withRetry(async () => {
      return new Promise((resolve, reject) => {
        this.billingClient.GetSubscription(
          { tenant_id: this.config.tenantId },
          this.metadata,
          { deadline: this.getDeadline() },
          (err: grpc.ServiceError | null, response: any) => {
            if (err) {
              if (err.code === grpc.status.NOT_FOUND) {
                resolve(null);
              } else {
                reject(this.handleError(err));
              }
            } else {
              resolve({
                id: response.subscription.id,
                tenantId: response.subscription.tenant_id,
                planId: response.subscription.plan_id,
                status: response.subscription.status,
                currentPeriodStart: new Date(response.subscription.current_period_start),
                currentPeriodEnd: new Date(response.subscription.current_period_end),
              });
            }
          }
        );
      });
    });
  }

  /**
   * Create a subscription
   */
  async createSubscription(planId: string, email: string): Promise<Subscription> {
    return this.withRetry(async () => {
      return new Promise((resolve, reject) => {
        this.billingClient.CreateSubscription(
          {
            tenant_id: this.config.tenantId,
            plan_id: planId,
            email,
          },
          this.metadata,
          { deadline: this.getDeadline() },
          (err: grpc.ServiceError | null, response: any) => {
            if (err) {
              reject(this.handleError(err));
            } else {
              this.emit('subscription:created', response.subscription.id);
              resolve({
                id: response.subscription.id,
                tenantId: response.subscription.tenant_id,
                planId: response.subscription.plan_id,
                status: response.subscription.status,
                currentPeriodStart: new Date(response.subscription.current_period_start),
                currentPeriodEnd: new Date(response.subscription.current_period_end),
                clientSecret: response.client_secret,
              });
            }
          }
        );
      });
    });
  }

  /**
   * Get current usage
   */
  async getUsage(): Promise<Usage> {
    return this.withRetry(async () => {
      return new Promise((resolve, reject) => {
        this.billingClient.GetUsage(
          { tenant_id: this.config.tenantId },
          this.metadata,
          { deadline: this.getDeadline() },
          (err: grpc.ServiceError | null, response: any) => {
            if (err) {
              reject(this.handleError(err));
            } else {
              resolve({
                predictionsCount: parseInt(response.usage.predictions_count, 10),
                predictionsLimit: parseInt(response.usage.predictions_limit, 10),
                usagePercentage: response.usage_percentage,
                periodStart: new Date(response.usage.period_start),
                periodEnd: new Date(response.usage.period_end),
                overageCount: parseInt(response.usage.overage_count || '0', 10),
                overageCostCents: parseInt(response.usage.overage_cost_cents || '0', 10),
              });
            }
          }
        );
      });
    });
  }

  /**
   * Create a checkout session for subscription upgrade
   */
  async createCheckoutSession(
    planId: string,
    successUrl: string,
    cancelUrl: string
  ): Promise<{ sessionId: string; checkoutUrl: string }> {
    return this.withRetry(async () => {
      return new Promise((resolve, reject) => {
        this.billingClient.CreateCheckoutSession(
          {
            tenant_id: this.config.tenantId,
            plan_id: planId,
            success_url: successUrl,
            cancel_url: cancelUrl,
          },
          this.metadata,
          { deadline: this.getDeadline() },
          (err: grpc.ServiceError | null, response: any) => {
            if (err) {
              reject(this.handleError(err));
            } else {
              resolve({
                sessionId: response.session_id,
                checkoutUrl: response.checkout_url,
              });
            }
          }
        );
      });
    });
  }

  // ============================================================================
  // Helper Methods
  // ============================================================================

  private getDeadline(): Date {
    return new Date(Date.now() + this.config.timeout);
  }

  private async withRetry<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: Error | null = null;

    for (let i = 0; i < this.config.retries; i++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;

        // Don't retry on certain errors
        if (error instanceof FHEGBDTError) {
          if (
            error.code === 'INVALID_ARGUMENT' ||
            error.code === 'NOT_FOUND' ||
            error.code === 'PERMISSION_DENIED'
          ) {
            throw error;
          }
        }

        if (i < this.config.retries - 1) {
          await this.sleep(this.config.retryDelay * Math.pow(2, i));
        }
      }
    }

    throw lastError;
  }

  private handleError(err: grpc.ServiceError): FHEGBDTError {
    const code = this.grpcStatusToCode(err.code);
    return new FHEGBDTError(err.message, code, err.details);
  }

  private grpcStatusToCode(status: grpc.status): string {
    const statusMap: { [key: number]: string } = {
      [grpc.status.OK]: 'OK',
      [grpc.status.CANCELLED]: 'CANCELLED',
      [grpc.status.UNKNOWN]: 'UNKNOWN',
      [grpc.status.INVALID_ARGUMENT]: 'INVALID_ARGUMENT',
      [grpc.status.DEADLINE_EXCEEDED]: 'TIMEOUT',
      [grpc.status.NOT_FOUND]: 'NOT_FOUND',
      [grpc.status.ALREADY_EXISTS]: 'ALREADY_EXISTS',
      [grpc.status.PERMISSION_DENIED]: 'PERMISSION_DENIED',
      [grpc.status.RESOURCE_EXHAUSTED]: 'RATE_LIMITED',
      [grpc.status.UNAVAILABLE]: 'UNAVAILABLE',
      [grpc.status.UNAUTHENTICATED]: 'UNAUTHENTICATED',
    };
    return statusMap[status] || 'UNKNOWN';
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Close the client connections
   */
  close(): void {
    if (this.gatewayClient) {
      grpc.closeClient(this.gatewayClient);
    }
    if (this.controlClient) {
      grpc.closeClient(this.controlClient);
    }
    if (this.billingClient) {
      grpc.closeClient(this.billingClient);
    }
  }
}

/**
 * Custom error class for FHE-GBDT errors
 */
export class FHEGBDTError extends Error {
  code: string;
  details?: string;

  constructor(message: string, code: string, details?: string) {
    super(message);
    this.name = 'FHEGBDTError';
    this.code = code;
    this.details = details;
  }
}

/**
 * Create a configured client instance
 */
export function createClient(config: Partial<FHEGBDTConfig> & { apiKey: string }): FHEGBDTClient {
  return new FHEGBDTClient(config);
}

// Default export
export default { createClient, FHEGBDTClient, FHEGBDTError };
