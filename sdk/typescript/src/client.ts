/**
 * FHE-GBDT TypeScript SDK Client
 * Aligned with TenSafe SDK implementation
 */

import {
  FHEGBDTClientOptions,
  RequestOptions,
  PredictRequest,
  PredictResponse,
  BatchPredictRequest,
  BatchPredictResponse,
  RegisterModelRequest,
  Model,
  CompileOptions,
  TrainingConfig,
  TrainingJob,
  KeyInfo,
  Package,
  PackageVerification,
  AuditLogEntry,
  FutureResponse,
  FutureStatus,
  OperationType,
  APIError,
  FHEGBDTError,
  AuthenticationError,
  RateLimitError,
  TimeoutError,
  FutureTimeoutError,
} from './types';

/** SDK Version */
export const VERSION = '1.0.0';

/** API Version */
export const API_VERSION = 'v1';

/**
 * Default client options
 */
const DEFAULT_OPTIONS: Required<FHEGBDTClientOptions> = {
  baseUrl: 'https://api.fhe-gbdt.example.com',
  apiKey: '',
  tenantId: '',
  timeout: 120000,
  maxRetries: 3,
  pollingInterval: 1000,
};

/**
 * Convert camelCase to snake_case for API
 */
function toSnakeCase(obj: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    const snakeKey = key.replace(/([A-Z])/g, '_$1').toLowerCase();
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      result[snakeKey] = toSnakeCase(value as Record<string, unknown>);
    } else {
      result[snakeKey] = value;
    }
  }
  return result;
}

/**
 * Convert snake_case to camelCase from API
 */
function toCamelCase(obj: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    const camelKey = key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      result[camelKey] = toCamelCase(value as Record<string, unknown>);
    } else {
      result[camelKey] = value;
    }
  }
  return result;
}

/**
 * FHE-GBDT API Client
 *
 * Main client for interacting with the FHE-GBDT platform.
 *
 * @example
 * ```typescript
 * const client = new FHEGBDTClient({
 *   apiKey: process.env.FHE_GBDT_API_KEY,
 *   tenantId: 'my-tenant',
 * });
 *
 * // Run encrypted prediction
 * const result = await client.predict({
 *   modelId: 'model-123',
 *   ciphertext: encryptedFeatures,
 * });
 * ```
 */
export class FHEGBDTClient {
  private readonly options: Required<FHEGBDTClientOptions>;

  constructor(options: FHEGBDTClientOptions = {}) {
    this.options = {
      ...DEFAULT_OPTIONS,
      ...options,
      apiKey: options.apiKey || process.env.FHE_GBDT_API_KEY || '',
      tenantId: options.tenantId || process.env.FHE_GBDT_TENANT_ID || '',
    };
  }

  /**
   * Make HTTP request to API
   */
  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    options: RequestOptions = {}
  ): Promise<T> {
    const url = `${this.options.baseUrl}/${API_VERSION}${path}`;
    const timeout = options.timeout || this.options.timeout;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-API-Key': this.options.apiKey,
      'X-Tenant-ID': this.options.tenantId,
      ...options.headers,
    };

    if (options.requestId) {
      headers['X-Request-ID'] = options.requestId;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(toSnakeCase(body as Record<string, unknown>)) : undefined,
        signal: options.signal || controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        await this.handleErrorResponse(response);
      }

      if (response.status === 204) {
        return {} as T;
      }

      const data = await response.json();
      return toCamelCase(data.data || data) as T;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof FHEGBDTError) {
        throw error;
      }

      if ((error as Error).name === 'AbortError') {
        throw new TimeoutError(`Request to ${path} timed out after ${timeout}ms`);
      }

      throw new FHEGBDTError(`Request failed: ${(error as Error).message}`, 'REQUEST_FAILED');
    }
  }

  /**
   * Handle error response from API
   */
  private async handleErrorResponse(response: Response): Promise<never> {
    let error: APIError;

    try {
      const data = await response.json();
      error = data.error || { code: 'UNKNOWN', message: 'Unknown error' };
    } catch {
      error = { code: 'UNKNOWN', message: response.statusText };
    }

    switch (response.status) {
      case 401:
        throw new AuthenticationError(error.message);
      case 429:
        const retryAfter = response.headers.get('Retry-After');
        throw new RateLimitError(error.message, retryAfter ? parseInt(retryAfter) : undefined);
      default:
        throw new FHEGBDTError(error.message, error.code, error.details);
    }
  }

  // ============== Prediction Methods ==============

  /**
   * Run encrypted prediction
   *
   * @param request - Prediction request
   * @param options - Request options
   * @returns Prediction response with encrypted result
   */
  async predict(request: PredictRequest, options?: RequestOptions): Promise<PredictResponse> {
    return this.request<PredictResponse>('POST', '/predict', request, options);
  }

  /**
   * Run batch encrypted predictions
   *
   * @param request - Batch prediction request
   * @param options - Request options
   * @returns Batch prediction response
   */
  async batchPredict(
    request: BatchPredictRequest,
    options?: RequestOptions
  ): Promise<BatchPredictResponse> {
    return this.request<BatchPredictResponse>('POST', '/batch/predict', request, options);
  }

  // ============== Model Methods ==============

  /**
   * List models
   *
   * @param limit - Maximum number of models to return
   * @param offset - Offset for pagination
   * @returns Array of models
   */
  async listModels(limit = 20, offset = 0): Promise<Model[]> {
    return this.request<Model[]>('GET', `/models?limit=${limit}&offset=${offset}`);
  }

  /**
   * Register a new model
   *
   * @param request - Model registration request
   * @returns Registered model
   */
  async registerModel(request: RegisterModelRequest): Promise<Model> {
    return this.request<Model>('POST', '/models', request);
  }

  /**
   * Get model by ID
   *
   * @param modelId - Model ID
   * @returns Model details
   */
  async getModel(modelId: string): Promise<Model> {
    return this.request<Model>('GET', `/models/${modelId}`);
  }

  /**
   * Delete model
   *
   * @param modelId - Model ID
   */
  async deleteModel(modelId: string): Promise<void> {
    await this.request<void>('DELETE', `/models/${modelId}`);
  }

  /**
   * Compile model for FHE
   *
   * @param modelId - Model ID
   * @param options - Compilation options
   * @returns Future for tracking compilation
   */
  async compileModel(
    modelId: string,
    options: CompileOptions = {}
  ): Promise<FutureResponse<Model>> {
    return this.request<FutureResponse<Model>>('POST', `/models/${modelId}/compile`, options);
  }

  /**
   * Get compilation status
   *
   * @param modelId - Model ID
   * @returns Compilation status
   */
  async getCompileStatus(modelId: string): Promise<{ status: string; progress: number }> {
    return this.request<{ status: string; progress: number }>(
      'GET',
      `/models/${modelId}/compile/status`
    );
  }

  /**
   * Compile model and wait for completion
   *
   * @param modelId - Model ID
   * @param options - Compilation options
   * @param timeout - Maximum time to wait (ms)
   * @returns Compiled model
   */
  async compileModelAndWait(
    modelId: string,
    options: CompileOptions = {},
    timeout = 300000
  ): Promise<Model> {
    await this.compileModel(modelId, options);

    const startTime = Date.now();
    while (Date.now() - startTime < timeout) {
      const status = await this.getCompileStatus(modelId);

      if (status.status === 'completed') {
        return this.getModel(modelId);
      }

      if (status.status === 'failed') {
        throw new FHEGBDTError('Compilation failed', 'COMPILE_FAILED');
      }

      await this.sleep(this.options.pollingInterval);
    }

    throw new FutureTimeoutError(modelId);
  }

  // ============== Training Methods ==============

  /**
   * Start training job
   *
   * @param config - Training configuration
   * @returns Training job
   */
  async startTraining(config: TrainingConfig): Promise<TrainingJob> {
    return this.request<TrainingJob>('POST', '/training/jobs', config);
  }

  /**
   * List training jobs
   *
   * @param limit - Maximum number of jobs to return
   * @param offset - Offset for pagination
   * @returns Array of training jobs
   */
  async listTrainingJobs(limit = 20, offset = 0): Promise<TrainingJob[]> {
    return this.request<TrainingJob[]>('GET', `/training/jobs?limit=${limit}&offset=${offset}`);
  }

  /**
   * Get training job status
   *
   * @param jobId - Job ID
   * @returns Training job details
   */
  async getTrainingJob(jobId: string): Promise<TrainingJob> {
    return this.request<TrainingJob>('GET', `/training/jobs/${jobId}`);
  }

  /**
   * Stop training job
   *
   * @param jobId - Job ID
   */
  async stopTraining(jobId: string): Promise<void> {
    await this.request<void>('DELETE', `/training/jobs/${jobId}`);
  }

  /**
   * Start training and wait for completion
   *
   * @param config - Training configuration
   * @param timeout - Maximum time to wait (ms)
   * @param onProgress - Progress callback
   * @returns Completed training job
   */
  async trainAndWait(
    config: TrainingConfig,
    timeout = 3600000,
    onProgress?: (job: TrainingJob) => void
  ): Promise<TrainingJob> {
    const job = await this.startTraining(config);

    const startTime = Date.now();
    while (Date.now() - startTime < timeout) {
      const status = await this.getTrainingJob(job.id);

      if (onProgress) {
        onProgress(status);
      }

      if (status.status === 'completed') {
        return status;
      }

      if (status.status === 'failed') {
        throw new FHEGBDTError(status.error || 'Training failed', 'TRAINING_FAILED');
      }

      if (status.status === 'cancelled') {
        throw new FHEGBDTError('Training was cancelled', 'TRAINING_CANCELLED');
      }

      await this.sleep(this.options.pollingInterval);
    }

    throw new FutureTimeoutError(job.id);
  }

  // ============== Key Management Methods ==============

  /**
   * Upload evaluation keys
   *
   * @param keyData - Base64 encoded key data
   * @returns Key information
   */
  async uploadKeys(keyData: string): Promise<KeyInfo> {
    return this.request<KeyInfo>('POST', '/keys', { key: keyData });
  }

  /**
   * Get key status
   *
   * @param keyId - Key ID
   * @returns Key information
   */
  async getKeyStatus(keyId: string): Promise<KeyInfo> {
    return this.request<KeyInfo>('GET', `/keys/${keyId}`);
  }

  /**
   * Rotate keys
   *
   * @param keyId - Key ID
   * @returns New key information
   */
  async rotateKeys(keyId: string): Promise<{ oldKeyId: string; newKeyId: string }> {
    return this.request<{ oldKeyId: string; newKeyId: string }>(
      'POST',
      `/keys/${keyId}/rotate`
    );
  }

  /**
   * Revoke keys
   *
   * @param keyId - Key ID
   */
  async revokeKeys(keyId: string): Promise<void> {
    await this.request<void>('DELETE', `/keys/${keyId}`);
  }

  // ============== Package Methods ==============

  /**
   * Create GBSP package
   *
   * @param modelId - Model ID
   * @param recipients - Optional recipient public keys
   * @returns Package information
   */
  async createPackage(modelId: string, recipients?: string[]): Promise<Package> {
    return this.request<Package>('POST', '/packages', { modelId, recipients });
  }

  /**
   * Get package status
   *
   * @param packageId - Package ID
   * @returns Package information
   */
  async getPackage(packageId: string): Promise<Package> {
    return this.request<Package>('GET', `/packages/${packageId}`);
  }

  /**
   * Verify GBSP package
   *
   * @param packageId - Package ID
   * @returns Verification result
   */
  async verifyPackage(packageId: string): Promise<PackageVerification> {
    return this.request<PackageVerification>('POST', `/packages/${packageId}/verify`);
  }

  // ============== Audit Methods ==============

  /**
   * Get audit logs
   *
   * @param limit - Maximum number of entries
   * @param offset - Offset for pagination
   * @returns Array of audit log entries
   */
  async getAuditLogs(limit = 100, offset = 0): Promise<AuditLogEntry[]> {
    return this.request<AuditLogEntry[]>('GET', `/audit/logs?limit=${limit}&offset=${offset}`);
  }

  /**
   * Export audit logs for compliance
   *
   * @param startDate - Start date
   * @param endDate - End date
   * @returns Export job ID
   */
  async exportAuditLogs(
    startDate: Date,
    endDate: Date
  ): Promise<{ exportId: string; status: string }> {
    return this.request<{ exportId: string; status: string }>('POST', '/audit/export', {
      startDate: startDate.toISOString(),
      endDate: endDate.toISOString(),
    });
  }

  // ============== Health Methods ==============

  /**
   * Check API health
   *
   * @returns Health status
   */
  async health(): Promise<{ status: string; time: string }> {
    return this.request<{ status: string; time: string }>('GET', '/health');
  }

  /**
   * Check API readiness
   *
   * @returns Readiness status
   */
  async ready(): Promise<{ status: string; time: string }> {
    return this.request<{ status: string; time: string }>('GET', '/ready');
  }

  // ============== Utility Methods ==============

  /**
   * Sleep for specified duration
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

/**
 * Create client instance
 */
export function createClient(options?: FHEGBDTClientOptions): FHEGBDTClient {
  return new FHEGBDTClient(options);
}

// Default export
export default FHEGBDTClient;
