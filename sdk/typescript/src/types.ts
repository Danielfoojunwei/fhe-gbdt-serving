/**
 * FHE-GBDT TypeScript SDK Type Definitions
 *
 * This module contains all TypeScript interfaces and types that map to
 * the FHE-GBDT protocol buffer definitions and API contracts.
 *
 * @packageDocumentation
 */

// ============================================================================
// Client Configuration Types
// ============================================================================

/**
 * Configuration options for the FHE-GBDT client.
 */
export interface FHEGBDTClientConfig {
  /**
   * API key for authentication.
   * Obtain from the FHE-GBDT dashboard.
   */
  apiKey: string;

  /**
   * Service endpoint URL.
   * @default "https://api.fhe-gbdt.dev"
   */
  endpoint?: string;

  /**
   * Request timeout in milliseconds.
   * @default 30000 (30 seconds)
   */
  timeout?: number;

  /**
   * Maximum number of retry attempts for transient failures.
   * @default 3
   */
  maxRetries?: number;

  /**
   * Base delay in milliseconds for exponential backoff.
   * @default 1000 (1 second)
   */
  retryBaseDelay?: number;

  /**
   * Maximum delay in milliseconds for exponential backoff.
   * @default 30000 (30 seconds)
   */
  retryMaxDelay?: number;

  /**
   * Tenant ID for multi-tenant deployments.
   * If not provided, will be derived from the API key.
   */
  tenantId?: string;

  /**
   * Enable debug logging.
   * @default false
   */
  debug?: boolean;

  /**
   * Custom gRPC channel options.
   */
  grpcOptions?: GrpcChannelOptions;
}

/**
 * gRPC channel configuration options.
 */
export interface GrpcChannelOptions {
  /**
   * Enable TLS/SSL for the connection.
   * @default true
   */
  secure?: boolean;

  /**
   * Path to custom CA certificate file.
   */
  caCertPath?: string;

  /**
   * Path to client certificate file for mTLS.
   */
  clientCertPath?: string;

  /**
   * Path to client private key file for mTLS.
   */
  clientKeyPath?: string;

  /**
   * Maximum message size in bytes for receiving.
   * @default 104857600 (100MB)
   */
  maxReceiveMessageSize?: number;

  /**
   * Maximum message size in bytes for sending.
   * @default 104857600 (100MB)
   */
  maxSendMessageSize?: number;

  /**
   * Keep-alive time in milliseconds.
   * @default 30000 (30 seconds)
   */
  keepAliveTimeMs?: number;
}

// ============================================================================
// Model Library Types
// ============================================================================

/**
 * Supported GBDT library types for model registration.
 */
export type LibraryType = 'xgboost' | 'lightgbm' | 'catboost';

/**
 * Optimization profile for model compilation and inference.
 */
export type OptimizationProfile = 'latency' | 'throughput';

/**
 * Status of a model compilation job.
 */
export type CompileStatus =
  | 'pending'
  | 'compiling'
  | 'optimizing'
  | 'completed'
  | 'failed';

// ============================================================================
// Control Service Types (Model Registration & Compilation)
// ============================================================================

/**
 * Request payload for registering a new model.
 */
export interface RegisterModelRequest {
  /**
   * Human-readable name for the model.
   * Must be unique within the tenant.
   */
  modelName: string;

  /**
   * Binary content of the model file.
   * Supported formats: XGBoost JSON/binary, LightGBM text/binary, CatBoost binary.
   */
  modelContent: Buffer | Uint8Array;

  /**
   * The GBDT library that produced this model.
   */
  libraryType: LibraryType;

  /**
   * Optional metadata to attach to the model.
   */
  metadata?: Record<string, string>;

  /**
   * Optional tags for organizing models.
   */
  tags?: string[];
}

/**
 * Response from a successful model registration.
 */
export interface RegisterModelResponse {
  /**
   * Unique identifier for the registered model.
   * Use this ID for compilation and management operations.
   */
  modelId: string;

  /**
   * Timestamp when the model was registered.
   */
  createdAt?: Date;
}

/**
 * Request payload for compiling a registered model for FHE inference.
 */
export interface CompileModelRequest {
  /**
   * The model ID returned from registration.
   */
  modelId: string;

  /**
   * Optimization profile for the compiled model.
   * - 'latency': Optimize for fastest single-prediction response time
   * - 'throughput': Optimize for maximum predictions per second
   */
  profile: OptimizationProfile;

  /**
   * Optional FHE scheme parameters override.
   */
  schemeParams?: FHESchemeParams;

  /**
   * Feature names to include in compilation.
   * If not provided, all features from the model are used.
   */
  featureNames?: string[];
}

/**
 * FHE scheme parameters for compilation.
 */
export interface FHESchemeParams {
  /**
   * Ring dimension (N). Must be a power of 2.
   * Higher values provide more security but slower computation.
   * @default 4096
   */
  ringDimension?: number;

  /**
   * Ciphertext modulus bit size.
   * @default 128
   */
  modulusBitSize?: number;

  /**
   * Plaintext scaling factor for fixed-point encoding.
   * @default 40
   */
  scalingFactor?: number;
}

/**
 * Response from a model compilation request.
 */
export interface CompileModelResponse {
  /**
   * Unique identifier for the compiled model.
   * Use this ID for inference requests.
   */
  compiledModelId: string;

  /**
   * Estimated time to completion in seconds.
   */
  estimatedTimeSeconds?: number;
}

/**
 * Request to check compilation status.
 */
export interface GetCompileStatusRequest {
  /**
   * The compiled model ID to check.
   */
  compiledModelId: string;
}

/**
 * Response with compilation status details.
 */
export interface GetCompileStatusResponse {
  /**
   * Current status of the compilation.
   */
  status: CompileStatus;

  /**
   * Unique identifier for the computation plan (available when completed).
   */
  planId?: string;

  /**
   * Error message if compilation failed.
   */
  errorMessage?: string;

  /**
   * Progress percentage (0-100).
   */
  progress?: number;

  /**
   * Compilation metrics (available when completed).
   */
  metrics?: CompilationMetrics;
}

/**
 * Metrics from a completed compilation.
 */
export interface CompilationMetrics {
  /**
   * Total compilation time in milliseconds.
   */
  compilationTimeMs: number;

  /**
   * Number of comparison operations in the circuit.
   */
  comparisonCount: number;

  /**
   * Number of tree nodes compiled.
   */
  nodeCount: number;

  /**
   * Estimated depth of the FHE circuit.
   */
  circuitDepth: number;

  /**
   * Estimated memory usage for inference in bytes.
   */
  estimatedMemoryBytes: number;
}

// ============================================================================
// Inference Service Types
// ============================================================================

/**
 * Request payload for encrypted prediction.
 */
export interface PredictRequest {
  /**
   * The compiled model ID to use for prediction.
   */
  compiledModelId: string;

  /**
   * Optimization profile for this request.
   * Should match the profile used during compilation.
   */
  profile: OptimizationProfile;

  /**
   * Encrypted feature batch for prediction.
   */
  batch: CiphertextBatch;

  /**
   * Optional request metadata.
   */
  metadata?: Record<string, string>;

  /**
   * Request priority (higher values = higher priority).
   * @default 0
   */
  priority?: number;
}

/**
 * A batch of encrypted feature values.
 */
export interface CiphertextBatch {
  /**
   * Identifier for the FHE scheme used for encryption.
   * @default "n2he_default"
   */
  schemeId: string;

  /**
   * Number of samples in this batch.
   */
  batchSize: number;

  /**
   * Identifier for the packing layout.
   * Describes how values are arranged in ciphertext slots.
   */
  packingLayoutId?: string;

  /**
   * The encrypted payload bytes.
   */
  payload: Buffer | Uint8Array;

  /**
   * Names of features in this batch, in order.
   */
  featureNames: string[];
}

/**
 * Response from an encrypted prediction request.
 */
export interface PredictResponse {
  /**
   * Encrypted prediction outputs.
   */
  outputs: CiphertextBatch;

  /**
   * Runtime statistics for this prediction.
   */
  stats: RuntimeStats;

  /**
   * Request ID for tracking and debugging.
   */
  requestId?: string;
}

/**
 * Runtime statistics from prediction execution.
 */
export interface RuntimeStats {
  /**
   * Number of comparison operations executed.
   */
  comparisons: number;

  /**
   * Number of FHE scheme switches performed.
   */
  schemeSwitches: number;

  /**
   * Number of bootstrapping operations performed.
   */
  bootstraps: number;

  /**
   * Number of rotation operations performed.
   */
  rotations: number;

  /**
   * Total runtime in milliseconds.
   */
  runtimeMs: number;

  /**
   * Server-side queue wait time in milliseconds.
   */
  queueTimeMs?: number;

  /**
   * Time spent in computation in milliseconds.
   */
  computeTimeMs?: number;
}

// ============================================================================
// Crypto Key Service Types
// ============================================================================

/**
 * Request to upload evaluation keys to the server.
 */
export interface UploadEvalKeysRequest {
  /**
   * The compiled model ID these keys are for.
   */
  compiledModelId: string;

  /**
   * Serialized evaluation keys.
   */
  evalKeys: Buffer | Uint8Array;

  /**
   * Key format identifier.
   * @default "n2he_v2"
   */
  keyFormat?: string;
}

/**
 * Response from evaluation key upload.
 */
export interface UploadEvalKeysResponse {
  /**
   * Whether the upload was successful.
   */
  success: boolean;

  /**
   * Unique identifier for the uploaded keys.
   */
  keyId?: string;

  /**
   * Expiration time for the uploaded keys.
   */
  expiresAt?: Date;
}

/**
 * Request to rotate evaluation keys.
 */
export interface RotateKeysRequest {
  /**
   * The model ID to rotate keys for.
   */
  modelId: string;
}

/**
 * Response from key rotation.
 */
export interface RotateKeysResponse {
  /**
   * Whether the rotation was successful.
   */
  success: boolean;

  /**
   * New key ID after rotation.
   */
  newKeyId?: string;
}

/**
 * Request to revoke evaluation keys.
 */
export interface RevokeKeysRequest {
  /**
   * The model ID to revoke keys for.
   */
  modelId: string;
}

/**
 * Response from key revocation.
 */
export interface RevokeKeysResponse {
  /**
   * Whether the revocation was successful.
   */
  success: boolean;
}

// ============================================================================
// Billing and Subscription Types
// ============================================================================

/**
 * Available subscription plans.
 */
export type PlanTier = 'free' | 'starter' | 'professional' | 'enterprise';

/**
 * Billing period for subscriptions.
 */
export type BillingPeriod = 'monthly' | 'yearly';

/**
 * Subscription information.
 */
export interface Subscription {
  /**
   * Unique subscription identifier.
   */
  subscriptionId: string;

  /**
   * Current plan tier.
   */
  plan: PlanTier;

  /**
   * Current billing period.
   */
  billingPeriod: BillingPeriod;

  /**
   * Subscription status.
   */
  status: 'active' | 'past_due' | 'canceled' | 'trialing';

  /**
   * When the current period started.
   */
  currentPeriodStart: Date;

  /**
   * When the current period ends.
   */
  currentPeriodEnd: Date;

  /**
   * Whether the subscription will renew.
   */
  cancelAtPeriodEnd: boolean;
}

/**
 * Plan details and limits.
 */
export interface Plan {
  /**
   * Plan tier identifier.
   */
  tier: PlanTier;

  /**
   * Human-readable plan name.
   */
  name: string;

  /**
   * Monthly price in cents (USD).
   */
  monthlyPriceCents: number;

  /**
   * Yearly price in cents (USD).
   */
  yearlyPriceCents: number;

  /**
   * Plan feature limits.
   */
  limits: PlanLimits;

  /**
   * List of included features.
   */
  features: string[];
}

/**
 * Resource limits for a plan.
 */
export interface PlanLimits {
  /**
   * Maximum predictions per month.
   */
  predictionsPerMonth: number;

  /**
   * Maximum registered models.
   */
  maxModels: number;

  /**
   * Maximum compiled models.
   */
  maxCompiledModels: number;

  /**
   * Maximum model size in bytes.
   */
  maxModelSizeBytes: number;

  /**
   * Maximum batch size for predictions.
   */
  maxBatchSize: number;

  /**
   * Maximum concurrent requests.
   */
  maxConcurrentRequests: number;

  /**
   * Evaluation key retention period in days.
   */
  keyRetentionDays: number;

  /**
   * Whether priority support is included.
   */
  prioritySupport: boolean;

  /**
   * Whether SLA guarantees are included.
   */
  slaGuarantee: boolean;
}

/**
 * Current usage statistics.
 */
export interface Usage {
  /**
   * Current billing period start.
   */
  periodStart: Date;

  /**
   * Current billing period end.
   */
  periodEnd: Date;

  /**
   * Number of predictions made this period.
   */
  predictionsUsed: number;

  /**
   * Prediction limit for this period.
   */
  predictionsLimit: number;

  /**
   * Number of registered models.
   */
  modelsUsed: number;

  /**
   * Model limit.
   */
  modelsLimit: number;

  /**
   * Number of compiled models.
   */
  compiledModelsUsed: number;

  /**
   * Compiled model limit.
   */
  compiledModelsLimit: number;

  /**
   * Total storage used in bytes.
   */
  storageUsedBytes: number;

  /**
   * Storage limit in bytes.
   */
  storageLimitBytes: number;

  /**
   * Compute time used in milliseconds.
   */
  computeTimeUsedMs: number;
}

/**
 * Usage breakdown by model.
 */
export interface ModelUsage {
  /**
   * Model identifier.
   */
  modelId: string;

  /**
   * Model name.
   */
  modelName: string;

  /**
   * Number of predictions for this model.
   */
  predictions: number;

  /**
   * Compute time used for this model in milliseconds.
   */
  computeTimeMs: number;

  /**
   * Last prediction timestamp.
   */
  lastUsedAt?: Date;
}

// ============================================================================
// Model Management Types
// ============================================================================

/**
 * Model information.
 */
export interface Model {
  /**
   * Unique model identifier.
   */
  modelId: string;

  /**
   * Human-readable model name.
   */
  modelName: string;

  /**
   * GBDT library type.
   */
  libraryType: LibraryType;

  /**
   * Model file size in bytes.
   */
  sizeBytes: number;

  /**
   * Number of trees in the model.
   */
  treeCount?: number;

  /**
   * Number of features.
   */
  featureCount?: number;

  /**
   * Feature names.
   */
  featureNames?: string[];

  /**
   * Model metadata.
   */
  metadata?: Record<string, string>;

  /**
   * Model tags.
   */
  tags?: string[];

  /**
   * When the model was registered.
   */
  createdAt: Date;

  /**
   * When the model was last updated.
   */
  updatedAt: Date;
}

/**
 * Compiled model information.
 */
export interface CompiledModel {
  /**
   * Unique compiled model identifier.
   */
  compiledModelId: string;

  /**
   * Source model identifier.
   */
  modelId: string;

  /**
   * Compilation profile used.
   */
  profile: OptimizationProfile;

  /**
   * Compilation status.
   */
  status: CompileStatus;

  /**
   * Computation plan identifier.
   */
  planId?: string;

  /**
   * Compilation metrics.
   */
  metrics?: CompilationMetrics;

  /**
   * When the model was compiled.
   */
  createdAt: Date;

  /**
   * When the compilation completed.
   */
  completedAt?: Date;
}

// ============================================================================
// Utility Types
// ============================================================================

/**
 * Pagination options for list operations.
 */
export interface PaginationOptions {
  /**
   * Maximum number of items to return.
   * @default 20
   */
  limit?: number;

  /**
   * Cursor for pagination.
   */
  cursor?: string;
}

/**
 * Paginated response wrapper.
 */
export interface PaginatedResponse<T> {
  /**
   * The items in this page.
   */
  items: T[];

  /**
   * Cursor for the next page, if available.
   */
  nextCursor?: string;

  /**
   * Whether there are more items.
   */
  hasMore: boolean;

  /**
   * Total count of items (if available).
   */
  totalCount?: number;
}

/**
 * Health check response.
 */
export interface HealthStatus {
  /**
   * Overall service status.
   */
  status: 'healthy' | 'degraded' | 'unhealthy';

  /**
   * Individual component statuses.
   */
  components: Record<string, ComponentHealth>;

  /**
   * Server timestamp.
   */
  timestamp: Date;

  /**
   * Server version.
   */
  version: string;
}

/**
 * Individual component health.
 */
export interface ComponentHealth {
  /**
   * Component status.
   */
  status: 'healthy' | 'degraded' | 'unhealthy';

  /**
   * Optional status message.
   */
  message?: string;

  /**
   * Last check timestamp.
   */
  lastCheck: Date;
}

/**
 * Generic API error response.
 */
export interface ApiErrorResponse {
  /**
   * Error code.
   */
  code: string;

  /**
   * Human-readable error message.
   */
  message: string;

  /**
   * Additional error details.
   */
  details?: Record<string, unknown>;

  /**
   * Request ID for support.
   */
  requestId?: string;
}

// ============================================================================
// SDK Client Configuration (Simplified)
// ============================================================================

/**
 * Configuration for FHE-GBDT SDK Client
 */
export interface FHEGBDTConfig {
  apiKey: string;
  endpoint: string;
  tenantId?: string;
  timeout: number;
  retries: number;
  retryDelay: number;
  useTLS?: boolean;
  controlEndpoint?: string;
  billingEndpoint?: string;
}

/**
 * Key pair for FHE operations
 */
export interface KeyPair {
  secretKey: Buffer;
  publicKey: Buffer;
  evalKeys: Buffer;
}

/**
 * Simple predict request for SDK
 */
export interface PredictRequest {
  compiledModelId: string;
  encryptedPayload: Buffer;
}

/**
 * Simple predict response for SDK
 */
export interface PredictResponse {
  encryptedResult: Buffer;
  latencyMs: number;
  requestId?: string;
}
