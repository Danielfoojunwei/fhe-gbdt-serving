/**
 * FHE-GBDT TypeScript SDK Type Definitions
 * Aligned with TenSafe SDK types
 */

// ============== Enums ==============

/**
 * Status of an asynchronous operation
 */
export enum FutureStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

/**
 * Types of operations that can be performed
 */
export enum OperationType {
  PREDICT = 'predict',
  BATCH_PREDICT = 'batch_predict',
  COMPILE = 'compile',
  TRAIN = 'train',
  PACKAGE = 'package',
}

/**
 * GBDT library types
 */
export enum GBDTLibrary {
  XGBOOST = 'xgboost',
  LIGHTGBM = 'lightgbm',
  CATBOOST = 'catboost',
}

/**
 * Execution profile for inference
 */
export enum ExecutionProfile {
  LATENCY = 'latency',
  THROUGHPUT = 'throughput',
}

/**
 * Model status
 */
export enum ModelStatus {
  REGISTERED = 'registered',
  COMPILING = 'compiling',
  READY = 'ready',
  FAILED = 'failed',
}

/**
 * Training job status
 */
export enum TrainingStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

/**
 * DP accountant type
 */
export enum DPAccountantType {
  RDP = 'rdp',
  PRV = 'prv',
  GDP = 'gdp',
}

// ============== Configuration Interfaces ==============

/**
 * Client configuration options
 */
export interface FHEGBDTClientOptions {
  /** Base URL of the API */
  baseUrl?: string;
  /** API key for authentication */
  apiKey?: string;
  /** Tenant ID */
  tenantId?: string;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Maximum number of retries */
  maxRetries?: number;
  /** Polling interval for async operations (ms) */
  pollingInterval?: number;
}

/**
 * Request options
 */
export interface RequestOptions {
  /** Request timeout override */
  timeout?: number;
  /** Custom headers */
  headers?: Record<string, string>;
  /** Request ID for tracing */
  requestId?: string;
  /** Signal for cancellation */
  signal?: AbortSignal;
}

/**
 * Crypto parameters for FHE
 */
export interface CryptoParams {
  /** Ring dimension (2048, 4096, 8192) */
  ringDimension?: number;
  /** Ciphertext modulus */
  ciphertextModulus?: string;
  /** Security level (standard, high, maximum) */
  securityLevel?: 'standard' | 'high' | 'maximum';
}

/**
 * Differential privacy configuration
 */
export interface DPConfig {
  /** Enable differential privacy */
  enabled: boolean;
  /** Epsilon budget */
  epsilon: number;
  /** Delta budget */
  delta: number;
  /** Noise type (laplace, gaussian) */
  noiseType?: 'laplace' | 'gaussian';
  /** Maximum gradient norm for clipping */
  maxGradNorm?: number;
}

/**
 * Training configuration
 */
export interface TrainingConfig {
  /** Training job name */
  name: string;
  /** Dataset path or URL */
  datasetPath: string;
  /** GBDT library to use */
  library: GBDTLibrary;
  /** Model hyperparameters */
  hyperparameters: TrainingHyperparameters;
  /** Differential privacy config */
  dpConfig?: DPConfig;
  /** Output path for model */
  outputPath?: string;
}

/**
 * Training hyperparameters
 */
export interface TrainingHyperparameters {
  /** Number of trees */
  nEstimators?: number;
  /** Maximum tree depth */
  maxDepth?: number;
  /** Learning rate */
  learningRate?: number;
  /** Subsample ratio */
  subsample?: number;
  /** Column sample ratio */
  colsampleBytree?: number;
  /** L1 regularization */
  regAlpha?: number;
  /** L2 regularization */
  regLambda?: number;
  /** Minimum child weight */
  minChildWeight?: number;
  /** Custom parameters */
  [key: string]: number | string | boolean | undefined;
}

/**
 * Compilation options
 */
export interface CompileOptions {
  /** Execution profile */
  profile?: ExecutionProfile;
  /** Crypto parameters */
  cryptoParams?: CryptoParams;
  /** Additional options */
  options?: Record<string, string>;
}

// ============== Request/Response Types ==============

/**
 * Prediction request
 */
export interface PredictRequest {
  /** Model ID */
  modelId?: string;
  /** Compiled model ID */
  compiledModelId?: string;
  /** Base64 encoded ciphertext */
  ciphertext: string;
  /** Ciphertext format */
  ciphertextFormat?: string;
  /** Execution profile */
  profile?: ExecutionProfile;
  /** Request metadata */
  metadata?: Record<string, string>;
}

/**
 * Prediction response
 */
export interface PredictResponse {
  /** Request ID */
  requestId: string;
  /** Model ID used */
  modelId: string;
  /** Base64 encoded result ciphertext */
  ciphertext: string;
  /** Runtime statistics */
  stats?: RuntimeStats;
  /** Processing timestamp */
  processedAt: string;
}

/**
 * Runtime statistics
 */
export interface RuntimeStats {
  /** Number of comparisons */
  comparisons: number;
  /** Number of scheme switches */
  schemeSwitches: number;
  /** Number of bootstraps */
  bootstraps: number;
  /** Number of rotations */
  rotations: number;
  /** Runtime in milliseconds */
  runtimeMs: number;
}

/**
 * Batch prediction request
 */
export interface BatchPredictRequest {
  /** Model ID */
  modelId?: string;
  /** Compiled model ID */
  compiledModelId?: string;
  /** Array of base64 encoded ciphertexts */
  ciphertexts: string[];
  /** Execution profile */
  profile?: ExecutionProfile;
}

/**
 * Batch prediction response
 */
export interface BatchPredictResponse {
  /** Request ID */
  requestId: string;
  /** Array of result ciphertexts */
  results: Array<{
    ciphertext: string;
    stats?: RuntimeStats;
  }>;
  /** Aggregate statistics */
  aggregateStats?: {
    totalRuntimeMs: number;
    avgRuntimeMs: number;
  };
}

/**
 * Model registration request
 */
export interface RegisterModelRequest {
  /** Model name */
  name: string;
  /** GBDT library */
  library: GBDTLibrary;
  /** Base64 encoded model */
  model: string;
  /** Model metadata */
  metadata?: Record<string, string>;
}

/**
 * Model information
 */
export interface Model {
  /** Model ID */
  id: string;
  /** Model name */
  name: string;
  /** Tenant ID */
  tenantId: string;
  /** GBDT library */
  library: GBDTLibrary;
  /** Model status */
  status: ModelStatus;
  /** Compiled model ID (if compiled) */
  compiledModelId?: string;
  /** Model metadata */
  metadata?: Record<string, string>;
  /** Creation timestamp */
  createdAt: string;
  /** Last update timestamp */
  updatedAt: string;
}

/**
 * Training job information
 */
export interface TrainingJob {
  /** Job ID */
  id: string;
  /** Job name */
  name: string;
  /** Job status */
  status: TrainingStatus;
  /** Progress percentage (0-100) */
  progress: number;
  /** Training metrics */
  metrics?: TrainingMetrics;
  /** Privacy spent (if DP enabled) */
  dpSpent?: DPSpent;
  /** Start timestamp */
  startedAt: string;
  /** Completion timestamp */
  completedAt?: string;
  /** Error message (if failed) */
  error?: string;
}

/**
 * Training metrics
 */
export interface TrainingMetrics {
  /** Training loss */
  trainLoss?: number;
  /** Validation loss */
  valLoss?: number;
  /** Training AUC */
  trainAuc?: number;
  /** Validation AUC */
  valAuc?: number;
  /** Custom metrics */
  [key: string]: number | undefined;
}

/**
 * Differential privacy spent
 */
export interface DPSpent {
  /** Epsilon spent */
  epsilon: number;
  /** Delta spent */
  delta: number;
}

/**
 * Key information
 */
export interface KeyInfo {
  /** Key ID */
  keyId: string;
  /** Key status */
  status: 'active' | 'expired' | 'revoked';
  /** Expiration timestamp */
  expiresAt: string;
  /** Creation timestamp */
  createdAt: string;
}

/**
 * GBSP Package information
 */
export interface Package {
  /** Package ID */
  id: string;
  /** Model ID */
  modelId: string;
  /** Package status */
  status: 'creating' | 'ready' | 'failed';
  /** Download URL (when ready) */
  downloadUrl?: string;
  /** Package hash */
  hash?: string;
  /** Creation timestamp */
  createdAt: string;
}

/**
 * Package verification result
 */
export interface PackageVerification {
  /** Package ID */
  packageId: string;
  /** Overall validity */
  valid: boolean;
  /** Individual check results */
  checks: {
    signature: boolean;
    integrity: boolean;
    policy: boolean;
    dpCertificate: boolean;
  };
}

/**
 * Audit log entry
 */
export interface AuditLogEntry {
  /** Entry ID */
  entryId: string;
  /** Timestamp */
  timestamp: string;
  /** Action performed */
  action: string;
  /** Tenant ID */
  tenantId: string;
  /** Request ID */
  requestId?: string;
  /** Resource ID */
  resourceId?: string;
  /** Status */
  status: string;
  /** Current hash (chain integrity) */
  currentHash: string;
  /** Previous hash (chain integrity) */
  previousHash: string;
}

// ============== Future/Async Types ==============

/**
 * Future response for async operations
 */
export interface FutureResponse<T> {
  /** Operation ID */
  id: string;
  /** Operation type */
  type: OperationType;
  /** Current status */
  status: FutureStatus;
  /** Progress percentage */
  progress?: number;
  /** Result (when completed) */
  result?: T;
  /** Error message (when failed) */
  error?: string;
  /** Creation timestamp */
  createdAt: string;
  /** Completion timestamp */
  completedAt?: string;
}

// ============== Error Types ==============

/**
 * API error response
 */
export interface APIError {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Additional details */
  details?: Record<string, string>;
}

/**
 * SDK error class
 */
export class FHEGBDTError extends Error {
  public readonly code: string;
  public readonly details?: Record<string, string>;

  constructor(message: string, code: string = 'UNKNOWN', details?: Record<string, string>) {
    super(message);
    this.name = 'FHEGBDTError';
    this.code = code;
    this.details = details;
  }
}

/**
 * Authentication error
 */
export class AuthenticationError extends FHEGBDTError {
  constructor(message: string = 'Authentication failed') {
    super(message, 'UNAUTHORIZED');
    this.name = 'AuthenticationError';
  }
}

/**
 * Rate limit error
 */
export class RateLimitError extends FHEGBDTError {
  public readonly retryAfter?: number;

  constructor(message: string = 'Rate limit exceeded', retryAfter?: number) {
    super(message, 'RATE_LIMITED');
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

/**
 * Timeout error
 */
export class TimeoutError extends FHEGBDTError {
  constructor(message: string = 'Request timed out') {
    super(message, 'TIMEOUT');
    this.name = 'TimeoutError';
  }
}

/**
 * Future timeout error
 */
export class FutureTimeoutError extends FHEGBDTError {
  constructor(operationId: string) {
    super(`Operation ${operationId} timed out`, 'FUTURE_TIMEOUT');
    this.name = 'FutureTimeoutError';
  }
}
