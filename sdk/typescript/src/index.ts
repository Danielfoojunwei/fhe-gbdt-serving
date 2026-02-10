/**
 * FHE-GBDT TypeScript SDK
 *
 * Privacy-preserving GBDT inference using Fully Homomorphic Encryption.
 *
 * Part of the unified Product Suite with TenSafe, sharing authentication
 * and identity management for seamless cross-product access.
 *
 * @example
 * ```typescript
 * import { FHEGBDTClient, UnifiedAuthClient, GBDTLibrary } from '@fhe-gbdt/sdk';
 *
 * // Unified auth - works across FHE-GBDT and TenSafe
 * const auth = new UnifiedAuthClient();
 * await auth.login('user@example.com', 'password');
 *
 * // FHE-GBDT client with unified auth
 * const client = new FHEGBDTClient({
 *   apiKey: process.env.FHE_GBDT_API_KEY,
 * });
 *
 * // Register a model
 * const model = await client.registerModel({
 *   name: 'my-classifier',
 *   library: GBDTLibrary.XGBOOST,
 *   model: base64EncodedModel,
 * });
 *
 * // Compile for FHE
 * await client.compileModelAndWait(model.id);
 *
 * // Run encrypted prediction
 * const result = await client.predict({
 *   modelId: model.id,
 *   ciphertext: encryptedFeatures,
 * });
 * ```
 *
 * @packageDocumentation
 */

// Client
export { FHEGBDTClient, createClient, VERSION, API_VERSION } from './client';
export type { default as FHEGBDTClientDefault } from './client';

// Unified Auth (Product Suite SSO)
export {
  // Auth Client
  UnifiedAuthClient,
  AuthClient,

  // Products & Permissions
  Product,
  GBDTPermission,
  TenSafePermission,
  PlatformPermission,
  ROLE_PERMISSIONS,

  // Auth Types
  TokenClaims,
  User,
  Organization,
  APIKeyMetadata,
  LoginResponse,
  Session,
  UnifiedAuthClientConfig,
  TokenStorage,

  // Storage Implementations
  MemoryStorage,
  LocalStorage,

  // Auth Errors
  AuthError,
  TokenExpiredError,
  InvalidCredentialsError,
  PermissionDeniedError,

  // Product Client Factory
  ProductClientConfig,
  PRODUCT_URLS,
  createProductFetch,
} from './auth';
export type { AuthType } from './auth';

// Types
export {
  // Enums
  FutureStatus,
  OperationType,
  GBDTLibrary,
  ExecutionProfile,
  ModelStatus,
  TrainingStatus,
  DPAccountantType,

  // Configuration
  FHEGBDTClientOptions,
  RequestOptions,
  CryptoParams,
  DPConfig,
  TrainingConfig,
  TrainingHyperparameters,
  CompileOptions,

  // Request/Response
  PredictRequest,
  PredictResponse,
  BatchPredictRequest,
  BatchPredictResponse,
  RegisterModelRequest,
  RuntimeStats,

  // Entities
  Model,
  TrainingJob,
  TrainingMetrics,
  DPSpent,
  KeyInfo,
  Package,
  PackageVerification,
  AuditLogEntry,

  // Async
  FutureResponse,

  // Errors
  APIError,
  FHEGBDTError,
  AuthenticationError,
  RateLimitError,
  TimeoutError,
  FutureTimeoutError,
} from './types';
