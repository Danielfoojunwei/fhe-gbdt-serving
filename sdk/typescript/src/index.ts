/**
 * FHE-GBDT TypeScript SDK
 *
 * Privacy-preserving GBDT inference using Fully Homomorphic Encryption.
 *
 * @example
 * ```typescript
 * import { FHEGBDTClient, GBDTLibrary } from '@fhe-gbdt/sdk';
 *
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
