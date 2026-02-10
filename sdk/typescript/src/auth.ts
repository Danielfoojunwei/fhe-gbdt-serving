/**
 * Unified Authentication Client for Product Suite
 *
 * This module provides SSO authentication that works across all products
 * in the suite (FHE-GBDT, TenSafe, etc.).
 *
 * @example
 * ```typescript
 * import { UnifiedAuthClient, Product } from '@fhe-gbdt/sdk';
 *
 * // Create auth client
 * const auth = new UnifiedAuthClient();
 *
 * // Login once
 * const session = await auth.login('user@example.com', 'password');
 *
 * // Use with FHE-GBDT
 * const gbdtClient = new FHEGBDTClient({
 *   authClient: auth,
 * });
 *
 * // Same auth works with TenSafe
 * const tensafeClient = new TenSafeClient({
 *   authClient: auth,
 * });
 * ```
 */

// =============================================================================
// Products and Permissions
// =============================================================================

/**
 * Products in the suite that share unified authentication
 */
export enum Product {
  FHE_GBDT = 'fhe_gbdt',
  TENSAFE = 'tensafe',
  PLATFORM = 'platform',
}

/**
 * Authentication type
 */
export type AuthType = 'user' | 'api_key';

/**
 * Permissions for FHE-GBDT product
 */
export enum GBDTPermission {
  PREDICT = 'gbdt:predict',
  TRAIN = 'gbdt:train',
  MODEL_UPLOAD = 'gbdt:model:upload',
  MODEL_DELETE = 'gbdt:model:delete',
  KEYS_MANAGE = 'gbdt:keys:manage',
}

/**
 * Permissions for TenSafe product
 */
export enum TenSafePermission {
  ADAPT = 'tensafe:adapt',
  INFERENCE = 'tensafe:inference',
  MODEL_UPLOAD = 'tensafe:model:upload',
  MODEL_DELETE = 'tensafe:model:delete',
  KEYS_MANAGE = 'tensafe:keys:manage',
}

/**
 * Platform-wide permissions
 */
export enum PlatformPermission {
  ADMIN = 'platform:admin',
  BILLING = 'platform:billing',
  AUDIT = 'platform:audit',
  ORG_MANAGE = 'org:manage',
  USER_MANAGE = 'user:manage',
}

/**
 * Common role definitions
 */
export const ROLE_PERMISSIONS: Record<string, string[]> = {
  viewer: [
    GBDTPermission.PREDICT,
    TenSafePermission.INFERENCE,
  ],
  developer: [
    GBDTPermission.PREDICT,
    GBDTPermission.TRAIN,
    GBDTPermission.MODEL_UPLOAD,
    TenSafePermission.ADAPT,
    TenSafePermission.INFERENCE,
    TenSafePermission.MODEL_UPLOAD,
  ],
  admin: [
    GBDTPermission.PREDICT,
    GBDTPermission.TRAIN,
    GBDTPermission.MODEL_UPLOAD,
    GBDTPermission.MODEL_DELETE,
    GBDTPermission.KEYS_MANAGE,
    TenSafePermission.ADAPT,
    TenSafePermission.INFERENCE,
    TenSafePermission.MODEL_UPLOAD,
    TenSafePermission.MODEL_DELETE,
    TenSafePermission.KEYS_MANAGE,
    PlatformPermission.ORG_MANAGE,
    PlatformPermission.USER_MANAGE,
  ],
};

// =============================================================================
// Data Types
// =============================================================================

/**
 * JWT token claims
 */
export interface TokenClaims {
  /** Subject (user_id or key_id) */
  sub: string;
  /** Issuer */
  iss: string;
  /** Audience (products) */
  aud: string[];
  /** Expiration timestamp */
  exp: number;
  /** Issued at timestamp */
  iat: number;
  /** JWT ID */
  jti: string;
  /** Organization ID */
  org_id: string;
  /** Authentication type */
  auth_type: AuthType;
  /** Permissions */
  permissions: string[];
  /** User email (for user auth) */
  email?: string;
  /** User name */
  name?: string;
  /** User role */
  role?: string;
}

/**
 * User information
 */
export interface User {
  user_id: string;
  org_id: string;
  email: string;
  name: string;
  role: string;
  permissions: string[];
  mfa_enabled: boolean;
  is_active: boolean;
  last_login_at?: string;
  created_at: string;
}

/**
 * Organization information
 */
export interface Organization {
  org_id: string;
  name: string;
  enabled_products: Product[];
  tier: 'free' | 'pro' | 'business' | 'enterprise';
  is_active: boolean;
  created_at: string;
}

/**
 * API key metadata
 */
export interface APIKeyMetadata {
  key_id: string;
  org_id: string;
  name: string;
  permissions: string[];
  created_at: string;
  expires_at?: string;
  last_used_at?: string;
  is_active: boolean;
}

/**
 * Login response
 */
export interface LoginResponse {
  access_token: string;
  refresh_token?: string;
  token_type: 'Bearer';
  expires_in: number;
  user: User;
}

/**
 * Session information
 */
export interface Session {
  accessToken: string;
  refreshToken?: string;
  expiresAt: Date;
  user: User;
}

/**
 * Auth client configuration
 */
export interface UnifiedAuthClientConfig {
  /** Identity provider URL */
  baseUrl?: string;
  /** Automatic token refresh */
  autoRefresh?: boolean;
  /** Storage for tokens */
  storage?: TokenStorage;
}

/**
 * Token storage interface
 */
export interface TokenStorage {
  get(key: string): string | null | Promise<string | null>;
  set(key: string, value: string): void | Promise<void>;
  remove(key: string): void | Promise<void>;
}

// =============================================================================
// Default Storage Implementations
// =============================================================================

/**
 * In-memory token storage
 */
export class MemoryStorage implements TokenStorage {
  private store: Map<string, string> = new Map();

  get(key: string): string | null {
    return this.store.get(key) ?? null;
  }

  set(key: string, value: string): void {
    this.store.set(key, value);
  }

  remove(key: string): void {
    this.store.delete(key);
  }
}

/**
 * Browser localStorage storage
 */
export class LocalStorage implements TokenStorage {
  private prefix: string;

  constructor(prefix: string = 'product_suite_') {
    this.prefix = prefix;
  }

  get(key: string): string | null {
    if (typeof localStorage === 'undefined') return null;
    return localStorage.getItem(this.prefix + key);
  }

  set(key: string, value: string): void {
    if (typeof localStorage === 'undefined') return;
    localStorage.setItem(this.prefix + key, value);
  }

  remove(key: string): void {
    if (typeof localStorage === 'undefined') return;
    localStorage.removeItem(this.prefix + key);
  }
}

// =============================================================================
// Auth Errors
// =============================================================================

/**
 * Authentication error
 */
export class AuthError extends Error {
  constructor(
    message: string,
    public readonly code: string = 'AUTH_ERROR',
    public readonly statusCode?: number
  ) {
    super(message);
    this.name = 'AuthError';
  }
}

/**
 * Token expired error
 */
export class TokenExpiredError extends AuthError {
  constructor() {
    super('Token has expired', 'TOKEN_EXPIRED', 401);
    this.name = 'TokenExpiredError';
  }
}

/**
 * Invalid credentials error
 */
export class InvalidCredentialsError extends AuthError {
  constructor() {
    super('Invalid email or password', 'INVALID_CREDENTIALS', 401);
    this.name = 'InvalidCredentialsError';
  }
}

/**
 * Permission denied error
 */
export class PermissionDeniedError extends AuthError {
  constructor(permission: string) {
    super(`Missing required permission: ${permission}`, 'PERMISSION_DENIED', 403);
    this.name = 'PermissionDeniedError';
  }
}

// =============================================================================
// Unified Auth Client
// =============================================================================

/**
 * Unified Authentication Client for the Product Suite
 *
 * Provides:
 * - User authentication with email/password
 * - API key authentication
 * - Token management and refresh
 * - SSO across all products
 */
export class UnifiedAuthClient {
  private baseUrl: string;
  private autoRefresh: boolean;
  private storage: TokenStorage;
  private session: Session | null = null;
  private refreshTimer: ReturnType<typeof setTimeout> | null = null;
  private apiKey: string | null = null;

  constructor(config: UnifiedAuthClientConfig = {}) {
    this.baseUrl = config.baseUrl || 'https://identity.product-suite.local';
    this.autoRefresh = config.autoRefresh ?? true;
    this.storage = config.storage || new MemoryStorage();
  }

  // =========================================================================
  // Authentication Methods
  // =========================================================================

  /**
   * Login with email and password
   */
  async login(email: string, password: string): Promise<Session> {
    const response = await this.fetch('/v1/auth/login', {
      method: 'POST',
      body: { email, password },
    });

    if (!response.ok) {
      if (response.status === 401) {
        throw new InvalidCredentialsError();
      }
      const error = await response.json().catch(() => ({}));
      throw new AuthError(error.message || 'Login failed', error.code, response.status);
    }

    const data: LoginResponse = await response.json();
    this.session = this.createSession(data);

    // Store tokens
    await this.storage.set('access_token', data.access_token);
    if (data.refresh_token) {
      await this.storage.set('refresh_token', data.refresh_token);
    }

    // Setup auto-refresh
    if (this.autoRefresh && data.refresh_token) {
      this.scheduleRefresh(data.expires_in);
    }

    return this.session;
  }

  /**
   * Authenticate with API key
   */
  setAPIKey(apiKey: string): void {
    this.apiKey = apiKey;
    this.session = null;
  }

  /**
   * Logout and clear session
   */
  async logout(): Promise<void> {
    if (this.session?.accessToken) {
      try {
        await this.fetch('/v1/auth/logout', {
          method: 'POST',
          headers: this.getAuthHeaders(),
        });
      } catch {
        // Ignore logout errors
      }
    }

    this.session = null;
    this.apiKey = null;
    this.clearRefreshTimer();

    await this.storage.remove('access_token');
    await this.storage.remove('refresh_token');
  }

  /**
   * Refresh the access token
   */
  async refreshToken(): Promise<Session> {
    const refreshToken = await this.storage.get('refresh_token');
    if (!refreshToken) {
      throw new AuthError('No refresh token available', 'NO_REFRESH_TOKEN');
    }

    const response = await this.fetch('/v1/auth/refresh', {
      method: 'POST',
      body: { refresh_token: refreshToken },
    });

    if (!response.ok) {
      // Clear invalid tokens
      await this.logout();
      throw new TokenExpiredError();
    }

    const data: LoginResponse = await response.json();
    this.session = this.createSession(data);

    await this.storage.set('access_token', data.access_token);
    if (data.refresh_token) {
      await this.storage.set('refresh_token', data.refresh_token);
    }

    if (this.autoRefresh && data.refresh_token) {
      this.scheduleRefresh(data.expires_in);
    }

    return this.session;
  }

  /**
   * Restore session from storage
   */
  async restoreSession(): Promise<Session | null> {
    const accessToken = await this.storage.get('access_token');
    if (!accessToken) {
      return null;
    }

    // Check if token is valid
    const claims = this.parseToken(accessToken);
    if (!claims) {
      await this.logout();
      return null;
    }

    // Check expiration
    if (claims.exp * 1000 < Date.now()) {
      // Try to refresh
      try {
        return await this.refreshToken();
      } catch {
        await this.logout();
        return null;
      }
    }

    // Restore session from token
    this.session = {
      accessToken,
      expiresAt: new Date(claims.exp * 1000),
      user: {
        user_id: claims.sub,
        org_id: claims.org_id,
        email: claims.email || '',
        name: claims.name || '',
        role: claims.role || 'developer',
        permissions: claims.permissions,
        mfa_enabled: false,
        is_active: true,
        created_at: new Date().toISOString(),
      },
    };

    // Setup refresh
    const expiresIn = Math.floor((claims.exp * 1000 - Date.now()) / 1000);
    if (this.autoRefresh && expiresIn > 60) {
      this.scheduleRefresh(expiresIn);
    }

    return this.session;
  }

  // =========================================================================
  // State Accessors
  // =========================================================================

  /**
   * Get current session
   */
  getSession(): Session | null {
    return this.session;
  }

  /**
   * Check if authenticated
   */
  isAuthenticated(): boolean {
    if (this.apiKey) return true;
    if (!this.session) return false;
    return this.session.expiresAt > new Date();
  }

  /**
   * Get current user
   */
  getUser(): User | null {
    return this.session?.user ?? null;
  }

  /**
   * Get organization ID
   */
  getOrgId(): string | null {
    if (this.apiKey) {
      // Extract org_id from API key format: org_id.secret
      return this.apiKey.split('.')[0] || null;
    }
    return this.session?.user.org_id ?? null;
  }

  /**
   * Get permissions from current auth
   */
  getPermissions(): string[] {
    if (this.session) {
      return this.session.user.permissions;
    }
    // For API keys, permissions are validated server-side
    return [];
  }

  /**
   * Check if user has a specific permission
   */
  hasPermission(permission: string): boolean {
    return this.getPermissions().includes(permission);
  }

  /**
   * Check if user has any of the given permissions
   */
  hasAnyPermission(permissions: string[]): boolean {
    const userPerms = this.getPermissions();
    return permissions.some((p) => userPerms.includes(p));
  }

  /**
   * Check if user has access to a product
   */
  hasProductAccess(product: Product): boolean {
    const prefix = `${product}:`;
    return this.getPermissions().some((p) => p.startsWith(prefix));
  }

  /**
   * Get authorization headers for API requests
   */
  getAuthHeaders(): Record<string, string> {
    if (this.apiKey) {
      return { 'X-API-Key': this.apiKey };
    }
    if (this.session?.accessToken) {
      return { Authorization: `Bearer ${this.session.accessToken}` };
    }
    return {};
  }

  // =========================================================================
  // User Management
  // =========================================================================

  /**
   * Get current user profile
   */
  async getProfile(): Promise<User> {
    const response = await this.fetch('/v1/users/me', {
      headers: this.getAuthHeaders(),
    });

    if (!response.ok) {
      throw new AuthError('Failed to get profile', 'PROFILE_ERROR', response.status);
    }

    return response.json();
  }

  /**
   * Update user profile
   */
  async updateProfile(updates: { name?: string }): Promise<User> {
    const response = await this.fetch('/v1/users/me', {
      method: 'PATCH',
      headers: this.getAuthHeaders(),
      body: updates,
    });

    if (!response.ok) {
      throw new AuthError('Failed to update profile', 'PROFILE_ERROR', response.status);
    }

    const user = await response.json();

    // Update session
    if (this.session) {
      this.session.user = user;
    }

    return user;
  }

  /**
   * Change password
   */
  async changePassword(oldPassword: string, newPassword: string): Promise<void> {
    const response = await this.fetch('/v1/users/me/password', {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: { old_password: oldPassword, new_password: newPassword },
    });

    if (!response.ok) {
      if (response.status === 400) {
        throw new AuthError('Invalid old password', 'INVALID_PASSWORD', 400);
      }
      throw new AuthError('Failed to change password', 'PASSWORD_ERROR', response.status);
    }
  }

  // =========================================================================
  // API Key Management
  // =========================================================================

  /**
   * List API keys for current organization
   */
  async listAPIKeys(): Promise<APIKeyMetadata[]> {
    const response = await this.fetch('/v1/api-keys', {
      headers: this.getAuthHeaders(),
    });

    if (!response.ok) {
      throw new AuthError('Failed to list API keys', 'API_KEY_ERROR', response.status);
    }

    return response.json();
  }

  /**
   * Create a new API key
   *
   * Note: The actual key value is only returned once
   */
  async createAPIKey(
    name: string,
    permissions?: string[],
    expiresInDays?: number
  ): Promise<{ metadata: APIKeyMetadata; key: string }> {
    const response = await this.fetch('/v1/api-keys', {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: {
        name,
        permissions,
        expires_in_days: expiresInDays,
      },
    });

    if (!response.ok) {
      throw new AuthError('Failed to create API key', 'API_KEY_ERROR', response.status);
    }

    return response.json();
  }

  /**
   * Revoke an API key
   */
  async revokeAPIKey(keyId: string): Promise<void> {
    const response = await this.fetch(`/v1/api-keys/${keyId}`, {
      method: 'DELETE',
      headers: this.getAuthHeaders(),
    });

    if (!response.ok) {
      throw new AuthError('Failed to revoke API key', 'API_KEY_ERROR', response.status);
    }
  }

  // =========================================================================
  // Organization Management
  // =========================================================================

  /**
   * Get current organization
   */
  async getOrganization(): Promise<Organization> {
    const response = await this.fetch('/v1/organizations/current', {
      headers: this.getAuthHeaders(),
    });

    if (!response.ok) {
      throw new AuthError('Failed to get organization', 'ORG_ERROR', response.status);
    }

    return response.json();
  }

  /**
   * List users in organization
   */
  async listOrganizationUsers(): Promise<User[]> {
    const response = await this.fetch('/v1/organizations/current/users', {
      headers: this.getAuthHeaders(),
    });

    if (!response.ok) {
      throw new AuthError('Failed to list users', 'ORG_ERROR', response.status);
    }

    return response.json();
  }

  /**
   * Invite user to organization
   */
  async inviteUser(email: string, role: string = 'developer'): Promise<void> {
    const response = await this.fetch('/v1/organizations/current/invitations', {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: { email, role },
    });

    if (!response.ok) {
      throw new AuthError('Failed to invite user', 'ORG_ERROR', response.status);
    }
  }

  // =========================================================================
  // Helper Methods
  // =========================================================================

  private async fetch(
    path: string,
    options: {
      method?: string;
      headers?: Record<string, string>;
      body?: unknown;
    } = {}
  ): Promise<Response> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    return fetch(url, {
      method: options.method || 'GET',
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
    });
  }

  private createSession(response: LoginResponse): Session {
    return {
      accessToken: response.access_token,
      refreshToken: response.refresh_token,
      expiresAt: new Date(Date.now() + response.expires_in * 1000),
      user: response.user,
    };
  }

  private parseToken(token: string): TokenClaims | null {
    try {
      const parts = token.split('.');
      if (parts.length !== 3) return null;
      const payload = JSON.parse(atob(parts[1]));
      return payload as TokenClaims;
    } catch {
      return null;
    }
  }

  private scheduleRefresh(expiresIn: number): void {
    this.clearRefreshTimer();

    // Refresh 5 minutes before expiration
    const refreshIn = Math.max(0, expiresIn - 300) * 1000;

    this.refreshTimer = setTimeout(async () => {
      try {
        await this.refreshToken();
      } catch (error) {
        console.error('Token refresh failed:', error);
      }
    }, refreshIn);
  }

  private clearRefreshTimer(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
      this.refreshTimer = null;
    }
  }
}

// =============================================================================
// Product Client Factory
// =============================================================================

/**
 * Configuration for creating product-specific clients
 */
export interface ProductClientConfig {
  product: Product;
  authClient: UnifiedAuthClient;
  baseUrl?: string;
}

/**
 * Product API URLs
 */
export const PRODUCT_URLS: Record<Product, string> = {
  [Product.FHE_GBDT]: 'https://api.fhe-gbdt.local',
  [Product.TENSAFE]: 'https://api.tensafe.local',
  [Product.PLATFORM]: 'https://api.platform.local',
};

/**
 * Create an authenticated fetch function for a product
 */
export function createProductFetch(config: ProductClientConfig) {
  const baseUrl = config.baseUrl || PRODUCT_URLS[config.product];

  return async function authenticatedFetch(
    path: string,
    options: RequestInit = {}
  ): Promise<Response> {
    // Check product access
    if (!config.authClient.hasProductAccess(config.product)) {
      throw new PermissionDeniedError(`Access to ${config.product}`);
    }

    const url = `${baseUrl}${path}`;
    const headers = new Headers(options.headers);

    // Add auth headers
    const authHeaders = config.authClient.getAuthHeaders();
    Object.entries(authHeaders).forEach(([key, value]) => {
      headers.set(key, value);
    });

    if (!headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json');
    }

    return fetch(url, {
      ...options,
      headers,
    });
  };
}

// =============================================================================
// Exports
// =============================================================================

export {
  UnifiedAuthClient as AuthClient,
};
