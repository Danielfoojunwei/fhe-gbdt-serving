"""
Single Sign-On (SSO) Middleware for Product Suite

This module provides middleware components for integrating the unified identity
provider with web frameworks and gRPC services.

Supported frameworks:
- FastAPI/Starlette
- Flask
- gRPC (Python)
- Generic HTTP middleware

Products using this module will share authentication, enabling:
- Single account across FHE-GBDT and TenSafe
- Seamless product switching without re-authentication
- Unified API key management
- Cross-product authorization
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Optional, Callable, Any, List, Dict

from .provider import (
    UnifiedIdentityProvider,
    TokenClaims,
    Product,
    Permission,
    get_identity_provider,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Authentication Context
# =============================================================================

@dataclass
class AuthContext:
    """
    Authentication context passed to request handlers.

    This object contains all authentication information needed to authorize
    the current request across any product in the suite.
    """
    claims: TokenClaims
    org_id: str
    user_id: Optional[str]  # None for API key auth
    auth_type: str  # "user" or "api_key"
    permissions: List[str]

    # Product-specific helpers
    @property
    def is_user(self) -> bool:
        return self.auth_type == "user"

    @property
    def is_api_key(self) -> bool:
        return self.auth_type == "api_key"

    def has_permission(self, permission: str) -> bool:
        """Check if the authenticated entity has a specific permission."""
        return permission in self.permissions

    def has_any_permission(self, permissions: List[str]) -> bool:
        """Check if the authenticated entity has any of the given permissions."""
        return any(p in self.permissions for p in permissions)

    def has_all_permissions(self, permissions: List[str]) -> bool:
        """Check if the authenticated entity has all of the given permissions."""
        return all(p in self.permissions for p in permissions)

    def has_product_access(self, product: Product) -> bool:
        """Check if the authenticated entity has any access to a product."""
        prefix = f"{product.value}:"
        return any(p.startswith(prefix) for p in self.permissions)

    # FHE-GBDT specific helpers
    def can_gbdt_predict(self) -> bool:
        return self.has_permission(Permission.GBDT_PREDICT.value)

    def can_gbdt_train(self) -> bool:
        return self.has_permission(Permission.GBDT_TRAIN.value)

    def can_gbdt_manage_models(self) -> bool:
        return self.has_any_permission([
            Permission.GBDT_MODEL_UPLOAD.value,
            Permission.GBDT_MODEL_DELETE.value,
        ])

    # TenSafe specific helpers
    def can_tensafe_adapt(self) -> bool:
        return self.has_permission(Permission.TENSAFE_ADAPT.value)

    def can_tensafe_inference(self) -> bool:
        return self.has_permission(Permission.TENSAFE_INFERENCE.value)

    # Platform helpers
    def is_platform_admin(self) -> bool:
        return self.has_permission(Permission.PLATFORM_ADMIN.value)

    def can_manage_org(self) -> bool:
        return self.has_permission(Permission.ORG_MANAGE.value)

    def can_manage_users(self) -> bool:
        return self.has_permission(Permission.USER_MANAGE.value)


# =============================================================================
# Generic SSO Handler
# =============================================================================

class SSOHandler:
    """
    Generic SSO handler that can be used with any web framework.

    Usage:
        handler = SSOHandler(product=Product.FHE_GBDT)
        auth_context = handler.authenticate(
            authorization="Bearer <token>",
            api_key="<api-key>",
        )
    """

    def __init__(
        self,
        product: Product,
        identity_provider: Optional[UnifiedIdentityProvider] = None,
    ):
        self.product = product
        self.idp = identity_provider or get_identity_provider()

    def authenticate(
        self,
        authorization: Optional[str] = None,
        api_key: Optional[str] = None,
        required_permission: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Optional[AuthContext]:
        """
        Authenticate a request and return an AuthContext.

        Args:
            authorization: Authorization header value (e.g., "Bearer <token>")
            api_key: X-API-Key header value
            required_permission: Permission required for this request
            ip_address: Client IP address for API key validation

        Returns:
            AuthContext if authenticated, None otherwise
        """
        claims = self.idp.authenticate_request(
            authorization_header=authorization,
            api_key_header=api_key,
            required_audience=self.product.value,
            required_permission=required_permission,
            ip_address=ip_address,
        )

        if not claims:
            return None

        return AuthContext(
            claims=claims,
            org_id=claims.org_id,
            user_id=claims.sub if claims.auth_type == "user" else None,
            auth_type=claims.auth_type,
            permissions=claims.permissions,
        )


# =============================================================================
# FastAPI/Starlette Integration
# =============================================================================

try:
    from fastapi import Depends, HTTPException, Security, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


if HAS_FASTAPI:
    class FastAPISSOMiddleware:
        """
        FastAPI/Starlette SSO middleware.

        Usage:
            from fastapi import FastAPI, Depends
            from services.identity.sso import FastAPISSOMiddleware, AuthContext

            app = FastAPI()
            sso = FastAPISSOMiddleware(Product.FHE_GBDT)

            @app.get("/predict")
            async def predict(auth: AuthContext = Depends(sso.require_auth())):
                if not auth.can_gbdt_predict():
                    raise HTTPException(403, "Missing gbdt:predict permission")
                return {"org_id": auth.org_id}
        """

        def __init__(
            self,
            product: Product,
            identity_provider: Optional[UnifiedIdentityProvider] = None,
        ):
            self.handler = SSOHandler(product, identity_provider)
            self.bearer_scheme = HTTPBearer(auto_error=False)
            self.api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

        def require_auth(
            self,
            required_permission: Optional[str] = None,
        ) -> Callable:
            """
            Dependency that requires authentication.

            Args:
                required_permission: Permission required for this endpoint

            Returns:
                Dependency function
            """
            async def dependency(
                request: Request,
                bearer: Optional[HTTPAuthorizationCredentials] = Security(self.bearer_scheme),
                api_key: Optional[str] = Security(self.api_key_scheme),
            ) -> AuthContext:
                authorization = f"Bearer {bearer.credentials}" if bearer else None

                # Get client IP
                ip_address = request.client.host if request.client else None

                auth_context = self.handler.authenticate(
                    authorization=authorization,
                    api_key=api_key,
                    required_permission=required_permission,
                    ip_address=ip_address,
                )

                if not auth_context:
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid or missing authentication",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

                return auth_context

            return dependency

        def optional_auth(self) -> Callable:
            """
            Dependency that allows optional authentication.

            Returns:
                Dependency function that returns AuthContext or None
            """
            async def dependency(
                request: Request,
                bearer: Optional[HTTPAuthorizationCredentials] = Security(self.bearer_scheme),
                api_key: Optional[str] = Security(self.api_key_scheme),
            ) -> Optional[AuthContext]:
                authorization = f"Bearer {bearer.credentials}" if bearer else None
                ip_address = request.client.host if request.client else None

                return self.handler.authenticate(
                    authorization=authorization,
                    api_key=api_key,
                    ip_address=ip_address,
                )

            return dependency

        def require_permission(self, permission: str) -> Callable:
            """
            Dependency that requires a specific permission.

            Args:
                permission: Permission string (e.g., "gbdt:predict")

            Returns:
                Dependency function
            """
            return self.require_auth(required_permission=permission)

        def require_any_permission(self, permissions: List[str]) -> Callable:
            """
            Dependency that requires any of the given permissions.

            Args:
                permissions: List of permission strings

            Returns:
                Dependency function
            """
            async def dependency(
                request: Request,
                bearer: Optional[HTTPAuthorizationCredentials] = Security(self.bearer_scheme),
                api_key: Optional[str] = Security(self.api_key_scheme),
            ) -> AuthContext:
                authorization = f"Bearer {bearer.credentials}" if bearer else None
                ip_address = request.client.host if request.client else None

                auth_context = self.handler.authenticate(
                    authorization=authorization,
                    api_key=api_key,
                    ip_address=ip_address,
                )

                if not auth_context:
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid or missing authentication",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

                if not auth_context.has_any_permission(permissions):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Missing required permission. Need one of: {permissions}",
                    )

                return auth_context

            return dependency


# =============================================================================
# Flask Integration
# =============================================================================

try:
    from flask import request as flask_request, g as flask_g
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


if HAS_FLASK:
    class FlaskSSOMiddleware:
        """
        Flask SSO middleware.

        Usage:
            from flask import Flask
            from services.identity.sso import FlaskSSOMiddleware, AuthContext

            app = Flask(__name__)
            sso = FlaskSSOMiddleware(app, Product.FHE_GBDT)

            @app.route("/predict")
            @sso.require_auth()
            def predict():
                auth = sso.get_auth_context()
                if not auth.can_gbdt_predict():
                    return {"error": "Forbidden"}, 403
                return {"org_id": auth.org_id}
        """

        def __init__(
            self,
            app=None,
            product: Product = Product.FHE_GBDT,
            identity_provider: Optional[UnifiedIdentityProvider] = None,
        ):
            self.handler = SSOHandler(product, identity_provider)
            if app:
                self.init_app(app)

        def init_app(self, app):
            """Initialize with a Flask app."""
            @app.before_request
            def authenticate_request():
                authorization = flask_request.headers.get("Authorization")
                api_key = flask_request.headers.get("X-API-Key")
                ip_address = flask_request.remote_addr

                flask_g.auth_context = self.handler.authenticate(
                    authorization=authorization,
                    api_key=api_key,
                    ip_address=ip_address,
                )

        def get_auth_context(self) -> Optional[AuthContext]:
            """Get the current auth context."""
            return getattr(flask_g, 'auth_context', None)

        def require_auth(self, required_permission: Optional[str] = None):
            """
            Decorator that requires authentication.

            Args:
                required_permission: Permission required for this endpoint

            Returns:
                Decorator function
            """
            def decorator(f):
                @wraps(f)
                def decorated_function(*args, **kwargs):
                    auth = self.get_auth_context()
                    if not auth:
                        return {"error": "Unauthorized"}, 401

                    if required_permission and not auth.has_permission(required_permission):
                        return {"error": f"Missing permission: {required_permission}"}, 403

                    return f(*args, **kwargs)
                return decorated_function
            return decorator

        def require_permission(self, permission: str):
            """
            Decorator that requires a specific permission.

            Args:
                permission: Permission string

            Returns:
                Decorator function
            """
            return self.require_auth(required_permission=permission)


# =============================================================================
# gRPC Integration
# =============================================================================

try:
    import grpc
    from grpc import ServerInterceptor
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False


if HAS_GRPC:
    class GRPCSSOInterceptor(ServerInterceptor):
        """
        gRPC server interceptor for SSO authentication.

        Usage:
            from grpc import server
            from services.identity.sso import GRPCSSOInterceptor, Product

            interceptor = GRPCSSOInterceptor(Product.FHE_GBDT)
            grpc_server = server(
                futures.ThreadPoolExecutor(),
                interceptors=[interceptor],
            )
        """

        AUTH_CONTEXT_KEY = "auth_context"

        def __init__(
            self,
            product: Product,
            identity_provider: Optional[UnifiedIdentityProvider] = None,
            skip_methods: Optional[List[str]] = None,
        ):
            self.handler = SSOHandler(product, identity_provider)
            self.skip_methods = skip_methods or ["/grpc.health.v1.Health/Check"]

        def intercept_service(self, continuation, handler_call_details):
            # Skip authentication for certain methods
            if handler_call_details.method in self.skip_methods:
                return continuation(handler_call_details)

            # Get metadata
            metadata = dict(handler_call_details.invocation_metadata or [])
            authorization = metadata.get("authorization")
            api_key = metadata.get("x-api-key")

            # Authenticate
            auth_context = self.handler.authenticate(
                authorization=authorization,
                api_key=api_key,
            )

            if not auth_context:
                return self._unauthenticated_handler()

            # Store auth context for the handler
            return self._authenticated_continuation(
                continuation,
                handler_call_details,
                auth_context,
            )

        def _unauthenticated_handler(self):
            """Return a handler that denies unauthenticated requests."""
            def handler(request, context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Authentication required")
            return grpc.unary_unary_rpc_method_handler(handler)

        def _authenticated_continuation(self, continuation, handler_call_details, auth_context):
            """Continue with authentication context."""
            # The auth context is passed via context in the actual handler
            # This is a simplified version - in practice, you'd use context.set_details
            return continuation(handler_call_details)


# =============================================================================
# Go Integration Helper
# =============================================================================

def generate_go_middleware() -> str:
    """
    Generate Go middleware code for the unified identity provider.

    This generates code that can be used in Go services like the gateway.
    """
    return '''
// Code generated by services/identity/sso.py. DO NOT EDIT.
// This file provides SSO middleware for Go services.

package sso

import (
    "context"
    "errors"
    "net/http"
    "strings"
    "time"

    "github.com/golang-jwt/jwt/v5"
)

var (
    ErrMissingAuth    = errors.New("missing authentication")
    ErrInvalidToken   = errors.New("invalid token")
    ErrExpiredToken   = errors.New("expired token")
    ErrMissingPerm    = errors.New("missing required permission")
    ErrInvalidProduct = errors.New("invalid product for token")
)

// Product represents a product in the suite
type Product string

const (
    ProductFHEGBDT  Product = "fhe_gbdt"
    ProductTenSafe  Product = "tensafe"
    ProductPlatform Product = "platform"
)

// TokenClaims represents the JWT claims
type TokenClaims struct {
    jwt.RegisteredClaims
    OrgID       string   `json:"org_id"`
    AuthType    string   `json:"auth_type"`
    Permissions []string `json:"permissions"`
    Email       string   `json:"email,omitempty"`
    Name        string   `json:"name,omitempty"`
    Role        string   `json:"role,omitempty"`
}

// AuthContext holds the authentication context for a request
type AuthContext struct {
    Claims      *TokenClaims
    OrgID       string
    UserID      *string // nil for API key auth
    AuthType    string
    Permissions map[string]bool
}

// HasPermission checks if the context has a specific permission
func (ac *AuthContext) HasPermission(perm string) bool {
    return ac.Permissions[perm]
}

// HasProductAccess checks if the context has any permission for a product
func (ac *AuthContext) HasProductAccess(product Product) bool {
    prefix := string(product) + ":"
    for perm := range ac.Permissions {
        if strings.HasPrefix(perm, prefix) {
            return true
        }
    }
    return false
}

// SSOConfig holds configuration for the SSO handler
type SSOConfig struct {
    JWTSecret string
    Issuer    string
    Product   Product
}

// SSOHandler handles SSO authentication
type SSOHandler struct {
    config SSOConfig
}

// NewSSOHandler creates a new SSO handler
func NewSSOHandler(config SSOConfig) *SSOHandler {
    return &SSOHandler{config: config}
}

// Authenticate validates the authentication headers and returns an AuthContext
func (h *SSOHandler) Authenticate(authHeader, apiKeyHeader string) (*AuthContext, error) {
    var token string

    // Try Bearer token first
    if strings.HasPrefix(authHeader, "Bearer ") {
        token = strings.TrimPrefix(authHeader, "Bearer ")
    } else if apiKeyHeader != "" {
        // For API keys, the identity provider would validate and issue a token
        // In this simplified version, we assume API keys are pre-validated
        return nil, ErrMissingAuth
    }

    if token == "" {
        return nil, ErrMissingAuth
    }

    // Parse and validate token
    claims := &TokenClaims{}
    parsedToken, err := jwt.ParseWithClaims(token, claims, func(t *jwt.Token) (interface{}, error) {
        return []byte(h.config.JWTSecret), nil
    })

    if err != nil || !parsedToken.Valid {
        return nil, ErrInvalidToken
    }

    // Check issuer
    if claims.Issuer != h.config.Issuer {
        return nil, ErrInvalidToken
    }

    // Check expiration
    if claims.ExpiresAt.Before(time.Now()) {
        return nil, ErrExpiredToken
    }

    // Check audience (product)
    found := false
    for _, aud := range claims.Audience {
        if aud == string(h.config.Product) {
            found = true
            break
        }
    }
    if !found {
        return nil, ErrInvalidProduct
    }

    // Build permission map
    perms := make(map[string]bool)
    for _, p := range claims.Permissions {
        perms[p] = true
    }

    // Build auth context
    ctx := &AuthContext{
        Claims:      claims,
        OrgID:       claims.OrgID,
        AuthType:    claims.AuthType,
        Permissions: perms,
    }

    if claims.AuthType == "user" {
        userID := claims.Subject
        ctx.UserID = &userID
    }

    return ctx, nil
}

// Middleware returns an HTTP middleware that authenticates requests
func (h *SSOHandler) Middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        authHeader := r.Header.Get("Authorization")
        apiKeyHeader := r.Header.Get("X-API-Key")

        authCtx, err := h.Authenticate(authHeader, apiKeyHeader)
        if err != nil {
            http.Error(w, err.Error(), http.StatusUnauthorized)
            return
        }

        // Store auth context in request context
        ctx := context.WithValue(r.Context(), "auth_context", authCtx)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

// RequirePermission returns middleware that requires a specific permission
func (h *SSOHandler) RequirePermission(perm string, next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        authCtx, ok := r.Context().Value("auth_context").(*AuthContext)
        if !ok || authCtx == nil {
            http.Error(w, "unauthorized", http.StatusUnauthorized)
            return
        }

        if !authCtx.HasPermission(perm) {
            http.Error(w, "forbidden: missing permission "+perm, http.StatusForbidden)
            return
        }

        next.ServeHTTP(w, r)
    })
}

// GetAuthContext retrieves the auth context from a request
func GetAuthContext(r *http.Request) *AuthContext {
    ctx, _ := r.Context().Value("auth_context").(*AuthContext)
    return ctx
}
'''


# =============================================================================
# TypeScript SDK Integration
# =============================================================================

def generate_typescript_auth() -> str:
    """
    Generate TypeScript types and utilities for the unified auth.

    This can be used in the TypeScript SDK for both products.
    """
    return '''
// Code generated by services/identity/sso.py. DO NOT EDIT.
// This file provides SSO types and utilities for TypeScript clients.

/**
 * Products in the suite that share unified authentication
 */
export enum Product {
  FHE_GBDT = 'fhe_gbdt',
  TENSAFE = 'tensafe',
  PLATFORM = 'platform',
}

/**
 * Authentication method
 */
export type AuthType = 'user' | 'api_key';

/**
 * JWT token claims
 */
export interface TokenClaims {
  sub: string;
  iss: string;
  aud: string[];
  exp: number;
  iat: number;
  jti: string;
  org_id: string;
  auth_type: AuthType;
  permissions: string[];
  email?: string;
  name?: string;
  role?: string;
}

/**
 * User information returned after login
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
}

/**
 * Login response
 */
export interface LoginResponse {
  access_token: string;
  token_type: 'Bearer';
  expires_in: number;
  user: User;
}

/**
 * API key metadata (secret is only returned once on creation)
 */
export interface APIKeyMetadata {
  key_id: string;
  org_id: string;
  name: string;
  created_at: string;
  expires_at?: string;
  last_used_at?: string;
  permissions: string[];
  is_active: boolean;
}

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
 * Unified authentication client for the product suite
 */
export class UnifiedAuthClient {
  private baseUrl: string;
  private accessToken?: string;

  constructor(baseUrl: string = 'https://identity.product-suite.local') {
    this.baseUrl = baseUrl;
  }

  /**
   * Login with email and password
   */
  async login(email: string, password: string): Promise<LoginResponse> {
    const response = await fetch(`${this.baseUrl}/v1/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      throw new Error(`Login failed: ${response.statusText}`);
    }

    const data = await response.json() as LoginResponse;
    this.accessToken = data.access_token;
    return data;
  }

  /**
   * Set access token directly (e.g., from storage)
   */
  setAccessToken(token: string): void {
    this.accessToken = token;
  }

  /**
   * Get current access token
   */
  getAccessToken(): string | undefined {
    return this.accessToken;
  }

  /**
   * Check if the client is authenticated
   */
  isAuthenticated(): boolean {
    if (!this.accessToken) return false;

    try {
      const payload = JSON.parse(atob(this.accessToken.split('.')[1]));
      return payload.exp * 1000 > Date.now();
    } catch {
      return false;
    }
  }

  /**
   * Get the current user's permissions from the token
   */
  getPermissions(): string[] {
    if (!this.accessToken) return [];

    try {
      const payload = JSON.parse(atob(this.accessToken.split('.')[1])) as TokenClaims;
      return payload.permissions;
    } catch {
      return [];
    }
  }

  /**
   * Check if the user has a specific permission
   */
  hasPermission(permission: string): boolean {
    return this.getPermissions().includes(permission);
  }

  /**
   * Check if the user has access to a product
   */
  hasProductAccess(product: Product): boolean {
    const prefix = `${product}:`;
    return this.getPermissions().some(p => p.startsWith(prefix));
  }

  /**
   * Logout (invalidate token on server)
   */
  async logout(): Promise<void> {
    if (this.accessToken) {
      await fetch(`${this.baseUrl}/v1/auth/logout`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${this.accessToken}` },
      });
    }
    this.accessToken = undefined;
  }

  /**
   * Get authorization headers for API requests
   */
  getAuthHeaders(): Record<string, string> {
    if (this.accessToken) {
      return { 'Authorization': `Bearer ${this.accessToken}` };
    }
    return {};
  }
}

/**
 * Create a client configured for a specific product
 */
export function createProductClient(
  product: Product,
  authClient: UnifiedAuthClient,
): { baseUrl: string; getHeaders: () => Record<string, string> } {
  const productUrls: Record<Product, string> = {
    [Product.FHE_GBDT]: 'https://api.fhe-gbdt.local',
    [Product.TENSAFE]: 'https://api.tensafe.local',
    [Product.PLATFORM]: 'https://api.platform.local',
  };

  return {
    baseUrl: productUrls[product],
    getHeaders: () => authClient.getAuthHeaders(),
  };
}
'''


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AuthContext",
    "SSOHandler",
    "generate_go_middleware",
    "generate_typescript_auth",
]

if HAS_FASTAPI:
    __all__.append("FastAPISSOMiddleware")

if HAS_FLASK:
    __all__.append("FlaskSSOMiddleware")

if HAS_GRPC:
    __all__.append("GRPCSSOInterceptor")
