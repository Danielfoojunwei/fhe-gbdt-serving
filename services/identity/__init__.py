"""
Unified Identity Service for Product Suite

This package provides a shared identity and authentication service that works
across all products in the suite (FHE-GBDT Serving, TenSafe, etc.).

Features:
- Single Sign-On (SSO) with JWT tokens
- Multi-product authorization with product scopes
- Organization/tenant management
- User management within organizations
- API key management for service accounts

Example usage:

    # Initialize the identity provider
    from services.identity import get_identity_provider, Product

    idp = get_identity_provider()

    # Create an organization (once per customer)
    org = idp.create_organization(
        name="Acme Corp",
        tier="business",
        enabled_products={Product.FHE_GBDT.value, Product.TENSAFE.value},
    )

    # Create a user in the organization
    user = idp.create_user(
        org_id=org.org_id,
        email="developer@acme.com",
        name="John Developer",
        password="secure-password-123",
        role="developer",
    )

    # Login and get tokens
    result = idp.login(email="developer@acme.com", password="secure-password-123")
    access_token = result["access_token"]

    # Use token across both products
    # - FHE-GBDT: Authorization: Bearer <access_token>
    # - TenSafe: Authorization: Bearer <access_token>

    # Or create an API key for service accounts
    api_key, raw_key = idp.create_api_key(
        org_id=org.org_id,
        name="CI/CD Pipeline",
        permissions=["gbdt:predict", "gbdt:train", "tensafe:adapt"],
    )
    # Use raw_key as: X-API-Key: <raw_key>

For web framework integration, see the sso module:

    # FastAPI
    from services.identity.sso import FastAPISSOMiddleware, Product
    sso = FastAPISSOMiddleware(Product.FHE_GBDT)

    @app.get("/predict")
    async def predict(auth = Depends(sso.require_auth())):
        ...

    # Flask
    from services.identity.sso import FlaskSSOMiddleware, Product
    sso = FlaskSSOMiddleware(app, Product.FHE_GBDT)

    @app.route("/predict")
    @sso.require_auth()
    def predict():
        ...
"""

from .provider import (
    # Core classes
    UnifiedIdentityProvider,
    IdentityProviderConfig,

    # Data models
    Organization,
    User,
    APIKey,
    TokenClaims,

    # Enums
    Product,
    Permission,

    # Store backends
    IdentityStore,
    InMemoryStore,
    FileStore,

    # Role permissions
    ROLE_PERMISSIONS,

    # Global instance
    get_identity_provider,
    init_identity_provider,

    # Utilities
    hash_password,
    verify_password,
    hash_api_key,
)

from .sso import (
    AuthContext,
    SSOHandler,
    generate_go_middleware,
    generate_typescript_auth,
)

# Conditional imports based on available frameworks
try:
    from .sso import FastAPISSOMiddleware
except ImportError:
    pass

try:
    from .sso import FlaskSSOMiddleware
except ImportError:
    pass

try:
    from .sso import GRPCSSOInterceptor
except ImportError:
    pass


__all__ = [
    # Provider
    "UnifiedIdentityProvider",
    "IdentityProviderConfig",
    "get_identity_provider",
    "init_identity_provider",

    # Models
    "Organization",
    "User",
    "APIKey",
    "TokenClaims",

    # Enums
    "Product",
    "Permission",
    "ROLE_PERMISSIONS",

    # Stores
    "IdentityStore",
    "InMemoryStore",
    "FileStore",

    # SSO
    "AuthContext",
    "SSOHandler",
    "generate_go_middleware",
    "generate_typescript_auth",

    # Utilities
    "hash_password",
    "verify_password",
    "hash_api_key",
]

__version__ = "1.0.0"
