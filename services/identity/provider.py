"""
Unified Identity Provider for Product Suite

This module provides a shared identity and authentication service that works
across all products in the suite (FHE-GBDT Serving, TenSafe, etc.).

Features:
- Single Sign-On (SSO) with JWT tokens
- Multi-product authorization with product scopes
- Organization/tenant management
- User management within organizations
- API key management for service accounts

Compliance: SOC2 CC6.1, CC6.2, CC6.7, ISO 27001 A.9.2, HIPAA 164.312(d)
"""

import os
import json
import time
import uuid
import hashlib
import secrets
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, List, Dict, Set, Any
from pathlib import Path

# Try to import JWT - handle any import errors gracefully
HAS_JWT = False
jwt = None
try:
    import jwt as _jwt
    jwt = _jwt
    HAS_JWT = True
except:
    pass

# Try to import cryptography - handle any import errors gracefully
HAS_CRYPTO = False
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTO = True
except:
    pass

import base64

logger = logging.getLogger(__name__)


# =============================================================================
# Product Definitions
# =============================================================================

class Product(Enum):
    """Products in the suite that share unified authentication."""
    FHE_GBDT = "fhe_gbdt"      # FHE-GBDT Serving
    TENSAFE = "tensafe"        # TenSafe LoRA Adaptation
    PLATFORM = "platform"      # Platform-wide admin access

    @classmethod
    def all_products(cls) -> Set[str]:
        return {p.value for p in cls}


class Permission(Enum):
    """Fine-grained permissions within each product."""
    # FHE-GBDT permissions
    GBDT_PREDICT = "gbdt:predict"
    GBDT_TRAIN = "gbdt:train"
    GBDT_MODEL_UPLOAD = "gbdt:model:upload"
    GBDT_MODEL_DELETE = "gbdt:model:delete"
    GBDT_KEYS_MANAGE = "gbdt:keys:manage"

    # TenSafe permissions
    TENSAFE_ADAPT = "tensafe:adapt"
    TENSAFE_INFERENCE = "tensafe:inference"
    TENSAFE_MODEL_UPLOAD = "tensafe:model:upload"
    TENSAFE_MODEL_DELETE = "tensafe:model:delete"
    TENSAFE_KEYS_MANAGE = "tensafe:keys:manage"

    # Platform permissions
    PLATFORM_ADMIN = "platform:admin"
    PLATFORM_BILLING = "platform:billing"
    PLATFORM_AUDIT = "platform:audit"
    ORG_MANAGE = "org:manage"
    USER_MANAGE = "user:manage"


# Default permission sets for common roles
ROLE_PERMISSIONS = {
    "viewer": [
        Permission.GBDT_PREDICT.value,
        Permission.TENSAFE_INFERENCE.value,
    ],
    "developer": [
        Permission.GBDT_PREDICT.value,
        Permission.GBDT_TRAIN.value,
        Permission.GBDT_MODEL_UPLOAD.value,
        Permission.TENSAFE_ADAPT.value,
        Permission.TENSAFE_INFERENCE.value,
        Permission.TENSAFE_MODEL_UPLOAD.value,
    ],
    "admin": [
        Permission.GBDT_PREDICT.value,
        Permission.GBDT_TRAIN.value,
        Permission.GBDT_MODEL_UPLOAD.value,
        Permission.GBDT_MODEL_DELETE.value,
        Permission.GBDT_KEYS_MANAGE.value,
        Permission.TENSAFE_ADAPT.value,
        Permission.TENSAFE_INFERENCE.value,
        Permission.TENSAFE_MODEL_UPLOAD.value,
        Permission.TENSAFE_MODEL_DELETE.value,
        Permission.TENSAFE_KEYS_MANAGE.value,
        Permission.ORG_MANAGE.value,
        Permission.USER_MANAGE.value,
    ],
    "platform_admin": [p.value for p in Permission],
}


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Organization:
    """An organization (tenant) that can have multiple users and products."""
    org_id: str
    name: str
    created_at: datetime
    updated_at: datetime

    # Products enabled for this organization
    enabled_products: Set[str] = field(default_factory=lambda: {Product.FHE_GBDT.value, Product.TENSAFE.value})

    # Subscription/billing tier
    tier: str = "free"  # free, pro, business, enterprise

    # Settings
    settings: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "org_id": self.org_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "enabled_products": list(self.enabled_products),
            "tier": self.tier,
            "settings": self.settings,
            "metadata": self.metadata,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Organization":
        return cls(
            org_id=data["org_id"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            enabled_products=set(data.get("enabled_products", [Product.FHE_GBDT.value, Product.TENSAFE.value])),
            tier=data.get("tier", "free"),
            settings=data.get("settings", {}),
            metadata=data.get("metadata", {}),
            is_active=data.get("is_active", True),
        )


@dataclass
class User:
    """A user within an organization."""
    user_id: str
    org_id: str
    email: str
    name: str
    created_at: datetime
    updated_at: datetime

    # Authentication
    password_hash: Optional[str] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None

    # Authorization
    role: str = "developer"  # viewer, developer, admin, platform_admin
    permissions: List[str] = field(default_factory=list)  # Additional permissions beyond role

    # Session tracking
    last_login_at: Optional[datetime] = None
    last_login_ip: Optional[str] = None

    # Status
    is_active: bool = True
    email_verified: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_effective_permissions(self) -> Set[str]:
        """Get all permissions including role-based and explicit."""
        perms = set(ROLE_PERMISSIONS.get(self.role, []))
        perms.update(self.permissions)
        return perms

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.get_effective_permissions()

    def has_product_access(self, product: Product) -> bool:
        """Check if user has any permission for a product."""
        # Map product enum to permission prefix
        product_to_prefix = {
            Product.FHE_GBDT: "gbdt:",
            Product.TENSAFE: "tensafe:",
            Product.PLATFORM: "platform:",
        }
        prefix = product_to_prefix.get(product, f"{product.value}:")
        return any(p.startswith(prefix) for p in self.get_effective_permissions())

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        result = {
            "user_id": self.user_id,
            "org_id": self.org_id,
            "email": self.email,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "role": self.role,
            "permissions": self.permissions,
            "mfa_enabled": self.mfa_enabled,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "is_active": self.is_active,
            "email_verified": self.email_verified,
            "metadata": self.metadata,
        }
        if include_sensitive:
            result["password_hash"] = self.password_hash
            result["mfa_secret"] = self.mfa_secret
            result["last_login_ip"] = self.last_login_ip
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        return cls(
            user_id=data["user_id"],
            org_id=data["org_id"],
            email=data["email"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            password_hash=data.get("password_hash"),
            mfa_enabled=data.get("mfa_enabled", False),
            mfa_secret=data.get("mfa_secret"),
            role=data.get("role", "developer"),
            permissions=data.get("permissions", []),
            last_login_at=datetime.fromisoformat(data["last_login_at"]) if data.get("last_login_at") else None,
            last_login_ip=data.get("last_login_ip"),
            is_active=data.get("is_active", True),
            email_verified=data.get("email_verified", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class APIKey:
    """An API key for service account authentication."""
    key_id: str
    org_id: str
    name: str
    key_hash: str  # SHA-256 hash of the actual key
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime] = None

    # Scopes/permissions
    permissions: List[str] = field(default_factory=list)

    # Restrictions
    allowed_ips: List[str] = field(default_factory=list)  # Empty = all allowed
    rate_limit: int = 1000  # Requests per minute

    # Status
    is_active: bool = True

    # Metadata
    created_by: Optional[str] = None  # user_id
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_id": self.key_id,
            "org_id": self.org_id,
            "name": self.name,
            "key_hash": self.key_hash,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "permissions": self.permissions,
            "allowed_ips": self.allowed_ips,
            "rate_limit": self.rate_limit,
            "is_active": self.is_active,
            "created_by": self.created_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIKey":
        return cls(
            key_id=data["key_id"],
            org_id=data["org_id"],
            name=data["name"],
            key_hash=data["key_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            last_used_at=datetime.fromisoformat(data["last_used_at"]) if data.get("last_used_at") else None,
            permissions=data.get("permissions", []),
            allowed_ips=data.get("allowed_ips", []),
            rate_limit=data.get("rate_limit", 1000),
            is_active=data.get("is_active", True),
            created_by=data.get("created_by"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TokenClaims:
    """Claims contained in a JWT token."""
    # Standard claims
    sub: str  # Subject (user_id or key_id)
    iss: str  # Issuer
    aud: List[str]  # Audience (products)
    exp: int  # Expiration time
    iat: int  # Issued at
    jti: str  # JWT ID

    # Custom claims
    org_id: str
    auth_type: str  # "user" or "api_key"
    permissions: List[str]

    # Optional claims
    email: Optional[str] = None
    name: Optional[str] = None
    role: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub": self.sub,
            "iss": self.iss,
            "aud": self.aud,
            "exp": self.exp,
            "iat": self.iat,
            "jti": self.jti,
            "org_id": self.org_id,
            "auth_type": self.auth_type,
            "permissions": self.permissions,
            "email": self.email,
            "name": self.name,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenClaims":
        return cls(
            sub=data["sub"],
            iss=data["iss"],
            aud=data["aud"],
            exp=data["exp"],
            iat=data["iat"],
            jti=data["jti"],
            org_id=data["org_id"],
            auth_type=data["auth_type"],
            permissions=data["permissions"],
            email=data.get("email"),
            name=data.get("name"),
            role=data.get("role"),
        )


# =============================================================================
# Storage Backend Interface
# =============================================================================

class IdentityStore(ABC):
    """Abstract interface for identity data storage."""

    @abstractmethod
    def get_organization(self, org_id: str) -> Optional[Organization]:
        pass

    @abstractmethod
    def save_organization(self, org: Organization) -> None:
        pass

    @abstractmethod
    def delete_organization(self, org_id: str) -> None:
        pass

    @abstractmethod
    def list_organizations(self, limit: int = 100, offset: int = 0) -> List[Organization]:
        pass

    @abstractmethod
    def get_user(self, user_id: str) -> Optional[User]:
        pass

    @abstractmethod
    def get_user_by_email(self, email: str) -> Optional[User]:
        pass

    @abstractmethod
    def save_user(self, user: User) -> None:
        pass

    @abstractmethod
    def delete_user(self, user_id: str) -> None:
        pass

    @abstractmethod
    def list_users_by_org(self, org_id: str, limit: int = 100, offset: int = 0) -> List[User]:
        pass

    @abstractmethod
    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        pass

    @abstractmethod
    def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        pass

    @abstractmethod
    def save_api_key(self, key: APIKey) -> None:
        pass

    @abstractmethod
    def delete_api_key(self, key_id: str) -> None:
        pass

    @abstractmethod
    def list_api_keys_by_org(self, org_id: str) -> List[APIKey]:
        pass


class InMemoryStore(IdentityStore):
    """In-memory storage for development and testing."""

    def __init__(self):
        self._orgs: Dict[str, Organization] = {}
        self._users: Dict[str, User] = {}
        self._users_by_email: Dict[str, str] = {}  # email -> user_id
        self._api_keys: Dict[str, APIKey] = {}
        self._api_keys_by_hash: Dict[str, str] = {}  # hash -> key_id

    def get_organization(self, org_id: str) -> Optional[Organization]:
        return self._orgs.get(org_id)

    def save_organization(self, org: Organization) -> None:
        self._orgs[org.org_id] = org

    def delete_organization(self, org_id: str) -> None:
        self._orgs.pop(org_id, None)

    def list_organizations(self, limit: int = 100, offset: int = 0) -> List[Organization]:
        orgs = list(self._orgs.values())
        return orgs[offset:offset + limit]

    def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    def get_user_by_email(self, email: str) -> Optional[User]:
        user_id = self._users_by_email.get(email.lower())
        return self._users.get(user_id) if user_id else None

    def save_user(self, user: User) -> None:
        self._users[user.user_id] = user
        self._users_by_email[user.email.lower()] = user.user_id

    def delete_user(self, user_id: str) -> None:
        user = self._users.pop(user_id, None)
        if user:
            self._users_by_email.pop(user.email.lower(), None)

    def list_users_by_org(self, org_id: str, limit: int = 100, offset: int = 0) -> List[User]:
        users = [u for u in self._users.values() if u.org_id == org_id]
        return users[offset:offset + limit]

    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        return self._api_keys.get(key_id)

    def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        key_id = self._api_keys_by_hash.get(key_hash)
        return self._api_keys.get(key_id) if key_id else None

    def save_api_key(self, key: APIKey) -> None:
        self._api_keys[key.key_id] = key
        self._api_keys_by_hash[key.key_hash] = key.key_id

    def delete_api_key(self, key_id: str) -> None:
        key = self._api_keys.pop(key_id, None)
        if key:
            self._api_keys_by_hash.pop(key.key_hash, None)

    def list_api_keys_by_org(self, org_id: str) -> List[APIKey]:
        return [k for k in self._api_keys.values() if k.org_id == org_id]


class FileStore(IdentityStore):
    """File-based storage for development and small deployments."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "orgs").mkdir(exist_ok=True)
        (self.base_path / "users").mkdir(exist_ok=True)
        (self.base_path / "api_keys").mkdir(exist_ok=True)

        # Index files
        self._load_indices()

    def _load_indices(self):
        self._users_by_email: Dict[str, str] = {}
        self._api_keys_by_hash: Dict[str, str] = {}

        # Build email index
        for user_file in (self.base_path / "users").glob("*.json"):
            with open(user_file) as f:
                data = json.load(f)
                self._users_by_email[data["email"].lower()] = data["user_id"]

        # Build key hash index
        for key_file in (self.base_path / "api_keys").glob("*.json"):
            with open(key_file) as f:
                data = json.load(f)
                self._api_keys_by_hash[data["key_hash"]] = data["key_id"]

    def get_organization(self, org_id: str) -> Optional[Organization]:
        path = self.base_path / "orgs" / f"{org_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return Organization.from_dict(json.load(f))

    def save_organization(self, org: Organization) -> None:
        path = self.base_path / "orgs" / f"{org.org_id}.json"
        with open(path, "w") as f:
            json.dump(org.to_dict(), f, indent=2)

    def delete_organization(self, org_id: str) -> None:
        path = self.base_path / "orgs" / f"{org_id}.json"
        path.unlink(missing_ok=True)

    def list_organizations(self, limit: int = 100, offset: int = 0) -> List[Organization]:
        orgs = []
        for path in sorted((self.base_path / "orgs").glob("*.json"))[offset:offset + limit]:
            with open(path) as f:
                orgs.append(Organization.from_dict(json.load(f)))
        return orgs

    def get_user(self, user_id: str) -> Optional[User]:
        path = self.base_path / "users" / f"{user_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return User.from_dict(json.load(f))

    def get_user_by_email(self, email: str) -> Optional[User]:
        user_id = self._users_by_email.get(email.lower())
        return self.get_user(user_id) if user_id else None

    def save_user(self, user: User) -> None:
        path = self.base_path / "users" / f"{user.user_id}.json"
        with open(path, "w") as f:
            json.dump(user.to_dict(include_sensitive=True), f, indent=2)
        self._users_by_email[user.email.lower()] = user.user_id

    def delete_user(self, user_id: str) -> None:
        user = self.get_user(user_id)
        if user:
            self._users_by_email.pop(user.email.lower(), None)
        path = self.base_path / "users" / f"{user_id}.json"
        path.unlink(missing_ok=True)

    def list_users_by_org(self, org_id: str, limit: int = 100, offset: int = 0) -> List[User]:
        users = []
        for path in (self.base_path / "users").glob("*.json"):
            with open(path) as f:
                data = json.load(f)
                if data["org_id"] == org_id:
                    users.append(User.from_dict(data))
        return users[offset:offset + limit]

    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        path = self.base_path / "api_keys" / f"{key_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return APIKey.from_dict(json.load(f))

    def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        key_id = self._api_keys_by_hash.get(key_hash)
        return self.get_api_key(key_id) if key_id else None

    def save_api_key(self, key: APIKey) -> None:
        path = self.base_path / "api_keys" / f"{key.key_id}.json"
        with open(path, "w") as f:
            json.dump(key.to_dict(), f, indent=2)
        self._api_keys_by_hash[key.key_hash] = key.key_id

    def delete_api_key(self, key_id: str) -> None:
        key = self.get_api_key(key_id)
        if key:
            self._api_keys_by_hash.pop(key.key_hash, None)
        path = self.base_path / "api_keys" / f"{key_id}.json"
        path.unlink(missing_ok=True)

    def list_api_keys_by_org(self, org_id: str) -> List[APIKey]:
        keys = []
        for path in (self.base_path / "api_keys").glob("*.json"):
            with open(path) as f:
                data = json.load(f)
                if data["org_id"] == org_id:
                    keys.append(APIKey.from_dict(data))
        return keys


# =============================================================================
# Password Hashing
# =============================================================================

def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Hash a password using PBKDF2-SHA256."""
    if salt is None:
        salt = secrets.token_bytes(32)

    if HAS_CRYPTO:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600000,  # OWASP recommended
            backend=default_backend()
        )
        key = kdf.derive(password.encode())
        return f"pbkdf2:sha256:600000${base64.b64encode(salt).decode()}${base64.b64encode(key).decode()}"
    else:
        # Fallback to hashlib (less secure but functional)
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 600000)
        return f"pbkdf2:sha256:600000${base64.b64encode(salt).decode()}${base64.b64encode(key).decode()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    try:
        parts = password_hash.split('$')
        if len(parts) != 3:
            return False

        header, salt_b64, key_b64 = parts
        salt = base64.b64decode(salt_b64)
        stored_key = base64.b64decode(key_b64)

        # Re-hash and compare
        if HAS_CRYPTO:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=600000,
                backend=default_backend()
            )
            try:
                kdf.verify(password.encode(), stored_key)
                return True
            except Exception:
                return False
        else:
            key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 600000)
            return secrets.compare_digest(key, stored_key)
    except Exception:
        return False


def hash_api_key(key: str) -> str:
    """Hash an API key using SHA-256."""
    return hashlib.sha256(key.encode()).hexdigest()


# =============================================================================
# Unified Identity Provider
# =============================================================================

class IdentityProviderConfig:
    """Configuration for the Identity Provider."""

    def __init__(
        self,
        issuer: str = "https://identity.product-suite.local",
        jwt_secret: Optional[str] = None,
        jwt_algorithm: str = "HS256",
        access_token_ttl: int = 3600,  # 1 hour
        refresh_token_ttl: int = 86400 * 30,  # 30 days
        store_type: str = "memory",  # memory, file
        store_path: Optional[str] = None,
    ):
        self.issuer = issuer
        self.jwt_secret = jwt_secret or os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
        self.jwt_algorithm = jwt_algorithm
        self.access_token_ttl = access_token_ttl
        self.refresh_token_ttl = refresh_token_ttl
        self.store_type = store_type
        self.store_path = store_path


class UnifiedIdentityProvider:
    """
    Unified Identity Provider for the product suite.

    Provides:
    - User authentication with password + optional MFA
    - API key authentication for service accounts
    - JWT token issuance for SSO across products
    - Organization/tenant management
    - Role-based access control
    """

    PRODUCT_SUITE_NAME = "ProductSuite"
    PRODUCT_SUITE_VERSION = "1.0.0"

    def __init__(self, config: Optional[IdentityProviderConfig] = None):
        self.config = config or IdentityProviderConfig()

        # Initialize store
        if self.config.store_type == "file":
            store_path = self.config.store_path or "/var/lib/identity"
            self.store = FileStore(store_path)
        else:
            self.store = InMemoryStore()

        # Token blacklist (for logout)
        self._token_blacklist: Set[str] = set()

        logger.info(f"Identity Provider initialized (issuer: {self.config.issuer})")

    # =========================================================================
    # Organization Management
    # =========================================================================

    def create_organization(
        self,
        name: str,
        tier: str = "free",
        enabled_products: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Organization:
        """Create a new organization."""
        org_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        org = Organization(
            org_id=org_id,
            name=name,
            created_at=now,
            updated_at=now,
            enabled_products=enabled_products or {Product.FHE_GBDT.value, Product.TENSAFE.value},
            tier=tier,
            metadata=metadata or {},
        )

        self.store.save_organization(org)
        logger.info(f"AUDIT: Created organization {org_id} ({name})")

        return org

    def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get an organization by ID."""
        return self.store.get_organization(org_id)

    def update_organization(
        self,
        org_id: str,
        name: Optional[str] = None,
        tier: Optional[str] = None,
        enabled_products: Optional[Set[str]] = None,
        settings: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Organization]:
        """Update an organization."""
        org = self.store.get_organization(org_id)
        if not org:
            return None

        if name is not None:
            org.name = name
        if tier is not None:
            org.tier = tier
        if enabled_products is not None:
            org.enabled_products = enabled_products
        if settings is not None:
            org.settings.update(settings)
        if is_active is not None:
            org.is_active = is_active

        org.updated_at = datetime.now(timezone.utc)
        self.store.save_organization(org)

        logger.info(f"AUDIT: Updated organization {org_id}")
        return org

    def delete_organization(self, org_id: str) -> bool:
        """Delete an organization and all associated data."""
        org = self.store.get_organization(org_id)
        if not org:
            return False

        # Delete all users in the organization
        for user in self.store.list_users_by_org(org_id):
            self.store.delete_user(user.user_id)

        # Delete all API keys
        for key in self.store.list_api_keys_by_org(org_id):
            self.store.delete_api_key(key.key_id)

        # Delete organization
        self.store.delete_organization(org_id)

        logger.info(f"AUDIT: Deleted organization {org_id}")
        return True

    # =========================================================================
    # User Management
    # =========================================================================

    def create_user(
        self,
        org_id: str,
        email: str,
        name: str,
        password: str,
        role: str = "developer",
        permissions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> User:
        """Create a new user in an organization."""
        # Verify organization exists
        org = self.store.get_organization(org_id)
        if not org:
            raise ValueError(f"Organization {org_id} not found")

        # Check email uniqueness
        if self.store.get_user_by_email(email):
            raise ValueError(f"Email {email} already registered")

        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        user = User(
            user_id=user_id,
            org_id=org_id,
            email=email,
            name=name,
            created_at=now,
            updated_at=now,
            password_hash=hash_password(password),
            role=role,
            permissions=permissions or [],
            metadata=metadata or {},
        )

        self.store.save_user(user)
        logger.info(f"AUDIT: Created user {user_id} ({email}) in org {org_id}")

        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self.store.get_user(user_id)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        return self.store.get_user_by_email(email)

    def update_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        role: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[User]:
        """Update a user."""
        user = self.store.get_user(user_id)
        if not user:
            return None

        if name is not None:
            user.name = name
        if role is not None:
            user.role = role
        if permissions is not None:
            user.permissions = permissions
        if is_active is not None:
            user.is_active = is_active

        user.updated_at = datetime.now(timezone.utc)
        self.store.save_user(user)

        logger.info(f"AUDIT: Updated user {user_id}")
        return user

    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change a user's password."""
        user = self.store.get_user(user_id)
        if not user or not user.password_hash:
            return False

        if not verify_password(old_password, user.password_hash):
            logger.warning(f"SECURITY: Failed password change for user {user_id} - invalid old password")
            return False

        user.password_hash = hash_password(new_password)
        user.updated_at = datetime.now(timezone.utc)
        self.store.save_user(user)

        logger.info(f"AUDIT: Password changed for user {user_id}")
        return True

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        user = self.store.get_user(user_id)
        if not user:
            return False

        self.store.delete_user(user_id)
        logger.info(f"AUDIT: Deleted user {user_id}")
        return True

    # =========================================================================
    # API Key Management
    # =========================================================================

    def create_api_key(
        self,
        org_id: str,
        name: str,
        permissions: Optional[List[str]] = None,
        expires_in_days: Optional[int] = 365,
        allowed_ips: Optional[List[str]] = None,
        created_by: Optional[str] = None,
    ) -> tuple[APIKey, str]:
        """
        Create a new API key.

        Returns:
            Tuple of (APIKey metadata, raw key string)

        Note: The raw key is only returned once and cannot be retrieved later.
        """
        # Verify organization exists
        org = self.store.get_organization(org_id)
        if not org:
            raise ValueError(f"Organization {org_id} not found")

        # Generate key
        key_id = str(uuid.uuid4())
        raw_key = f"{org_id}.{secrets.token_urlsafe(32)}"
        key_hash_value = hash_api_key(raw_key)

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=expires_in_days) if expires_in_days else None

        # Default permissions based on organization tier
        if permissions is None:
            permissions = ROLE_PERMISSIONS.get("developer", [])

        api_key = APIKey(
            key_id=key_id,
            org_id=org_id,
            name=name,
            key_hash=key_hash_value,
            created_at=now,
            expires_at=expires_at,
            permissions=permissions,
            allowed_ips=allowed_ips or [],
            created_by=created_by,
        )

        self.store.save_api_key(api_key)
        logger.info(f"AUDIT: Created API key {key_id} for org {org_id}")

        return api_key, raw_key

    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key metadata by ID."""
        return self.store.get_api_key(key_id)

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        key = self.store.get_api_key(key_id)
        if not key:
            return False

        key.is_active = False
        self.store.save_api_key(key)

        logger.info(f"AUDIT: Revoked API key {key_id}")
        return True

    def delete_api_key(self, key_id: str) -> bool:
        """Delete an API key."""
        key = self.store.get_api_key(key_id)
        if not key:
            return False

        self.store.delete_api_key(key_id)
        logger.info(f"AUDIT: Deleted API key {key_id}")
        return True

    # =========================================================================
    # Authentication
    # =========================================================================

    def authenticate_user(
        self,
        email: str,
        password: str,
        ip_address: Optional[str] = None,
    ) -> Optional[User]:
        """
        Authenticate a user with email and password.

        Returns:
            User object if authentication succeeds, None otherwise.
        """
        user = self.store.get_user_by_email(email)
        if not user:
            logger.warning(f"SECURITY: Login attempt for unknown email {email}")
            return None

        if not user.is_active:
            logger.warning(f"SECURITY: Login attempt for inactive user {user.user_id}")
            return None

        if not user.password_hash:
            logger.warning(f"SECURITY: Login attempt for user without password {user.user_id}")
            return None

        if not verify_password(password, user.password_hash):
            logger.warning(f"SECURITY: Failed login for user {user.user_id} - invalid password")
            return None

        # Update last login
        user.last_login_at = datetime.now(timezone.utc)
        user.last_login_ip = ip_address
        self.store.save_user(user)

        logger.info(f"AUDIT: User {user.user_id} authenticated successfully")
        return user

    def authenticate_api_key(
        self,
        raw_key: str,
        ip_address: Optional[str] = None,
    ) -> Optional[APIKey]:
        """
        Authenticate with an API key.

        Returns:
            APIKey object if authentication succeeds, None otherwise.
        """
        key_hash_value = hash_api_key(raw_key)
        api_key = self.store.get_api_key_by_hash(key_hash_value)

        if not api_key:
            logger.warning("SECURITY: Authentication attempt with unknown API key")
            return None

        if not api_key.is_active:
            logger.warning(f"SECURITY: Authentication attempt with revoked API key {api_key.key_id}")
            return None

        # Check expiration
        if api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
            logger.warning(f"SECURITY: Authentication attempt with expired API key {api_key.key_id}")
            return None

        # Check IP restriction
        if api_key.allowed_ips and ip_address:
            if ip_address not in api_key.allowed_ips:
                logger.warning(f"SECURITY: API key {api_key.key_id} used from unauthorized IP {ip_address}")
                return None

        # Update last used
        api_key.last_used_at = datetime.now(timezone.utc)
        self.store.save_api_key(api_key)

        logger.info(f"AUDIT: API key {api_key.key_id} authenticated successfully")
        return api_key

    # =========================================================================
    # Token Management
    # =========================================================================

    def issue_token(
        self,
        user: Optional[User] = None,
        api_key: Optional[APIKey] = None,
        audiences: Optional[List[str]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """
        Issue a JWT token for a user or API key.

        Args:
            user: User to issue token for
            api_key: API key to issue token for
            audiences: List of products this token is valid for
            ttl: Token TTL in seconds (overrides config)

        Returns:
            JWT token string
        """
        if not HAS_JWT:
            raise RuntimeError("PyJWT is required for token issuance")

        if not user and not api_key:
            raise ValueError("Either user or api_key must be provided")

        now = int(time.time())
        token_ttl = ttl or self.config.access_token_ttl

        # Determine subject and permissions
        if user:
            org = self.store.get_organization(user.org_id)
            subject = user.user_id
            org_id = user.org_id
            auth_type = "user"
            permissions = list(user.get_effective_permissions())
            email = user.email
            name = user.name
            role = user.role

            # Filter audiences to enabled products
            if audiences:
                audiences = [a for a in audiences if a in org.enabled_products]
            else:
                audiences = list(org.enabled_products)
        else:
            org = self.store.get_organization(api_key.org_id)
            subject = api_key.key_id
            org_id = api_key.org_id
            auth_type = "api_key"
            permissions = api_key.permissions
            email = None
            name = api_key.name
            role = None

            # Filter audiences to enabled products
            if audiences:
                audiences = [a for a in audiences if a in org.enabled_products]
            else:
                audiences = list(org.enabled_products)

        claims = TokenClaims(
            sub=subject,
            iss=self.config.issuer,
            aud=audiences,
            exp=now + token_ttl,
            iat=now,
            jti=str(uuid.uuid4()),
            org_id=org_id,
            auth_type=auth_type,
            permissions=permissions,
            email=email,
            name=name,
            role=role,
        )

        token = jwt.encode(
            claims.to_dict(),
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm,
        )

        logger.debug(f"AUDIT: Issued token for {auth_type} {subject}")
        return token

    def verify_token(
        self,
        token: str,
        required_audience: Optional[str] = None,
        required_permission: Optional[str] = None,
    ) -> Optional[TokenClaims]:
        """
        Verify a JWT token.

        Args:
            token: JWT token string
            required_audience: Required product audience (e.g., "fhe_gbdt")
            required_permission: Required permission (e.g., "gbdt:predict")

        Returns:
            TokenClaims if valid, None otherwise
        """
        if not HAS_JWT:
            raise RuntimeError("PyJWT is required for token verification")

        try:
            # Decode and verify
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
                issuer=self.config.issuer,
            )

            claims = TokenClaims.from_dict(payload)

            # Check blacklist
            if claims.jti in self._token_blacklist:
                logger.warning(f"SECURITY: Blacklisted token used: {claims.jti}")
                return None

            # Check audience
            if required_audience and required_audience not in claims.aud:
                logger.warning(f"SECURITY: Token missing required audience: {required_audience}")
                return None

            # Check permission
            if required_permission and required_permission not in claims.permissions:
                logger.warning(f"SECURITY: Token missing required permission: {required_permission}")
                return None

            # Verify organization is still active
            org = self.store.get_organization(claims.org_id)
            if not org or not org.is_active:
                logger.warning(f"SECURITY: Token for inactive organization: {claims.org_id}")
                return None

            return claims

        except jwt.ExpiredSignatureError:
            logger.warning("SECURITY: Expired token used")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"SECURITY: Invalid token: {e}")
            return None

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token by adding it to the blacklist.

        Note: In production, use Redis or similar for distributed blacklist.
        """
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
                options={"verify_exp": False},
            )
            jti = payload.get("jti")
            if jti:
                self._token_blacklist.add(jti)
                logger.info(f"AUDIT: Revoked token {jti}")
                return True
        except jwt.InvalidTokenError:
            pass
        return False

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def login(
        self,
        email: str,
        password: str,
        ip_address: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Complete login flow: authenticate and issue tokens.

        Returns:
            Dict with access_token, token_type, expires_in, user info
        """
        user = self.authenticate_user(email, password, ip_address)
        if not user:
            return None

        access_token = self.issue_token(user=user)

        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": self.config.access_token_ttl,
            "user": user.to_dict(),
        }

    def authenticate_request(
        self,
        authorization_header: Optional[str] = None,
        api_key_header: Optional[str] = None,
        required_audience: Optional[str] = None,
        required_permission: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Optional[TokenClaims]:
        """
        Authenticate a request using either Bearer token or API key.

        This is the main entry point for request authentication in services.
        """
        # Try Bearer token first
        if authorization_header and authorization_header.startswith("Bearer "):
            token = authorization_header[7:]
            return self.verify_token(
                token,
                required_audience=required_audience,
                required_permission=required_permission,
            )

        # Try API key
        if api_key_header:
            api_key = self.authenticate_api_key(api_key_header, ip_address)
            if api_key:
                # Check permission
                if required_permission and required_permission not in api_key.permissions:
                    logger.warning(f"SECURITY: API key missing permission: {required_permission}")
                    return None

                # Issue a short-lived token for the request
                token = self.issue_token(api_key=api_key, audiences=[required_audience] if required_audience else None)
                return self.verify_token(token)

        return None


# =============================================================================
# Global Instance
# =============================================================================

_identity_provider: Optional[UnifiedIdentityProvider] = None


def get_identity_provider() -> UnifiedIdentityProvider:
    """Get the global identity provider instance."""
    global _identity_provider
    if _identity_provider is None:
        config = IdentityProviderConfig(
            issuer=os.getenv("IDENTITY_ISSUER", "https://identity.product-suite.local"),
            jwt_secret=os.getenv("JWT_SECRET"),
            store_type=os.getenv("IDENTITY_STORE_TYPE", "memory"),
            store_path=os.getenv("IDENTITY_STORE_PATH"),
        )
        _identity_provider = UnifiedIdentityProvider(config)
    return _identity_provider


def init_identity_provider(config: IdentityProviderConfig) -> UnifiedIdentityProvider:
    """Initialize the global identity provider with custom config."""
    global _identity_provider
    _identity_provider = UnifiedIdentityProvider(config)
    return _identity_provider
