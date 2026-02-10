"""
QA Tests for Unified Identity Service

Tests the unified identity provider that enables single sign-on across
all products in the suite (FHE-GBDT, TenSafe, etc.).
"""

import os
import sys
import json
import time
import tempfile
import unittest
import hashlib
import secrets
import base64
from datetime import datetime, timezone, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Check for JWT availability - must use bare except due to pyo3 panic
HAS_JWT = False
jwt = None
try:
    import jwt as _jwt
    jwt = _jwt
    HAS_JWT = True
except:
    print("WARNING: PyJWT not available, JWT tests will be skipped")

from services.identity.provider import (
    UnifiedIdentityProvider,
    IdentityProviderConfig,
    Organization,
    User,
    APIKey,
    TokenClaims,
    Product,
    Permission,
    ROLE_PERMISSIONS,
    get_identity_provider,
    init_identity_provider,
    InMemoryStore,
    FileStore,
    hash_password,
    verify_password,
    hash_api_key,
)


class TestPasswordHashing(unittest.TestCase):
    """Test password hashing functionality."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "secure-password-123"
        hashed = hash_password(password)

        self.assertIsInstance(hashed, str)
        self.assertTrue(hashed.startswith("pbkdf2:sha256:"))
        self.assertNotEqual(hashed, password)

    def test_verify_password_correct(self):
        """Test verifying correct password."""
        password = "my-secret-password"
        hashed = hash_password(password)

        self.assertTrue(verify_password(password, hashed))

    def test_verify_password_incorrect(self):
        """Test verifying incorrect password."""
        password = "correct-password"
        hashed = hash_password(password)

        self.assertFalse(verify_password("wrong-password", hashed))

    def test_hash_api_key(self):
        """Test API key hashing."""
        key = "org-123.secret-key-value"
        hashed = hash_api_key(key)

        self.assertIsInstance(hashed, str)
        self.assertEqual(len(hashed), 64)  # SHA-256 hex


class TestInMemoryStore(unittest.TestCase):
    """Test in-memory storage backend."""

    def setUp(self):
        self.store = InMemoryStore()

    def test_organization_crud(self):
        """Test organization CRUD operations."""
        now = datetime.now(timezone.utc)
        org = Organization(
            org_id="org-123",
            name="Test Org",
            created_at=now,
            updated_at=now,
        )

        # Create
        self.store.save_organization(org)

        # Read
        retrieved = self.store.get_organization("org-123")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Test Org")

        # List
        orgs = self.store.list_organizations()
        self.assertEqual(len(orgs), 1)

        # Delete
        self.store.delete_organization("org-123")
        self.assertIsNone(self.store.get_organization("org-123"))

    def test_user_crud(self):
        """Test user CRUD operations."""
        now = datetime.now(timezone.utc)
        user = User(
            user_id="user-123",
            org_id="org-123",
            email="test@example.com",
            name="Test User",
            created_at=now,
            updated_at=now,
        )

        # Create
        self.store.save_user(user)

        # Read by ID
        retrieved = self.store.get_user("user-123")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.email, "test@example.com")

        # Read by email
        retrieved_by_email = self.store.get_user_by_email("test@example.com")
        self.assertIsNotNone(retrieved_by_email)
        self.assertEqual(retrieved_by_email.user_id, "user-123")

        # List by org
        users = self.store.list_users_by_org("org-123")
        self.assertEqual(len(users), 1)

        # Delete
        self.store.delete_user("user-123")
        self.assertIsNone(self.store.get_user("user-123"))

    def test_api_key_crud(self):
        """Test API key CRUD operations."""
        now = datetime.now(timezone.utc)
        api_key = APIKey(
            key_id="key-123",
            org_id="org-123",
            name="Test Key",
            key_hash="abc123hash",
            created_at=now,
            expires_at=now + timedelta(days=365),
        )

        # Create
        self.store.save_api_key(api_key)

        # Read by ID
        retrieved = self.store.get_api_key("key-123")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Test Key")

        # Read by hash
        retrieved_by_hash = self.store.get_api_key_by_hash("abc123hash")
        self.assertIsNotNone(retrieved_by_hash)
        self.assertEqual(retrieved_by_hash.key_id, "key-123")

        # List by org
        keys = self.store.list_api_keys_by_org("org-123")
        self.assertEqual(len(keys), 1)

        # Delete
        self.store.delete_api_key("key-123")
        self.assertIsNone(self.store.get_api_key("key-123"))


class TestFileStore(unittest.TestCase):
    """Test file-based storage backend."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = FileStore(self.temp_dir)

    def test_organization_persistence(self):
        """Test organization persistence to file."""
        now = datetime.now(timezone.utc)
        org = Organization(
            org_id="org-file-123",
            name="File Test Org",
            created_at=now,
            updated_at=now,
            tier="business",
        )

        # Save
        self.store.save_organization(org)

        # Verify file exists
        import os
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "orgs", "org-file-123.json")))

        # Reload from disk
        new_store = FileStore(self.temp_dir)
        retrieved = new_store.get_organization("org-file-123")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "File Test Org")
        self.assertEqual(retrieved.tier, "business")


class TestUnifiedIdentityProvider(unittest.TestCase):
    """Test unified identity provider."""

    def setUp(self):
        config = IdentityProviderConfig(
            issuer="https://test.identity.local",
            jwt_secret="test-secret-key-for-testing",
            store_type="memory",
        )
        self.idp = UnifiedIdentityProvider(config)

    def test_create_organization(self):
        """Test organization creation."""
        org = self.idp.create_organization(
            name="Acme Corp",
            tier="business",
            enabled_products={Product.FHE_GBDT.value, Product.TENSAFE.value},
        )

        self.assertIsNotNone(org.org_id)
        self.assertEqual(org.name, "Acme Corp")
        self.assertEqual(org.tier, "business")
        self.assertIn(Product.FHE_GBDT.value, org.enabled_products)
        self.assertIn(Product.TENSAFE.value, org.enabled_products)

    def test_create_user(self):
        """Test user creation."""
        # Create org first
        org = self.idp.create_organization(name="Test Org")

        # Create user
        user = self.idp.create_user(
            org_id=org.org_id,
            email="developer@test.com",
            name="Test Developer",
            password="secure-pass-123",
            role="developer",
        )

        self.assertIsNotNone(user.user_id)
        self.assertEqual(user.email, "developer@test.com")
        self.assertEqual(user.role, "developer")
        self.assertIsNotNone(user.password_hash)

    def test_create_user_duplicate_email(self):
        """Test that duplicate emails are rejected."""
        org = self.idp.create_organization(name="Test Org")

        self.idp.create_user(
            org_id=org.org_id,
            email="unique@test.com",
            name="First User",
            password="pass123",
        )

        with self.assertRaises(ValueError) as ctx:
            self.idp.create_user(
                org_id=org.org_id,
                email="unique@test.com",
                name="Second User",
                password="pass456",
            )

        self.assertIn("already registered", str(ctx.exception))

    def test_authenticate_user(self):
        """Test user authentication."""
        org = self.idp.create_organization(name="Auth Test Org")
        self.idp.create_user(
            org_id=org.org_id,
            email="auth@test.com",
            name="Auth User",
            password="correct-password",
        )

        # Correct password
        user = self.idp.authenticate_user("auth@test.com", "correct-password")
        self.assertIsNotNone(user)
        self.assertEqual(user.email, "auth@test.com")

        # Wrong password
        user = self.idp.authenticate_user("auth@test.com", "wrong-password")
        self.assertIsNone(user)

        # Unknown email
        user = self.idp.authenticate_user("unknown@test.com", "any-password")
        self.assertIsNone(user)

    def test_create_api_key(self):
        """Test API key creation."""
        org = self.idp.create_organization(name="API Key Test Org")

        api_key, raw_key = self.idp.create_api_key(
            org_id=org.org_id,
            name="CI/CD Key",
            permissions=[Permission.GBDT_PREDICT.value, Permission.GBDT_TRAIN.value],
            expires_in_days=90,
        )

        self.assertIsNotNone(api_key.key_id)
        self.assertEqual(api_key.name, "CI/CD Key")
        self.assertIn(Permission.GBDT_PREDICT.value, api_key.permissions)
        self.assertTrue(raw_key.startswith(org.org_id + "."))

    def test_authenticate_api_key(self):
        """Test API key authentication."""
        org = self.idp.create_organization(name="API Key Auth Org")
        api_key, raw_key = self.idp.create_api_key(
            org_id=org.org_id,
            name="Test Key",
        )

        # Valid key
        authenticated = self.idp.authenticate_api_key(raw_key)
        self.assertIsNotNone(authenticated)
        self.assertEqual(authenticated.key_id, api_key.key_id)

        # Invalid key
        authenticated = self.idp.authenticate_api_key("invalid.key.value")
        self.assertIsNone(authenticated)

    def test_revoke_api_key(self):
        """Test API key revocation."""
        org = self.idp.create_organization(name="Revoke Test Org")
        api_key, raw_key = self.idp.create_api_key(
            org_id=org.org_id,
            name="To Be Revoked",
        )

        # Revoke
        result = self.idp.revoke_api_key(api_key.key_id)
        self.assertTrue(result)

        # Try to authenticate
        authenticated = self.idp.authenticate_api_key(raw_key)
        self.assertIsNone(authenticated)


class TestTokenManagement(unittest.TestCase):
    """Test JWT token management."""

    def setUp(self):
        config = IdentityProviderConfig(
            issuer="https://test.identity.local",
            jwt_secret="test-jwt-secret-key",
            access_token_ttl=3600,
            store_type="memory",
        )
        self.idp = UnifiedIdentityProvider(config)

        # Create test org and user
        self.org = self.idp.create_organization(
            name="Token Test Org",
            enabled_products={Product.FHE_GBDT.value, Product.TENSAFE.value},
        )
        self.user = self.idp.create_user(
            org_id=self.org.org_id,
            email="token@test.com",
            name="Token User",
            password="token-pass",
            role="developer",
        )

    def test_issue_user_token(self):
        """Test issuing token for user."""
        if not HAS_JWT:
            self.skipTest("PyJWT not available")

        token = self.idp.issue_token(user=self.user)

        self.assertIsNotNone(token)
        self.assertIsInstance(token, str)

        # Decode and verify
        decoded = jwt.decode(
            token,
            "test-jwt-secret-key",
            algorithms=["HS256"],
            issuer="https://test.identity.local",
        )

        self.assertEqual(decoded["sub"], self.user.user_id)
        self.assertEqual(decoded["org_id"], self.org.org_id)
        self.assertEqual(decoded["auth_type"], "user")
        self.assertIn(Product.FHE_GBDT.value, decoded["aud"])

    def test_issue_api_key_token(self):
        """Test issuing token for API key."""
        if not HAS_JWT:
            self.skipTest("PyJWT not available")

        api_key, _ = self.idp.create_api_key(
            org_id=self.org.org_id,
            name="Token Test Key",
            permissions=[Permission.GBDT_PREDICT.value],
        )

        token = self.idp.issue_token(api_key=api_key)
        decoded = jwt.decode(
            token,
            "test-jwt-secret-key",
            algorithms=["HS256"],
            issuer="https://test.identity.local",
        )

        self.assertEqual(decoded["sub"], api_key.key_id)
        self.assertEqual(decoded["auth_type"], "api_key")

    def test_verify_token(self):
        """Test token verification."""
        if not HAS_JWT:
            self.skipTest("PyJWT not available")

        token = self.idp.issue_token(user=self.user)

        # Valid token
        claims = self.idp.verify_token(token)
        self.assertIsNotNone(claims)
        self.assertEqual(claims.sub, self.user.user_id)

        # Verify with required audience
        claims = self.idp.verify_token(
            token,
            required_audience=Product.FHE_GBDT.value,
        )
        self.assertIsNotNone(claims)

        # Verify with required permission
        claims = self.idp.verify_token(
            token,
            required_permission=Permission.GBDT_PREDICT.value,
        )
        self.assertIsNotNone(claims)

        # Invalid audience
        claims = self.idp.verify_token(
            token,
            required_audience="invalid_product",
        )
        self.assertIsNone(claims)

    def test_revoke_token(self):
        """Test token revocation."""
        if not HAS_JWT:
            self.skipTest("PyJWT not available")

        token = self.idp.issue_token(user=self.user)

        # Token is valid
        claims = self.idp.verify_token(token)
        self.assertIsNotNone(claims)

        # Revoke
        result = self.idp.revoke_token(token)
        self.assertTrue(result)

        # Token is now invalid
        claims = self.idp.verify_token(token)
        self.assertIsNone(claims)


class TestLoginFlow(unittest.TestCase):
    """Test complete login flow."""

    def setUp(self):
        config = IdentityProviderConfig(
            issuer="https://test.identity.local",
            jwt_secret="test-login-secret",
            store_type="memory",
        )
        self.idp = UnifiedIdentityProvider(config)

        # Create test org and user
        self.org = self.idp.create_organization(name="Login Test Org")
        self.user = self.idp.create_user(
            org_id=self.org.org_id,
            email="login@test.com",
            name="Login User",
            password="login-password",
            role="admin",
        )

    def test_login_success(self):
        """Test successful login."""
        if not HAS_JWT:
            self.skipTest("PyJWT not available")

        result = self.idp.login(
            email="login@test.com",
            password="login-password",
        )

        self.assertIsNotNone(result)
        self.assertIn("access_token", result)
        self.assertEqual(result["token_type"], "Bearer")
        self.assertIn("user", result)
        self.assertEqual(result["user"]["email"], "login@test.com")

    def test_login_failure(self):
        """Test failed login."""
        result = self.idp.login(
            email="login@test.com",
            password="wrong-password",
        )
        self.assertIsNone(result)

        result = self.idp.login(
            email="nonexistent@test.com",
            password="any-password",
        )
        self.assertIsNone(result)


class TestCrossProductAccess(unittest.TestCase):
    """Test cross-product access with unified auth."""

    def setUp(self):
        config = IdentityProviderConfig(
            issuer="https://test.identity.local",
            jwt_secret="cross-product-secret",
            store_type="memory",
        )
        self.idp = UnifiedIdentityProvider(config)

        # Create org with both products
        self.org = self.idp.create_organization(
            name="Cross Product Org",
            enabled_products={Product.FHE_GBDT.value, Product.TENSAFE.value},
        )

        # Create user with permissions for both products
        self.user = self.idp.create_user(
            org_id=self.org.org_id,
            email="crossproduct@test.com",
            name="Cross Product User",
            password="cross-pass",
            role="developer",
        )

    def test_single_token_multiple_products(self):
        """Test that a single token works for multiple products."""
        if not HAS_JWT:
            self.skipTest("PyJWT not available")

        token = self.idp.issue_token(user=self.user)

        # Token works for FHE-GBDT
        claims = self.idp.verify_token(
            token,
            required_audience=Product.FHE_GBDT.value,
        )
        self.assertIsNotNone(claims)

        # Same token works for TenSafe
        claims = self.idp.verify_token(
            token,
            required_audience=Product.TENSAFE.value,
        )
        self.assertIsNotNone(claims)

    def test_user_has_both_product_permissions(self):
        """Test that user has permissions for both products."""
        permissions = self.user.get_effective_permissions()

        # FHE-GBDT permissions
        self.assertIn(Permission.GBDT_PREDICT.value, permissions)
        self.assertIn(Permission.GBDT_TRAIN.value, permissions)

        # TenSafe permissions
        self.assertIn(Permission.TENSAFE_ADAPT.value, permissions)
        self.assertIn(Permission.TENSAFE_INFERENCE.value, permissions)

    def test_product_access_helpers(self):
        """Test product access helper methods."""
        self.assertTrue(self.user.has_product_access(Product.FHE_GBDT))
        self.assertTrue(self.user.has_product_access(Product.TENSAFE))
        self.assertFalse(self.user.has_product_access(Product.PLATFORM))


class TestRolePermissions(unittest.TestCase):
    """Test role-based permissions."""

    def test_viewer_permissions(self):
        """Test viewer role permissions."""
        perms = ROLE_PERMISSIONS["viewer"]

        # Can predict/inference
        self.assertIn(Permission.GBDT_PREDICT.value, perms)
        self.assertIn(Permission.TENSAFE_INFERENCE.value, perms)

        # Cannot train/adapt
        self.assertNotIn(Permission.GBDT_TRAIN.value, perms)
        self.assertNotIn(Permission.TENSAFE_ADAPT.value, perms)

    def test_developer_permissions(self):
        """Test developer role permissions."""
        perms = ROLE_PERMISSIONS["developer"]

        # Can predict and train
        self.assertIn(Permission.GBDT_PREDICT.value, perms)
        self.assertIn(Permission.GBDT_TRAIN.value, perms)
        self.assertIn(Permission.GBDT_MODEL_UPLOAD.value, perms)

        # Cannot delete or manage keys
        self.assertNotIn(Permission.GBDT_MODEL_DELETE.value, perms)
        self.assertNotIn(Permission.GBDT_KEYS_MANAGE.value, perms)

    def test_admin_permissions(self):
        """Test admin role permissions."""
        perms = ROLE_PERMISSIONS["admin"]

        # Has all product permissions
        self.assertIn(Permission.GBDT_PREDICT.value, perms)
        self.assertIn(Permission.GBDT_MODEL_DELETE.value, perms)
        self.assertIn(Permission.GBDT_KEYS_MANAGE.value, perms)

        # Has org management
        self.assertIn(Permission.ORG_MANAGE.value, perms)
        self.assertIn(Permission.USER_MANAGE.value, perms)

        # No platform admin
        self.assertNotIn(Permission.PLATFORM_ADMIN.value, perms)

    def test_platform_admin_permissions(self):
        """Test platform admin role permissions."""
        perms = ROLE_PERMISSIONS["platform_admin"]

        # Has everything
        for perm in Permission:
            self.assertIn(perm.value, perms)


def run_tests():
    """Run all unified identity tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPasswordHashing))
    suite.addTests(loader.loadTestsFromTestCase(TestInMemoryStore))
    suite.addTests(loader.loadTestsFromTestCase(TestFileStore))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedIdentityProvider))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestLoginFlow))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossProductAccess))
    suite.addTests(loader.loadTestsFromTestCase(TestRolePermissions))

    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
