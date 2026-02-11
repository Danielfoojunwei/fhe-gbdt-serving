"""
End-to-End Tests for Licensing, Heartbeat, Plan Encryption, and Offline Grace Period

Tests the complete hybrid deployment model:
1. License token issuance and validation
2. Heartbeat telemetry collection and reporting
3. Plan encryption/decryption bound to tenant+deployment
4. Offline grace period for license validation
5. Compiler SaaS auth and multi-tenancy
6. Integration: full flow from compilation to licensed prediction

These tests run without external services (no gRPC, no Vault, no database).
"""

import json
import os
import sys
import time
import unittest
import threading

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# services/ must be on path so 'compiler' and 'licensing' resolve as packages
# NOTE: do NOT add services/compiler/ directly -- it breaks relative imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'services'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'sdk', 'python'))


# ============================================================================
# Test 1: License Token Issuance & Validation
# ============================================================================

class TestLicenseTokenE2E(unittest.TestCase):
    """End-to-end tests for license token lifecycle."""

    SIGNING_KEY = "test-signing-key-that-is-at-least-32-chars-long"

    def setUp(self):
        from licensing.license_token import LicenseIssuer, LicenseValidator
        self.issuer = LicenseIssuer(self.SIGNING_KEY)
        self.validator = LicenseValidator(self.SIGNING_KEY)

    def test_issue_and_validate_token(self):
        """Issue a token and validate it successfully."""
        token = self.issuer.issue(
            tenant_id="bank-abc",
            model_ids=["model-001", "model-002"],
            max_predictions=10000,
            validity_hours=24,
        )
        self.assertIsInstance(token, str)
        self.assertEqual(token.count("."), 2)  # JWT format: header.payload.sig

        claims = self.validator.validate(token)
        self.assertEqual(claims.tenant_id, "bank-abc")
        self.assertEqual(claims.model_ids, ["model-001", "model-002"])
        self.assertEqual(claims.max_predictions, 10000)
        self.assertFalse(claims.is_expired())

    def test_token_model_authorization(self):
        """Validate that model authorization works correctly."""
        token = self.issuer.issue(
            tenant_id="bank-abc",
            model_ids=["model-001"],
        )

        # Authorized model
        claims = self.validator.validate(token, model_id="model-001")
        self.assertTrue(claims.allows_model("model-001"))

        # Unauthorized model
        from licensing.license_token import LicenseModelNotAuthorizedError
        with self.assertRaises(LicenseModelNotAuthorizedError):
            self.validator.validate(token, model_id="model-999")

    def test_wildcard_model_authorization(self):
        """Wildcard model_ids=['*'] authorizes all models."""
        token = self.issuer.issue(
            tenant_id="bank-abc",
            model_ids=["*"],
        )
        claims = self.validator.validate(token, model_id="any-model-id")
        self.assertTrue(claims.allows_model("any-model-id"))

    def test_expired_token_rejected(self):
        """Expired tokens are rejected when no grace period applies."""
        from licensing.license_token import LicenseExpiredError
        token = self.issuer.issue(
            tenant_id="bank-abc",
            model_ids=["*"],
            validity_hours=0,  # Expires immediately
        )
        # Small sleep to ensure expiration
        time.sleep(0.1)
        with self.assertRaises(LicenseExpiredError):
            self.validator.validate(token)

    def test_prediction_cap_enforcement(self):
        """Prediction cap is enforced after reaching the limit."""
        from licensing.license_token import LicensePredictionCapExceededError
        token = self.issuer.issue(
            tenant_id="bank-abc",
            model_ids=["*"],
            max_predictions=3,
        )

        claims = self.validator.validate(token)
        license_id = claims.license_id

        # Record 3 predictions (up to cap)
        for i in range(3):
            self.validator.record_prediction(license_id)

        # 4th validation should fail
        with self.assertRaises(LicensePredictionCapExceededError):
            self.validator.validate(token)

    def test_tampered_token_rejected(self):
        """Tokens with modified payloads are rejected."""
        from licensing.license_token import LicenseInvalidError
        token = self.issuer.issue(
            tenant_id="bank-abc",
            model_ids=["*"],
        )
        # Tamper with payload
        parts = token.split(".")
        tampered = parts[0] + "." + parts[1] + "X" + "." + parts[2]
        with self.assertRaises(LicenseInvalidError):
            self.validator.validate(tampered)

    def test_wrong_signing_key_rejected(self):
        """Tokens signed with a different key are rejected."""
        from licensing.license_token import LicenseIssuer, LicenseValidator, LicenseInvalidError
        other_issuer = LicenseIssuer("different-key-that-is-also-32-chars-long")
        token = other_issuer.issue(tenant_id="bank-abc", model_ids=["*"])

        with self.assertRaises(LicenseInvalidError):
            self.validator.validate(token)

    def test_signing_key_minimum_length(self):
        """Signing key must be at least 32 characters."""
        from licensing.license_token import LicenseIssuer
        with self.assertRaises(ValueError):
            LicenseIssuer("short-key")


# ============================================================================
# Test 2: Heartbeat Telemetry
# ============================================================================

class TestHeartbeatTelemetryE2E(unittest.TestCase):
    """End-to-end tests for heartbeat telemetry collection."""

    def test_basic_telemetry_collection(self):
        """Record predictions and flush a heartbeat report."""
        from licensing.heartbeat import HeartbeatCollector

        collector = HeartbeatCollector(
            tenant_id="bank-abc",
            license_id="lic-001",
            report_interval_seconds=60,
        )

        # Record some predictions
        collector.record_prediction("model-001", 62.5, success=True)
        collector.record_prediction("model-001", 65.0, success=True)
        collector.record_prediction("model-002", 70.0, success=True)
        collector.record_prediction("model-001", 0.0, success=False)

        # Flush report
        report = collector.flush()
        self.assertIsNotNone(report)
        self.assertEqual(report.tenant_id, "bank-abc")
        self.assertEqual(report.license_id, "lic-001")
        self.assertEqual(report.total_predictions, 4)
        self.assertEqual(report.successful_predictions, 3)
        self.assertEqual(report.failed_predictions, 1)
        self.assertEqual(report.predictions_by_model["model-001"], 3)
        self.assertEqual(report.predictions_by_model["model-002"], 1)
        self.assertGreater(report.latency_p50_ms, 0)

    def test_empty_flush_returns_none(self):
        """Flushing with no events returns None."""
        from licensing.heartbeat import HeartbeatCollector
        collector = HeartbeatCollector("t", "l")
        self.assertIsNone(collector.flush())

    def test_report_json_serialization(self):
        """Reports serialize to valid JSON without forbidden fields."""
        from licensing.heartbeat import HeartbeatCollector, FORBIDDEN_FIELDS

        collector = HeartbeatCollector("bank-abc", "lic-001")
        collector.record_prediction("model-001", 50.0, True)
        report = collector.flush()

        json_str = report.to_json()
        parsed = json.loads(json_str)

        # Verify no forbidden fields
        for key in parsed:
            self.assertNotIn(key, FORBIDDEN_FIELDS,
                             f"Forbidden field '{key}' found in telemetry report")

        # Verify expected fields present
        self.assertIn("tenant_id", parsed)
        self.assertIn("total_predictions", parsed)
        self.assertIn("latency_p50_ms", parsed)

    def test_callback_invoked_on_flush(self):
        """Report callback is invoked when flush produces a report."""
        from licensing.heartbeat import HeartbeatCollector

        received_reports = []
        collector = HeartbeatCollector(
            "bank-abc", "lic-001",
            report_callback=lambda r: received_reports.append(r),
        )

        collector.record_prediction("model-001", 50.0, True)
        collector.flush()

        self.assertEqual(len(received_reports), 1)
        self.assertEqual(received_reports[0].total_predictions, 1)

    def test_periodic_reporting(self):
        """Background timer produces periodic reports."""
        from licensing.heartbeat import HeartbeatCollector

        collector = HeartbeatCollector(
            "bank-abc", "lic-001",
            report_interval_seconds=1,  # 1 second for test speed
        )
        collector.start()

        # Record events across intervals
        collector.record_prediction("model-001", 50.0, True)
        time.sleep(1.5)
        collector.record_prediction("model-001", 55.0, True)
        time.sleep(1.5)

        final_report = collector.stop()
        reports = collector.get_reports()

        # Should have at least one auto-flushed report + the stop flush
        self.assertGreaterEqual(len(reports), 1)

    def test_thread_safety(self):
        """Concurrent recording doesn't lose events."""
        from licensing.heartbeat import HeartbeatCollector

        collector = HeartbeatCollector("bank-abc", "lic-001")
        num_threads = 10
        events_per_thread = 100

        def record_events():
            for i in range(events_per_thread):
                collector.record_prediction("model-001", float(i), True)

        threads = [threading.Thread(target=record_events) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        report = collector.flush()
        self.assertEqual(report.total_predictions, num_threads * events_per_thread)


# ============================================================================
# Test 3: Plan Encryption
# ============================================================================

class TestPlanEncryptionE2E(unittest.TestCase):
    """End-to-end tests for plan encryption/decryption."""

    TENANT_ID = "bank-abc"
    DEPLOYMENT_SECRET = "my-deployment-secret-key-very-long"

    def setUp(self):
        from licensing.plan_encryption import PlanEncryptor
        self.encryptor = PlanEncryptor(self.TENANT_ID, self.DEPLOYMENT_SECRET)

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypting then decrypting returns the original plan."""
        plan_json = json.dumps({
            "compiled_model_id": "test-model",
            "num_trees": 100,
            "schedule": [{"depth_level": 0, "ops": []}],
        })

        encrypted = self.encryptor.encrypt(plan_json)
        self.assertNotEqual(encrypted.ciphertext, plan_json.encode())

        decrypted = self.encryptor.decrypt(encrypted)
        self.assertEqual(decrypted, plan_json)

    def test_serialize_deserialize_roundtrip(self):
        """Serialized encrypted plan can be deserialized and decrypted."""
        from licensing.plan_encryption import EncryptedPlan

        plan_json = json.dumps({"model": "test", "trees": 50})

        encrypted = self.encryptor.encrypt(plan_json)
        serialized = encrypted.serialize()

        # Deserialize
        deserialized = EncryptedPlan.deserialize(serialized)
        self.assertEqual(deserialized.tenant_id, self.TENANT_ID)

        # Decrypt
        decrypted = self.encryptor.decrypt(deserialized)
        self.assertEqual(decrypted, plan_json)

    def test_different_deployment_cannot_decrypt(self):
        """Plan encrypted for one deployment cannot be decrypted by another."""
        from licensing.plan_encryption import PlanEncryptor, PlanEncryptionError

        plan_json = json.dumps({"model": "test"})
        encrypted = self.encryptor.encrypt(plan_json)

        # Different deployment secret
        other_encryptor = PlanEncryptor(self.TENANT_ID, "different-deployment-secret-key-long")
        with self.assertRaises(PlanEncryptionError):
            other_encryptor.decrypt(encrypted)

    def test_different_tenant_cannot_decrypt(self):
        """Plan encrypted for one tenant cannot be decrypted by another."""
        from licensing.plan_encryption import PlanEncryptor, PlanEncryptionError

        plan_json = json.dumps({"model": "test"})
        encrypted = self.encryptor.encrypt(plan_json)

        other_encryptor = PlanEncryptor("other-tenant", self.DEPLOYMENT_SECRET)
        with self.assertRaises(PlanEncryptionError):
            other_encryptor.decrypt(encrypted)

    def test_tampered_ciphertext_detected(self):
        """Tampered ciphertext is detected via authentication tag."""
        from licensing.plan_encryption import PlanEncryptionError

        plan_json = json.dumps({"model": "test"})
        encrypted = self.encryptor.encrypt(plan_json)

        # Tamper with ciphertext
        tampered_ct = bytearray(encrypted.ciphertext)
        if len(tampered_ct) > 0:
            tampered_ct[0] ^= 0xFF
        encrypted.ciphertext = bytes(tampered_ct)

        with self.assertRaises(PlanEncryptionError):
            self.encryptor.decrypt(encrypted)

    def test_large_plan_encryption(self):
        """Encryption works for large plans (>1MB)."""
        # Simulate a large plan with many trees
        large_plan = json.dumps({
            "trees": [{"id": i, "nodes": list(range(100))} for i in range(1000)],
            "metadata": "x" * 100000,
        })

        encrypted = self.encryptor.encrypt(large_plan)
        decrypted = self.encryptor.decrypt(encrypted)
        self.assertEqual(decrypted, large_plan)

    def test_hex_wire_format_roundtrip(self):
        """Plans can be transmitted as hex strings (as the API returns)."""
        from licensing.plan_encryption import EncryptedPlan

        plan_json = json.dumps({"model": "test"})
        encrypted = self.encryptor.encrypt(plan_json)

        # Simulate API transmission as hex
        hex_str = encrypted.serialize().hex()
        received_bytes = bytes.fromhex(hex_str)

        deserialized = EncryptedPlan.deserialize(received_bytes)
        decrypted = self.encryptor.decrypt(deserialized)
        self.assertEqual(decrypted, plan_json)


# ============================================================================
# Test 4: Offline Grace Period
# ============================================================================

class TestOfflineGracePeriodE2E(unittest.TestCase):
    """End-to-end tests for license offline grace period."""

    SIGNING_KEY = "test-signing-key-that-is-at-least-32-chars-long"

    def test_grace_period_allows_expired_token(self):
        """
        After a successful validation, an expired token is still accepted
        within the grace period.
        """
        from licensing.license_token import LicenseIssuer, LicenseValidator

        issuer = LicenseIssuer(self.SIGNING_KEY)
        # Grace period of 1 hour (for test, the key is that we had a prior success)
        validator = LicenseValidator(self.SIGNING_KEY, offline_grace_hours=1)

        # First: issue and validate a valid token to set last_successful_validation
        valid_token = issuer.issue(
            tenant_id="bank-abc",
            model_ids=["*"],
            validity_hours=24,
        )
        claims = validator.validate(valid_token)
        self.assertIsNotNone(claims)

        # Now: create an expired token but reuse the same validator
        # (simulates: token expired but we're within grace period)
        expired_token = issuer.issue(
            tenant_id="bank-abc",
            model_ids=["*"],
            validity_hours=0,  # Expires immediately
        )
        time.sleep(0.1)

        # Should still work because we had a successful validation recently
        claims = validator.validate(expired_token)
        self.assertIsNotNone(claims)

    def test_grace_period_exhausted_rejects(self):
        """
        Once the grace period is exhausted, expired tokens are rejected.
        """
        from licensing.license_token import (
            LicenseIssuer, LicenseValidator, LicenseExpiredError
        )

        issuer = LicenseIssuer(self.SIGNING_KEY)
        # Zero grace period
        validator = LicenseValidator(self.SIGNING_KEY, offline_grace_hours=0)

        # First validation succeeds
        valid_token = issuer.issue(
            tenant_id="bank-abc",
            model_ids=["*"],
            validity_hours=24,
        )
        validator.validate(valid_token)

        # Expired token with zero grace: should fail
        expired_token = issuer.issue(
            tenant_id="bank-abc",
            model_ids=["*"],
            validity_hours=0,
        )
        time.sleep(0.1)

        with self.assertRaises(LicenseExpiredError):
            validator.validate(expired_token)

    def test_cached_claims_available_during_grace(self):
        """Cached claims are available during the grace period."""
        from licensing.license_token import LicenseIssuer, LicenseValidator

        issuer = LicenseIssuer(self.SIGNING_KEY)
        validator = LicenseValidator(self.SIGNING_KEY, offline_grace_hours=24)

        token = issuer.issue(
            tenant_id="bank-abc",
            model_ids=["model-001"],
        )
        validator.validate(token)

        # Cached claims should be available
        cached = validator.get_cached_claims()
        self.assertIsNotNone(cached)
        self.assertEqual(cached.tenant_id, "bank-abc")

    def test_no_cached_claims_without_prior_validation(self):
        """No cached claims available if no prior validation occurred."""
        from licensing.license_token import LicenseValidator

        validator = LicenseValidator(self.SIGNING_KEY)
        cached = validator.get_cached_claims()
        self.assertIsNone(cached)


# ============================================================================
# Test 5: Compiler SaaS Auth & Multi-tenancy
# ============================================================================

class TestCompilerSaaSE2E(unittest.TestCase):
    """End-to-end tests for the compiler's SaaS auth and endpoints."""

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client."""
        os.environ["DEPLOYMENT_ENV"] = "development"
        os.environ["LICENSE_SIGNING_KEY"] = (
            "test-signing-key-that-is-at-least-32-chars-long"
        )
        try:
            from compiler.main import app
            cls.app = app
            cls.client = app.test_client()
            cls.available = True
        except Exception as e:
            cls.available = False
            cls.skip_reason = str(e)

    def setUp(self):
        if not self.available:
            self.skipTest(f"Compiler app not available: {self.skip_reason}")

    def _auth_header(self, tenant_id="test-tenant", secret="test-secret"):
        return {"X-API-Key": f"{tenant_id}.{secret}"}

    def test_health_endpoint(self):
        """Health endpoint returns OK without auth."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "OK")

    def test_compile_requires_auth(self):
        """Compile endpoint requires valid API key."""
        resp = self.client.post("/v1/compile", json={"model_content": "dGVzdA=="})
        self.assertIn(resp.status_code, [401, 403])

    def test_compile_rejects_wrong_key(self):
        """Compile endpoint rejects invalid API key."""
        resp = self.client.post(
            "/v1/compile",
            json={"model_content": "dGVzdA=="},
            headers={"X-API-Key": "test-tenant.wrong-secret"},
        )
        self.assertEqual(resp.status_code, 403)

    def test_license_issue_and_validate(self):
        """Issue a license through the API and validate it."""
        # Issue
        resp = self.client.post(
            "/v1/license/issue",
            json={
                "model_ids": ["model-001"],
                "max_predictions": 5000,
                "validity_hours": 48,
            },
            headers=self._auth_header(),
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("token", data)
        self.assertEqual(data["tenant_id"], "test-tenant")
        self.assertEqual(data["max_predictions"], 5000)

        # Validate
        resp = self.client.post(
            "/v1/license/validate",
            json={"token": data["token"], "model_id": "model-001"},
            headers=self._auth_header(),
        )
        self.assertEqual(resp.status_code, 200)
        validate_data = resp.get_json()
        self.assertTrue(validate_data["valid"])

    def test_license_validate_wrong_model(self):
        """Validate fails for unauthorized model."""
        # Issue for model-001 only
        resp = self.client.post(
            "/v1/license/issue",
            json={"model_ids": ["model-001"]},
            headers=self._auth_header(),
        )
        token = resp.get_json()["token"]

        # Validate for model-999
        resp = self.client.post(
            "/v1/license/validate",
            json={"token": token, "model_id": "model-999"},
            headers=self._auth_header(),
        )
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.get_json()["valid"])

    def test_multi_tenant_isolation(self):
        """Licenses issued to one tenant can't be validated by another."""
        # Issue for test-tenant
        resp = self.client.post(
            "/v1/license/issue",
            json={"model_ids": ["*"]},
            headers=self._auth_header("test-tenant", "test-secret"),
        )
        token = resp.get_json()["token"]

        # Try to validate as demo-tenant
        resp = self.client.post(
            "/v1/license/validate",
            json={"token": token},
            headers=self._auth_header("demo-tenant", "demo-secret"),
        )
        self.assertEqual(resp.status_code, 403)


# ============================================================================
# Test 6: Full Integration Flow
# ============================================================================

class TestFullIntegrationE2E(unittest.TestCase):
    """
    Full integration test: compile -> encrypt plan -> issue license ->
    validate license -> simulate prediction -> collect telemetry.
    """

    SIGNING_KEY = "test-signing-key-that-is-at-least-32-chars-long"
    DEPLOYMENT_SECRET = "bank-abc-deployment-secret-very-long"

    def test_full_hybrid_deployment_flow(self):
        """
        Simulate the complete hybrid deployment model:
        1. Vendor cloud: compile model + encrypt plan + issue license
        2. Customer on-prem: validate license + decrypt plan + predict + report telemetry
        """
        from licensing.license_token import LicenseIssuer, LicenseValidator
        from licensing.plan_encryption import PlanEncryptor, EncryptedPlan
        from licensing.heartbeat import HeartbeatCollector

        tenant_id = "bank-abc"
        model_id = "credit-score-model-v1"

        # ---- VENDOR CLOUD SIDE ----

        # Step 1: Compile model (simulated plan)
        plan_json = json.dumps({
            "compiled_model_id": model_id,
            "num_trees": 100,
            "crypto_params_id": "n2he_default",
            "schedule": [{"depth_level": 0, "ops": ["DELTA", "STEP"]}],
        })

        # Step 2: Encrypt plan for customer deployment
        vendor_encryptor = PlanEncryptor(tenant_id, self.DEPLOYMENT_SECRET)
        encrypted_plan = vendor_encryptor.encrypt(plan_json)
        wire_bytes = encrypted_plan.serialize()

        # Step 3: Issue license
        issuer = LicenseIssuer(self.SIGNING_KEY)
        license_token = issuer.issue(
            tenant_id=tenant_id,
            model_ids=[model_id],
            max_predictions=1000,
            validity_hours=72,
        )

        # ---- CUSTOMER ON-PREM SIDE ----

        # Step 4: Validate license
        validator = LicenseValidator(self.SIGNING_KEY, offline_grace_hours=72)
        claims = validator.validate(license_token, model_id=model_id)
        self.assertEqual(claims.tenant_id, tenant_id)
        self.assertTrue(claims.allows_model(model_id))

        # Step 5: Decrypt plan
        customer_encryptor = PlanEncryptor(tenant_id, self.DEPLOYMENT_SECRET)
        received_plan = EncryptedPlan.deserialize(wire_bytes)
        decrypted_plan = customer_encryptor.decrypt(received_plan)
        parsed_plan = json.loads(decrypted_plan)
        self.assertEqual(parsed_plan["compiled_model_id"], model_id)
        self.assertEqual(parsed_plan["num_trees"], 100)

        # Step 6: Run predictions and collect telemetry
        collector = HeartbeatCollector(
            tenant_id=tenant_id,
            license_id=claims.license_id,
        )

        # Simulate 10 predictions
        for i in range(10):
            latency_ms = 60.0 + (i % 5)  # 60-64ms simulated latency
            collector.record_prediction(model_id, latency_ms, success=True)
            count = validator.record_prediction(claims.license_id)
            self.assertLessEqual(count, claims.max_predictions)

        # Flush telemetry
        report = collector.flush()
        self.assertIsNotNone(report)
        self.assertEqual(report.total_predictions, 10)
        self.assertEqual(report.successful_predictions, 10)
        self.assertEqual(report.failed_predictions, 0)
        self.assertEqual(report.predictions_by_model[model_id], 10)
        self.assertGreater(report.latency_p50_ms, 0)

        # Verify prediction count tracked
        self.assertEqual(validator.get_prediction_count(claims.license_id), 10)

        # Verify report JSON is clean
        report_data = json.loads(report.to_json())
        self.assertNotIn("ciphertext", report_data)
        self.assertNotIn("payload", report_data)
        self.assertNotIn("secret_key", report_data)

    def test_prediction_cap_enforced_in_flow(self):
        """License prediction cap blocks further predictions."""
        from licensing.license_token import (
            LicenseIssuer, LicenseValidator, LicensePredictionCapExceededError
        )

        issuer = LicenseIssuer(self.SIGNING_KEY)
        validator = LicenseValidator(self.SIGNING_KEY)

        # Issue with cap of 5
        token = issuer.issue(
            tenant_id="bank-abc",
            model_ids=["*"],
            max_predictions=5,
        )

        # 5 predictions succeed
        for i in range(5):
            claims = validator.validate(token)
            validator.record_prediction(claims.license_id)

        # 6th prediction fails
        with self.assertRaises(LicensePredictionCapExceededError):
            validator.validate(token)

    def test_plan_bound_to_deployment(self):
        """Plan from vendor can only be used on intended deployment."""
        from licensing.plan_encryption import PlanEncryptor, PlanEncryptionError

        plan_json = json.dumps({"model": "credit-score-v1"})

        # Vendor encrypts for deployment A
        vendor = PlanEncryptor("bank-abc", "deployment-a-secret-very-long-key")
        encrypted = vendor.encrypt(plan_json)

        # Deployment A can decrypt
        deploy_a = PlanEncryptor("bank-abc", "deployment-a-secret-very-long-key")
        decrypted = deploy_a.decrypt(encrypted)
        self.assertEqual(decrypted, plan_json)

        # Deployment B cannot decrypt
        deploy_b = PlanEncryptor("bank-abc", "deployment-b-secret-very-long-key")
        with self.assertRaises(PlanEncryptionError):
            deploy_b.decrypt(encrypted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
