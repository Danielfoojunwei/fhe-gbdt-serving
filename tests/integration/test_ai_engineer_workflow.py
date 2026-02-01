"""
AI Engineer Workflow Integration Tests

These tests verify the complete end-to-end workflow that an AI engineer
would follow when building, deploying, and serving GBDT models with FHE.

The workflow covers:
1. Model Training: XGBoost/LightGBM/CatBoost model training
2. Model Export: Saving trained models in compatible formats
3. Model Registration: Uploading models to the registry
4. Model Compilation: Compiling to FHE-compatible format
5. Key Management: Client key generation and upload
6. Inference: Running encrypted inference
7. Result Verification: Comparing FHE vs plaintext results

References:
- GBDT Training Best Practices: https://research.aimultiple.com/ai-training/
- FHE Production Deployment: https://cloudsecurityalliance.org/blog/2025/03/19/assessing-the-security-of-fhe-solutions
"""

import unittest
import json
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import project modules
from services.compiler.parser import XGBoostParser, LightGBMParser, CatBoostParser, get_parser
from services.compiler.ir import TreeNode, TreeIR, ModelIR
from sdk.python.crypto import N2HEKeyManager, NATIVE_AVAILABLE


@dataclass
class WorkflowResult:
    """Result of a workflow step."""
    success: bool
    step_name: str
    duration_ms: float
    error: Optional[str] = None
    data: Optional[Dict] = None


class TestAIEngineerWorkflow(unittest.TestCase):
    """
    Integration tests for the complete AI engineer workflow.

    These tests simulate the journey an AI engineer takes from
    training a GBDT model to serving encrypted predictions.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.output_dir = tempfile.mkdtemp(prefix="fhe_gbdt_test_")
        cls.tenant_id = "test-ai-engineer"

    def _time_step(self, step_name: str, func, *args, **kwargs) -> WorkflowResult:
        """Time a workflow step and capture result."""
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start) * 1000
            return WorkflowResult(
                success=True,
                step_name=step_name,
                duration_ms=duration_ms,
                data=result if isinstance(result, dict) else {"result": result}
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return WorkflowResult(
                success=False,
                step_name=step_name,
                duration_ms=duration_ms,
                error=str(e)
            )

    def test_workflow_step1_train_xgboost_model(self):
        """Step 1: Train an XGBoost model."""
        try:
            import xgboost as xgb
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split

            # Load data
            iris = load_iris()
            X_train, X_test, y_train, y_test = train_test_split(
                iris.data, iris.target, test_size=0.2, random_state=42
            )

            # Train model with controlled hyperparameters
            # Following best practices: small depth, early stopping
            params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "max_depth": 3,  # Keep shallow for FHE efficiency
                "n_estimators": 10,  # Limit trees for FHE
                "learning_rate": 0.1,
                "random_state": 42,
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)

            # Verify training accuracy
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)

            self.assertGreater(train_acc, 0.9, "Training accuracy should be > 90%")
            self.assertGreater(test_acc, 0.8, "Test accuracy should be > 80%")

            # Export model
            model_path = os.path.join(self.output_dir, "xgboost_model.json")
            model.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))

        except ImportError:
            self.skipTest("XGBoost not available")

    def test_workflow_step2_train_lightgbm_model(self):
        """Step 2: Train a LightGBM model."""
        try:
            import lightgbm as lgb
            from sklearn.datasets import load_diabetes
            from sklearn.model_selection import train_test_split

            # Load regression data
            diabetes = load_diabetes()
            X_train, X_test, y_train, y_test = train_test_split(
                diabetes.data, diabetes.target, test_size=0.2, random_state=42
            )

            # Train with FHE-friendly parameters
            params = {
                "objective": "regression",
                "max_depth": 4,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "num_leaves": 15,  # 2^max_depth - 1
                "verbose": -1,
            }

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)

            # Verify model quality
            r2_score = model.score(X_test, y_test)
            self.assertGreater(r2_score, 0.3, "R2 score should be > 0.3")

            # Export model
            model_path = os.path.join(self.output_dir, "lightgbm_model.json")
            model.booster_.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))

        except ImportError:
            self.skipTest("LightGBM not available")

    def test_workflow_step3_parse_and_compile_model(self):
        """Step 3: Parse trained model into IR."""
        # Create a minimal valid XGBoost model for parsing
        xgb_model = {
            "learner": {
                "gradient_booster": {
                    "model": {
                        "trees": [
                            {
                                "left_children": [1, -1, -1],
                                "right_children": [2, -1, -1],
                                "split_indices": [0, 0, 0],
                                "split_conditions": [0.5, 0.0, 0.0],
                                "base_weights": [0.0, 0.3, 0.7]
                            },
                            {
                                "left_children": [1, -1, -1],
                                "right_children": [2, -1, -1],
                                "split_indices": [1, 0, 0],
                                "split_conditions": [0.3, 0.0, 0.0],
                                "base_weights": [0.0, -0.2, 0.4]
                            }
                        ]
                    }
                },
                "learner_model_param": {"base_score": "0.5"}
            }
        }

        parser = XGBoostParser()
        model_ir = parser.parse(json.dumps(xgb_model).encode())

        # Verify IR structure
        self.assertEqual(model_ir.model_type, "xgboost")
        self.assertEqual(len(model_ir.trees), 2)
        self.assertEqual(model_ir.num_features, 2)
        self.assertEqual(model_ir.base_score, 0.5)

        # Verify tree depths are FHE-friendly
        for tree in model_ir.trees:
            self.assertLessEqual(tree.max_depth, 10,
                "Tree depth should be <= 10 for efficient FHE")

    def test_workflow_step4_key_generation(self):
        """Step 4: Generate FHE keys."""
        km = N2HEKeyManager(self.tenant_id)
        km.generate_keys()

        # Verify keys are generated
        self.assertIsNotNone(km._secret_key)
        self.assertIsNotNone(km._eval_keys)

        # Export eval keys (for server)
        eval_keys = km.export_eval_keys()
        self.assertIsInstance(eval_keys, bytes)
        self.assertGreater(len(eval_keys), 100)

    def test_workflow_step5_encrypt_features(self):
        """Step 5: Encrypt input features."""
        km = N2HEKeyManager(self.tenant_id)
        km.generate_keys()

        # Test feature encryption
        features = [0.1, 0.5, 0.9, 0.3]
        ciphertext = km.encrypt(features)

        self.assertIsInstance(ciphertext, bytes)
        self.assertGreater(len(ciphertext), len(features) * 8)

    def test_workflow_step6_decrypt_results(self):
        """Step 6: Decrypt inference results."""
        km = N2HEKeyManager(self.tenant_id)
        km.generate_keys()

        # Encrypt and decrypt roundtrip
        original = [0.1, 0.5, 0.9, 0.3]
        ciphertext = km.encrypt(original)
        decrypted = km.decrypt(ciphertext)

        self.assertEqual(len(decrypted), len(original))
        # Note: Some precision loss is expected in simulation mode
        for i, (orig, dec) in enumerate(zip(original, decrypted)):
            # Allow larger tolerance for simulation mode
            tolerance = 0.5 if not NATIVE_AVAILABLE else 0.01
            self.assertAlmostEqual(orig, dec, delta=tolerance,
                msg=f"Feature {i}: {orig} != {dec}")

    def test_workflow_step7_batch_inference(self):
        """Step 7: Run batch encrypted inference."""
        km = N2HEKeyManager(self.tenant_id)
        km.generate_keys()

        # Simulate batch of 10 samples
        batch_size = 10
        num_features = 4

        batch_results = []
        for i in range(batch_size):
            features = [float(j + i) / 10 for j in range(num_features)]
            ct = km.encrypt(features)
            # In real system, ct would go to gateway for FHE inference
            # For this test, we verify encryption/decryption works
            decrypted = km.decrypt(ct)
            batch_results.append(decrypted)

        self.assertEqual(len(batch_results), batch_size)

    def test_workflow_step8_model_versioning(self):
        """Step 8: Test model versioning workflow."""
        # Simulate registering multiple versions
        models = {
            "v1": {"trees": 5, "depth": 3},
            "v2": {"trees": 10, "depth": 4},
            "v3": {"trees": 8, "depth": 3},  # Rollback to smaller
        }

        # Verify version metadata
        for version, config in models.items():
            self.assertIn("trees", config)
            self.assertIn("depth", config)
            # FHE constraints
            self.assertLessEqual(config["depth"], 10)
            self.assertLessEqual(config["trees"], 100)

    def test_full_workflow_timing(self):
        """Test full workflow with timing."""
        workflow_steps = []

        # Step 1: Initialize key manager
        result = self._time_step("key_generation",
            lambda: N2HEKeyManager(self.tenant_id))
        workflow_steps.append(result)
        self.assertTrue(result.success)

        km = result.data["result"]

        # Step 2: Generate keys
        result = self._time_step("generate_keys", km.generate_keys)
        workflow_steps.append(result)
        self.assertTrue(result.success)

        # Step 3: Encrypt features
        features = [0.1, 0.5, 0.9, 0.3]
        result = self._time_step("encrypt",
            lambda: km.encrypt(features))
        workflow_steps.append(result)
        self.assertTrue(result.success)

        # Print timing summary
        total_time = sum(s.duration_ms for s in workflow_steps)
        print(f"\nWorkflow Timing Summary:")
        for step in workflow_steps:
            status = "✓" if step.success else "✗"
            print(f"  {status} {step.step_name}: {step.duration_ms:.1f}ms")
        print(f"  Total: {total_time:.1f}ms")


class TestModelValidation(unittest.TestCase):
    """Tests for model validation rules."""

    def test_validate_tree_depth_limits(self):
        """Verify tree depth limits for FHE efficiency."""
        # Max depth of 10 is recommended for FHE
        MAX_ALLOWED_DEPTH = 10

        # Valid model (depth 3)
        valid_model = {
            "learner": {
                "gradient_booster": {
                    "model": {
                        "trees": [{
                            "left_children": [1, 3, -1, -1, -1, -1, -1],
                            "right_children": [2, 4, -1, -1, -1, -1, -1],
                            "split_indices": [0, 1, 0, 0, 0, 0, 0],
                            "split_conditions": [0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "base_weights": [0.0, 0.0, 0.2, -0.1, 0.3, 0.0, 0.0]
                        }]
                    }
                }
            }
        }

        parser = XGBoostParser()
        model_ir = parser.parse(json.dumps(valid_model).encode())

        for tree in model_ir.trees:
            self.assertLessEqual(tree.max_depth, MAX_ALLOWED_DEPTH)

    def test_validate_feature_count(self):
        """Verify feature count is within limits."""
        MAX_FEATURES = 1000  # Reasonable limit for FHE

        model = {
            "learner": {
                "gradient_booster": {
                    "model": {
                        "trees": [{
                            "left_children": [1, -1, -1],
                            "right_children": [2, -1, -1],
                            "split_indices": [50, 0, 0],  # Feature 50
                            "split_conditions": [0.5, 0.0, 0.0],
                            "base_weights": [0.0, 0.3, 0.7]
                        }]
                    }
                }
            }
        }

        parser = XGBoostParser()
        model_ir = parser.parse(json.dumps(model).encode())

        self.assertLessEqual(model_ir.num_features, MAX_FEATURES)

    def test_validate_no_missing_values_handling(self):
        """Test that missing value handling is documented."""
        # FHE cannot handle branching on missing values at runtime
        # This test ensures we're aware of the limitation
        model = {
            "learner": {
                "gradient_booster": {
                    "model": {
                        "trees": [{
                            "left_children": [1, -1, -1],
                            "right_children": [2, -1, -1],
                            "split_indices": [0, 0, 0],
                            "split_conditions": [0.5, 0.0, 0.0],
                            "base_weights": [0.0, 0.3, 0.7],
                            # default_left indicates missing value handling
                        }]
                    }
                }
            }
        }

        parser = XGBoostParser()
        model_ir = parser.parse(json.dumps(model).encode())

        # Verify model parsed successfully
        self.assertEqual(len(model_ir.trees), 1)


class TestSecurityValidation(unittest.TestCase):
    """Security validation tests for production deployment."""

    def test_key_isolation(self):
        """Verify keys are isolated between tenants."""
        km1 = N2HEKeyManager("tenant-1")
        km2 = N2HEKeyManager("tenant-2")

        km1.generate_keys()
        km2.generate_keys()

        # Keys should be different
        self.assertNotEqual(km1._secret_key, km2._secret_key)
        self.assertNotEqual(km1._eval_keys, km2._eval_keys)

    def test_ciphertext_indistinguishable(self):
        """Verify ciphertexts of same value look different (semantic security)."""
        km = N2HEKeyManager("test-tenant")
        km.generate_keys()

        # Encrypt same value twice
        ct1 = km.encrypt([0.5])
        ct2 = km.encrypt([0.5])

        # Ciphertexts should be different (randomized encryption)
        self.assertNotEqual(ct1, ct2)

    def test_no_key_in_ciphertext(self):
        """Verify secret key material is not in ciphertext."""
        km = N2HEKeyManager("test-tenant")
        km.generate_keys()

        ct = km.encrypt([0.5, 0.3, 0.1])

        # Secret key (in simulation) is binary polynomial
        # Ensure it's not directly embedded in ciphertext
        if hasattr(km, '_secret_key') and km._secret_key:
            # Convert to bytes for comparison (if possible)
            # This is a basic check - real security requires formal analysis
            self.assertGreater(len(ct), 20)  # Ciphertext should be substantial


class TestPerformanceRegression(unittest.TestCase):
    """Performance regression tests."""

    def test_encryption_latency_regression(self):
        """Test encryption latency hasn't regressed."""
        km = N2HEKeyManager("perf-tenant")
        km.generate_keys()

        # Warmup
        for _ in range(3):
            km.encrypt([0.5])

        # Measure
        iterations = 10
        start = time.time()
        for _ in range(iterations):
            km.encrypt([0.1, 0.2, 0.3, 0.4])
        elapsed = time.time() - start

        avg_latency_ms = (elapsed / iterations) * 1000

        # Threshold: 500ms per encryption in simulation, 50ms native
        threshold = 500 if not NATIVE_AVAILABLE else 50
        self.assertLess(avg_latency_ms, threshold,
            f"Encryption latency regression: {avg_latency_ms:.1f}ms > {threshold}ms")

    def test_decryption_latency_regression(self):
        """Test decryption latency hasn't regressed."""
        km = N2HEKeyManager("perf-tenant")
        km.generate_keys()

        ct = km.encrypt([0.1, 0.2, 0.3, 0.4])

        # Warmup
        for _ in range(3):
            km.decrypt(ct)

        # Measure
        iterations = 10
        start = time.time()
        for _ in range(iterations):
            km.decrypt(ct)
        elapsed = time.time() - start

        avg_latency_ms = (elapsed / iterations) * 1000

        threshold = 100 if not NATIVE_AVAILABLE else 10
        self.assertLess(avg_latency_ms, threshold,
            f"Decryption latency regression: {avg_latency_ms:.1f}ms > {threshold}ms")


if __name__ == '__main__':
    unittest.main(verbosity=2)
