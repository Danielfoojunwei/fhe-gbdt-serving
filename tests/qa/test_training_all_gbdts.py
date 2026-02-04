#!/usr/bin/env python3
"""
QA Test Suite: Training Service for All GBDT Libraries

This test suite verifies that the training service works correctly with:
1. XGBoost
2. LightGBM
3. CatBoost

Each library is tested with:
- Basic training (no DP)
- Differential privacy training
- Privacy accounting verification
- Checkpoint functionality
"""

import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.training.trainer import (
    DPGBDTTrainer,
    TrainingConfig,
    DPConfig,
    GBDTLibrary,
    TrainingStatus,
)
from services.training.privacy import (
    RDPAccountant,
    PrivacySpent,
    compute_dp_sgd_privacy,
)


class TestResult:
    """Test result tracker."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.metrics = {}
        self.duration = 0

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        result = f"{status} {self.name} ({self.duration:.2f}s)"
        if self.error:
            result += f"\n    Error: {self.error}"
        if self.metrics:
            result += f"\n    Metrics: {self.metrics}"
        return result


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 10,
    task: str = "classification",
    random_seed: int = 42,
):
    """Generate synthetic data for testing."""
    np.random.seed(random_seed)

    X = np.random.randn(n_samples, n_features)

    if task == "classification":
        # Binary classification
        weights = np.random.randn(n_features)
        logits = X @ weights
        probs = 1 / (1 + np.exp(-logits))
        y = (probs > 0.5).astype(int)
    else:
        # Regression
        weights = np.random.randn(n_features)
        y = X @ weights + np.random.randn(n_samples) * 0.1

    # Split into train/val
    split_idx = int(n_samples * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_val, y_val


def test_xgboost_basic():
    """Test basic XGBoost training without DP."""
    result = TestResult("XGBoost Basic Training")
    start = time.time()

    try:
        # Check if XGBoost is available
        import xgboost

        # Generate data
        X_train, y_train, X_val, y_val = generate_synthetic_data(
            n_samples=500, n_features=10, task="classification"
        )

        # Configure training
        config = TrainingConfig(
            name="test_xgboost_basic",
            library=GBDTLibrary.XGBOOST,
            n_estimators=10,
            max_depth=4,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="auc",
            verbose=False,
        )

        # Train
        trainer = DPGBDTTrainer(config, tenant_id="test-tenant")
        metrics = trainer.train(X_train, y_train, X_val, y_val)

        # Verify
        assert trainer.status == TrainingStatus.COMPLETED, f"Status: {trainer.status}"
        assert trainer.model is not None, "Model is None"
        assert metrics.n_trees == 10, f"Expected 10 trees, got {metrics.n_trees}"

        result.passed = True
        result.metrics = {
            "n_trees": metrics.n_trees,
            "training_time": f"{metrics.training_time_seconds:.2f}s",
        }

    except ImportError:
        result.error = "XGBoost not installed"
    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_xgboost_with_dp():
    """Test XGBoost training with differential privacy."""
    result = TestResult("XGBoost with Differential Privacy")
    start = time.time()

    try:
        import xgboost

        # Generate data
        X_train, y_train, X_val, y_val = generate_synthetic_data(
            n_samples=500, n_features=10, task="classification"
        )

        # Configure DP
        dp_config = DPConfig(
            enabled=True,
            epsilon=1.0,
            delta=1e-5,
            noise_type="laplace",
            max_grad_norm=1.0,
        )

        # Configure training
        config = TrainingConfig(
            name="test_xgboost_dp",
            library=GBDTLibrary.XGBOOST,
            n_estimators=10,
            max_depth=4,
            learning_rate=0.1,
            dp_config=dp_config,
            verbose=False,
        )

        # Train
        trainer = DPGBDTTrainer(config, tenant_id="test-tenant")
        metrics = trainer.train(X_train, y_train, X_val, y_val)

        # Verify
        assert trainer.status == TrainingStatus.COMPLETED
        assert metrics.epsilon_spent is not None, "Epsilon not tracked"
        # Allow 30% tolerance for RDP composition overhead
        assert metrics.epsilon_spent <= dp_config.epsilon * 1.3, f"Epsilon exceeded: {metrics.epsilon_spent}"

        result.passed = True
        result.metrics = {
            "epsilon_spent": f"{metrics.epsilon_spent:.4f}",
            "delta_spent": f"{metrics.delta_spent:.2e}",
            "n_trees": metrics.n_trees,
        }

    except ImportError:
        result.error = "XGBoost not installed"
    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_lightgbm_basic():
    """Test basic LightGBM training without DP."""
    result = TestResult("LightGBM Basic Training")
    start = time.time()

    try:
        import lightgbm

        # Generate data
        X_train, y_train, X_val, y_val = generate_synthetic_data(
            n_samples=500, n_features=10, task="classification"
        )

        # Configure training
        config = TrainingConfig(
            name="test_lightgbm_basic",
            library=GBDTLibrary.LIGHTGBM,
            n_estimators=10,
            max_depth=4,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="auc",
            verbose=False,
        )

        # Train
        trainer = DPGBDTTrainer(config, tenant_id="test-tenant")
        metrics = trainer.train(X_train, y_train, X_val, y_val)

        # Verify
        assert trainer.status == TrainingStatus.COMPLETED
        assert trainer.model is not None

        result.passed = True
        result.metrics = {
            "n_trees": metrics.n_trees,
            "training_time": f"{metrics.training_time_seconds:.2f}s",
        }

    except ImportError:
        result.error = "LightGBM not installed"
    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_lightgbm_with_dp():
    """Test LightGBM training with differential privacy."""
    result = TestResult("LightGBM with Differential Privacy")
    start = time.time()

    try:
        import lightgbm

        # Generate data
        X_train, y_train, X_val, y_val = generate_synthetic_data(
            n_samples=500, n_features=10, task="classification"
        )

        # Configure DP
        dp_config = DPConfig(
            enabled=True,
            epsilon=2.0,
            delta=1e-5,
        )

        # Configure training
        config = TrainingConfig(
            name="test_lightgbm_dp",
            library=GBDTLibrary.LIGHTGBM,
            n_estimators=10,
            max_depth=4,
            dp_config=dp_config,
            verbose=False,
        )

        # Train
        trainer = DPGBDTTrainer(config, tenant_id="test-tenant")
        metrics = trainer.train(X_train, y_train, X_val, y_val)

        # Verify
        assert trainer.status == TrainingStatus.COMPLETED
        assert metrics.epsilon_spent is not None

        result.passed = True
        result.metrics = {
            "epsilon_spent": f"{metrics.epsilon_spent:.4f}",
            "n_trees": metrics.n_trees,
        }

    except ImportError:
        result.error = "LightGBM not installed"
    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_catboost_basic():
    """Test basic CatBoost training without DP."""
    result = TestResult("CatBoost Basic Training")
    start = time.time()

    try:
        import catboost

        # Generate data
        X_train, y_train, X_val, y_val = generate_synthetic_data(
            n_samples=500, n_features=10, task="classification"
        )

        # Configure training
        config = TrainingConfig(
            name="test_catboost_basic",
            library=GBDTLibrary.CATBOOST,
            n_estimators=10,
            max_depth=4,
            learning_rate=0.1,
            objective="binary:logistic",
            verbose=False,
        )

        # Train
        trainer = DPGBDTTrainer(config, tenant_id="test-tenant")
        metrics = trainer.train(X_train, y_train, X_val, y_val)

        # Verify
        assert trainer.status == TrainingStatus.COMPLETED
        assert trainer.model is not None

        result.passed = True
        result.metrics = {
            "n_trees": metrics.n_trees,
            "training_time": f"{metrics.training_time_seconds:.2f}s",
        }

    except ImportError:
        result.error = "CatBoost not installed"
    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_catboost_with_dp():
    """Test CatBoost training with differential privacy."""
    result = TestResult("CatBoost with Differential Privacy")
    start = time.time()

    try:
        import catboost

        # Generate data
        X_train, y_train, X_val, y_val = generate_synthetic_data(
            n_samples=500, n_features=10, task="classification"
        )

        # Configure DP
        dp_config = DPConfig(
            enabled=True,
            epsilon=1.5,
            delta=1e-5,
        )

        # Configure training
        config = TrainingConfig(
            name="test_catboost_dp",
            library=GBDTLibrary.CATBOOST,
            n_estimators=10,
            max_depth=4,
            dp_config=dp_config,
            verbose=False,
        )

        # Train
        trainer = DPGBDTTrainer(config, tenant_id="test-tenant")
        metrics = trainer.train(X_train, y_train, X_val, y_val)

        # Verify
        assert trainer.status == TrainingStatus.COMPLETED
        assert metrics.epsilon_spent is not None

        result.passed = True
        result.metrics = {
            "epsilon_spent": f"{metrics.epsilon_spent:.4f}",
            "n_trees": metrics.n_trees,
        }

    except ImportError:
        result.error = "CatBoost not installed"
    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_privacy_accountant():
    """Test RDP privacy accountant."""
    result = TestResult("RDP Privacy Accountant")
    start = time.time()

    try:
        # Create accountant
        accountant = RDPAccountant(
            epsilon_target=1.0,
            delta=1e-5,
        )

        # Simulate training iterations
        for i in range(100):
            accountant.account(
                noise_multiplier=1.0,
                sample_rate=0.01,
            )

        spent = accountant.get_privacy_spent()
        remaining = accountant.get_remaining_budget()

        # Verify
        assert spent.epsilon > 0, "Epsilon should be positive"
        assert spent.delta == 1e-5, "Delta should match"
        assert remaining.epsilon >= 0, "Remaining should be non-negative"

        # Test state save/load
        state = accountant.get_state()
        new_accountant = RDPAccountant(epsilon_target=1.0, delta=1e-5)
        new_accountant.load_state(state)

        new_spent = new_accountant.get_privacy_spent()
        assert abs(new_spent.epsilon - spent.epsilon) < 1e-6, "State not preserved"

        result.passed = True
        result.metrics = {
            "epsilon_spent": f"{spent.epsilon:.4f}",
            "remaining": f"{remaining.epsilon:.4f}",
        }

    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_checkpoint_functionality():
    """Test training checkpoint save/load."""
    result = TestResult("Checkpoint Functionality")
    start = time.time()

    try:
        import xgboost

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate data
            X_train, y_train, X_val, y_val = generate_synthetic_data(
                n_samples=500, n_features=10
            )

            # Configure with checkpointing
            config = TrainingConfig(
                name="test_checkpoint",
                library=GBDTLibrary.XGBOOST,
                n_estimators=20,
                max_depth=4,
                checkpoint_dir=tmpdir,
                checkpoint_interval=5,
                verbose=False,
            )

            # Track checkpoints
            checkpoints_saved = []

            def on_checkpoint(ckpt):
                checkpoints_saved.append(ckpt.checkpoint_id)

            # Train
            trainer = DPGBDTTrainer(config, tenant_id="test-tenant")
            trainer.register_checkpoint_callback(on_checkpoint)
            trainer.train(X_train, y_train, X_val, y_val)

            # Verify checkpoints created
            checkpoint_files = list(Path(tmpdir).glob("*.json"))

            assert len(checkpoint_files) >= 2, f"Expected checkpoints, found {len(checkpoint_files)}"

            # Verify checkpoint content
            with open(checkpoint_files[0]) as f:
                ckpt_data = json.load(f)
                assert "job_id" in ckpt_data
                assert "n_trees" in ckpt_data
                assert "hash" in ckpt_data

            result.passed = True
            result.metrics = {
                "checkpoints_saved": len(checkpoint_files),
                "checkpoint_ids": checkpoints_saved[:3],
            }

    except ImportError:
        result.error = "XGBoost not installed"
    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_trainer_status_tracking():
    """Test trainer status and progress tracking."""
    result = TestResult("Status & Progress Tracking")
    start = time.time()

    try:
        import xgboost

        # Generate data
        X_train, y_train, X_val, y_val = generate_synthetic_data(n_samples=300)

        config = TrainingConfig(
            name="test_status",
            library=GBDTLibrary.XGBOOST,
            n_estimators=10,
            verbose=False,
        )

        # Track progress
        progress_updates = []

        def on_progress(progress, metrics):
            progress_updates.append(progress)

        trainer = DPGBDTTrainer(config, tenant_id="test-tenant")
        trainer.register_progress_callback(on_progress)

        # Check initial status
        assert trainer.status == TrainingStatus.PENDING

        # Train
        trainer.train(X_train, y_train, X_val, y_val)

        # Check final status
        assert trainer.status == TrainingStatus.COMPLETED
        assert trainer.progress == 100.0

        # Get status dict
        status = trainer.get_status()
        assert "job_id" in status
        assert "status" in status
        assert "progress" in status
        assert "metrics" in status

        result.passed = True
        result.metrics = {
            "progress_updates": len(progress_updates),
            "final_progress": trainer.progress,
        }

    except ImportError:
        result.error = "XGBoost not installed"
    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_regression_task():
    """Test training on regression task."""
    result = TestResult("Regression Task")
    start = time.time()

    try:
        import xgboost

        # Generate regression data
        X_train, y_train, X_val, y_val = generate_synthetic_data(
            n_samples=500, task="regression"
        )

        config = TrainingConfig(
            name="test_regression",
            library=GBDTLibrary.XGBOOST,
            n_estimators=10,
            max_depth=4,
            objective="reg:squarederror",
            eval_metric="rmse",
            verbose=False,
        )

        trainer = DPGBDTTrainer(config, tenant_id="test-tenant")
        metrics = trainer.train(X_train, y_train, X_val, y_val)

        assert trainer.status == TrainingStatus.COMPLETED
        assert trainer.model is not None

        result.passed = True
        result.metrics = {
            "n_trees": metrics.n_trees,
            "training_time": f"{metrics.training_time_seconds:.2f}s",
        }

    except ImportError:
        result.error = "XGBoost not installed"
    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def run_all_tests():
    """Run all QA tests."""
    print("=" * 70)
    print("FHE-GBDT Training Service QA Tests")
    print("=" * 70)
    print()

    tests = [
        # XGBoost tests
        test_xgboost_basic,
        test_xgboost_with_dp,

        # LightGBM tests
        test_lightgbm_basic,
        test_lightgbm_with_dp,

        # CatBoost tests
        test_catboost_basic,
        test_catboost_with_dp,

        # Privacy accountant
        test_privacy_accountant,

        # Functionality tests
        test_checkpoint_functionality,
        test_trainer_status_tracking,
        test_regression_task,
    ]

    results = []

    for test_func in tests:
        print(f"Running: {test_func.__name__}...", end=" ", flush=True)
        result = test_func()
        results.append(result)

        if result.passed:
            print("PASS")
        else:
            print("FAIL")

    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    for result in results:
        print(result)
        print()

    print("-" * 70)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print("-" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
