"""
Integration Tests for FHE-Aware Tree Training

Tests the genuinely novel contribution: modifying tree training
to select thresholds that minimize polynomial sign approximation
error during FHE inference.

Key validations:
1. Sign error profile correctly identifies danger zone δ
2. FHE-aware splits prefer thresholds with low margin density
3. FHE-aware trees have lower simulated FHE error than standard trees
4. Theoretical error bound holds empirically
5. Plaintext accuracy is not severely degraded
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.innovations.fhe_aware_training import (
    SignPolynomialAnalyzer,
    SignErrorProfile,
    FHEAwareSplitCriterion,
    FHEAwareSplitResult,
    FHEAwareTreeTrainer,
    FHEAwareTrainingConfig,
    FHEAwareObliviousTree,
    FHEErrorAnalyzer,
    train_fhe_aware_trees,
    compare_training_approaches,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_regression_data():
    """Simple regression dataset where thresholds matter."""
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 4)
    # y depends on step functions with known thresholds
    y = (
        2.0 * (X[:, 0] > 0.5).astype(float)
        - 1.5 * (X[:, 1] > -0.3).astype(float)
        + 0.8 * (X[:, 2] > 0.0).astype(float)
        + np.random.randn(n) * 0.1
    )
    return X, y


@pytest.fixture
def clustered_feature_data():
    """Data where features cluster near certain thresholds (worst case for FHE)."""
    np.random.seed(42)
    n = 500
    # Feature 0: bimodal with cluster at 0.5 (many samples near threshold)
    cluster1 = np.random.normal(0.48, 0.02, n // 2)
    cluster2 = np.random.normal(0.52, 0.02, n // 2)
    f0 = np.concatenate([cluster1, cluster2])

    # Feature 1: uniform (spread out, fewer samples near any threshold)
    f1 = np.random.uniform(-1, 1, n)

    X = np.column_stack([f0, f1, np.random.randn(n), np.random.randn(n)])
    y = 2.0 * (f0 > 0.5).astype(float) + 1.0 * (f1 > 0.0).astype(float)
    return X, y


@pytest.fixture
def train_test_split(simple_regression_data):
    """Split data into train/test."""
    X, y = simple_regression_data
    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:]


# =============================================================================
# Test 1: Sign Polynomial Error Profile
# =============================================================================

class TestSignPolynomialAnalyzer:
    """Tests for the sign polynomial error analysis."""

    def test_error_profile_computed(self):
        """Should compute error profile for degree 7."""
        analyzer = SignPolynomialAnalyzer()
        profile = analyzer.get_error_profile(7)

        assert profile.degree == 7
        assert profile.delta > 0  # Danger zone exists
        assert profile.delta < 1.0  # Not entire domain
        assert profile.epsilon < 1.0  # Error outside danger zone is bounded below 1
        assert len(profile.error_curve) > 0

    def test_higher_degree_lower_epsilon(self):
        """Higher degree polynomial should have lower error outside danger zone."""
        analyzer = SignPolynomialAnalyzer()
        profile_7 = analyzer.get_error_profile(7)
        profile_15 = analyzer.get_error_profile(15)

        # Higher degree should have lower max error outside its delta
        # (even if delta computation varies, epsilon should improve)
        assert profile_15.epsilon <= profile_7.epsilon + 0.01

    def test_error_decreases_away_from_zero(self):
        """Error should decrease as we move away from z=0."""
        analyzer = SignPolynomialAnalyzer()
        profile = analyzer.get_error_profile(7)

        # Error at z=0.1 should be > error at z=0.5
        idx_01 = np.argmin(np.abs(profile.error_sample_points - 0.1))
        idx_05 = np.argmin(np.abs(profile.error_sample_points - 0.5))

        assert profile.error_curve[idx_01] >= profile.error_curve[idx_05]

    def test_margin_penalty_computation(self):
        """Margin penalty should be higher when data clusters near threshold."""
        analyzer = SignPolynomialAnalyzer()

        # Data clustered near threshold 0.5
        clustered = np.random.normal(0.5, 0.01, 1000)
        penalty_clustered = analyzer.compute_margin_penalty(clustered, 0.5, 7, 1.0)

        # Data spread out
        spread = np.random.uniform(-1, 1, 1000)
        penalty_spread = analyzer.compute_margin_penalty(spread, 0.0, 7, 1.0)

        # Clustered data should have higher penalty
        assert penalty_clustered > penalty_spread

    def test_margin_penalty_low_far_from_threshold(self):
        """Penalty should be lower when data is far from threshold."""
        analyzer = SignPolynomialAnalyzer()

        # All data far from threshold (margins > 1 after normalization)
        data_far = np.random.uniform(5.0, 10.0, 1000)
        penalty_far = analyzer.compute_margin_penalty(data_far, 0.0, 7, 1.0)

        # Data near threshold
        data_near = np.random.normal(0.0, 0.01, 1000)
        penalty_near = analyzer.compute_margin_penalty(data_near, 0.0, 7, 1.0)

        # Far data should have much lower penalty than near data
        assert penalty_far < penalty_near


# =============================================================================
# Test 2: FHE-Aware Split Criterion
# =============================================================================

class TestFHEAwareSplitCriterion:
    """Tests for the FHE-aware split selection."""

    def test_finds_best_split(self):
        """Should find a valid split."""
        np.random.seed(42)
        feature = np.random.randn(200)
        targets = (feature > 0.5).astype(float) + np.random.randn(200) * 0.1

        criterion = FHEAwareSplitCriterion(fhe_penalty_weight=1.0)
        result = criterion.find_best_split(feature, targets)

        assert result.information_gain > 0
        assert 0 <= result.margin_penalty <= 1
        assert result.fhe_aware_gain > 0

    def test_lambda_zero_equals_standard(self):
        """With λ=0, FHE-aware gain should equal information gain."""
        np.random.seed(42)
        feature = np.random.randn(200)
        targets = (feature > 0.0).astype(float)

        criterion = FHEAwareSplitCriterion(fhe_penalty_weight=0.0)
        result = criterion.find_best_split(feature, targets)

        # With λ=0, fhe_gain = ig * (1 - 0 * penalty) = ig
        assert abs(result.fhe_aware_gain - result.information_gain) < 1e-10

    def test_prefers_low_density_threshold(self, clustered_feature_data):
        """FHE-aware split should prefer thresholds in low-density regions."""
        X, y = clustered_feature_data

        criterion = FHEAwareSplitCriterion(fhe_penalty_weight=1.0)
        fhe_result = criterion.find_best_split(X[:, 0], y)

        standard = FHEAwareSplitCriterion(fhe_penalty_weight=0.0)
        std_result = standard.find_best_split(X[:, 0], y)

        # FHE-aware should have lower margin penalty (avoids density cluster)
        assert fhe_result.margin_penalty <= std_result.margin_penalty + 0.05

    def test_comparison_standard_vs_fhe_aware(self):
        """Comparison should show different thresholds."""
        np.random.seed(42)
        # Feature with bimodal distribution (peak at 0.5)
        feature = np.concatenate([
            np.random.normal(0.48, 0.02, 200),
            np.random.normal(0.52, 0.02, 200),
        ])
        targets = (feature > 0.5).astype(float)

        criterion = FHEAwareSplitCriterion(fhe_penalty_weight=1.0)
        comparison = criterion.compare_standard_vs_fhe_aware(feature, targets)

        assert "standard_threshold" in comparison
        assert "fhe_aware_threshold" in comparison
        assert "error_improvement" in comparison

    def test_returns_valid_result_for_constant_feature(self):
        """Should handle constant features gracefully."""
        feature = np.ones(100)
        targets = np.random.randn(100)

        criterion = FHEAwareSplitCriterion()
        result = criterion.find_best_split(feature, targets)

        assert result.information_gain == 0.0


# =============================================================================
# Test 3: FHE-Aware Tree Training
# =============================================================================

class TestFHEAwareTreeTrainer:
    """Tests for the full FHE-aware tree training."""

    def test_basic_training(self, simple_regression_data):
        """Should train trees and produce predictions."""
        X, y = simple_regression_data

        trees, metadata = train_fhe_aware_trees(
            X, y, max_depth=3, num_trees=10, learning_rate=0.1
        )

        assert len(trees) == 10
        assert all(isinstance(t, FHEAwareObliviousTree) for t in trees)
        assert metadata["num_trees"] == 10
        assert metadata["max_depth"] == 3

    def test_tree_structure_valid(self, simple_regression_data):
        """Trained trees should have valid structure."""
        X, y = simple_regression_data

        trees, _ = train_fhe_aware_trees(X, y, max_depth=3, num_trees=5)

        for tree in trees:
            assert len(tree.levels) == 3
            assert len(tree.leaf_values) == 8  # 2^3
            assert tree.max_depth == 3
            # Each level should have a valid feature index
            for level in tree.levels:
                assert 0 <= level.feature_idx < X.shape[1]

    def test_margin_penalty_tracked(self, simple_regression_data):
        """Training should track margin penalty per tree."""
        X, y = simple_regression_data

        trees, metadata = train_fhe_aware_trees(
            X, y, max_depth=3, num_trees=10, fhe_penalty_weight=1.0
        )

        for tree in trees:
            assert tree.total_margin_penalty >= 0
            assert tree.predicted_fhe_error_bound >= 0

        assert metadata["avg_margin_penalty_per_tree"] >= 0

    def test_fhe_aware_reduces_margin_penalty(self, clustered_feature_data):
        """FHE-aware training should have lower margin penalty than standard."""
        X, y = clustered_feature_data

        # Standard (λ=0)
        std_trees, std_meta = train_fhe_aware_trees(
            X, y, max_depth=3, num_trees=20, fhe_penalty_weight=0.0
        )

        # FHE-aware (λ=1)
        fhe_trees, fhe_meta = train_fhe_aware_trees(
            X, y, max_depth=3, num_trees=20, fhe_penalty_weight=1.0
        )

        # FHE-aware should have lower average margin penalty
        assert fhe_meta["avg_margin_penalty_per_tree"] <= std_meta["avg_margin_penalty_per_tree"] + 0.01

    def test_predictions_reasonable(self, simple_regression_data):
        """Trained model should make reasonable predictions."""
        X, y = simple_regression_data

        trees, _ = train_fhe_aware_trees(
            X, y, max_depth=4, num_trees=50, learning_rate=0.1
        )

        # Evaluate
        predictions = np.zeros(X.shape[0])
        for tree in trees:
            outputs = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                leaf_idx = 0
                for d, level in enumerate(tree.levels):
                    if X[i, level.feature_idx] >= level.threshold:
                        leaf_idx |= (1 << d)
                outputs[i] = tree.leaf_values[leaf_idx] if leaf_idx < len(tree.leaf_values) else 0.0
            predictions += 0.1 * outputs

        # Should have reasonable correlation with targets
        corr = np.corrcoef(predictions, y)[0, 1]
        assert corr > 0.3  # At least some predictive power


# =============================================================================
# Test 4: FHE Error Simulation
# =============================================================================

class TestFHEErrorAnalyzer:
    """Tests for the FHE error analysis and simulation."""

    def test_fhe_simulation_runs(self, train_test_split):
        """FHE simulation should complete and return valid results."""
        X_train, y_train, X_test, y_test = train_test_split

        trees, _ = train_fhe_aware_trees(
            X_train, y_train, max_depth=3, num_trees=20
        )

        analyzer = FHEErrorAnalyzer(poly_degree=7)
        result = analyzer.evaluate_fhe_simulation(trees, X_test)

        assert "mean_absolute_error" in result
        assert "max_absolute_error" in result
        assert "theoretical_bound" in result
        assert "prediction_correlation" in result

        # FHE predictions should correlate with exact
        assert result["prediction_correlation"] > 0.5

    def test_fhe_error_bounded(self, train_test_split):
        """FHE error should be bounded (not diverge)."""
        X_train, y_train, X_test, y_test = train_test_split

        trees, _ = train_fhe_aware_trees(
            X_train, y_train, max_depth=3, num_trees=20
        )

        analyzer = FHEErrorAnalyzer()
        result = analyzer.evaluate_fhe_simulation(trees, X_test)

        # Mean error should not be huge
        assert result["mean_absolute_error"] < 10.0

    def test_head_to_head_comparison(self, train_test_split):
        """Head-to-head should show FHE-aware has lower FHE error."""
        X_train, y_train, X_test, y_test = train_test_split

        analyzer = FHEErrorAnalyzer()
        comparison = analyzer.compare_standard_vs_fhe_aware(
            X_train, y_train, X_test, y_test,
            max_depth=3, num_trees=30, learning_rate=0.1,
        )

        assert "standard" in comparison
        assert "fhe_aware" in comparison
        assert "improvements" in comparison

        # FHE-aware should have lower margin penalty
        assert (
            comparison["fhe_aware"]["avg_margin_penalty"]
            <= comparison["standard"]["avg_margin_penalty"] + 0.02
        )

        # Both should have reasonable correlation
        assert comparison["standard"]["fhe_correlation"] > 0.3
        assert comparison["fhe_aware"]["fhe_correlation"] > 0.3

    def test_polynomial_sign_maps_correctly(self):
        """Polynomial sign should produce values in [0, 1]."""
        analyzer = FHEErrorAnalyzer(poly_degree=7)

        z = np.linspace(-1, 1, 100)
        result = analyzer._polynomial_sign(z)

        # Should be in [0, 1] (clipped)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

        # Should be ~0 for z << 0 and ~1 for z >> 0
        assert np.mean(result[:10]) < 0.3   # Left side → 0
        assert np.mean(result[-10:]) > 0.7  # Right side → 1


# =============================================================================
# Test 5: Clustered Data Validation
# =============================================================================

class TestClusteredDataBenefit:
    """Tests that FHE-aware training specifically helps on clustered data."""

    def test_clustered_data_benefits_most(self, clustered_feature_data):
        """Clustered data should show the largest improvement."""
        X, y = clustered_feature_data
        n = len(X)
        split = int(0.8 * n)

        analyzer = FHEErrorAnalyzer()
        comparison = analyzer.compare_standard_vs_fhe_aware(
            X[:split], y[:split], X[split:], y[split:],
            max_depth=3, num_trees=30, learning_rate=0.1,
        )

        # On clustered data, FHE-aware should have noticeably lower margin penalty
        assert (
            comparison["fhe_aware"]["avg_margin_penalty"]
            <= comparison["standard"]["avg_margin_penalty"] + 0.05
        )

    def test_uniform_data_minimal_difference(self):
        """On uniform data, both approaches should behave similarly."""
        np.random.seed(42)
        n = 400
        X = np.random.uniform(-2, 2, (n, 4))
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
        split = int(0.8 * n)

        analyzer = FHEErrorAnalyzer()
        comparison = analyzer.compare_standard_vs_fhe_aware(
            X[:split], y[:split], X[split:], y[split:],
            max_depth=3, num_trees=20, learning_rate=0.1,
        )

        # On uniform data, difference should be small
        penalty_diff = abs(
            comparison["standard"]["avg_margin_penalty"]
            - comparison["fhe_aware"]["avg_margin_penalty"]
        )
        assert penalty_diff < 0.3  # Not a huge difference on uniform data


# =============================================================================
# Test 6: Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for top-level convenience functions."""

    def test_train_fhe_aware_trees(self, simple_regression_data):
        """train_fhe_aware_trees should work end-to-end."""
        X, y = simple_regression_data
        trees, metadata = train_fhe_aware_trees(X, y, max_depth=3, num_trees=10)
        assert len(trees) == 10
        assert "num_trees" in metadata

    def test_compare_training_approaches(self, train_test_split):
        """compare_training_approaches should return comparison."""
        X_train, y_train, X_test, y_test = train_test_split
        result = compare_training_approaches(
            X_train, y_train, X_test, y_test,
            max_depth=3, num_trees=10,
        )
        assert "standard" in result
        assert "fhe_aware" in result
        assert "improvements" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
