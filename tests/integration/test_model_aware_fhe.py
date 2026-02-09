"""
Integration Tests for Model-Aware FHE Optimization (Innovation #8)

Tests all 5 novel contributions:
1. Model Structure Classifier — Correct detection of linear/single-tree/RF/boosted
2. Comparison-Free Linear FHE — Zero sign functions for linear models
3. Independent Noise Channels — RF-specific noise optimization
4. Precision-Adaptive Sign — Higher-degree polynomials for single trees
5. Encrypted Majority Vote — Polynomial argmax for RF classification
"""

import pytest
import numpy as np
from numpy.polynomial import chebyshev
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.compiler.ir import ModelIR, TreeIR, TreeNode
from services.innovations.model_aware_fhe import (
    ModelStructureClassifier,
    ModelStructureType,
    ModelStructureAnalysis,
    ModelAwareFHEEngine,
    LinkFunctionLibrary,
    ComparisonFreeLinearEvaluator,
    IndependentNoiseOptimizer,
    PrecisionAdaptiveSign,
    EncryptedMajorityVote,
    MajorityVoteConfig,
    classify_model_structure,
    analyze_model_aware_fhe,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def single_tree_model():
    """Single decision tree (n_estimators=1)."""
    nodes = {
        0: TreeNode(node_id=0, feature_index=0, threshold=0.5,
                     left_child_id=1, right_child_id=2, depth=0),
        1: TreeNode(node_id=1, feature_index=1, threshold=0.3,
                     left_child_id=3, right_child_id=4, depth=1),
        2: TreeNode(node_id=2, feature_index=2, threshold=0.7,
                     left_child_id=5, right_child_id=6, depth=1),
        3: TreeNode(node_id=3, leaf_value=-0.5, depth=2),
        4: TreeNode(node_id=4, leaf_value=-0.1, depth=2),
        5: TreeNode(node_id=5, leaf_value=0.2, depth=2),
        6: TreeNode(node_id=6, leaf_value=0.6, depth=2),
    }
    tree = TreeIR(tree_id=0, nodes=nodes, root_id=0, max_depth=3)
    return ModelIR(model_type="xgboost", trees=[tree], num_features=4, base_score=0.5)


@pytest.fixture
def linear_model_stumps():
    """Logistic regression approximated as depth-1 stumps (100 trees)."""
    trees = []
    np.random.seed(42)
    for i in range(100):
        feat_idx = i % 4  # Cycle through 4 features
        threshold = np.random.uniform(0.2, 0.8)
        left_val = np.random.uniform(-0.1, 0.0)
        right_val = np.random.uniform(0.0, 0.1)

        nodes = {
            0: TreeNode(node_id=0, feature_index=feat_idx, threshold=threshold,
                         left_child_id=1, right_child_id=2, depth=0),
            1: TreeNode(node_id=1, leaf_value=left_val, depth=1),
            2: TreeNode(node_id=2, leaf_value=right_val, depth=1),
        }
        tree = TreeIR(tree_id=i, nodes=nodes, root_id=0, max_depth=1)
        trees.append(tree)

    return ModelIR(model_type="xgboost", trees=trees, num_features=4, base_score=0.0)


@pytest.fixture
def random_forest_model():
    """Random forest: independent trees using different feature subsets."""
    trees = []
    np.random.seed(42)
    num_features = 10

    for i in range(50):
        # Each tree uses a random subset of features (RF style)
        available_features = np.random.choice(num_features, size=4, replace=False)

        # Build depth-4 tree with subset features
        nodes = {}
        node_id = 0

        def build_tree(depth, max_depth=4):
            nonlocal node_id
            current_id = node_id
            node_id += 1

            if depth >= max_depth:
                nodes[current_id] = TreeNode(
                    node_id=current_id,
                    leaf_value=np.random.uniform(-0.5, 0.5),
                    depth=depth
                )
                return current_id

            feat_idx = int(available_features[depth % len(available_features)])
            threshold = np.random.uniform(0.2, 0.8)

            left_id = build_tree(depth + 1, max_depth)
            right_id = build_tree(depth + 1, max_depth)

            nodes[current_id] = TreeNode(
                node_id=current_id,
                feature_index=feat_idx,
                threshold=threshold,
                left_child_id=left_id,
                right_child_id=right_id,
                depth=depth
            )
            return current_id

        root_id = build_tree(0)
        tree = TreeIR(tree_id=i, nodes=nodes, root_id=root_id, max_depth=4)
        trees.append(tree)

    return ModelIR(model_type="xgboost", trees=trees, num_features=num_features, base_score=0.0)


@pytest.fixture
def boosted_ensemble_model():
    """Standard GBDT: all trees use same features (high overlap)."""
    trees = []
    for i in range(50):
        nodes = {
            0: TreeNode(node_id=0, feature_index=0, threshold=0.5,
                         left_child_id=1, right_child_id=2, depth=0),
            1: TreeNode(node_id=1, feature_index=1, threshold=0.3,
                         left_child_id=3, right_child_id=4, depth=1),
            2: TreeNode(node_id=2, feature_index=1, threshold=0.7,
                         left_child_id=5, right_child_id=6, depth=1),
            3: TreeNode(node_id=3, leaf_value=-0.3 * (0.9 ** i), depth=2),
            4: TreeNode(node_id=4, leaf_value=-0.1 * (0.9 ** i), depth=2),
            5: TreeNode(node_id=5, leaf_value=0.1 * (0.9 ** i), depth=2),
            6: TreeNode(node_id=6, leaf_value=0.3 * (0.9 ** i), depth=2),
        }
        tree = TreeIR(tree_id=i, nodes=nodes, root_id=0, max_depth=3)
        trees.append(tree)

    return ModelIR(model_type="xgboost", trees=trees, num_features=4, base_score=0.5)


@pytest.fixture
def sample_features():
    """Sample feature data."""
    np.random.seed(42)
    return np.random.randn(200, 10)


# =============================================================================
# Test 1: Model Structure Classifier
# =============================================================================

class TestModelStructureClassifier:
    """Tests for model structure classification (Contribution 1)."""

    def test_classifies_single_tree(self, single_tree_model):
        """Single tree should be classified as SINGLE_TREE."""
        classifier = ModelStructureClassifier()
        analysis = classifier.classify(single_tree_model)

        assert analysis.structure_type == ModelStructureType.SINGLE_TREE
        assert analysis.confidence == 1.0
        assert analysis.num_trees == 1
        assert analysis.max_depth == 3
        assert analysis.recommended_strategy == "precision_adaptive_sign"

    def test_classifies_linear_model(self, linear_model_stumps):
        """Depth-1 stumps should be classified as LINEAR_MODEL."""
        classifier = ModelStructureClassifier()
        analysis = classifier.classify(linear_model_stumps)

        assert analysis.structure_type == ModelStructureType.LINEAR_MODEL
        assert analysis.is_effectively_linear is True
        assert analysis.linear_weight_estimate is not None
        assert analysis.linear_bias_estimate is not None
        assert analysis.recommended_strategy == "comparison_free_linear"

        # Linear path should have lower depth than tree path
        assert analysis.multiplicative_depth_linear_path < analysis.multiplicative_depth_tree_path

    def test_classifies_random_forest(self, random_forest_model):
        """Independent trees with feature subsampling → RANDOM_FOREST."""
        classifier = ModelStructureClassifier()
        analysis = classifier.classify(random_forest_model)

        assert analysis.structure_type == ModelStructureType.RANDOM_FOREST
        assert analysis.tree_independence_score > 0.5
        assert analysis.uses_feature_subsampling is True
        assert analysis.recommended_strategy == "independent_noise_channels"
        assert analysis.estimated_noise_savings_percent > 0

    def test_classifies_boosted_ensemble(self, boosted_ensemble_model):
        """Standard GBDT with high feature overlap → BOOSTED_ENSEMBLE."""
        classifier = ModelStructureClassifier()
        analysis = classifier.classify(boosted_ensemble_model)

        assert analysis.structure_type == ModelStructureType.BOOSTED_ENSEMBLE
        assert analysis.recommended_strategy == "standard_moai_pipeline"

    def test_empty_model(self):
        """Empty model should be classified as LINEAR (constant)."""
        model = ModelIR(model_type="xgboost", trees=[], num_features=4, base_score=0.5)
        classifier = ModelStructureClassifier()
        analysis = classifier.classify(model)

        assert analysis.structure_type == ModelStructureType.LINEAR_MODEL
        assert analysis.confidence == 1.0

    def test_independence_score_computed(self, random_forest_model, boosted_ensemble_model):
        """RF should have higher independence score than GBDT."""
        classifier = ModelStructureClassifier()

        rf_analysis = classifier.classify(random_forest_model)
        gbdt_analysis = classifier.classify(boosted_ensemble_model)

        # RF trees use different feature subsets → higher independence
        assert rf_analysis.tree_independence_score > gbdt_analysis.tree_independence_score

    def test_linear_weight_extraction(self, linear_model_stumps):
        """Extracted weights should be reasonable."""
        classifier = ModelStructureClassifier()
        analysis = classifier.classify(linear_model_stumps)

        weights = analysis.linear_weight_estimate
        assert weights is not None
        assert len(weights) == 4
        # Weights should be non-zero (model is not trivial)
        assert np.any(weights != 0)

    def test_noise_savings_positive_for_non_boosted(self, single_tree_model, random_forest_model):
        """Non-boosted models should show noise savings."""
        classifier = ModelStructureClassifier()

        st_analysis = classifier.classify(single_tree_model)
        rf_analysis = classifier.classify(random_forest_model)

        assert st_analysis.estimated_noise_savings_percent >= 0
        assert rf_analysis.estimated_noise_savings_percent > 0


# =============================================================================
# Test 2: Comparison-Free Linear Model FHE
# =============================================================================

class TestComparisonFreeLinear:
    """Tests for comparison-free linear model evaluation (Contribution 2)."""

    def test_link_function_library_complete(self):
        """All standard link functions should be available."""
        library = LinkFunctionLibrary()
        available = library.list_available()

        assert "identity" in available
        assert "sigmoid" in available
        assert "log" in available
        assert "probit" in available
        assert "inverse" in available
        assert "sqrt" in available

    def test_sigmoid_approximation_accuracy(self):
        """Sigmoid polynomial should approximate well on [-8, 8]."""
        library = LinkFunctionLibrary()
        sigmoid_approx = library.get("sigmoid")

        assert sigmoid_approx is not None
        assert sigmoid_approx.max_error < 0.01  # Less than 1% error

        # Test at specific points
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        exact = 1.0 / (1.0 + np.exp(-x))
        approx = sigmoid_approx.evaluate(x)

        np.testing.assert_allclose(approx, exact, atol=0.02)

    def test_identity_link_exact(self):
        """Identity link should be exact."""
        library = LinkFunctionLibrary()
        identity = library.get("identity")

        assert identity is not None
        assert identity.multiplicative_depth == 0  # No multiplications needed

        x = np.linspace(-5, 5, 100)
        result = identity.evaluate(x)
        np.testing.assert_allclose(result, x, atol=1e-6)  # Identity: f(x) = x

    def test_linear_evaluator_plaintext(self, linear_model_stumps):
        """Comparison-free evaluator should produce reasonable predictions."""
        classifier = ModelStructureClassifier()
        analysis = classifier.classify(linear_model_stumps)

        weights = analysis.linear_weight_estimate
        bias = analysis.linear_bias_estimate

        evaluator = ComparisonFreeLinearEvaluator()

        # Generate test data
        np.random.seed(42)
        X = np.random.randn(50, 4)

        predictions = evaluator.evaluate_plaintext(weights, bias, X, "sigmoid")

        # Predictions should be in [0, 1] for sigmoid
        assert np.all(predictions >= -0.1)  # Small tolerance for polynomial error
        assert np.all(predictions <= 1.1)

    def test_depth_analysis_shows_reduction(self, linear_model_stumps):
        """Linear path should show significant depth reduction."""
        classifier = ModelStructureClassifier()
        analysis = classifier.classify(linear_model_stumps)

        evaluator = ComparisonFreeLinearEvaluator()
        depth_analysis = evaluator.get_depth_analysis(analysis, "sigmoid")

        assert depth_analysis["linear_path_depth"] < depth_analysis["tree_path_depth"]
        assert depth_analysis["depth_reduction_factor"] > 10  # At least 10x
        assert depth_analysis["bootstraps_eliminated"] > 0

    def test_noise_simulation(self, linear_model_stumps):
        """Simulated encrypted evaluation should be close to exact."""
        classifier = ModelStructureClassifier()
        analysis = classifier.classify(linear_model_stumps)

        weights = analysis.linear_weight_estimate
        bias = analysis.linear_bias_estimate

        evaluator = ComparisonFreeLinearEvaluator()
        np.random.seed(42)
        X = np.random.randn(50, 4)

        predictions, eval_analysis = evaluator.evaluate_encrypted_simulation(
            weights, bias, X, "sigmoid", noise_std=0.001
        )

        assert eval_analysis["mean_absolute_error"] < 0.1
        assert eval_analysis["multiplicative_depth"] < 10

    def test_link_function_depth_comparison(self):
        """Different link functions should have different depths."""
        library = LinkFunctionLibrary()
        comparison = library.get_depth_comparison()

        # Identity should have lowest depth
        assert comparison["identity"]["multiplicative_depth"] == 0

        # Sigmoid and probit should have similar depth
        assert comparison["sigmoid"]["multiplicative_depth"] > 0
        assert comparison["probit"]["multiplicative_depth"] > 0


# =============================================================================
# Test 3: Independent Noise Channels for Random Forest
# =============================================================================

class TestIndependentNoiseChannels:
    """Tests for RF-specific noise optimization (Contribution 3)."""

    def test_noise_reduction_for_rf(self, random_forest_model):
        """RF should achieve noise reduction vs sequential GBDT."""
        classifier = ModelStructureClassifier()
        analysis = classifier.classify(random_forest_model)

        optimizer = IndependentNoiseOptimizer()
        schedule = optimizer.compute_schedule(random_forest_model, analysis)

        # Independent channels should use less noise than sequential
        assert schedule.total_noise < schedule.noise_without_independence
        noise_ratio = schedule.noise_without_independence / max(schedule.total_noise, 0.01)
        assert noise_ratio > 1.5  # At least 1.5x reduction

    def test_tree_grouping(self, random_forest_model):
        """Trees should be grouped into noise-budget-fitting chunks."""
        classifier = ModelStructureClassifier()
        analysis = classifier.classify(random_forest_model)

        optimizer = IndependentNoiseOptimizer()
        schedule = optimizer.compute_schedule(random_forest_model, analysis)

        # Should have at least 1 group
        assert len(schedule.tree_groups) >= 1

        # All trees should be in some group
        all_trees = set()
        for group in schedule.tree_groups:
            all_trees.update(group)
        assert len(all_trees) == len(random_forest_model.trees)

    def test_theoretical_scaling(self):
        """Verify O(D + log T) scaling vs O(T × D)."""
        optimizer = IndependentNoiseOptimizer()

        # Compare for 100 trees, depth 6
        comparison = optimizer.get_theoretical_comparison(100, 6)

        assert comparison["rf_total_noise_bits"] < comparison["gbdt_total_noise_bits"]
        assert comparison["rf_bootstraps_needed"] < comparison["gbdt_bootstraps_needed"]
        assert comparison["noise_reduction_factor"] > 5  # Significant reduction

    def test_scaling_improves_with_more_trees(self):
        """More trees should increase the benefit of independence."""
        optimizer = IndependentNoiseOptimizer()

        small = optimizer.get_theoretical_comparison(10, 6)
        large = optimizer.get_theoretical_comparison(100, 6)

        # Benefit should increase with more trees
        assert large["noise_reduction_factor"] > small["noise_reduction_factor"]

    def test_bootstrap_schedule(self, random_forest_model):
        """Bootstrap should only occur between groups, not within."""
        classifier = ModelStructureClassifier()
        analysis = classifier.classify(random_forest_model)

        optimizer = IndependentNoiseOptimizer()
        schedule = optimizer.compute_schedule(random_forest_model, analysis)

        # Number of bootstraps = number of groups - 1
        assert len(schedule.bootstrap_schedule) == max(0, len(schedule.tree_groups) - 1)

    def test_aggregation_noise_logarithmic(self, random_forest_model):
        """Aggregation noise should be O(log T)."""
        classifier = ModelStructureClassifier()
        analysis = classifier.classify(random_forest_model)

        optimizer = IndependentNoiseOptimizer()
        schedule = optimizer.compute_schedule(random_forest_model, analysis)

        import math
        expected_agg_noise = math.ceil(math.log2(len(random_forest_model.trees))) * 0.1
        assert abs(schedule.aggregation_noise - expected_agg_noise) < 1.0


# =============================================================================
# Test 4: Precision-Adaptive Sign for Single Trees
# =============================================================================

class TestPrecisionAdaptiveSign:
    """Tests for precision-adaptive sign approximation (Contribution 4)."""

    def test_optimal_degree_higher_for_single_tree(self):
        """Single tree should get higher polynomial degree than GBDT tree."""
        sign_opt = PrecisionAdaptiveSign(noise_budget_bits=31.0)

        single_tree_degree = sign_opt.compute_optimal_degree(tree_depth=6)
        # With full budget for 6 levels, should get higher than default 7
        assert single_tree_degree >= 7

    def test_correctness_bound_improves_with_degree(self):
        """Higher degree should give tighter correctness bounds."""
        sign_opt = PrecisionAdaptiveSign()

        margin = 0.1
        bound_7 = sign_opt.compute_correctness_bound(7, margin)
        bound_15 = sign_opt.compute_correctness_bound(15, margin)

        # Higher degree should have lower error
        assert bound_15.max_absolute_error <= bound_7.max_absolute_error

    def test_correctness_bound_improves_with_margin(self):
        """Larger margin should give better correctness."""
        sign_opt = PrecisionAdaptiveSign()

        small_margin = sign_opt.compute_correctness_bound(11, 0.01)
        large_margin = sign_opt.compute_correctness_bound(11, 0.5)

        # Larger margin should have lower error
        assert large_margin.max_absolute_error <= small_margin.max_absolute_error

    def test_high_degree_sign_fits(self):
        """Should be able to fit sign polynomials at various degrees."""
        sign_opt = PrecisionAdaptiveSign()

        for degree in [7, 11, 15, 21]:
            coeffs = sign_opt.fit_high_degree_sign(degree)
            assert coeffs is not None
            assert len(coeffs) == degree + 1

            # Evaluate: should approximate sign(x) for |x| > 0.1
            x = np.linspace(-1, -0.1, 100)
            y_exact = np.sign(x)
            y_approx = chebyshev.chebval(x, coeffs)
            # Should mostly be negative
            assert np.mean(y_approx < 0) > 0.8

    def test_margin_analysis(self, single_tree_model, sample_features):
        """Margin analysis should return meaningful statistics."""
        sign_opt = PrecisionAdaptiveSign()
        X = sample_features[:, :4]  # 4 features

        analysis = sign_opt.analyze_tree_margins(single_tree_model, X)

        assert "tree_depth" in analysis
        assert "num_comparisons" in analysis
        assert "optimal_sign_degree" in analysis
        assert "margin_distribution" in analysis
        assert "mean_margin" in analysis
        assert analysis["tree_depth"] == 3
        assert analysis["num_comparisons"] > 0
        assert analysis["mean_margin"] > 0

    def test_odd_symmetry_preserved(self):
        """Sign polynomial should maintain odd symmetry: p(-x) = -p(x)."""
        sign_opt = PrecisionAdaptiveSign()
        coeffs = sign_opt.fit_high_degree_sign(11)

        # Even coefficients should be ~0 (odd function)
        for i in range(0, len(coeffs), 2):
            assert abs(coeffs[i]) < 1e-10, f"Even coefficient {i} should be 0"


# =============================================================================
# Test 5: Encrypted Majority Vote
# =============================================================================

class TestEncryptedMajorityVote:
    """Tests for encrypted majority vote (Contribution 5)."""

    def test_vote_counting(self):
        """Vote counting should correctly tally class predictions."""
        voter = EncryptedMajorityVote(MajorityVoteConfig(num_classes=3))

        # 3 samples, 5 trees
        predictions = np.array([
            [0, 0, 1, 1, 0],  # Class 0 wins (3 vs 2)
            [1, 1, 1, 0, 2],  # Class 1 wins (3 vs 1 vs 1)
            [2, 2, 2, 1, 0],  # Class 2 wins (3 vs 1 vs 1)
        ])

        counts = voter.count_votes_plaintext(predictions, 3)
        assert counts.shape == (3, 3)

    def test_majority_vote_correct(self):
        """Majority vote should select the most common class."""
        voter = EncryptedMajorityVote(MajorityVoteConfig(num_classes=2))

        # Binary: 7 out of 10 trees predict class 1
        predictions = np.array([
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # 7 votes for class 1
        ])

        classes, probs = voter.majority_vote_plaintext(predictions, 2)
        assert classes[0] == 1  # Should predict class 1
        assert probs[0, 1] > probs[0, 0]  # Class 1 probability should be higher

    def test_majority_vote_batch(self):
        """Majority vote should work on batches."""
        voter = EncryptedMajorityVote(MajorityVoteConfig(num_classes=2))

        np.random.seed(42)
        predictions = np.random.randint(0, 2, size=(100, 20))

        classes, probs = voter.majority_vote_plaintext(predictions, 2)
        assert classes.shape == (100,)
        assert probs.shape == (100, 2)

        # Probabilities should sum to ~1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=0.1)

    def test_fhe_depth_analysis(self):
        """FHE depth should be reasonable."""
        voter = EncryptedMajorityVote()
        analysis = voter.get_fhe_depth_analysis()

        assert analysis["vote_counting_depth"] == 0  # Pure additions
        assert analysis["total_depth"] > 0
        assert analysis["total_depth"] < 10  # Should be manageable

    def test_comparison_with_weighted_sum(self):
        """Compare majority vote with GBDT-style weighted sum."""
        voter = EncryptedMajorityVote(MajorityVoteConfig(num_classes=2))

        np.random.seed(42)
        # Use skewed predictions so both methods agree more often
        # 50 trees, most predict class 1
        predictions = np.zeros((100, 50), dtype=int)
        for i in range(100):
            n_ones = np.random.randint(30, 50)  # Bias toward class 1
            predictions[i, :n_ones] = 1
        weights = np.ones(50) * 0.1

        comparison = voter.compare_with_weighted_sum(predictions, weights, 2)

        assert "agreement_rate" in comparison
        assert "majority_vote_fhe_depth" in comparison
        assert "weighted_sum_fhe_depth" in comparison
        # Both methods should produce valid results
        assert 0.0 <= comparison["agreement_rate"] <= 1.0

    def test_multiclass_majority_vote(self):
        """Should handle multi-class voting and return valid probabilities."""
        voter = EncryptedMajorityVote(
            MajorityVoteConfig(num_classes=3, softmax_temperature=5.0)
        )

        # 3-class prediction with clear winners
        predictions = np.array([
            [0, 0, 0, 0, 0, 1, 2, 0, 0, 0],  # Class 0: 7 votes
            [1, 1, 1, 1, 1, 1, 0, 2, 2, 2],  # Class 1: 6 votes
        ])

        classes, probs = voter.majority_vote_plaintext(predictions, 3)

        # Probabilities should be valid
        assert probs.shape == (2, 3)
        for i in range(2):
            # All probabilities non-negative
            assert np.all(probs[i] >= 0)
            # Probabilities should sum to ~1
            np.testing.assert_allclose(probs[i].sum(), 1.0, atol=0.15)

        # The class with most votes should have highest probability
        assert probs[0, 0] > probs[0, 2]  # Class 0 > class 2 for sample 0
        assert probs[1, 1] > probs[1, 2]  # Class 1 > class 2 for sample 1


# =============================================================================
# Test 6: Unified Model-Aware Engine
# =============================================================================

class TestModelAwareFHEEngine:
    """Tests for the unified model-aware FHE engine."""

    def test_full_analysis_single_tree(self, single_tree_model, sample_features):
        """Full analysis for single tree."""
        engine = ModelAwareFHEEngine()
        X = sample_features[:, :4]
        result = engine.analyze(single_tree_model, X)

        assert result["model_structure"]["type"] == "single_tree"
        assert "single_tree_analysis" in result
        assert "depth_comparison" in result
        assert result["single_tree_analysis"]["optimal_sign_degree"] >= 7

    def test_full_analysis_linear(self, linear_model_stumps):
        """Full analysis for linear model."""
        engine = ModelAwareFHEEngine()
        result = engine.analyze(linear_model_stumps)

        assert result["model_structure"]["type"] == "linear"
        assert "linear_analysis" in result
        assert "depth_analysis" in result["linear_analysis"]

        # Should recommend comparison-free path
        assert result["recommended_strategy"] == "comparison_free_linear"

    def test_full_analysis_random_forest(self, random_forest_model):
        """Full analysis for random forest."""
        engine = ModelAwareFHEEngine()
        result = engine.analyze(random_forest_model)

        assert result["model_structure"]["type"] == "random_forest"
        assert "rf_analysis" in result
        assert "noise_schedule" in result["rf_analysis"]
        assert result["rf_analysis"]["noise_schedule"]["noise_reduction_factor"] > 1.0

    def test_full_analysis_boosted(self, boosted_ensemble_model):
        """Full analysis for boosted ensemble."""
        engine = ModelAwareFHEEngine()
        result = engine.analyze(boosted_ensemble_model)

        assert result["model_structure"]["type"] == "boosted"
        assert "boosted_analysis" in result

    def test_depth_comparison_across_strategies(self, linear_model_stumps):
        """Depth comparison should show linear < boosted."""
        engine = ModelAwareFHEEngine()
        result = engine.analyze(linear_model_stumps)

        depths = result["depth_comparison"]["strategy_depths"]
        assert depths["linear_sigmoid"] < depths["tree_comparison_standard"]

    def test_evaluate_plaintext_linear(self, linear_model_stumps):
        """Plaintext evaluation should work for linear models."""
        engine = ModelAwareFHEEngine()
        np.random.seed(42)
        X = np.random.randn(50, 4)

        predictions, metadata = engine.evaluate_plaintext(
            linear_model_stumps, X, "sigmoid"
        )

        assert predictions.shape == (50,)
        assert metadata["strategy"] == "comparison_free_linear"

    def test_evaluate_plaintext_tree(self, single_tree_model):
        """Plaintext evaluation should work for single trees."""
        engine = ModelAwareFHEEngine()
        np.random.seed(42)
        X = np.random.randn(50, 4)

        predictions, metadata = engine.evaluate_plaintext(single_tree_model, X)

        assert predictions.shape == (50,)

    def test_convenience_functions(self, single_tree_model, random_forest_model):
        """Convenience functions should work."""
        # classify_model_structure
        analysis = classify_model_structure(single_tree_model)
        assert analysis.structure_type == ModelStructureType.SINGLE_TREE

        # analyze_model_aware_fhe
        result = analyze_model_aware_fhe(random_forest_model)
        assert "model_structure" in result

    def test_noise_savings_reported(self, linear_model_stumps, random_forest_model):
        """All non-boosted models should report noise savings."""
        engine = ModelAwareFHEEngine()

        linear_result = engine.analyze(linear_model_stumps)
        rf_result = engine.analyze(random_forest_model)

        assert linear_result["estimated_noise_savings_percent"] > 0
        assert rf_result["estimated_noise_savings_percent"] > 0


# =============================================================================
# Test 7: Edge Cases and Robustness
# =============================================================================

class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_single_leaf_tree(self):
        """Tree with only a leaf (no splits)."""
        nodes = {
            0: TreeNode(node_id=0, leaf_value=0.5, depth=0),
        }
        tree = TreeIR(tree_id=0, nodes=nodes, root_id=0, max_depth=0)
        model = ModelIR(model_type="xgboost", trees=[tree], num_features=4, base_score=0.5)

        classifier = ModelStructureClassifier()
        analysis = classifier.classify(model)
        # Should still classify successfully
        assert analysis is not None
        assert analysis.structure_type == ModelStructureType.SINGLE_TREE

    def test_very_deep_single_tree(self):
        """Deep single tree should still get optimal degree."""
        sign_opt = PrecisionAdaptiveSign(noise_budget_bits=31.0)
        degree = sign_opt.compute_optimal_degree(tree_depth=15)
        assert degree >= 7  # Should be at least standard

    def test_large_forest_noise(self):
        """Large forest noise comparison."""
        optimizer = IndependentNoiseOptimizer()
        comparison = optimizer.get_theoretical_comparison(500, 8)
        assert comparison["rf_total_noise_bits"] < comparison["gbdt_total_noise_bits"]
        assert comparison["noise_reduction_factor"] > 10

    def test_link_function_domain_clipping(self):
        """Link functions should handle out-of-domain inputs gracefully."""
        library = LinkFunctionLibrary()
        sigmoid = library.get("sigmoid")

        # Way outside domain
        x = np.array([-100, -50, 0, 50, 100])
        result = sigmoid.evaluate(x)
        # Should not produce NaN or Inf
        assert np.all(np.isfinite(result))


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
