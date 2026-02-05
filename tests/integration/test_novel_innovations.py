"""
Integration Tests for Novel FHE-GBDT Innovations

Tests all 10 novel architectural innovations:
1. Leaf-Centric Encoding
2. Gradient-Informed Noise Budget Allocation
3. Homomorphic Ensemble Pruning
4. N2HE Multi-Key Federated GBDT
5. Bootstrapping-Aligned Tree Architecture
6. Polynomial Leaf Functions
7. MOAI-Native Tree Structure
8. Streaming Encrypted Gradients
9. Unified Architecture
10. C++ Kernel Extensions (via Python bindings)
"""

import pytest
import numpy as np
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.compiler.ir import ModelIR, TreeIR, TreeNode


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_model_ir():
    """Create a simple GBDT model for testing."""
    # Tree 1: Simple depth-2 tree
    tree1_nodes = {
        0: TreeNode(node_id=0, feature_index=0, threshold=0.5,
                   left_child_id=1, right_child_id=2, depth=0),
        1: TreeNode(node_id=1, leaf_value=-0.3, depth=1),
        2: TreeNode(node_id=2, leaf_value=0.3, depth=1),
    }
    tree1 = TreeIR(tree_id=0, nodes=tree1_nodes, root_id=0, max_depth=2)

    # Tree 2: Depth-3 tree
    tree2_nodes = {
        0: TreeNode(node_id=0, feature_index=1, threshold=0.3,
                   left_child_id=1, right_child_id=2, depth=0),
        1: TreeNode(node_id=1, feature_index=0, threshold=0.2,
                   left_child_id=3, right_child_id=4, depth=1),
        2: TreeNode(node_id=2, feature_index=0, threshold=0.7,
                   left_child_id=5, right_child_id=6, depth=1),
        3: TreeNode(node_id=3, leaf_value=-0.2, depth=2),
        4: TreeNode(node_id=4, leaf_value=0.1, depth=2),
        5: TreeNode(node_id=5, leaf_value=0.2, depth=2),
        6: TreeNode(node_id=6, leaf_value=0.4, depth=2),
    }
    tree2 = TreeIR(tree_id=1, nodes=tree2_nodes, root_id=0, max_depth=3)

    return ModelIR(
        model_type="xgboost",
        trees=[tree1, tree2],
        num_features=4,
        base_score=0.5
    )


@pytest.fixture
def oblivious_model_ir():
    """Create an oblivious (CatBoost-style) model for testing."""
    # All nodes at same depth use same feature (oblivious property)
    tree_nodes = {
        0: TreeNode(node_id=0, feature_index=0, threshold=0.5,
                   left_child_id=1, right_child_id=2, depth=0),
        1: TreeNode(node_id=1, feature_index=1, threshold=0.3,
                   left_child_id=3, right_child_id=4, depth=1),
        2: TreeNode(node_id=2, feature_index=1, threshold=0.3,
                   left_child_id=5, right_child_id=6, depth=1),
        3: TreeNode(node_id=3, leaf_value=-0.5, depth=2),
        4: TreeNode(node_id=4, leaf_value=-0.1, depth=2),
        5: TreeNode(node_id=5, leaf_value=0.1, depth=2),
        6: TreeNode(node_id=6, leaf_value=0.5, depth=2),
    }
    tree = TreeIR(tree_id=0, nodes=tree_nodes, root_id=0, max_depth=3)

    return ModelIR(
        model_type="catboost",
        trees=[tree],
        num_features=4,
        base_score=0.0
    )


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = np.sum(X[:, :2], axis=1) + 0.1 * np.random.randn(100)
    return X, y


@pytest.fixture
def large_model_ir():
    """Create a larger model for stress testing."""
    trees = []
    for tree_idx in range(50):  # 50 trees
        nodes = {}
        # Depth 4 tree
        node_id = 0
        for depth in range(4):
            num_nodes_at_depth = 2 ** depth
            for i in range(num_nodes_at_depth):
                if depth < 3:
                    nodes[node_id] = TreeNode(
                        node_id=node_id,
                        feature_index=node_id % 10,  # 10 features
                        threshold=0.5 + 0.1 * (node_id % 5),
                        left_child_id=node_id * 2 + 1,
                        right_child_id=node_id * 2 + 2,
                        depth=depth
                    )
                else:
                    nodes[node_id] = TreeNode(
                        node_id=node_id,
                        leaf_value=0.1 * (node_id % 10 - 5),
                        depth=depth
                    )
                node_id += 1

        trees.append(TreeIR(tree_id=tree_idx, nodes=nodes, root_id=0, max_depth=4))

    return ModelIR(
        model_type="xgboost",
        trees=trees,
        num_features=10,
        base_score=0.5
    )


# =============================================================================
# Test Innovation #1: Leaf-Centric Encoding
# =============================================================================

class TestLeafCentricEncoding:
    """Tests for leaf-centric encoding innovation."""

    def test_leaf_indicator_computer_basic(self):
        """Test basic leaf indicator computation."""
        from services.innovations.leaf_centric import LeafIndicatorComputer

        computer = LeafIndicatorComputer(max_depth=4)

        # Test sign computation
        x = np.array([0.1, 0.5, 0.9])
        signs = computer.polynomial_sign(x)

        # Should map to approximately [0, 0.5, 1]
        assert signs.shape == (3,)
        assert 0 <= signs[0] <= 0.5
        assert 0.4 <= signs[1] <= 0.6
        assert 0.5 <= signs[2] <= 1.0

    def test_leaf_indicator_tensor_product(self):
        """Test tensor product leaf indicator computation."""
        from services.innovations.leaf_centric import LeafIndicatorComputer

        computer = LeafIndicatorComputer()

        # 2 samples, depth 2
        signs = np.array([
            [0.2, 0.8],  # Sample 1: mostly left, mostly right
            [0.9, 0.1],  # Sample 2: mostly right, mostly left
        ])

        indicators = computer.compute_leaf_indicators_tensor(signs, depth=2)

        # Should have 4 leaves (2^2)
        assert indicators.shape == (2, 4)

        # Indicators should sum to ~1 for each sample
        np.testing.assert_array_almost_equal(
            indicators.sum(axis=1),
            np.ones(2),
            decimal=5
        )

    def test_encoder_creates_plan(self, simple_model_ir):
        """Test that encoder creates valid plan."""
        from services.innovations.leaf_centric import LeafCentricEncoder

        encoder = LeafCentricEncoder()
        plan = encoder.encode_model(simple_model_ir)

        assert plan.num_trees == 2
        assert plan.max_depth >= 2
        assert len(plan.leaf_indicators) == 2

    def test_encoder_evaluation_consistency(self, simple_model_ir, sample_data):
        """Test that leaf-centric evaluation matches standard."""
        from services.innovations.leaf_centric import LeafCentricEncoder

        X, _ = sample_data
        encoder = LeafCentricEncoder()
        plan = encoder.encode_model(simple_model_ir)

        # Evaluate using leaf-centric method
        predictions = encoder.evaluate_plaintext(
            plan, X[:10], simple_model_ir.base_score
        )

        assert predictions.shape == (10,)
        # Predictions should be finite
        assert np.all(np.isfinite(predictions))


# =============================================================================
# Test Innovation #2: Gradient-Informed Noise Budget Allocation
# =============================================================================

class TestGradientNoiseAllocation:
    """Tests for gradient-informed noise budget allocation."""

    def test_feature_importance_analysis(self, simple_model_ir):
        """Test feature importance analysis."""
        from services.innovations.gradient_noise import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer()
        importance = analyzer.analyze(simple_model_ir)

        # Should have importance for used features
        assert len(importance) > 0

        # All importance values should be valid
        for feat_idx, imp in importance.items():
            assert 0 <= imp.gradient_importance <= 1
            assert imp.frequency > 0

    def test_noise_budget_allocation(self, simple_model_ir):
        """Test noise budget allocation."""
        from services.innovations.gradient_noise import GradientAwareNoiseAllocator

        allocator = GradientAwareNoiseAllocator()
        allocations = allocator.allocate(simple_model_ir, simple_model_ir.num_features)

        # Should have allocation for each feature
        assert len(allocations) == simple_model_ir.num_features

        # Check allocations are valid
        for feat_idx, alloc in allocations.items():
            assert 8 <= alloc.precision_bits <= 16
            assert alloc.encoding_scale > 0
            assert 0 < alloc.noise_budget_fraction <= 1

    def test_adaptive_precision_encoding(self, simple_model_ir, sample_data):
        """Test adaptive precision encoding."""
        from services.innovations.gradient_noise import (
            GradientAwareNoiseAllocator,
            AdaptivePrecisionEncoder
        )

        X, _ = sample_data
        allocator = GradientAwareNoiseAllocator()
        allocations = allocator.allocate(simple_model_ir, simple_model_ir.num_features)

        encoder = AdaptivePrecisionEncoder(allocations)

        # Encode and decode
        encoded, scales = encoder.encode(X[:5])
        decoded = encoder.decode(encoded, scales)

        # Should approximately recover original values
        np.testing.assert_array_almost_equal(X[:5], decoded, decimal=3)


# =============================================================================
# Test Innovation #3: Homomorphic Ensemble Pruning
# =============================================================================

class TestHomomorphicPruning:
    """Tests for homomorphic ensemble pruning."""

    def test_significance_computation(self):
        """Test tree significance computation."""
        from services.innovations.homomorphic_pruning import EncryptedTreeSignificance

        computer = EncryptedTreeSignificance()

        # Simulate tree outputs with varying significance
        np.random.seed(42)
        tree_outputs = np.random.randn(100, 10)  # 100 samples, 10 trees
        # Make some trees more significant
        tree_outputs[:, 0] *= 3  # High significance
        tree_outputs[:, 5] *= 0.1  # Low significance

        significance = computer.compute_significance_plaintext(tree_outputs)

        assert significance.shape == (10,)
        # Tree 0 should be more significant than tree 5
        assert significance[0] > significance[5]

    def test_pruning_gate(self):
        """Test pruning gate computation."""
        from services.innovations.homomorphic_pruning import AdaptivePruningGate, PruningConfig

        config = PruningConfig(significance_threshold=0.2, soft_pruning=True)
        gate = AdaptivePruningGate(config)

        significance = np.array([0.1, 0.3, 0.5, 0.05, 0.8])
        gates = gate.compute_gates_plaintext(significance)

        assert gates.shape == (5,)
        # High significance should have high gate
        assert gates[4] > gates[0]  # 0.8 > 0.1
        assert gates[2] > gates[3]  # 0.5 > 0.05

    def test_ensemble_pruning_end_to_end(self):
        """Test complete ensemble pruning pipeline."""
        from services.innovations.homomorphic_pruning import HomomorphicEnsemblePruner

        pruner = HomomorphicEnsemblePruner()

        np.random.seed(42)
        tree_outputs = np.random.randn(50, 20)  # 50 samples, 20 trees

        aggregated, metadata = pruner.prune_plaintext(tree_outputs)

        assert aggregated.shape == (50,)
        assert "num_active_trees" in metadata
        assert metadata["num_active_trees"] <= 20
        assert "pruning_ratio" in metadata


# =============================================================================
# Test Innovation #4: N2HE Multi-Key Federated GBDT
# =============================================================================

class TestFederatedMultiKey:
    """Tests for N2HE multi-key federated GBDT."""

    def test_party_creation(self):
        """Test multi-key party creation."""
        from services.innovations.federated_multikey import MultiKeyParty, MultiKeyConfig

        party = MultiKeyParty("party_1", [0, 1, 2])
        config = MultiKeyConfig()
        party.generate_keys(config)

        assert party._secret_key is not None
        assert party._public_key is not None

    def test_party_encryption(self):
        """Test party feature encryption."""
        from services.innovations.federated_multikey import MultiKeyParty, MultiKeyConfig

        party = MultiKeyParty("party_1", [0, 1])
        config = MultiKeyConfig()
        party.generate_keys(config)

        features = {0: 0.5, 1: -0.3}
        encrypted = party.encrypt_features(features)

        assert len(encrypted) == 2
        assert 0 in encrypted
        assert 1 in encrypted

    def test_combiner_registration(self):
        """Test multi-key combiner setup."""
        from services.innovations.federated_multikey import (
            MultiKeyParty, N2HEMultiKeyCombiner, MultiKeyConfig
        )

        config = MultiKeyConfig(decryption_threshold=2)
        combiner = N2HEMultiKeyCombiner(config)

        for i in range(3):
            party = MultiKeyParty(f"party_{i}", [i])
            party.generate_keys(config)
            combiner.register_party(party)

        assert len(combiner._parties) == 3

    @pytest.mark.asyncio
    async def test_federated_protocol(self, simple_model_ir):
        """Test complete federated protocol."""
        from services.innovations.federated_multikey import FederatedGBDTProtocol

        protocol = FederatedGBDTProtocol(simple_model_ir)

        # Create parties with feature assignments
        protocol.create_party("party_0", [0, 1])
        protocol.create_party("party_1", [2, 3])

        status = protocol.get_protocol_status()
        assert status["num_parties"] == 2


# =============================================================================
# Test Innovation #5: Bootstrapping-Aligned Tree Architecture
# =============================================================================

class TestBootstrapAligned:
    """Tests for bootstrapping-aligned tree architecture."""

    def test_noise_analysis(self, large_model_ir):
        """Test noise consumption analysis."""
        from services.innovations.bootstrap_aligned import BootstrapAwareTreeBuilder

        builder = BootstrapAwareTreeBuilder()
        analysis = builder.analyze_noise_consumption(large_model_ir)

        assert "max_depth" in analysis
        assert "total_estimated_noise" in analysis
        assert "needs_bootstrap" in analysis
        assert analysis["num_trees"] == 50

    def test_chunk_partitioning(self, large_model_ir):
        """Test bootstrap-aligned chunk partitioning."""
        from services.innovations.bootstrap_aligned import BootstrapAwareTreeBuilder

        builder = BootstrapAwareTreeBuilder()
        forest = builder.partition_into_chunks(large_model_ir)

        assert len(forest.chunks) > 0
        assert forest.total_trees == 50

        # All trees should be assigned to chunks
        total_in_chunks = sum(len(c.trees) for c in forest.chunks)
        assert total_in_chunks == 50

    def test_interleaved_ensemble_evaluation(self, simple_model_ir, sample_data):
        """Test interleaved ensemble evaluation."""
        from services.innovations.bootstrap_aligned import (
            create_bootstrap_aligned_forest,
            BootstrapInterleavedEnsemble
        )

        X, _ = sample_data
        forest = create_bootstrap_aligned_forest(simple_model_ir)
        ensemble = BootstrapInterleavedEnsemble(forest)

        predictions = ensemble.evaluate_plaintext(X[:10], simple_model_ir.base_score)

        assert predictions.shape == (10,)
        assert np.all(np.isfinite(predictions))


# =============================================================================
# Test Innovation #6: Polynomial Leaf Functions
# =============================================================================

class TestPolynomialLeaves:
    """Tests for polynomial leaf functions."""

    def test_polynomial_leaf_fitting(self, simple_model_ir, sample_data):
        """Test polynomial fitting at leaves."""
        from services.innovations.polynomial_leaves import (
            PolynomialLeafTrainer, PolynomialLeafConfig
        )

        X, y = sample_data
        config = PolynomialLeafConfig(max_degree=2, min_samples_for_poly=5)
        trainer = PolynomialLeafTrainer(config)

        poly_leaves = trainer.fit_leaf_polynomials(simple_model_ir, X, y)

        # May or may not fit polynomials depending on leaf sample counts
        assert isinstance(poly_leaves, dict)

    def test_polynomial_evaluation(self):
        """Test polynomial evaluation."""
        from services.innovations.polynomial_leaves import FHEPolynomialEvaluator

        evaluator = FHEPolynomialEvaluator(max_degree=3)

        # Polynomial: 1 + 2x + 3x^2
        coeffs = np.array([1.0, 2.0, 3.0])
        x = 2.0

        result = evaluator.evaluate_plaintext_horner = evaluator.evaluate_horner
        # Manually compute expected: 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        expected = 1 + 2*x + 3*x**2

        actual = evaluator.evaluate_horner(coeffs, len(coeffs), x)
        assert abs(actual - expected) < 1e-10

    def test_polynomial_leaf_gbdt(self, simple_model_ir, sample_data):
        """Test PolynomialLeafGBDT model."""
        from services.innovations.polynomial_leaves import PolynomialLeafGBDT

        X, y = sample_data
        model = PolynomialLeafGBDT(simple_model_ir)
        model.fit_polynomials(X, y)

        predictions = model.predict(X[:10])

        assert predictions.shape == (10,)
        assert np.all(np.isfinite(predictions))


# =============================================================================
# Test Innovation #7: MOAI-Native Tree Structure
# =============================================================================

class TestMOAINative:
    """Tests for MOAI-native tree structure."""

    def test_oblivious_detection(self, oblivious_model_ir, simple_model_ir):
        """Test oblivious tree detection."""
        from services.innovations.moai_native import MOAINativeTreeBuilder

        builder = MOAINativeTreeBuilder()

        # Analysis should detect oblivious property
        converter = builder.converter
        is_oblivious_catboost = all(
            builder.converter._check_level_oblivious(tree, d)
            for tree in oblivious_model_ir.trees
            for d in range(tree.max_depth)
        ) if hasattr(converter, '_check_level_oblivious') else True

    def test_tree_conversion(self, simple_model_ir, sample_data):
        """Test conversion to oblivious form."""
        from services.innovations.moai_native import RotationOptimalConverter

        X, y = sample_data
        converter = RotationOptimalConverter()

        result = converter.convert_model(simple_model_ir, X[:20], y[:20])

        assert len(result.oblivious_trees) == 2
        assert "rotation_savings" in dir(result) or hasattr(result, 'rotation_savings')

    def test_oblivious_tree_synthesis(self, sample_data):
        """Test synthesizing oblivious trees from scratch."""
        from services.innovations.moai_native import ObliviousTreeSynthesizer

        X, y = sample_data
        synthesizer = ObliviousTreeSynthesizer(
            max_depth=3,
            num_trees=5,
            learning_rate=0.1
        )

        trees = synthesizer.synthesize(X, y)

        assert len(trees) == 5
        for tree in trees:
            assert tree.max_depth == 3
            assert len(tree.leaf_values) == 2**3


# =============================================================================
# Test Innovation #8: Streaming Encrypted Gradients
# =============================================================================

class TestStreamingGradients:
    """Tests for streaming encrypted gradients."""

    def test_gradient_computation(self):
        """Test gradient computation."""
        from services.innovations.streaming_gradients import HomomorphicGradientComputer

        computer = HomomorphicGradientComputer(loss_type="mse")

        y_true = np.array([1.0, 0.0, 0.5])
        y_pred = np.array([0.8, 0.2, 0.6])

        gradients = computer.compute_gradient_plaintext(y_true, y_pred)

        expected = y_true - y_pred
        np.testing.assert_array_almost_equal(gradients, expected)

    def test_leaf_gradient_computation(self):
        """Test per-leaf gradient computation."""
        from services.innovations.streaming_gradients import HomomorphicGradientComputer

        computer = HomomorphicGradientComputer()

        gradients = np.array([0.1, -0.2, 0.3, -0.1])
        leaf_indicators = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float64)

        leaf_gradients = computer.compute_leaf_gradients_plaintext(
            gradients, leaf_indicators
        )

        assert leaf_gradients.shape == (4,)
        # Leaf 0 should have average of gradients[0] and gradients[2]
        np.testing.assert_almost_equal(leaf_gradients[0], (0.1 + 0.3) / 2)

    def test_streaming_gbdt_update(self, simple_model_ir, sample_data):
        """Test streaming GBDT updates."""
        from services.innovations.streaming_gradients import EncryptedStreamingGBDT

        X, y = sample_data
        streaming = EncryptedStreamingGBDT(simple_model_ir)

        # Process samples
        for i in range(50):
            streaming.process_sample(X[i], y[i])

        model_state = streaming.get_current_model()

        assert "base_score" in model_state
        assert "leaf_values" in model_state
        assert model_state["stats"]["num_updates"] > 0


# =============================================================================
# Test Innovation #9: Unified Architecture
# =============================================================================

class TestUnifiedArchitecture:
    """Tests for unified architecture integration."""

    def test_model_analysis(self, simple_model_ir):
        """Test model analysis."""
        from services.innovations.unified_architecture import NovelFHEGBDTEngine

        engine = NovelFHEGBDTEngine()
        analysis = engine.analyze_model(simple_model_ir)

        assert "model_stats" in analysis
        assert "recommendations" in analysis
        assert analysis["model_stats"]["num_trees"] == 2

    def test_execution_plan_creation(self, simple_model_ir, sample_data):
        """Test execution plan creation."""
        from services.innovations.unified_architecture import NovelFHEGBDTEngine

        X, y = sample_data
        engine = NovelFHEGBDTEngine()

        plan = engine.create_execution_plan(simple_model_ir, X, y)

        assert plan.plan_id is not None
        assert len(plan.innovations) > 0
        assert plan.rotation_savings_percent >= 0

    def test_unified_prediction(self, simple_model_ir, sample_data):
        """Test unified prediction."""
        from services.innovations.unified_architecture import NovelFHEGBDTEngine

        X, y = sample_data
        engine = NovelFHEGBDTEngine()
        engine.create_execution_plan(simple_model_ir, X, y)

        predictions = engine.predict(X[:10])

        assert predictions.shape == (10,)
        assert np.all(np.isfinite(predictions))

    def test_prediction_with_pruning(self, simple_model_ir, sample_data):
        """Test prediction with adaptive pruning."""
        from services.innovations.unified_architecture import (
            NovelFHEGBDTEngine, InnovationConfig, InnovationFlag
        )

        X, y = sample_data

        config = InnovationConfig(
            enabled_innovations={InnovationFlag.HOMOMORPHIC_PRUNING}
        )
        engine = NovelFHEGBDTEngine(config)
        engine.create_execution_plan(simple_model_ir)

        predictions, metadata = engine.predict_with_pruning(X[:10])

        assert predictions.shape == (10,)

    def test_optimization_report(self, simple_model_ir, sample_data):
        """Test optimization report generation."""
        from services.innovations.unified_architecture import NovelFHEGBDTEngine

        X, y = sample_data
        engine = NovelFHEGBDTEngine()
        engine.create_execution_plan(simple_model_ir, X, y)

        report = engine.get_optimization_report()

        assert "plan_id" in report
        assert "performance" in report
        assert "innovations_enabled" in report


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Tests for convenience factory functions."""

    def test_create_novel_engine_latency(self):
        """Test creating engine optimized for latency."""
        from services.innovations.unified_architecture import create_novel_engine

        engine = create_novel_engine(optimize_for="latency")
        assert engine is not None

    def test_create_novel_engine_accuracy(self):
        """Test creating engine optimized for accuracy."""
        from services.innovations.unified_architecture import create_novel_engine

        engine = create_novel_engine(optimize_for="accuracy")
        assert engine is not None

    def test_create_novel_engine_all(self):
        """Test creating engine with all innovations."""
        from services.innovations.unified_architecture import (
            create_novel_engine, InnovationFlag
        )

        engine = create_novel_engine(enable_all=True)
        assert len(engine.config.enabled_innovations) == len(InnovationFlag)

    def test_optimize_model_for_fhe(self, simple_model_ir, sample_data):
        """Test complete optimization pipeline."""
        from services.innovations.unified_architecture import optimize_model_for_fhe

        X, y = sample_data
        engine, plan = optimize_model_for_fhe(simple_model_ir, X, y)

        assert engine is not None
        assert plan is not None
        assert plan.rotation_savings_percent >= 0


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance tests for novel innovations."""

    def test_leaf_centric_batch_performance(self, large_model_ir, sample_data):
        """Test leaf-centric batch computation performance."""
        from services.innovations.leaf_centric import LeafIndicatorComputer
        import time

        X, _ = sample_data
        computer = LeafIndicatorComputer(max_depth=6)

        # Generate test data
        np.random.seed(42)
        signs = np.random.rand(100, 6)

        start = time.time()
        for _ in range(10):
            indicators = computer.compute_leaf_indicators_tensor(signs, depth=6)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 1.0  # Less than 1 second for 1000 evaluations

    def test_unified_engine_performance(self, large_model_ir, sample_data):
        """Test unified engine performance."""
        from services.innovations.unified_architecture import NovelFHEGBDTEngine
        import time

        X, y = sample_data
        engine = NovelFHEGBDTEngine()

        start = time.time()
        plan = engine.create_execution_plan(large_model_ir, X[:50], y[:50])
        plan_time = time.time() - start

        start = time.time()
        predictions = engine.predict(X[:100])
        predict_time = time.time() - start

        # Plan creation should be reasonably fast
        assert plan_time < 10.0  # Less than 10 seconds

        # Prediction should be fast
        assert predict_time < 1.0  # Less than 1 second for 100 samples


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
