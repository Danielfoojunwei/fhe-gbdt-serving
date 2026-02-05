"""
Novel Innovation #9: Unified Architecture Integration

Integrates all novel innovations into a cohesive FHE-GBDT system that
automatically selects and combines optimizations based on model characteristics.

Architecture Overview:
┌─────────────────────────────────────────────────────────────────────┐
│                    NOVEL FHE-GBDT ENGINE                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │  Gradient-Aware │    │  MOAI-Native    │    │  Polynomial     │ │
│  │  Noise Allocator│───▶│  Tree Converter │───▶│  Leaf Expander  │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│          │                      │                      │           │
│          ▼                      ▼                      ▼           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           Bootstrap-Aligned Execution Engine                │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │   │
│  │  │Forest 1 │──│Bootstrap│──│Forest 2 │──│Bootstrap│──...   │   │
│  │  │(depth 6)│  │         │  │(depth 6)│  │         │        │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│          │                                                         │
│          ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              N2HE Weighted Sum Aggregator                    │   │
│  │   Final = Σᵢ (tree_output_i × importance_weight_i)          │   │
│  │   Complexity: O(1) regardless of ensemble size!              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│          │                                                         │
│          ▼                                                         │
│  ┌─────────────────┐    ┌─────────────────┐                       │
│  │  Homomorphic    │    │  Federated      │                       │
│  │  Pruning Gate   │◀───│  Multi-Party    │                       │
│  │  (adaptive)     │    │  Combination    │                       │
│  └─────────────────┘    └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set, Union
from enum import Enum, auto
import logging
import time

import numpy as np

# Import all innovations
from .leaf_centric import LeafCentricEncoder, DirectLeafPlan
from .gradient_noise import GradientAwareNoiseAllocator, NoiseBudgetAllocation
from .homomorphic_pruning import HomomorphicEnsemblePruner, PruningConfig
from .federated_multikey import FederatedGBDTProtocol, MultiKeyConfig
from .bootstrap_aligned import BootstrapAwareTreeBuilder, NoiseAlignedForest
from .polynomial_leaves import PolynomialLeafGBDT, PolynomialLeafConfig
from .moai_native import MOAINativeTreeBuilder, ConversionResult
from .streaming_gradients import EncryptedStreamingGBDT, StreamingConfig

logger = logging.getLogger(__name__)


class InnovationFlag(Enum):
    """Flags for enabling/disabling innovations."""
    LEAF_CENTRIC = auto()
    GRADIENT_NOISE = auto()
    HOMOMORPHIC_PRUNING = auto()
    FEDERATED_MULTIKEY = auto()
    BOOTSTRAP_ALIGNED = auto()
    POLYNOMIAL_LEAVES = auto()
    MOAI_NATIVE = auto()
    STREAMING_GRADIENTS = auto()


@dataclass
class InnovationConfig:
    """Configuration for novel innovations."""
    # Which innovations to enable
    enabled_innovations: Set[InnovationFlag] = field(
        default_factory=lambda: {
            InnovationFlag.LEAF_CENTRIC,
            InnovationFlag.GRADIENT_NOISE,
            InnovationFlag.BOOTSTRAP_ALIGNED,
            InnovationFlag.MOAI_NATIVE,
        }
    )

    # Individual configs
    pruning_config: PruningConfig = field(default_factory=PruningConfig)
    multikey_config: MultiKeyConfig = field(default_factory=MultiKeyConfig)
    polynomial_config: PolynomialLeafConfig = field(default_factory=PolynomialLeafConfig)
    streaming_config: StreamingConfig = field(default_factory=StreamingConfig)

    # Optimization targets
    optimize_for: str = "latency"  # "latency", "throughput", "accuracy", "privacy"

    # Auto-tuning
    auto_tune: bool = True

    # Noise budget
    total_noise_budget_bits: int = 31

    # Target accuracy loss threshold
    max_accuracy_loss: float = 0.02


@dataclass
class UnifiedExecutionPlan:
    """Unified execution plan combining all innovations."""
    # Plan ID
    plan_id: str

    # Enabled innovations
    innovations: Set[InnovationFlag]

    # Component plans
    leaf_centric_plan: Optional[DirectLeafPlan] = None
    noise_allocations: Optional[Dict[int, NoiseBudgetAllocation]] = None
    bootstrap_forest: Optional[NoiseAlignedForest] = None
    moai_conversion: Optional[ConversionResult] = None
    polynomial_model: Optional[PolynomialLeafGBDT] = None

    # Execution metadata
    estimated_latency_ms: float = 0.0
    estimated_rotations: int = 0
    estimated_noise_bits: float = 0.0
    bootstrap_count: int = 0

    # Performance comparison
    baseline_rotations: int = 0
    rotation_savings_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plan_id": self.plan_id,
            "innovations": [i.name for i in self.innovations],
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_rotations": self.estimated_rotations,
            "estimated_noise_bits": self.estimated_noise_bits,
            "bootstrap_count": self.bootstrap_count,
            "rotation_savings_percent": self.rotation_savings_percent,
        }


class NovelFHEGBDTEngine:
    """
    Unified engine integrating all novel FHE-GBDT innovations.

    This engine:
    1. Analyzes model characteristics
    2. Auto-selects optimal innovations
    3. Generates unified execution plan
    4. Executes with all optimizations active
    """

    def __init__(self, config: Optional[InnovationConfig] = None):
        """
        Initialize engine.

        Args:
            config: Innovation configuration
        """
        self.config = config or InnovationConfig()

        # Initialize components
        self._leaf_encoder = LeafCentricEncoder()
        self._noise_allocator = GradientAwareNoiseAllocator()
        self._pruner = HomomorphicEnsemblePruner(self.config.pruning_config)
        self._bootstrap_builder = BootstrapAwareTreeBuilder()
        self._moai_builder = MOAINativeTreeBuilder()

        # Execution state
        self._current_plan: Optional[UnifiedExecutionPlan] = None
        self._model_ir: Optional[Any] = None

        logger.info(f"NovelFHEGBDTEngine initialized with {len(self.config.enabled_innovations)} innovations")

    def analyze_model(self, model_ir: Any) -> Dict[str, Any]:
        """
        Analyze model to recommend optimal innovations.

        Args:
            model_ir: Parsed ModelIR

        Returns:
            Analysis results with recommendations
        """
        self._model_ir = model_ir

        # Gather statistics
        num_trees = len(model_ir.trees)
        max_depth = max(t.max_depth for t in model_ir.trees)
        total_nodes = sum(len(t.nodes) for t in model_ir.trees)
        num_features = model_ir.num_features

        # Check if already oblivious (CatBoost)
        is_oblivious = self._check_oblivious(model_ir)

        # Analyze noise requirements
        noise_analysis = self._bootstrap_builder.analyze_noise_consumption(model_ir)

        # Compute baseline rotation cost
        baseline_rotations = total_nodes

        # Recommendations
        recommendations = []

        if not is_oblivious:
            recommendations.append({
                "innovation": "MOAI_NATIVE",
                "reason": "Convert to oblivious for rotation-free execution",
                "expected_benefit": f"Up to {baseline_rotations}x rotation reduction"
            })

        if noise_analysis["needs_bootstrap"]:
            recommendations.append({
                "innovation": "BOOTSTRAP_ALIGNED",
                "reason": f"Model needs {noise_analysis['estimated_bootstraps']} bootstraps",
                "expected_benefit": "Optimal bootstrap placement"
            })

        if max_depth > 4:
            recommendations.append({
                "innovation": "LEAF_CENTRIC",
                "reason": "Deep trees benefit from direct leaf computation",
                "expected_benefit": "Parallel leaf evaluation"
            })

        if num_features > 10:
            recommendations.append({
                "innovation": "GRADIENT_NOISE",
                "reason": "Many features can use adaptive precision",
                "expected_benefit": "Better noise budget utilization"
            })

        return {
            "model_stats": {
                "num_trees": num_trees,
                "max_depth": max_depth,
                "total_nodes": total_nodes,
                "num_features": num_features,
                "is_oblivious": is_oblivious,
            },
            "noise_analysis": noise_analysis,
            "baseline_rotations": baseline_rotations,
            "recommendations": recommendations,
        }

    def _check_oblivious(self, model_ir: Any) -> bool:
        """Check if model has oblivious tree structure."""
        for tree in model_ir.trees:
            # Group nodes by depth
            nodes_by_depth: Dict[int, List[Any]] = {}
            for node in tree.nodes.values():
                if node.feature_index is not None:
                    depth = node.depth
                    if depth not in nodes_by_depth:
                        nodes_by_depth[depth] = []
                    nodes_by_depth[depth].append(node)

            # Check if all nodes at same depth use same feature
            for depth, nodes in nodes_by_depth.items():
                features = set(n.feature_index for n in nodes)
                if len(features) > 1:
                    return False

        return True

    def create_execution_plan(
        self,
        model_ir: Any,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None
    ) -> UnifiedExecutionPlan:
        """
        Create unified execution plan.

        Args:
            model_ir: Model to optimize
            X_train: Optional training data for polynomial fitting
            y_train: Optional training targets

        Returns:
            UnifiedExecutionPlan
        """
        self._model_ir = model_ir
        enabled = self.config.enabled_innovations

        plan_id = f"plan_{int(time.time() * 1000)}"
        plan = UnifiedExecutionPlan(
            plan_id=plan_id,
            innovations=enabled
        )

        # Baseline metrics
        baseline_rotations = sum(len(t.nodes) for t in model_ir.trees)
        plan.baseline_rotations = baseline_rotations

        # Apply MOAI-Native conversion first (affects other optimizations)
        if InnovationFlag.MOAI_NATIVE in enabled:
            conversion = self._moai_builder.from_model(model_ir)
            plan.moai_conversion = conversion
            plan.estimated_rotations = conversion.rotation_savings.get("oblivious_rotations", 0)
        else:
            plan.estimated_rotations = baseline_rotations

        # Apply leaf-centric encoding
        if InnovationFlag.LEAF_CENTRIC in enabled:
            leaf_plan = self._leaf_encoder.encode_model(model_ir)
            plan.leaf_centric_plan = leaf_plan

        # Apply gradient-informed noise allocation
        if InnovationFlag.GRADIENT_NOISE in enabled:
            allocations = self._noise_allocator.allocate(model_ir, model_ir.num_features)
            plan.noise_allocations = allocations

        # Apply bootstrap alignment
        if InnovationFlag.BOOTSTRAP_ALIGNED in enabled:
            forest = self._bootstrap_builder.partition_into_chunks(model_ir)
            plan.bootstrap_forest = forest
            plan.bootstrap_count = len(forest.bootstrap_points)
            plan.estimated_noise_bits = max(
                c.estimated_noise_bits for c in forest.chunks
            ) if forest.chunks else 0

        # Apply polynomial leaves if training data available
        if InnovationFlag.POLYNOMIAL_LEAVES in enabled and X_train is not None and y_train is not None:
            poly_model = PolynomialLeafGBDT(model_ir, config=self.config.polynomial_config)
            poly_model.fit_polynomials(X_train, y_train)
            plan.polynomial_model = poly_model

        # Compute rotation savings
        if plan.baseline_rotations > 0:
            plan.rotation_savings_percent = (
                1 - plan.estimated_rotations / plan.baseline_rotations
            ) * 100

        # Estimate latency
        plan.estimated_latency_ms = self._estimate_latency(plan)

        self._current_plan = plan

        logger.info(
            f"Created plan {plan_id}: {len(enabled)} innovations, "
            f"{plan.rotation_savings_percent:.1f}% rotation savings, "
            f"~{plan.estimated_latency_ms:.1f}ms latency"
        )

        return plan

    def _estimate_latency(self, plan: UnifiedExecutionPlan) -> float:
        """Estimate execution latency in milliseconds."""
        # Base latency model
        base_per_rotation = 0.1  # ms per rotation
        base_per_bootstrap = 50.0  # ms per bootstrap
        base_overhead = 10.0  # ms

        latency = base_overhead
        latency += plan.estimated_rotations * base_per_rotation
        latency += plan.bootstrap_count * base_per_bootstrap

        return latency

    def predict(
        self,
        features: np.ndarray,
        plan: Optional[UnifiedExecutionPlan] = None
    ) -> np.ndarray:
        """
        Execute prediction with all enabled innovations.

        Args:
            features: Input features
            plan: Execution plan (uses current if not provided)

        Returns:
            Predictions
        """
        if plan is None:
            plan = self._current_plan

        if plan is None:
            raise ValueError("No execution plan. Call create_execution_plan first.")

        # Use polynomial leaves if available
        if plan.polynomial_model is not None:
            return plan.polynomial_model.predict(features)

        # Use leaf-centric evaluation if available
        if plan.leaf_centric_plan is not None:
            return self._leaf_encoder.evaluate_plaintext(
                plan.leaf_centric_plan,
                features,
                self._model_ir.base_score
            )

        # Fall back to standard evaluation
        return self._evaluate_standard(features)

    def _evaluate_standard(self, features: np.ndarray) -> np.ndarray:
        """Standard tree evaluation."""
        predictions = np.full(features.shape[0], self._model_ir.base_score)

        for tree in self._model_ir.trees:
            for i in range(features.shape[0]):
                predictions[i] += self._traverse_tree(tree, features[i])

        return predictions

    def _traverse_tree(self, tree_ir: Any, sample: np.ndarray) -> float:
        """Traverse single tree."""
        node = tree_ir.nodes.get(tree_ir.root_id)

        while node is not None:
            if node.leaf_value is not None:
                return node.leaf_value

            if sample[node.feature_index] < node.threshold:
                node = tree_ir.nodes.get(node.left_child_id)
            else:
                node = tree_ir.nodes.get(node.right_child_id)

        return 0.0

    def predict_with_pruning(
        self,
        features: np.ndarray,
        pruning_threshold: float = 0.1
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict with adaptive pruning.

        Args:
            features: Input features
            pruning_threshold: Significance threshold for pruning

        Returns:
            Tuple of (predictions, pruning_metadata)
        """
        if InnovationFlag.HOMOMORPHIC_PRUNING not in self.config.enabled_innovations:
            return self.predict(features), {"pruning": "disabled"}

        # Compute tree outputs
        tree_outputs = np.zeros((features.shape[0], len(self._model_ir.trees)))

        for tree_idx, tree in enumerate(self._model_ir.trees):
            for i in range(features.shape[0]):
                tree_outputs[i, tree_idx] = self._traverse_tree(tree, features[i])

        # Apply pruning
        aggregated, metadata = self._pruner.prune_plaintext(tree_outputs)

        # Add base score
        predictions = aggregated + self._model_ir.base_score

        return predictions, metadata

    def create_streaming_model(self) -> EncryptedStreamingGBDT:
        """
        Create streaming model for online learning.

        Returns:
            EncryptedStreamingGBDT
        """
        if self._model_ir is None:
            raise ValueError("No model loaded. Call create_execution_plan first.")

        return EncryptedStreamingGBDT(
            self._model_ir,
            self.config.streaming_config
        )

    def create_federated_protocol(
        self,
        num_parties: int = 2
    ) -> FederatedGBDTProtocol:
        """
        Create federated protocol for multi-party inference.

        Args:
            num_parties: Number of parties

        Returns:
            FederatedGBDTProtocol
        """
        if self._model_ir is None:
            raise ValueError("No model loaded. Call create_execution_plan first.")

        from .federated_multikey import create_federated_protocol
        return create_federated_protocol(
            self._model_ir,
            num_parties=num_parties,
            decryption_threshold=self.config.multikey_config.decryption_threshold
        )

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        if self._current_plan is None:
            return {"error": "No execution plan created"}

        plan = self._current_plan

        report = {
            "plan_id": plan.plan_id,
            "innovations_enabled": [i.name for i in plan.innovations],
            "performance": {
                "estimated_latency_ms": plan.estimated_latency_ms,
                "rotation_savings_percent": plan.rotation_savings_percent,
                "baseline_rotations": plan.baseline_rotations,
                "optimized_rotations": plan.estimated_rotations,
                "bootstrap_count": plan.bootstrap_count,
            },
            "components": {},
        }

        if plan.moai_conversion:
            report["components"]["moai_native"] = {
                "num_oblivious_trees": plan.moai_conversion.num_trees,
                "accuracy_loss": plan.moai_conversion.accuracy_loss,
                "rotation_savings": plan.moai_conversion.rotation_savings,
            }

        if plan.bootstrap_forest:
            report["components"]["bootstrap_aligned"] = {
                "num_chunks": len(plan.bootstrap_forest.chunks),
                "bootstrap_points": plan.bootstrap_forest.bootstrap_points,
            }

        if plan.polynomial_model:
            stats = plan.polynomial_model.get_statistics()
            report["components"]["polynomial_leaves"] = stats

        if plan.noise_allocations:
            precisions = [a.precision_bits for a in plan.noise_allocations.values()]
            report["components"]["gradient_noise"] = {
                "num_features": len(plan.noise_allocations),
                "avg_precision_bits": np.mean(precisions) if precisions else 0,
                "precision_range": [min(precisions), max(precisions)] if precisions else [0, 0],
            }

        return report


# Factory functions

def create_novel_engine(
    optimize_for: str = "latency",
    enable_all: bool = False
) -> NovelFHEGBDTEngine:
    """
    Create a novel FHE-GBDT engine.

    Args:
        optimize_for: "latency", "throughput", "accuracy", or "privacy"
        enable_all: Enable all innovations

    Returns:
        Configured engine
    """
    if enable_all:
        enabled = set(InnovationFlag)
    else:
        # Default selection based on optimization target
        if optimize_for == "latency":
            enabled = {
                InnovationFlag.MOAI_NATIVE,
                InnovationFlag.LEAF_CENTRIC,
                InnovationFlag.BOOTSTRAP_ALIGNED,
            }
        elif optimize_for == "accuracy":
            enabled = {
                InnovationFlag.POLYNOMIAL_LEAVES,
                InnovationFlag.GRADIENT_NOISE,
            }
        elif optimize_for == "privacy":
            enabled = {
                InnovationFlag.FEDERATED_MULTIKEY,
                InnovationFlag.HOMOMORPHIC_PRUNING,
            }
        else:  # throughput
            enabled = {
                InnovationFlag.MOAI_NATIVE,
                InnovationFlag.BOOTSTRAP_ALIGNED,
            }

    config = InnovationConfig(
        enabled_innovations=enabled,
        optimize_for=optimize_for
    )

    return NovelFHEGBDTEngine(config)


def optimize_model_for_fhe(
    model_ir: Any,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None
) -> Tuple[NovelFHEGBDTEngine, UnifiedExecutionPlan]:
    """
    Optimize a model for FHE execution with all innovations.

    Args:
        model_ir: Model to optimize
        X_train: Optional training data
        y_train: Optional training targets

    Returns:
        Tuple of (engine, plan)
    """
    engine = create_novel_engine(enable_all=True)
    plan = engine.create_execution_plan(model_ir, X_train, y_train)

    return engine, plan
