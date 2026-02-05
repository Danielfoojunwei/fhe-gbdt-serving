"""
Novel Innovation #1: Leaf-Centric Encoding

Instead of path-centric tree traversal, this approach computes leaf indicators
directly using tensor products of sign functions.

Mathematical Foundation:
A GBDT prediction is: f(x) = Σᵢ Σⱼ wᵢⱼ · 𝟙[x ∈ Rᵢⱼ]
where 𝟙[x ∈ Rᵢⱼ] is an indicator for leaf region Rᵢⱼ.

For oblivious trees (CatBoost), each leaf indicator is a product of d sign functions:
  Leaf k = Π_{d=0}^{D-1} sign_d(f_{d} - t_{d}) ^ bit(k, d)

Benefits with N2HE:
- N2HE's weighted sum primitive aggregates all leaves in O(1)
- Parallel computation of all 2^d leaf indicators
- Perfect fit for MOAI column packing (rotation-free)
- Tensor product structure enables massive parallelism
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LeafIndicator:
    """Representation of a leaf indicator computation."""
    tree_id: int
    leaf_id: int
    leaf_value: float
    # Sign conditions: list of (feature_idx, threshold, negate)
    conditions: List[Tuple[int, float, bool]] = field(default_factory=list)

    def to_binary_path(self) -> int:
        """Convert conditions to binary path encoding."""
        path = 0
        for i, (_, _, negate) in enumerate(self.conditions):
            if not negate:  # Right branch = 1
                path |= (1 << i)
        return path


@dataclass
class DirectLeafPlan:
    """
    Execution plan for direct leaf indicator computation.

    Instead of levelized traversal, computes all leaf indicators in parallel
    then aggregates using N2HE weighted sum.
    """
    num_trees: int
    max_depth: int
    total_leaves: int

    # Per-tree leaf indicators
    leaf_indicators: List[List[LeafIndicator]] = field(default_factory=list)

    # Feature access pattern for column packing
    feature_access_per_level: List[List[int]] = field(default_factory=list)

    # Precomputed tensor product patterns
    tensor_patterns: Optional[np.ndarray] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class LeafIndicatorComputer:
    """
    Computes leaf indicators directly using tensor products.

    Key Insight:
    For an oblivious tree of depth D with levels using features f_0, ..., f_{D-1}
    and thresholds t_0, ..., t_{D-1}:

    sign_d = polynomial_sign(f_d - t_d)  ∈ {-1, +1} → mapped to {0, 1}

    Leaf indicator for leaf k (binary encoding):
        I_k = Π_{d=0}^{D-1} [(sign_d if bit(k,d)==1 else (1-sign_d))]

    This is equivalent to a tensor product across all levels!
    """

    # Chebyshev coefficients for sign approximation
    SIGN_POLY_CHEBYSHEV = [0.0, 1.1963, 0.0, -0.2849, 0.0, 0.0951, 0.0]

    # Minimax coefficients (higher accuracy)
    SIGN_POLY_MINIMAX = [0.0, 1.5708, 0.0, -0.6460, 0.0, 0.0796, 0.0]

    def __init__(self, poly_type: str = "minimax", max_depth: int = 10):
        """
        Initialize leaf indicator computer.

        Args:
            poly_type: "chebyshev" or "minimax" for sign approximation
            max_depth: Maximum supported tree depth
        """
        self.poly_coeffs = (
            self.SIGN_POLY_MINIMAX if poly_type == "minimax"
            else self.SIGN_POLY_CHEBYSHEV
        )
        self.max_depth = max_depth

        # Precompute binary patterns for all leaf indices up to 2^max_depth
        self._precompute_patterns()

        logger.info(f"LeafIndicatorComputer: poly={poly_type}, max_depth={max_depth}")

    def _precompute_patterns(self):
        """Precompute tensor product patterns for all leaf indices."""
        max_leaves = 2 ** self.max_depth
        self.patterns = np.zeros((max_leaves, self.max_depth), dtype=np.int8)

        for leaf_idx in range(max_leaves):
            for d in range(self.max_depth):
                # Extract bit d from leaf_idx
                self.patterns[leaf_idx, d] = (leaf_idx >> d) & 1

    def polynomial_sign(self, x: np.ndarray) -> np.ndarray:
        """
        Compute polynomial approximation of sign function.

        sign(x) ≈ x · (c_1 + c_3·x² + c_5·x⁴ + ...)

        Maps to [0, 1] range for FHE: (sign(x) + 1) / 2
        """
        result = np.zeros_like(x)
        x_power = x.copy()

        for i, coeff in enumerate(self.poly_coeffs):
            if coeff != 0:
                result += coeff * x_power
            x_power = x_power * (x ** 2) if i % 2 == 0 else x_power

        # Map from [-1, 1] to [0, 1]
        return (result + 1) / 2

    def compute_level_signs(
        self,
        features: np.ndarray,
        level_thresholds: List[Tuple[int, float]]
    ) -> np.ndarray:
        """
        Compute sign functions for all levels.

        Args:
            features: Shape (batch_size, num_features)
            level_thresholds: List of (feature_idx, threshold) per level

        Returns:
            Shape (batch_size, depth) array of sign values in [0, 1]
        """
        batch_size = features.shape[0]
        depth = len(level_thresholds)

        signs = np.zeros((batch_size, depth))

        for d, (feat_idx, threshold) in enumerate(level_thresholds):
            delta = features[:, feat_idx] - threshold
            # Normalize to [-1, 1] range for stable polynomial
            delta_norm = np.clip(delta / (np.abs(delta).max() + 1e-8), -1, 1)
            signs[:, d] = self.polynomial_sign(delta_norm)

        return signs

    def compute_leaf_indicators(
        self,
        signs: np.ndarray,
        num_leaves: int
    ) -> np.ndarray:
        """
        Compute all leaf indicators using tensor product.

        Args:
            signs: Shape (batch_size, depth) sign values in [0, 1]
            num_leaves: Number of leaves (2^depth)

        Returns:
            Shape (batch_size, num_leaves) leaf indicator values
        """
        batch_size, depth = signs.shape

        if num_leaves > 2 ** depth:
            raise ValueError(f"num_leaves {num_leaves} exceeds 2^{depth}")

        indicators = np.zeros((batch_size, num_leaves))

        for leaf_idx in range(num_leaves):
            # Compute product for this leaf
            indicator = np.ones(batch_size)

            for d in range(depth):
                bit = (leaf_idx >> d) & 1
                if bit == 1:
                    indicator *= signs[:, d]
                else:
                    indicator *= (1 - signs[:, d])

            indicators[:, leaf_idx] = indicator

        return indicators

    def compute_leaf_indicators_tensor(
        self,
        signs: np.ndarray,
        depth: int
    ) -> np.ndarray:
        """
        Compute leaf indicators using efficient tensor product.

        This is the key innovation: instead of looping over leaves,
        use tensor operations that map naturally to FHE SIMD.

        Args:
            signs: Shape (batch_size, depth) sign values in [0, 1]
            depth: Tree depth

        Returns:
            Shape (batch_size, 2^depth) leaf indicators
        """
        batch_size = signs.shape[0]
        num_leaves = 2 ** depth

        # Create complementary signs: [s_0, 1-s_0, s_1, 1-s_1, ...]
        # Shape: (batch_size, depth, 2)
        sign_pairs = np.stack([1 - signs, signs], axis=-1)

        # Tensor product across all levels
        # This creates all 2^depth combinations
        result = sign_pairs[:, 0, :]  # Start with first level

        for d in range(1, depth):
            # Outer product with next level
            result = np.einsum('bi,bj->bij', result, sign_pairs[:, d, :])
            result = result.reshape(batch_size, -1)

        return result[:, :num_leaves]


class LeafCentricEncoder:
    """
    Encodes GBDT models for leaf-centric FHE evaluation.

    This encoder transforms standard GBDT models (XGBoost, LightGBM, CatBoost)
    into a leaf-centric format optimized for N2HE weighted sum aggregation.
    """

    def __init__(
        self,
        indicator_computer: Optional[LeafIndicatorComputer] = None
    ):
        """
        Initialize encoder.

        Args:
            indicator_computer: LeafIndicatorComputer instance
        """
        self.computer = indicator_computer or LeafIndicatorComputer()

    def encode_model(self, model_ir: Any) -> DirectLeafPlan:
        """
        Encode a ModelIR into a DirectLeafPlan.

        Args:
            model_ir: Parsed ModelIR from compiler

        Returns:
            DirectLeafPlan for leaf-centric evaluation
        """
        num_trees = len(model_ir.trees)
        max_depth = max(t.max_depth for t in model_ir.trees)

        all_leaf_indicators = []
        feature_access_per_level = [[] for _ in range(max_depth)]

        for tree_idx, tree in enumerate(model_ir.trees):
            tree_indicators = self._encode_tree(
                tree, tree_idx, feature_access_per_level
            )
            all_leaf_indicators.append(tree_indicators)

        total_leaves = sum(len(indicators) for indicators in all_leaf_indicators)

        # Compute tensor patterns for efficient evaluation
        tensor_patterns = self._compute_tensor_patterns(max_depth)

        plan = DirectLeafPlan(
            num_trees=num_trees,
            max_depth=max_depth,
            total_leaves=total_leaves,
            leaf_indicators=all_leaf_indicators,
            feature_access_per_level=feature_access_per_level,
            tensor_patterns=tensor_patterns,
            metadata={
                "model_type": model_ir.model_type,
                "num_features": model_ir.num_features,
                "base_score": model_ir.base_score,
                "encoding": "leaf_centric",
            }
        )

        logger.info(
            f"Encoded model: {num_trees} trees, {total_leaves} total leaves, "
            f"max_depth={max_depth}"
        )

        return plan

    def _encode_tree(
        self,
        tree_ir: Any,
        tree_idx: int,
        feature_access: List[List[int]]
    ) -> List[LeafIndicator]:
        """Encode a single tree into leaf indicators."""
        indicators = []

        # Collect all leaf nodes
        leaves = [
            node for node in tree_ir.nodes.values()
            if node.leaf_value is not None
        ]

        for leaf in leaves:
            # Build path conditions from root to this leaf
            conditions = self._build_path_conditions(tree_ir, leaf)

            # Track feature access per level
            for depth, (feat_idx, _, _) in enumerate(conditions):
                if depth < len(feature_access) and feat_idx not in feature_access[depth]:
                    feature_access[depth].append(feat_idx)

            indicator = LeafIndicator(
                tree_id=tree_idx,
                leaf_id=leaf.node_id,
                leaf_value=leaf.leaf_value,
                conditions=conditions
            )
            indicators.append(indicator)

        return indicators

    def _build_path_conditions(
        self,
        tree_ir: Any,
        leaf_node: Any
    ) -> List[Tuple[int, float, bool]]:
        """
        Build path conditions from root to leaf.

        Returns list of (feature_idx, threshold, is_left_branch) tuples.
        """
        conditions = []

        # Find path from root to leaf
        target_id = leaf_node.node_id
        path = self._find_path(tree_ir, tree_ir.root_id, target_id)

        if path is None:
            return conditions

        # Convert path to conditions
        for i in range(len(path) - 1):
            current_id = path[i]
            next_id = path[i + 1]
            current_node = tree_ir.nodes[current_id]

            if current_node.feature_index is not None:
                is_left = (next_id == current_node.left_child_id)
                conditions.append((
                    current_node.feature_index,
                    current_node.threshold,
                    is_left  # negate=True for left branch
                ))

        return conditions

    def _find_path(
        self,
        tree_ir: Any,
        current_id: int,
        target_id: int
    ) -> Optional[List[int]]:
        """Find path from current node to target node."""
        if current_id == target_id:
            return [current_id]

        node = tree_ir.nodes.get(current_id)
        if node is None or node.leaf_value is not None:
            return None

        # Try left subtree
        if node.left_child_id is not None:
            left_path = self._find_path(tree_ir, node.left_child_id, target_id)
            if left_path is not None:
                return [current_id] + left_path

        # Try right subtree
        if node.right_child_id is not None:
            right_path = self._find_path(tree_ir, node.right_child_id, target_id)
            if right_path is not None:
                return [current_id] + right_path

        return None

    def _compute_tensor_patterns(self, max_depth: int) -> np.ndarray:
        """Precompute tensor product patterns for efficient GPU execution."""
        num_leaves = 2 ** max_depth
        patterns = np.zeros((num_leaves, max_depth), dtype=np.float32)

        for leaf_idx in range(num_leaves):
            for d in range(max_depth):
                # 1 if bit d is set, -1 otherwise
                patterns[leaf_idx, d] = 1.0 if ((leaf_idx >> d) & 1) else -1.0

        return patterns

    def evaluate_plaintext(
        self,
        plan: DirectLeafPlan,
        features: np.ndarray,
        base_score: float = 0.0
    ) -> np.ndarray:
        """
        Evaluate plan in plaintext (for validation).

        Args:
            plan: DirectLeafPlan from encode_model
            features: Shape (batch_size, num_features)
            base_score: Base prediction score

        Returns:
            Shape (batch_size,) predictions
        """
        batch_size = features.shape[0]
        predictions = np.full(batch_size, base_score)

        for tree_indicators in plan.leaf_indicators:
            # Build level thresholds for this tree
            if not tree_indicators:
                continue

            # Get conditions from first leaf (all leaves share same structure for oblivious)
            first_indicator = tree_indicators[0]
            level_thresholds = [
                (cond[0], cond[1]) for cond in first_indicator.conditions
            ]

            if not level_thresholds:
                # Single-leaf tree
                predictions += tree_indicators[0].leaf_value
                continue

            # Compute level signs
            signs = self.computer.compute_level_signs(features, level_thresholds)

            # Compute leaf indicators using tensor product
            depth = len(level_thresholds)
            num_leaves = len(tree_indicators)
            indicators = self.computer.compute_leaf_indicators_tensor(signs, depth)

            # Weighted sum of leaf values
            leaf_values = np.array([ind.leaf_value for ind in tree_indicators])
            tree_output = np.dot(indicators[:, :num_leaves], leaf_values[:num_leaves])

            predictions += tree_output

        return predictions


# Convenience functions

def create_leaf_centric_plan(model_ir: Any) -> DirectLeafPlan:
    """
    Create a leaf-centric execution plan from a ModelIR.

    Args:
        model_ir: Parsed ModelIR

    Returns:
        DirectLeafPlan for leaf-centric evaluation
    """
    encoder = LeafCentricEncoder()
    return encoder.encode_model(model_ir)


def evaluate_leaf_centric(
    plan: DirectLeafPlan,
    features: np.ndarray,
    base_score: float = 0.0
) -> np.ndarray:
    """
    Evaluate a leaf-centric plan on features.

    Args:
        plan: DirectLeafPlan
        features: Input features
        base_score: Base prediction score

    Returns:
        Predictions
    """
    encoder = LeafCentricEncoder()
    return encoder.evaluate_plaintext(plan, features, base_score)
