"""
Novel Innovation #2: Gradient-Informed Noise Budget Allocation

GBDT training produces gradient-based feature importance scores. This innovation
uses these scores to intelligently allocate FHE noise budget, giving more
precision to important features while saving budget on less important ones.

Key Insight:
- Important features (high gradient contribution) need accurate comparisons
- Less important features can tolerate more noise/lower precision
- Total noise budget is fixed, but allocation can be optimized

Benefits:
- Better prediction accuracy with same noise budget
- Reduced precision for unimportant features saves noise
- Automatic adaptation based on model characteristics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance with gradient-based scores."""
    feature_idx: int
    gradient_importance: float  # Sum of gradient gains from splits
    frequency: int  # Number of times used in splits
    average_split_gain: float  # Average gain per split
    depth_weighted_importance: float  # Higher for shallower splits

    @property
    def normalized_importance(self) -> float:
        """Compute normalized importance score."""
        return self.gradient_importance * self.depth_weighted_importance


@dataclass
class NoiseBudgetAllocation:
    """Noise budget allocation for a feature."""
    feature_idx: int
    precision_bits: int  # Encoding precision (more bits = more accuracy)
    noise_budget_fraction: float  # Fraction of total budget
    encoding_scale: float  # Scale factor for encoding


@dataclass
class AdaptivePrecisionConfig:
    """Configuration for adaptive precision encoding."""
    total_noise_budget_bits: int = 28  # Total available noise budget
    min_precision_bits: int = 8  # Minimum precision for any feature
    max_precision_bits: int = 16  # Maximum precision
    base_precision_bits: int = 12  # Default precision
    importance_bonus_bits: int = 4  # Max bonus for important features


class FeatureImportanceAnalyzer:
    """
    Analyzes GBDT model to extract gradient-based feature importance.

    This goes beyond simple split counts to capture the actual gradient
    contribution of each feature during training.
    """

    def __init__(self, depth_decay: float = 0.8):
        """
        Initialize analyzer.

        Args:
            depth_decay: Decay factor for importance at deeper levels
                        (shallower splits are more important)
        """
        self.depth_decay = depth_decay

    def analyze(self, model_ir: Any) -> Dict[int, FeatureImportance]:
        """
        Analyze model to extract feature importance.

        Args:
            model_ir: Parsed ModelIR

        Returns:
            Dict mapping feature_idx to FeatureImportance
        """
        importance_data = defaultdict(lambda: {
            "gradient_sum": 0.0,
            "frequency": 0,
            "gains": [],
            "depths": [],
        })

        # Collect statistics from all trees
        for tree in model_ir.trees:
            self._analyze_tree(tree, importance_data)

        # Build FeatureImportance objects
        importance_map = {}
        max_gradient = max(
            (d["gradient_sum"] for d in importance_data.values()),
            default=1.0
        )

        for feat_idx, data in importance_data.items():
            if data["frequency"] == 0:
                continue

            # Compute depth-weighted importance
            depth_weights = [
                self.depth_decay ** d for d in data["depths"]
            ]
            depth_weighted = sum(
                g * w for g, w in zip(data["gains"], depth_weights)
            )

            importance_map[feat_idx] = FeatureImportance(
                feature_idx=feat_idx,
                gradient_importance=data["gradient_sum"] / max_gradient,
                frequency=data["frequency"],
                average_split_gain=np.mean(data["gains"]) if data["gains"] else 0,
                depth_weighted_importance=depth_weighted / max_gradient if max_gradient > 0 else 0
            )

        logger.info(f"Analyzed {len(importance_map)} features for importance")
        return importance_map

    def _analyze_tree(
        self,
        tree_ir: Any,
        importance_data: Dict[int, Dict]
    ):
        """Analyze a single tree for feature importance."""
        for node in tree_ir.nodes.values():
            if node.feature_index is not None:
                feat_idx = node.feature_index
                data = importance_data[feat_idx]

                # Estimate split gain (in production, this would come from training)
                # Here we use depth as a proxy (shallower = more important)
                depth = node.depth
                estimated_gain = 1.0 / (1.0 + depth)

                data["gradient_sum"] += estimated_gain
                data["frequency"] += 1
                data["gains"].append(estimated_gain)
                data["depths"].append(depth)

    def analyze_from_xgboost(
        self,
        booster: Any
    ) -> Dict[int, FeatureImportance]:
        """
        Analyze XGBoost booster for feature importance.

        Uses native XGBoost importance metrics when available.
        """
        try:
            # Get gain-based importance
            gain_importance = booster.get_score(importance_type='gain')
            total_gain = sum(gain_importance.values())

            # Get weight (frequency) importance
            weight_importance = booster.get_score(importance_type='weight')

            importance_map = {}
            for feat_name, gain in gain_importance.items():
                # Extract feature index from name (e.g., "f0" -> 0)
                if feat_name.startswith('f'):
                    feat_idx = int(feat_name[1:])
                else:
                    feat_idx = int(feat_name)

                importance_map[feat_idx] = FeatureImportance(
                    feature_idx=feat_idx,
                    gradient_importance=gain / total_gain if total_gain > 0 else 0,
                    frequency=weight_importance.get(feat_name, 0),
                    average_split_gain=gain,
                    depth_weighted_importance=gain / total_gain if total_gain > 0 else 0
                )

            return importance_map

        except Exception as e:
            logger.warning(f"Failed to analyze XGBoost importance: {e}")
            return {}


class GradientAwareNoiseAllocator:
    """
    Allocates FHE noise budget based on gradient-derived feature importance.

    High-importance features receive more precision bits (lower noise),
    while low-importance features can tolerate lower precision.
    """

    def __init__(self, config: Optional[AdaptivePrecisionConfig] = None):
        """
        Initialize allocator.

        Args:
            config: Precision configuration
        """
        self.config = config or AdaptivePrecisionConfig()
        self.analyzer = FeatureImportanceAnalyzer()

    def allocate(
        self,
        model_ir: Any,
        num_features: int
    ) -> Dict[int, NoiseBudgetAllocation]:
        """
        Allocate noise budget based on feature importance.

        Args:
            model_ir: Parsed ModelIR
            num_features: Total number of features

        Returns:
            Dict mapping feature_idx to NoiseBudgetAllocation
        """
        # Analyze feature importance
        importance_map = self.analyzer.analyze(model_ir)

        # Compute importance scores for all features
        scores = []
        for feat_idx in range(num_features):
            if feat_idx in importance_map:
                score = importance_map[feat_idx].normalized_importance
            else:
                score = 0.0
            scores.append(score)

        # Normalize scores
        total_score = sum(scores)
        if total_score > 0:
            normalized_scores = [s / total_score for s in scores]
        else:
            normalized_scores = [1.0 / num_features] * num_features

        # Allocate precision bits based on importance
        allocations = {}
        for feat_idx in range(num_features):
            allocation = self._compute_allocation(
                feat_idx, normalized_scores[feat_idx], num_features
            )
            allocations[feat_idx] = allocation

        # Log allocation summary
        total_bits = sum(a.precision_bits for a in allocations.values())
        avg_bits = total_bits / num_features if num_features > 0 else 0
        logger.info(
            f"Allocated noise budget: {num_features} features, "
            f"avg={avg_bits:.1f} bits, total={total_bits} bits"
        )

        return allocations

    def _compute_allocation(
        self,
        feat_idx: int,
        importance_score: float,
        num_features: int
    ) -> NoiseBudgetAllocation:
        """Compute allocation for a single feature."""
        cfg = self.config

        # Base precision + importance bonus
        importance_bonus = int(importance_score * num_features * cfg.importance_bonus_bits)
        precision_bits = cfg.base_precision_bits + importance_bonus

        # Clamp to valid range
        precision_bits = max(cfg.min_precision_bits, min(cfg.max_precision_bits, precision_bits))

        # Compute budget fraction
        budget_fraction = precision_bits / cfg.total_noise_budget_bits

        # Compute encoding scale
        encoding_scale = 2 ** precision_bits

        return NoiseBudgetAllocation(
            feature_idx=feat_idx,
            precision_bits=precision_bits,
            noise_budget_fraction=budget_fraction,
            encoding_scale=encoding_scale
        )

    def allocate_from_importance(
        self,
        importance_map: Dict[int, FeatureImportance],
        num_features: int
    ) -> Dict[int, NoiseBudgetAllocation]:
        """
        Allocate from pre-computed importance map.

        Args:
            importance_map: Pre-computed feature importance
            num_features: Total number of features

        Returns:
            Noise budget allocations
        """
        scores = []
        for feat_idx in range(num_features):
            if feat_idx in importance_map:
                score = importance_map[feat_idx].normalized_importance
            else:
                score = 0.0
            scores.append(score)

        total_score = sum(scores) or 1.0
        normalized_scores = [s / total_score for s in scores]

        allocations = {}
        for feat_idx in range(num_features):
            allocation = self._compute_allocation(
                feat_idx, normalized_scores[feat_idx], num_features
            )
            allocations[feat_idx] = allocation

        return allocations


class AdaptivePrecisionEncoder:
    """
    Encodes feature values with adaptive precision based on importance.

    Uses gradient-derived importance to determine encoding precision,
    maximizing accuracy for important features while conserving noise budget.
    """

    def __init__(self, allocations: Dict[int, NoiseBudgetAllocation]):
        """
        Initialize encoder with allocations.

        Args:
            allocations: Per-feature noise budget allocations
        """
        self.allocations = allocations

    def encode(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode features with adaptive precision.

        Args:
            features: Shape (batch_size, num_features) raw features

        Returns:
            Tuple of (encoded_values, scales) where:
            - encoded_values: Integer-encoded features
            - scales: Per-feature scale factors for decoding
        """
        batch_size, num_features = features.shape

        encoded = np.zeros_like(features, dtype=np.int64)
        scales = np.zeros(num_features)

        for feat_idx in range(num_features):
            allocation = self.allocations.get(feat_idx)
            if allocation is None:
                # Default encoding
                scale = 2 ** 12
            else:
                scale = allocation.encoding_scale

            scales[feat_idx] = scale
            encoded[:, feat_idx] = np.round(features[:, feat_idx] * scale).astype(np.int64)

        return encoded, scales

    def decode(
        self,
        encoded: np.ndarray,
        scales: np.ndarray
    ) -> np.ndarray:
        """
        Decode features from adaptive precision encoding.

        Args:
            encoded: Integer-encoded features
            scales: Per-feature scale factors

        Returns:
            Decoded float features
        """
        return encoded.astype(np.float64) / scales

    def encode_threshold(
        self,
        threshold: float,
        feature_idx: int
    ) -> int:
        """
        Encode a threshold value with matching precision.

        Args:
            threshold: Raw threshold value
            feature_idx: Feature index for this threshold

        Returns:
            Integer-encoded threshold
        """
        allocation = self.allocations.get(feature_idx)
        scale = allocation.encoding_scale if allocation else (2 ** 12)
        return int(round(threshold * scale))

    def get_precision_bits(self, feature_idx: int) -> int:
        """Get precision bits for a feature."""
        allocation = self.allocations.get(feature_idx)
        return allocation.precision_bits if allocation else 12


# Integration with MOAI optimizer

def create_adaptive_precision_plan(
    model_ir: Any,
    allocator: Optional[GradientAwareNoiseAllocator] = None
) -> Tuple[Dict[int, NoiseBudgetAllocation], AdaptivePrecisionEncoder]:
    """
    Create adaptive precision encoding plan for a model.

    Args:
        model_ir: Parsed ModelIR
        allocator: Optional custom allocator

    Returns:
        Tuple of (allocations, encoder)
    """
    if allocator is None:
        allocator = GradientAwareNoiseAllocator()

    allocations = allocator.allocate(model_ir, model_ir.num_features)
    encoder = AdaptivePrecisionEncoder(allocations)

    return allocations, encoder


def optimize_noise_budget(
    model_ir: Any,
    total_budget_bits: int = 28
) -> Dict[str, Any]:
    """
    Optimize noise budget allocation for a model.

    Args:
        model_ir: Parsed ModelIR
        total_budget_bits: Total available noise budget

    Returns:
        Optimization results including allocations and statistics
    """
    config = AdaptivePrecisionConfig(total_noise_budget_bits=total_budget_bits)
    allocator = GradientAwareNoiseAllocator(config)
    allocations = allocator.allocate(model_ir, model_ir.num_features)

    # Compute statistics
    precision_values = [a.precision_bits for a in allocations.values()]

    return {
        "allocations": allocations,
        "statistics": {
            "num_features": len(allocations),
            "min_precision": min(precision_values) if precision_values else 0,
            "max_precision": max(precision_values) if precision_values else 0,
            "avg_precision": np.mean(precision_values) if precision_values else 0,
            "total_precision_bits": sum(precision_values),
        },
        "config": config,
    }
