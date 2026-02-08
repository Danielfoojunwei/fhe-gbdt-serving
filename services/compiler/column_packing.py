"""
MOAI Column Packing Implementation

Production-hardened implementation based on:
"MOAI: Module-Optimizing Architecture for Non-Interactive Secure
Transformer Inference" by Digital Trust Centre, NTU Singapore.
(IACR ePrint 2025/991, NDSS 2025)

Column packing replicates each feature value across ciphertext slots,
enabling rotation-free comparison operations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import logging

# Configure module logger
logger = logging.getLogger(__name__)

# Lazy numpy import for environments without it
_np = None

def _get_numpy():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


@dataclass
class ColumnPackedLayout:
    """
    Column-packed feature layout for rotation-free FHE operations.

    Traditional row packing:
        ct[0] = [f0, f1, f2, ..., f_n]  # All features in one ciphertext

    Column packing:
        ct[0] = [f0, f0, f0, ..., f0]   # Feature 0 replicated
        ct[1] = [f1, f1, f1, ..., f1]   # Feature 1 replicated

    Benefits:
    - Comparing f_i against threshold requires NO rotation
    - Batch multiple samples in the replication dimension
    - Consistent format across all tree levels
    """

    num_features: int
    slots_per_ciphertext: int
    num_ciphertexts: int = field(init=False)
    feature_to_ct_index: Dict[int, int] = field(default_factory=dict)

    # Limits for production safety
    MAX_FEATURES = 65536
    MAX_SLOTS = 32768
    MIN_SLOTS = 1

    def __post_init__(self):
        # Validate inputs
        if self.num_features <= 0:
            raise ValueError(f"num_features must be positive, got {self.num_features}")
        if self.num_features > self.MAX_FEATURES:
            raise ValueError(f"num_features exceeds max {self.MAX_FEATURES}")
        if self.slots_per_ciphertext < self.MIN_SLOTS:
            raise ValueError(f"slots_per_ciphertext must be >= {self.MIN_SLOTS}")
        if self.slots_per_ciphertext > self.MAX_SLOTS:
            raise ValueError(f"slots_per_ciphertext exceeds max {self.MAX_SLOTS}")

        self.num_ciphertexts = self.num_features
        self.feature_to_ct_index = {i: i for i in range(self.num_features)}

        logger.debug(f"ColumnPackedLayout: {self.num_features} features, "
                    f"{self.slots_per_ciphertext} slots/ct")

    def get_ciphertext_index(self, feature_id: int) -> int:
        """Get the ciphertext index for a given feature."""
        if feature_id < 0 or feature_id >= self.num_features:
            raise ValueError(f"feature_id {feature_id} out of range [0, {self.num_features})")
        return self.feature_to_ct_index.get(feature_id, feature_id)

    def pack_single_sample(self, features: List[float]) -> List[List[float]]:
        """
        Pack a single sample into column format.

        Args:
            features: List of feature values [f0, f1, ..., f_n]

        Returns:
            List of slot arrays, one per feature, each replicated

        Raises:
            ValueError: If features length doesn't match num_features
        """
        if len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {len(features)}")

        packed = []
        for i, val in enumerate(features):
            if not isinstance(val, (int, float)):
                raise TypeError(f"Feature {i} must be numeric, got {type(val)}")
            slots = [float(val)] * self.slots_per_ciphertext
            packed.append(slots)
        return packed

    def pack_batch(self, samples: Any) -> List[List[float]]:
        """
        Pack a batch of samples into column format.

        Args:
            samples: Shape (batch_size, num_features) - numpy array or list

        Returns:
            List of slot arrays, one per feature
        """
        np = _get_numpy()

        # Convert to numpy if needed
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples, dtype=np.float64)

        if samples.ndim != 2:
            raise ValueError(f"samples must be 2D, got {samples.ndim}D")

        batch_size, num_features = samples.shape

        if num_features != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {num_features}")

        if batch_size > self.slots_per_ciphertext:
            raise ValueError(
                f"Batch size {batch_size} exceeds slots {self.slots_per_ciphertext}")

        packed = []
        for feat_idx in range(num_features):
            feature_values = samples[:, feat_idx].tolist()

            # Pad to slots_per_ciphertext
            if len(feature_values) < self.slots_per_ciphertext:
                feature_values.extend(
                    [0.0] * (self.slots_per_ciphertext - len(feature_values)))

            packed.append(feature_values[:self.slots_per_ciphertext])

        return packed

    def unpack_batch(self, packed: List[List[float]], batch_size: int) -> Any:
        """
        Unpack column format back to samples.

        Args:
            packed: List of slot arrays from pack_batch
            batch_size: Original batch size

        Returns:
            Shape (batch_size, num_features)
        """
        np = _get_numpy()

        if not packed:
            raise ValueError("packed cannot be empty")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        num_features = len(packed)
        samples = np.zeros((batch_size, num_features), dtype=np.float64)

        for feat_idx, slots in enumerate(packed):
            if len(slots) < batch_size:
                raise ValueError(
                    f"Feature {feat_idx} has {len(slots)} slots, need {batch_size}")
            for sample_idx in range(batch_size):
                samples[sample_idx, feat_idx] = slots[sample_idx]

        return samples


@dataclass
class ColumnPackedThresholds:
    """
    Pre-encoded thresholds in column format for rotation-free comparison.
    """

    thresholds: Dict[Tuple[int, int, int], List[float]] = field(default_factory=dict)

    def add_threshold(
        self,
        tree_idx: int,
        depth: int,
        node_idx: int,
        threshold: float,
        slots: int
    ) -> None:
        """Add a replicated threshold for a tree node."""
        if slots <= 0:
            raise ValueError(f"slots must be positive, got {slots}")
        if not isinstance(threshold, (int, float)):
            raise TypeError(f"threshold must be numeric, got {type(threshold)}")

        key = (tree_idx, depth, node_idx)
        self.thresholds[key] = [float(threshold)] * slots

    def get_threshold(
        self,
        tree_idx: int,
        depth: int,
        node_idx: int
    ) -> Optional[List[float]]:
        """Get the replicated threshold for a tree node."""
        return self.thresholds.get((tree_idx, depth, node_idx))

    def __len__(self) -> int:
        return len(self.thresholds)


class ColumnPackingOptimizer:
    """
    MOAI-style optimizer that generates column-packed execution plans.

    Thread-safe and production-hardened.
    """

    DEFAULT_SLOTS = 2048
    MAX_SLOTS = 32768
    MIN_SLOTS = 1

    def __init__(self, slots_per_ciphertext: int = DEFAULT_SLOTS):
        if slots_per_ciphertext < self.MIN_SLOTS:
            raise ValueError(f"slots_per_ciphertext must be >= {self.MIN_SLOTS}")
        if slots_per_ciphertext > self.MAX_SLOTS:
            raise ValueError(f"slots_per_ciphertext exceeds max {self.MAX_SLOTS}")

        self.slots_per_ciphertext = slots_per_ciphertext
        logger.info(f"ColumnPackingOptimizer initialized: {slots_per_ciphertext} slots")

    def create_layout(self, num_features: int) -> ColumnPackedLayout:
        """Create a column-packed layout for the given number of features."""
        return ColumnPackedLayout(
            num_features=num_features,
            slots_per_ciphertext=self.slots_per_ciphertext
        )

    def prepare_thresholds(
        self,
        trees: List[dict],
        slots: Optional[int] = None
    ) -> ColumnPackedThresholds:
        """
        Pre-compute all thresholds in column format.

        Args:
            trees: List of tree structures with nodes
            slots: Number of slots (defaults to optimizer's slots_per_ciphertext)

        Returns:
            ColumnPackedThresholds with all thresholds replicated
        """
        if slots is None:
            slots = self.slots_per_ciphertext

        packed_thresholds = ColumnPackedThresholds()

        for tree_idx, tree in enumerate(trees):
            nodes = tree.get('nodes', {})
            if isinstance(nodes, dict):
                nodes = nodes.values()

            for node in nodes:
                threshold = node.get('threshold')
                if threshold is not None:
                    packed_thresholds.add_threshold(
                        tree_idx=tree_idx,
                        depth=node.get('depth', 0),
                        node_idx=node.get('node_id', 0),
                        threshold=threshold,
                        slots=slots
                    )

        logger.debug(f"Prepared {len(packed_thresholds)} thresholds")
        return packed_thresholds

    def compute_rotation_savings(
        self,
        num_trees: int,
        max_depth: int,
        num_features: int
    ) -> Dict[str, float]:
        """
        Compute rotation savings from column packing.

        Returns comparison between traditional and MOAI approach.
        """
        np = _get_numpy()

        if num_trees <= 0 or max_depth <= 0 or num_features <= 0:
            return {
                "traditional_rotations": 0,
                "moai_rotations": 0,
                "savings_percent": 0.0,
                "speedup_factor": 1.0
            }

        # Traditional: worst case 1 rotation per node
        traditional_rotations = num_trees * (2 ** max_depth - 1)

        # MOAI: only aggregation rotations
        moai_rotations = int(np.ceil(np.log2(max(num_trees, 1)))) if num_trees > 1 else 0

        savings_percent = (1 - moai_rotations / max(traditional_rotations, 1)) * 100
        speedup_factor = traditional_rotations / max(moai_rotations, 1)

        return {
            "traditional_rotations": traditional_rotations,
            "moai_rotations": moai_rotations,
            "savings_percent": round(savings_percent, 2),
            "speedup_factor": round(speedup_factor, 2)
        }


def create_column_packed_plan(
    num_features: int,
    num_trees: int,
    max_depth: int,
    slots: int = 2048
) -> Tuple[ColumnPackedLayout, Dict[str, float]]:
    """
    Create a column-packed execution plan.

    Args:
        num_features: Number of input features
        num_trees: Number of trees in the ensemble
        max_depth: Maximum tree depth
        slots: Slots per ciphertext (default 2048)

    Returns:
        Tuple of (layout, rotation_savings)

    Raises:
        ValueError: If inputs are invalid
    """
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    if num_trees <= 0:
        raise ValueError("num_trees must be positive")
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")

    optimizer = ColumnPackingOptimizer(slots_per_ciphertext=slots)
    layout = optimizer.create_layout(num_features)
    savings = optimizer.compute_rotation_savings(num_trees, max_depth, num_features)

    logger.info(f"Created MOAI plan: {num_features} features, {num_trees} trees, "
               f"depth {max_depth} -> {savings['savings_percent']:.1f}% rotation savings")

    return layout, savings
