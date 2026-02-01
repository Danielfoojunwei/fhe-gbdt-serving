"""
MOAI Column Packing Implementation

Based on: "MOAI: Module-Optimizing Architecture for Non-Interactive Secure
Transformer Inference" by Digital Trust Centre, NTU Singapore.

Column packing replicates each feature value across ciphertext slots,
enabling rotation-free comparison operations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class ColumnPackedLayout:
    """
    Column-packed feature layout for rotation-free FHE operations.

    Traditional row packing:
        ct[0] = [f0, f1, f2, ..., f_n]  # All features in one ciphertext

    Column packing:
        ct[0] = [f0, f0, f0, ..., f0]   # Feature 0 replicated
        ct[1] = [f1, f1, f1, ..., f1]   # Feature 1 replicated
        ...

    Benefits:
    - Comparing f_i against threshold requires NO rotation
    - Batch multiple samples in the replication dimension
    - Consistent format across all tree levels
    """

    num_features: int
    slots_per_ciphertext: int  # Ring dimension / 2 for CKKS, or batch_size
    num_ciphertexts: int = field(init=False)
    feature_to_ct_index: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        # One ciphertext per feature (or group if features > slots)
        self.num_ciphertexts = self.num_features
        for i in range(self.num_features):
            self.feature_to_ct_index[i] = i

    def get_ciphertext_index(self, feature_id: int) -> int:
        """Get the ciphertext index for a given feature."""
        return self.feature_to_ct_index.get(feature_id, feature_id)

    def pack_single_sample(self, features: List[float]) -> List[List[float]]:
        """
        Pack a single sample into column format.

        Args:
            features: List of feature values [f0, f1, ..., f_n]

        Returns:
            List of slot arrays, one per feature, each replicated
        """
        packed = []
        for i, val in enumerate(features):
            # Replicate feature value across all slots
            slots = [val] * self.slots_per_ciphertext
            packed.append(slots)
        return packed

    def pack_batch(self, samples: np.ndarray) -> List[List[float]]:
        """
        Pack a batch of samples into column format.

        Args:
            samples: Shape (batch_size, num_features)

        Returns:
            List of slot arrays, one per feature
            Each array has samples interleaved: [s0_f0, s1_f0, s2_f0, ...]
        """
        batch_size, num_features = samples.shape
        packed = []

        for feat_idx in range(num_features):
            # Extract this feature from all samples
            feature_values = samples[:, feat_idx].tolist()

            # Pad to slots_per_ciphertext if needed
            if len(feature_values) < self.slots_per_ciphertext:
                feature_values.extend([0.0] * (self.slots_per_ciphertext - len(feature_values)))

            packed.append(feature_values[:self.slots_per_ciphertext])

        return packed

    def unpack_batch(self, packed: List[List[float]], batch_size: int) -> np.ndarray:
        """
        Unpack column format back to samples.

        Args:
            packed: List of slot arrays from pack_batch
            batch_size: Original batch size

        Returns:
            Shape (batch_size, num_features)
        """
        num_features = len(packed)
        samples = np.zeros((batch_size, num_features))

        for feat_idx, slots in enumerate(packed):
            for sample_idx in range(batch_size):
                samples[sample_idx, feat_idx] = slots[sample_idx]

        return samples


@dataclass
class ColumnPackedThresholds:
    """
    Pre-encoded thresholds in column format for rotation-free comparison.

    For each tree node, the threshold is replicated across slots to match
    the column-packed feature format.
    """

    thresholds: Dict[Tuple[int, int, int], List[float]] = field(default_factory=dict)
    # Key: (tree_idx, depth, node_idx), Value: replicated threshold

    def add_threshold(
        self,
        tree_idx: int,
        depth: int,
        node_idx: int,
        threshold: float,
        slots: int
    ):
        """Add a replicated threshold for a tree node."""
        key = (tree_idx, depth, node_idx)
        self.thresholds[key] = [threshold] * slots

    def get_threshold(
        self,
        tree_idx: int,
        depth: int,
        node_idx: int
    ) -> Optional[List[float]]:
        """Get the replicated threshold for a tree node."""
        key = (tree_idx, depth, node_idx)
        return self.thresholds.get(key)


class ColumnPackingOptimizer:
    """
    MOAI-style optimizer that generates column-packed execution plans.
    """

    def __init__(self, slots_per_ciphertext: int = 2048):
        self.slots_per_ciphertext = slots_per_ciphertext

    def create_layout(self, num_features: int) -> ColumnPackedLayout:
        """Create a column-packed layout for the given number of features."""
        return ColumnPackedLayout(
            num_features=num_features,
            slots_per_ciphertext=self.slots_per_ciphertext
        )

    def prepare_thresholds(
        self,
        trees: List[dict],
        slots: int
    ) -> ColumnPackedThresholds:
        """
        Pre-compute all thresholds in column format.

        Args:
            trees: List of tree structures with nodes
            slots: Number of slots per ciphertext

        Returns:
            ColumnPackedThresholds with all thresholds replicated
        """
        packed_thresholds = ColumnPackedThresholds()

        for tree_idx, tree in enumerate(trees):
            for node in tree.get('nodes', {}).values():
                if node.get('threshold') is not None:
                    packed_thresholds.add_threshold(
                        tree_idx=tree_idx,
                        depth=node.get('depth', 0),
                        node_idx=node.get('node_id', 0),
                        threshold=node['threshold'],
                        slots=slots
                    )

        return packed_thresholds

    def compute_rotation_savings(
        self,
        num_trees: int,
        max_depth: int,
        num_features: int
    ) -> Dict[str, int]:
        """
        Compute rotation savings from column packing.

        Returns comparison between traditional and MOAI approach.
        """
        # Traditional: Each comparison may need rotation
        # Worst case: every node needs rotation to access its feature
        traditional_rotations = num_trees * (2 ** max_depth - 1)

        # MOAI column packing: No rotations for feature access
        # Only rotations needed for tree aggregation
        moai_rotations = int(np.log2(num_trees)) + 1 if num_trees > 1 else 0

        return {
            "traditional_rotations": traditional_rotations,
            "moai_rotations": moai_rotations,
            "savings_percent": (1 - moai_rotations / max(traditional_rotations, 1)) * 100,
            "speedup_factor": traditional_rotations / max(moai_rotations, 1)
        }


# Convenience function for integration
def create_column_packed_plan(
    num_features: int,
    num_trees: int,
    max_depth: int,
    slots: int = 2048
) -> Tuple[ColumnPackedLayout, Dict[str, int]]:
    """
    Create a column-packed execution plan.

    Returns:
        Tuple of (layout, rotation_savings)
    """
    optimizer = ColumnPackingOptimizer(slots_per_ciphertext=slots)
    layout = optimizer.create_layout(num_features)
    savings = optimizer.compute_rotation_savings(num_trees, max_depth, num_features)

    return layout, savings
