"""
MOAI Optimizer for FHE-GBDT

Production-hardened optimizer that generates execution plans using MOAI techniques:
- Column packing for rotation-free comparisons
- Levelized execution schedule
- Feature frequency analysis for optimal packing

Based on: "MOAI: Module-Optimizing Architecture for Non-Interactive Secure
Transformer Inference" by Digital Trust Centre, NTU Singapore.
(IACR ePrint 2025/991, NDSS 2025)
"""

from typing import List, Dict, Optional, Tuple
from .ir import ModelIR, ObliviousPlanIR, ScheduleBlock, OpSequence, PackingLayout
from .column_packing import ColumnPackingOptimizer, create_column_packed_plan
import hashlib
import logging
import warnings

logger = logging.getLogger(__name__)


class MOAIOptimizer:
    """
    MOAI-style optimizer for FHE-GBDT execution plans.

    Generates optimized execution plans with:
    - Column packing (rotation-free comparisons)
    - Levelized execution (all nodes at same depth together)
    - Feature frequency analysis (hot features in lower slots)
    """

    # Batch size presets by target
    BATCH_SIZES = {
        ("cpu", "latency"): 1,
        ("cpu", "throughput"): 256,
        ("gpu", "latency"): 2048,
        ("gpu", "throughput"): 4096,
    }

    MAX_FEATURES = 65536
    MAX_TREES = 10000
    MAX_DEPTH = 50

    def __init__(
        self,
        profile: str = "latency",
        target: str = "cpu",
        use_column_packing: bool = True
    ):
        """
        Initialize MOAI optimizer.

        Args:
            profile: "latency" or "throughput"
            target: "cpu" or "gpu"
            use_column_packing: Enable MOAI column packing (default True)
        """
        self.profile = profile.lower()
        self.target = target.lower()
        self.use_column_packing = use_column_packing

        if self.profile not in ("latency", "throughput"):
            raise ValueError(f"Invalid profile: {profile}")
        if self.target not in ("cpu", "gpu"):
            raise ValueError(f"Invalid target: {target}")

        self.batch_size = self.BATCH_SIZES.get(
            (self.target, self.profile),
            256  # Default fallback
        )

        logger.info(f"MOAIOptimizer: target={target}, profile={profile}, "
                   f"batch_size={self.batch_size}, column_packing={use_column_packing}")

    def optimize(self, model: ModelIR) -> ObliviousPlanIR:
        """
        Generate optimized execution plan for a model.

        Args:
            model: Parsed ModelIR from parser

        Returns:
            ObliviousPlanIR execution plan

        Raises:
            ValueError: If model is invalid
        """
        self._validate_model(model)

        # 1. Feature Frequency Analysis
        feature_counts = self._analyze_frequency(model)
        sorted_features = sorted(feature_counts, key=feature_counts.get, reverse=True)

        # 2. Create packing layout
        feature_map, layout = self._create_packing_layout(sorted_features, model.num_features)

        # 3. Build levelized schedule
        schedule = self._build_schedule(model, feature_map)

        # Generate compiled model ID
        compiled_id = hashlib.sha256(
            f"{model.model_type}:{model.num_features}:{len(model.trees)}".encode()
        ).hexdigest()[:16]

        # Compute rotation savings for metadata
        rotation_savings = self._compute_rotation_savings(model)

        plan = ObliviousPlanIR(
            compiled_model_id=compiled_id,
            crypto_params_id="n2he_default",
            packing_layout=layout,
            schedule=schedule,
            base_score=model.base_score,
            num_trees=len(model.trees),
            metadata={
                "optimizer": "MOAI",
                "profile": self.profile,
                "target": self.target,
                "column_packing": self.use_column_packing,
                "rotation_savings": rotation_savings,
            }
        )

        logger.info(f"Generated MOAI plan: {len(model.trees)} trees, "
                   f"{len(schedule)} levels, {rotation_savings['savings_percent']:.1f}% rotation savings")

        return plan

    def _validate_model(self, model: ModelIR) -> None:
        """Validate model constraints."""
        if not model.trees:
            raise ValueError("Model has no trees")
        if model.num_features <= 0:
            raise ValueError("Model has no features")
        if model.num_features > self.MAX_FEATURES:
            raise ValueError(f"Model has {model.num_features} features, exceeds max {self.MAX_FEATURES}")
        if len(model.trees) > self.MAX_TREES:
            raise ValueError(f"Model has {len(model.trees)} trees, exceeds max {self.MAX_TREES}")

        max_depth = max((t.max_depth for t in model.trees), default=0)
        if max_depth > self.MAX_DEPTH:
            raise ValueError(f"Model has depth {max_depth}, exceeds max {self.MAX_DEPTH}")

    def _analyze_frequency(self, model: ModelIR) -> Dict[int, int]:
        """
        Analyze feature usage frequency across all trees.

        Returns dict of feature_id -> usage_count
        """
        counts: Dict[int, int] = {}
        for tree in model.trees:
            for node in tree.nodes.values():
                # Check if this is a split node (not a leaf)
                if node.feature_index is not None:
                    counts[node.feature_index] = counts.get(node.feature_index, 0) + 1
        return counts

    def _create_packing_layout(
        self,
        sorted_features: List[int],
        num_features: int
    ) -> Tuple[Dict[int, int], PackingLayout]:
        """
        Create packing layout with hot features in lower slots.

        Returns (feature_map, PackingLayout)
        """
        feature_map: Dict[int, int] = {}
        overflow_features: List[int] = []

        for idx, fid in enumerate(sorted_features):
            if idx < self.batch_size:
                feature_map[fid] = idx
            else:
                overflow_features.append(fid)

        # Handle overflow with multi-ciphertext packing
        if overflow_features:
            warnings.warn(
                f"Model has {len(sorted_features)} features but batch_size={self.batch_size}. "
                f"{len(overflow_features)} features will use additional ciphertexts. "
                f"Consider GPU profile (batch_size=4096) for large feature sets.",
                RuntimeWarning
            )
            for i, fid in enumerate(overflow_features):
                ct_index = 1 + (i // self.batch_size)
                slot_index = i % self.batch_size
                feature_map[fid] = ct_index * self.batch_size + slot_index

        layout_type = "moai_column" if self.use_column_packing else "frequency_sorted"

        layout = PackingLayout(
            layout_type=layout_type,
            feature_to_ciphertext=feature_map,
            slots=self.batch_size
        )

        return feature_map, layout

    def _build_schedule(
        self,
        model: ModelIR,
        feature_map: Dict[int, int]
    ) -> List[ScheduleBlock]:
        """
        Build levelized execution schedule.

        Groups all nodes at the same depth together.
        """
        max_depth = max((t.max_depth for t in model.trees), default=0)
        schedule: List[ScheduleBlock] = []

        for depth in range(max_depth):
            ops = self._schedule_level(model, depth, feature_map)

            if ops:  # Only add non-empty levels
                schedule.append(ScheduleBlock(
                    depth_level=depth,
                    node_group_id=0,
                    ops=ops
                ))

        return schedule

    def _schedule_level(
        self,
        model: ModelIR,
        depth: int,
        feature_map: Dict[int, int]
    ) -> List[OpSequence]:
        """
        Schedule operations for a single depth level.

        With MOAI column packing, rotations are only needed for non-column-packed
        access. With column packing enabled, we emit 0-offset rotations.
        """
        # Group nodes by feature for efficient batching
        feature_groups: Dict[int, List[Tuple[int, int, float]]] = {}

        for tree_idx, tree in enumerate(model.trees):
            nodes_at_depth = [n for n in tree.nodes.values() if n.depth == depth]

            for node in nodes_at_depth:
                if node.feature_index is not None and node.threshold is not None:
                    feat_idx = node.feature_index
                    if feat_idx not in feature_groups:
                        feature_groups[feat_idx] = []
                    feature_groups[feat_idx].append(
                        (tree_idx, node.node_id, node.threshold)
                    )

        ops: List[OpSequence] = []

        if self.use_column_packing:
            # MOAI: No rotations needed for feature access
            for feat_idx, items in feature_groups.items():
                thresholds = [thresh for _, _, thresh in items]
                tree_ids = [tid for tid, _, _ in items]

                ops.append(OpSequence(
                    op_type="COMPARE_BATCH",
                    params={
                        "feature_idx": feat_idx,
                        "thresholds": thresholds,
                        "tree_ids": tree_ids,
                        "size": len(items),
                        "rotation_free": True
                    }
                ))
        else:
            # Traditional: Group by rotation offset
            rotation_groups: Dict[int, List[Tuple[int, float]]] = {}

            for feat_idx, items in feature_groups.items():
                f_slot = feature_map.get(feat_idx, 0)

                for tree_idx, node_id, threshold in items:
                    t_slot = tree_idx % self.batch_size
                    offset = (f_slot - t_slot) % self.batch_size

                    if offset not in rotation_groups:
                        rotation_groups[offset] = []
                    rotation_groups[offset].append((tree_idx, threshold))

            for offset, items in rotation_groups.items():
                if offset != 0:
                    ops.append(OpSequence(
                        op_type="ROTATE",
                        params={"offset": offset}
                    ))

                ops.append(OpSequence(
                    op_type="COMPARE_BATCH",
                    params={
                        "size": len(items),
                        "thresholds": [t for _, t in items],
                        "tree_ids": [tid for tid, _ in items],
                        "rotation_free": False
                    }
                ))

        return ops

    def _compute_rotation_savings(self, model: ModelIR) -> Dict[str, float]:
        """Compute rotation savings from MOAI optimization."""
        num_trees = len(model.trees)
        max_depth = max((t.max_depth for t in model.trees), default=0)

        if not self.use_column_packing:
            return {
                "traditional_rotations": 0,
                "moai_rotations": 0,
                "savings_percent": 0.0,
                "speedup_factor": 1.0
            }

        # Traditional: worst case 1 rotation per node
        total_nodes = sum(len(t.nodes) for t in model.trees)
        traditional_rotations = total_nodes

        # MOAI: only aggregation rotations (log2(num_trees))
        import math
        moai_rotations = int(math.ceil(math.log2(max(num_trees, 1)))) if num_trees > 1 else 0

        savings_percent = (1 - moai_rotations / max(traditional_rotations, 1)) * 100
        speedup_factor = traditional_rotations / max(moai_rotations, 1)

        return {
            "traditional_rotations": traditional_rotations,
            "moai_rotations": moai_rotations,
            "savings_percent": round(savings_percent, 2),
            "speedup_factor": round(speedup_factor, 2)
        }


def optimize_model(
    model: ModelIR,
    profile: str = "latency",
    target: str = "cpu",
    use_column_packing: bool = True
) -> ObliviousPlanIR:
    """
    Convenience function to optimize a model.

    Args:
        model: Parsed ModelIR
        profile: "latency" or "throughput"
        target: "cpu" or "gpu"
        use_column_packing: Enable MOAI column packing

    Returns:
        Optimized execution plan
    """
    optimizer = MOAIOptimizer(
        profile=profile,
        target=target,
        use_column_packing=use_column_packing
    )
    return optimizer.optimize(model)
