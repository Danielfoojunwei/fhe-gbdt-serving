"""
Random Forest Optimizer for FHE

Generates FHE execution plans for Random Forest models.
Leverages innovations:
- MOAI Native (moai_native.py): Convert arbitrary RF trees to oblivious form
- Bootstrap Aligned (bootstrap_aligned.py): Noise budget for ensembles
- Column Packing: Rotation-free comparisons

Key differences from GBDT:
- Aggregation = MEAN (average of trees) instead of SUM
- Trees are independent (not boosted) → can parallelize
- Each tree votes equally → uniform noise budget across trees
"""

import hashlib
import logging
import math
from typing import Dict, List, Optional, Tuple, Any

from .ir import (
    ModelIR, ModelFamily, Aggregation, ObliviousPlanIR,
    ScheduleBlock, OpSequence, PackingLayout,
)

logger = logging.getLogger(__name__)

# Try to import innovations
try:
    from services.innovations.moai_native import (
        RotationOptimalConverter,
        ConversionConfig,
    )
    MOAI_NATIVE_AVAILABLE = True
except ImportError:
    MOAI_NATIVE_AVAILABLE = False

try:
    from services.innovations.bootstrap_aligned import BootstrapAwareTreeBuilder
    BOOTSTRAP_AVAILABLE = True
except ImportError:
    BOOTSTRAP_AVAILABLE = False


class RandomForestOptimizer:
    """
    Optimizer for Random Forest models under FHE.

    Generates ObliviousPlanIR with:
    - MOAI oblivious conversion of each tree (rotation-free)
    - MEAN aggregation instead of SUM
    - Uniform noise budget across trees (all equally important)
    """

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
        convert_to_oblivious: bool = True,
    ):
        self.profile = profile.lower()
        self.target = target.lower()
        self.convert_to_oblivious = convert_to_oblivious and MOAI_NATIVE_AVAILABLE
        self.batch_size = self.BATCH_SIZES.get(
            (self.target, self.profile), 256
        )

        logger.info(
            f"RandomForestOptimizer: profile={profile}, target={target}, "
            f"oblivious={self.convert_to_oblivious}"
        )

    def optimize(self, model: ModelIR) -> ObliviousPlanIR:
        """
        Generate FHE execution plan for a random forest.

        Args:
            model: Parsed ModelIR with model_family=RANDOM_FOREST

        Returns:
            ObliviousPlanIR execution plan
        """
        self._validate(model)

        num_trees = len(model.trees)

        # 1. Feature frequency analysis
        feature_counts = self._analyze_frequency(model)
        sorted_features = sorted(feature_counts, key=feature_counts.get, reverse=True)

        # 2. Create packing layout
        feature_map, layout = self._create_packing_layout(sorted_features, model.num_features)

        # 3. Optionally convert trees to oblivious form
        conversion_metadata = {}
        if self.convert_to_oblivious:
            converter = RotationOptimalConverter(ConversionConfig(
                feature_strategy="dominant",
                threshold_strategy="median",
            ))
            conversion_result = converter.convert_model(model)
            conversion_metadata = {
                "oblivious_conversion": True,
                "accuracy_loss": conversion_result.accuracy_loss,
                "rotation_savings": conversion_result.rotation_savings,
            }

        # 4. Build levelized schedule (same as GBDT but with MEAN agg)
        schedule = self._build_schedule(model, feature_map)

        # 5. Add MEAN aggregation op at the end
        schedule.append(ScheduleBlock(
            depth_level=-1,  # Post-tree aggregation
            node_group_id=0,
            ops=[OpSequence(
                op_type="AGGREGATE_MEAN",
                params={
                    "num_trees": num_trees,
                    "divisor": float(num_trees),
                },
            )],
        ))

        # Generate compiled model ID
        compiled_id = hashlib.sha256(
            f"rf:{model.num_features}:{num_trees}".encode()
        ).hexdigest()[:16]

        # Compute rotation savings
        rotation_savings = self._compute_rotation_savings(model)

        plan = ObliviousPlanIR(
            compiled_model_id=compiled_id,
            crypto_params_id="n2he_default",
            packing_layout=layout,
            schedule=schedule,
            base_score=0.0,
            num_trees=num_trees,
            metadata={
                "optimizer": "RandomForestOptimizer",
                "profile": self.profile,
                "target": self.target,
                "model_family": "random_forest",
                "aggregation": "mean",
                "column_packing": True,
                "rotation_savings": rotation_savings,
                **conversion_metadata,
            },
        )

        logger.info(
            f"Generated RF plan: {num_trees} trees, "
            f"{model.num_features} features, "
            f"MEAN aggregation, "
            f"{rotation_savings.get('savings_percent', 0):.1f}% rotation savings"
        )

        return plan

    def _validate(self, model: ModelIR) -> None:
        if not model.trees:
            raise ValueError("Random forest has no trees")
        if model.num_features <= 0:
            raise ValueError("Model has no features")
        if model.num_features > self.MAX_FEATURES:
            raise ValueError(f"Too many features: {model.num_features}")
        if len(model.trees) > self.MAX_TREES:
            raise ValueError(f"Too many trees: {len(model.trees)}")

    def _analyze_frequency(self, model: ModelIR) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for tree in model.trees:
            for node in tree.nodes.values():
                if node.feature_index is not None:
                    counts[node.feature_index] = counts.get(node.feature_index, 0) + 1
        return counts

    def _create_packing_layout(
        self, sorted_features: List[int], num_features: int
    ) -> Tuple[Dict[int, int], PackingLayout]:
        feature_map = {fid: idx for idx, fid in enumerate(sorted_features)}
        layout = PackingLayout(
            layout_type="moai_column",
            feature_to_ciphertext=feature_map,
            slots=self.batch_size,
        )
        return feature_map, layout

    def _build_schedule(
        self, model: ModelIR, feature_map: Dict[int, int]
    ) -> List[ScheduleBlock]:
        max_depth = max((t.max_depth for t in model.trees), default=0)
        schedule: List[ScheduleBlock] = []

        for depth in range(max_depth):
            feature_groups: Dict[int, List[Tuple[int, int, float]]] = {}

            for tree_idx, tree in enumerate(model.trees):
                nodes_at_depth = [
                    n for n in tree.nodes.values()
                    if n.depth == depth and n.feature_index is not None
                ]
                for node in nodes_at_depth:
                    feat_idx = node.feature_index
                    if feat_idx not in feature_groups:
                        feature_groups[feat_idx] = []
                    feature_groups[feat_idx].append(
                        (tree_idx, node.node_id, node.threshold)
                    )

            if feature_groups:
                ops = []
                for feat_idx, items in feature_groups.items():
                    ops.append(OpSequence(
                        op_type="COMPARE_BATCH",
                        params={
                            "feature_idx": feat_idx,
                            "thresholds": [t for _, _, t in items],
                            "tree_ids": [tid for tid, _, _ in items],
                            "size": len(items),
                            "rotation_free": True,
                        },
                    ))
                schedule.append(ScheduleBlock(
                    depth_level=depth,
                    node_group_id=0,
                    ops=ops,
                ))

        return schedule

    def _compute_rotation_savings(self, model: ModelIR) -> Dict[str, float]:
        num_trees = len(model.trees)
        total_nodes = sum(len(t.nodes) for t in model.trees)
        traditional_rotations = total_nodes
        moai_rotations = int(math.ceil(math.log2(max(num_trees, 1)))) if num_trees > 1 else 0
        savings_percent = (1 - moai_rotations / max(traditional_rotations, 1)) * 100

        return {
            "traditional_rotations": traditional_rotations,
            "moai_rotations": moai_rotations,
            "savings_percent": round(savings_percent, 2),
            "speedup_factor": round(traditional_rotations / max(moai_rotations, 1), 2),
        }
