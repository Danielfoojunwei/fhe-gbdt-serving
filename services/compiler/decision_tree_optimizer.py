"""
Single Decision Tree Optimizer for FHE

Generates FHE execution plans for single decision trees.
Designed for regulatory models requiring:
- Reason codes (adverse action notices under ECOA/Reg B)
- Full interpretability (path tracing)
- Compliance audit trails

Leverages innovations:
- MOAI Native: Convert to oblivious form for rotation-free FHE
- Column Packing: Rotation-free comparisons

Key difference: Single tree → no ensemble aggregation, but adds
REASON_CODES operation to extract the decision path features
for adverse action explanations.
"""

import hashlib
import logging
import math
from typing import Dict, List, Optional, Tuple, Any

from .ir import (
    ModelIR, ModelFamily, ObliviousPlanIR,
    ScheduleBlock, OpSequence, PackingLayout,
)

logger = logging.getLogger(__name__)

try:
    from services.innovations.moai_native import (
        RotationOptimalConverter,
        ConversionConfig,
    )
    MOAI_NATIVE_AVAILABLE = True
except ImportError:
    MOAI_NATIVE_AVAILABLE = False


class DecisionTreeOptimizer:
    """
    Optimizer for single decision tree models under FHE.

    Generates ObliviousPlanIR with:
    - Single tree oblivious conversion (rotation-free)
    - REASON_CODES operation for adverse action
    - No ensemble aggregation
    """

    MAX_FEATURES = 65536
    MAX_DEPTH = 50

    def __init__(
        self,
        profile: str = "latency",
        target: str = "cpu",
        enable_reason_codes: bool = True,
    ):
        self.profile = profile.lower()
        self.target = target.lower()
        self.enable_reason_codes = enable_reason_codes

        logger.info(
            f"DecisionTreeOptimizer: profile={profile}, target={target}, "
            f"reason_codes={enable_reason_codes}"
        )

    def optimize(self, model: ModelIR) -> ObliviousPlanIR:
        """
        Generate FHE execution plan for a single decision tree.

        Args:
            model: Parsed ModelIR with model_family=SINGLE_TREE

        Returns:
            ObliviousPlanIR execution plan
        """
        self._validate(model)

        tree = model.trees[0]  # Single tree

        # 1. Feature analysis
        feature_counts = self._analyze_frequency(model)
        sorted_features = sorted(feature_counts, key=feature_counts.get, reverse=True)

        # 2. Packing layout
        feature_map = {fid: idx for idx, fid in enumerate(sorted_features)}
        layout = PackingLayout(
            layout_type="moai_column",
            feature_to_ciphertext=feature_map,
            slots=1,
        )

        # 3. Optionally convert to oblivious form
        conversion_metadata = {}
        if MOAI_NATIVE_AVAILABLE:
            converter = RotationOptimalConverter(ConversionConfig(
                feature_strategy="dominant",
                threshold_strategy="median",
            ))
            conversion_result = converter.convert_model(model)
            conversion_metadata = {
                "oblivious_conversion": True,
                "accuracy_loss": conversion_result.accuracy_loss,
            }

        # 4. Build schedule
        schedule = self._build_schedule(model, feature_map)

        # 5. Add reason codes operation (for adverse action notices)
        if self.enable_reason_codes:
            reason_features = self._extract_reason_code_features(tree, model.feature_names)
            schedule.append(ScheduleBlock(
                depth_level=-1,
                node_group_id=0,
                ops=[OpSequence(
                    op_type="REASON_CODES",
                    params={
                        "features_by_depth": reason_features,
                        "feature_names": model.feature_names or [],
                        "num_top_reasons": min(4, len(reason_features)),
                    },
                )],
            ))

        # 6. Generate plan ID
        compiled_id = hashlib.sha256(
            f"dt:{model.num_features}:{tree.max_depth}".encode()
        ).hexdigest()[:16]

        plan = ObliviousPlanIR(
            compiled_model_id=compiled_id,
            crypto_params_id="n2he_default",
            packing_layout=layout,
            schedule=schedule,
            base_score=0.0,
            num_trees=1,
            metadata={
                "optimizer": "DecisionTreeOptimizer",
                "profile": self.profile,
                "target": self.target,
                "model_family": "single_tree",
                "aggregation": "none",
                "reason_codes_enabled": self.enable_reason_codes,
                "supports_adverse_action": True,
                "max_depth": tree.max_depth,
                **conversion_metadata,
            },
        )

        logger.info(
            f"Generated DT plan: depth={tree.max_depth}, "
            f"{model.num_features} features, "
            f"reason_codes={'enabled' if self.enable_reason_codes else 'disabled'}"
        )

        return plan

    def _validate(self, model: ModelIR) -> None:
        if not model.trees:
            raise ValueError("Decision tree model has no trees")
        if len(model.trees) != 1:
            raise ValueError(
                f"Expected single tree, got {len(model.trees)}. "
                f"Use RandomForestOptimizer for ensembles."
            )
        if model.num_features <= 0:
            raise ValueError("Model has no features")
        if model.num_features > self.MAX_FEATURES:
            raise ValueError(f"Too many features: {model.num_features}")

        tree = model.trees[0]
        if tree.max_depth > self.MAX_DEPTH:
            raise ValueError(f"Tree depth {tree.max_depth} exceeds max {self.MAX_DEPTH}")

    def _analyze_frequency(self, model: ModelIR) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for tree in model.trees:
            for node in tree.nodes.values():
                if node.feature_index is not None:
                    counts[node.feature_index] = counts.get(node.feature_index, 0) + 1
        return counts

    def _build_schedule(
        self, model: ModelIR, feature_map: Dict[int, int]
    ) -> List[ScheduleBlock]:
        tree = model.trees[0]
        schedule: List[ScheduleBlock] = []

        for depth in range(tree.max_depth):
            nodes_at_depth = [
                n for n in tree.nodes.values()
                if n.depth == depth and n.feature_index is not None
            ]

            if nodes_at_depth:
                ops = []
                for node in nodes_at_depth:
                    ops.append(OpSequence(
                        op_type="COMPARE",
                        params={
                            "feature_idx": node.feature_index,
                            "threshold": node.threshold,
                            "node_id": node.node_id,
                            "rotation_free": True,
                        },
                    ))
                schedule.append(ScheduleBlock(
                    depth_level=depth,
                    node_group_id=0,
                    ops=ops,
                ))

        return schedule

    def _extract_reason_code_features(
        self,
        tree,
        feature_names: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Extract features used at each depth level for reason code generation.

        In adverse action notices (ECOA/Reg B), the lender must provide
        the top reasons for denial. The features used at shallow depths
        in the decision tree are the most impactful reasons.
        """
        reason_features = []
        for depth in range(tree.max_depth):
            nodes = [
                n for n in tree.nodes.values()
                if n.depth == depth and n.feature_index is not None
            ]
            for node in nodes:
                name = (
                    feature_names[node.feature_index]
                    if feature_names and node.feature_index < len(feature_names)
                    else f"feature_{node.feature_index}"
                )
                reason_features.append({
                    "depth": depth,
                    "feature_index": node.feature_index,
                    "feature_name": name,
                    "threshold": node.threshold,
                    "importance_rank": depth,  # Shallower = more important
                })

        return reason_features
