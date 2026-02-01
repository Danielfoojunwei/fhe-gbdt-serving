from typing import List, Dict
from .ir import ModelIR, ObliviousPlanIR, ScheduleBlock, OpSequence, PackingLayout, TreeIR
import hashlib

class MOAIOptimizer:
    def __init__(self, profile: str = "latency"):
        self.profile = profile
        self.batch_size = 1 if profile == "latency" else 256

    def optimize(self, model: ModelIR) -> ObliviousPlanIR:
        # 1. Define Packing Layout
        # For feature-major packing, each feature is in a separate ciphertext (or packed if slots allow)
        # For now, let's assume 1 feature per ciphertext for simplicity
        layout = PackingLayout(
            layout_type="feature_major",
            feature_to_ciphertext={i: i for i in range(model.num_features)},
            slots=self.batch_size
        )

        schedule = []
        max_depth = max(tree.max_depth for tree in model.trees) if model.trees else 0

        # 2. Levelization
        for depth in range(max_depth):
            # Group nodes from all trees at this depth
            # In direct evaluation, we need to compare X[f] - t for all active nodes
            
            # Simplified: Collect all unique (feature, threshold) at this depth
            # In a real FHE-GBDT, we'd batch these comparisons
            ops = []
            
            # Add DELTA ops (X[f] - t)
            # Add STEP ops (activation)
            # Add ROUTE ops (weight update)
            
            # This is where the MOAI optimization would happen: 
            # sorting features to minimize rotations or reloading.
            
            schedule.append(ScheduleBlock(
                depth_level=depth,
                node_group_id=0, # Simplified
                ops=ops
            ))

        compiled_id = hashlib.sha256(str(model).encode()).hexdigest()[:16]

        return ObliviousPlanIR(
            compiled_model_id=compiled_id,
            crypto_params_id="n2he_default",
            packing_layout=layout,
            schedule=schedule,
            base_score=model.base_score,
            num_trees=len(model.trees)
        )
