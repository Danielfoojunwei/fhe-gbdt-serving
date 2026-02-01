from typing import List, Dict
from .ir import ModelIR, ObliviousPlanIR, ScheduleBlock, OpSequence, PackingLayout, TreeIR
import hashlib

class MOAIOptimizer:
    def __init__(self, profile: str = "latency", target: str = "cpu"):
        self.profile = profile
        self.target = target
        
        if self.target == "gpu":
            # GPU architecture favors massive data parallelism
            self.batch_size = 4096
        else:
            self.batch_size = 1 if profile == "latency" else 256

    def optimize(self, model: ModelIR) -> ObliviousPlanIR:
        # 1. Feature Frequency Analysis
        feature_counts = self._analyze_frequency(model)
        sorted_features = sorted(feature_counts, key=feature_counts.get, reverse=True)
        
        # 2. Define Packing Layout (Hot features -> Lower slots)
        feature_map = {}
        for idx, fid in enumerate(sorted_features):
            if idx < self.batch_size:
                feature_map[fid] = idx
            else:
                # Handle overflow or multi-ciphertext packing in future
                pass

        layout = PackingLayout(
            layout_type="frequency_sorted",
            feature_to_ciphertext=feature_map,
            slots=self.batch_size
        )

        schedule = []
        max_depth = max(tree.max_depth for tree in model.trees) if model.trees else 0

        # 3. Levelization with Node Grouping
        for depth in range(max_depth):
            ops = self._schedule_level(model, depth, feature_map)
            
            schedule.append(ScheduleBlock(
                depth_level=depth,
                node_group_id=0, 
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

    def _analyze_frequency(self, model: ModelIR) -> Dict[int, int]:
        counts = {}
        for tree in model.trees:
            for node in tree.nodes.values():
                if not node.default_left: # 'is_leaf' is not in IR, usually inferred from feature_index is None
                     # Actually check if feature_index is None for leaf
                    if node.feature_index is not None:
                        counts[node.feature_index] = counts.get(node.feature_index, 0) + 1
        return counts

    def _schedule_level(self, model: ModelIR, depth: int, feature_map: Dict[int, int]) -> List[str]:
        # Identify all active splits at this depth
        # Group them by required Rotation Offset
        rotation_groups = {} # offset -> list of (tree_id, node_id, threshold)

        for tree_idx, tree in enumerate(model.trees):
            # Find nodes at this depth
            nodes_at_depth = [n for n in tree.nodes.values() if n.depth == depth]
            
            for node in nodes_at_depth:
                if node.feature_index is not None:
                    f_slot = feature_map.get(node.feature_index, 0)
                    t_slot = tree_idx % self.batch_size # Simple tree->slot mapping
                    
                    offset = (f_slot - t_slot) % self.batch_size
                    
                    if offset not in rotation_groups:
                        rotation_groups[offset] = []
                    rotation_groups[offset].append((tree_idx, node.threshold))

        # Emit Ops: One Rotate per group
        ops = []
        for offset, items in rotation_groups.items():
            ops.append(f"ROTATE({offset})")
            ops.append(f"COMPARE_BATCH(size={len(items)})")
        
        return ops
