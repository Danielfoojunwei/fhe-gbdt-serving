import json
from typing import Dict, List
from .ir import TreeIR, TreeNode, ModelIR

class BaseParser:
    def parse(self, content: bytes) -> ModelIR:
        raise NotImplementedError

class XGBoostParser(BaseParser):
    def parse(self, content: bytes) -> ModelIR:
        data = json.loads(content.decode('utf-8'))
        
        # XGBoost models can be in different formats; we expect the 'learner' structure
        # typically seen in models saved via `save_model('model.json')`
        if 'learner' not in data:
            raise ValueError("Invalid XGBoost JSON: 'learner' key not found")
            
        booster = data['learner']['gradient_booster']
        model = booster['model']
        trees_json = model['trees']
        
        model_ir_trees = []
        max_feature_id = 0
        
        for tree_id, tree_data in enumerate(trees_json):
            # XGBoost tree nodes are typically flat lists in 'left_children', 'right_children', etc.
            lefts = tree_data['left_children']
            rights = tree_data['right_children']
            indices = tree_data['split_indices']
            conditions = tree_data['split_conditions']
            leaf_values = tree_data['base_weights'] # or 'leaf_values' depending on version
            
            nodes = {}
            
            def build_node(node_idx: int, depth: int) -> TreeNode:
                nonlocal max_feature_id
                if lefts[node_idx] == -1: # Leaf node
                    return TreeNode(
                        node_id=node_idx,
                        feature_index=-1,
                        threshold=0.0,
                        left_child=None,
                        right_child=None,
                        is_leaf=True,
                        leaf_value=leaf_values[node_idx],  # Use base_weights for leaf value
                        depth=depth
                    )

                
                feat_idx = indices[node_idx]
                max_feature_id = max(max_feature_id, feat_idx)
                
                left_node = build_node(lefts[node_idx], depth + 1)
                right_node = build_node(rights[node_idx], depth + 1)
                
                return TreeNode(
                    node_id=node_idx,
                    feature_index=feat_idx,
                    threshold=conditions[node_idx],
                    left_child=left_node,
                    right_child=right_node,
                    is_leaf=False,
                    leaf_value=0.0,
                    depth=depth
                )

            root = build_node(0, 0)
            model_ir_trees.append(TreeIR(tree_id=tree_id, root=root))
            
        return ModelIR(model_type="xgboost", trees=model_ir_trees, num_features=max_feature_id + 1)

class LightGBMParser(BaseParser):
    def parse(self, content: bytes) -> ModelIR:
        data = json.loads(content.decode('utf-8'))
        
        # LightGBM JSON dump (via model.to_dict() or dump_model())
        if 'tree_info' not in data:
            raise ValueError("Invalid LightGBM JSON: 'tree_info' not found")
            
        trees_json = data['tree_info']
        model_ir_trees = []
        max_feature_id = 0
        
        for tree_id, tree_data in enumerate(trees_json):
            def build_node(node_data: Dict, depth: int) -> TreeNode:
                nonlocal max_feature_id
                if 'leaf_value' in node_data:
                    return TreeNode(
                        node_id=-1, # LightGBM doesn't always provide IDs in dict
                        feature_index=-1,
                        threshold=0.0,
                        left_child=None,
                        right_child=None,
                        is_leaf=True,
                        leaf_value=float(node_data['leaf_value']),
                        depth=depth
                    )
                
                feat_idx = int(node_data['split_feature'])
                max_feature_id = max(max_feature_id, feat_idx)
                
                left_node = build_node(node_data['left_child'], depth + 1)
                right_node = build_node(node_data['right_child'], depth + 1)
                
                return TreeNode(
                    node_id=-1,
                    feature_index=feat_idx,
                    threshold=float(node_data['threshold']),
                    left_child=left_node,
                    right_child=right_node,
                    is_leaf=False,
                    leaf_value=0.0,
                    depth=depth
                )

            root = build_node(tree_data['tree_structure'], 0)
            model_ir_trees.append(TreeIR(tree_id=tree_id, root=root))
            
        return ModelIR(model_type="lightgbm", trees=model_ir_trees, num_features=max_feature_id + 1)

class CatBoostParser(BaseParser):
    def parse(self, content: bytes) -> ModelIR:
        data = json.loads(content.decode('utf-8'))
        
        # CatBoost models have "oblivious" trees (symmetric)
        # We parse them into our standard TreeIR for unified execution
        if 'oblivious_trees' not in data:
            raise ValueError("Invalid CatBoost JSON: 'oblivious_trees' not found")
            
        trees_data = data['oblivious_trees']
        model_ir_trees = []
        max_feature_id = 0
        
        for tree_id, tree_data in enumerate(trees_data):
            # CatBoost symmetric tree structure:
            # Splits are shared across levels
            splits = tree_data['splits']
            leaf_values = tree_data['leaf_values']
            
            # Convert symmetric tree to binary tree structure
            def build_symmetric(level: int, leaf_idx_offset: int, depth: int) -> TreeNode:
                nonlocal max_feature_id
                if level < 0:
                    return TreeNode(
                        node_id=-1,
                        feature_index=-1,
                        threshold=0.0,
                        left_child=None,
                        right_child=None,
                        is_leaf=True,
                        leaf_value=leaf_values[leaf_idx_offset],
                        depth=depth
                    )
                
                split = splits[level]
                feat_idx = split['float_feature_index']
                max_feature_id = max(max_feature_id, feat_idx)
                
                # In CatBoost, 'left' usually corresponds to 'condition is false'
                # but we map it to binary tree branches symmetrically
                left = build_symmetric(level - 1, leaf_idx_offset, depth + 1)
                right = build_symmetric(level - 1, leaf_idx_offset + (1 << level), depth + 1)
                
                return TreeNode(
                    node_id=-1,
                    feature_index=feat_idx,
                    threshold=split['border'],
                    left_child=left,
                    right_child=right,
                    is_leaf=False,
                    leaf_value=0.0,
                    depth=depth
                )

            root = build_symmetric(len(splits) - 1, 0, 0)
            model_ir_trees.append(TreeIR(tree_id=tree_id, root=root))
            
        return ModelIR(model_type="catboost", trees=model_ir_trees, num_features=max_feature_id + 1)

def get_parser(library_type: str) -> BaseParser:
    parsers = {
        "xgboost": XGBoostParser(),
        "lightgbm": LightGBMParser(),
        "catboost": CatBoostParser()
    }
    return parsers.get(library_type.lower())
