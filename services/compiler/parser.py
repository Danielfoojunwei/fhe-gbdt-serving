"""
Model Parser for GBDT Libraries

Production-hardened parsers for XGBoost, LightGBM, and CatBoost models.
Converts library-specific formats to unified TreeIR for MOAI execution.
"""

import json
import logging
from functools import lru_cache
from typing import Dict, List, Optional
from .ir import TreeIR, TreeNode, ModelIR

logger = logging.getLogger(__name__)


class BaseParser:
    """Base class for model parsers."""

    def parse(self, content: bytes) -> ModelIR:
        raise NotImplementedError

    def _validate_content(self, content: bytes) -> dict:
        """Validate and parse JSON content."""
        if not content:
            raise ValueError("Empty content")
        try:
            return json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")


class XGBoostParser(BaseParser):
    """Parser for XGBoost models saved via save_model('model.json')."""

    def parse(self, content: bytes) -> ModelIR:
        data = self._validate_content(content)

        if 'learner' not in data:
            raise ValueError("Invalid XGBoost JSON: 'learner' key not found")

        booster = data['learner']['gradient_booster']
        model = booster['model']
        trees_json = model['trees']

        model_ir_trees = []
        max_feature_id = 0

        for tree_id, tree_data in enumerate(trees_json):
            lefts = tree_data['left_children']
            rights = tree_data['right_children']
            indices = tree_data['split_indices']
            conditions = tree_data['split_conditions']
            leaf_values = tree_data.get('base_weights', tree_data.get('leaf_values', []))

            nodes: Dict[int, TreeNode] = {}
            max_depth = 0

            def build_node(node_idx: int, depth: int) -> int:
                nonlocal max_feature_id, max_depth
                max_depth = max(max_depth, depth)

                is_leaf = lefts[node_idx] == -1

                if is_leaf:
                    node = TreeNode(
                        node_id=node_idx,
                        feature_index=None,
                        threshold=None,
                        left_child_id=None,
                        right_child_id=None,
                        leaf_value=float(leaf_values[node_idx]) if node_idx < len(leaf_values) else 0.0,
                        default_left=True,
                        depth=depth
                    )
                else:
                    feat_idx = indices[node_idx]
                    max_feature_id = max(max_feature_id, feat_idx)

                    left_id = build_node(lefts[node_idx], depth + 1)
                    right_id = build_node(rights[node_idx], depth + 1)

                    node = TreeNode(
                        node_id=node_idx,
                        feature_index=feat_idx,
                        threshold=float(conditions[node_idx]),
                        left_child_id=left_id,
                        right_child_id=right_id,
                        leaf_value=None,
                        default_left=True,
                        depth=depth
                    )

                nodes[node_idx] = node
                return node_idx

            root_id = build_node(0, 0)
            model_ir_trees.append(TreeIR(
                tree_id=tree_id,
                nodes=nodes,
                root_id=root_id,
                max_depth=max_depth
            ))

        base_score = float(data.get('learner', {}).get('learner_model_param', {}).get('base_score', 0.5))

        logger.info(f"Parsed XGBoost model: {len(model_ir_trees)} trees, {max_feature_id + 1} features")

        return ModelIR(
            model_type="xgboost",
            trees=model_ir_trees,
            num_features=max_feature_id + 1,
            base_score=base_score
        )


class LightGBMParser(BaseParser):
    """Parser for LightGBM models from model.dump_model() or model_to_dict()."""

    def parse(self, content: bytes) -> ModelIR:
        data = self._validate_content(content)

        if 'tree_info' not in data:
            raise ValueError("Invalid LightGBM JSON: 'tree_info' not found")

        trees_json = data['tree_info']
        model_ir_trees = []
        max_feature_id = 0

        for tree_id, tree_data in enumerate(trees_json):
            nodes: Dict[int, TreeNode] = {}
            node_counter = [0]  # Use list for nonlocal mutation
            max_depth = 0

            def build_node(node_data: Dict, depth: int) -> int:
                nonlocal max_feature_id, max_depth
                max_depth = max(max_depth, depth)

                node_id = node_counter[0]
                node_counter[0] += 1

                if 'leaf_value' in node_data:
                    node = TreeNode(
                        node_id=node_id,
                        feature_index=None,
                        threshold=None,
                        left_child_id=None,
                        right_child_id=None,
                        leaf_value=float(node_data['leaf_value']),
                        default_left=True,
                        depth=depth
                    )
                else:
                    feat_idx = int(node_data['split_feature'])
                    max_feature_id = max(max_feature_id, feat_idx)

                    left_id = build_node(node_data['left_child'], depth + 1)
                    right_id = build_node(node_data['right_child'], depth + 1)

                    node = TreeNode(
                        node_id=node_id,
                        feature_index=feat_idx,
                        threshold=float(node_data['threshold']),
                        left_child_id=left_id,
                        right_child_id=right_id,
                        leaf_value=None,
                        default_left=node_data.get('default_left', True),
                        depth=depth
                    )

                nodes[node_id] = node
                return node_id

            root_id = build_node(tree_data['tree_structure'], 0)
            model_ir_trees.append(TreeIR(
                tree_id=tree_id,
                nodes=nodes,
                root_id=root_id,
                max_depth=max_depth
            ))

        logger.info(f"Parsed LightGBM model: {len(model_ir_trees)} trees, {max_feature_id + 1} features")

        return ModelIR(
            model_type="lightgbm",
            trees=model_ir_trees,
            num_features=max_feature_id + 1,
            base_score=0.0
        )


class CatBoostParser(BaseParser):
    """Parser for CatBoost oblivious (symmetric) trees."""

    def parse(self, content: bytes) -> ModelIR:
        data = self._validate_content(content)

        if 'oblivious_trees' not in data:
            raise ValueError("Invalid CatBoost JSON: 'oblivious_trees' not found")

        trees_data = data['oblivious_trees']
        model_ir_trees = []
        max_feature_id = 0

        for tree_id, tree_data in enumerate(trees_data):
            splits = tree_data['splits']
            leaf_values = tree_data['leaf_values']

            nodes: Dict[int, TreeNode] = {}
            node_counter = [0]
            max_depth = len(splits)

            def build_symmetric(level: int, leaf_idx_offset: int, depth: int) -> int:
                nonlocal max_feature_id

                node_id = node_counter[0]
                node_counter[0] += 1

                if level < 0:
                    leaf_val = leaf_values[leaf_idx_offset] if leaf_idx_offset < len(leaf_values) else 0.0
                    node = TreeNode(
                        node_id=node_id,
                        feature_index=None,
                        threshold=None,
                        left_child_id=None,
                        right_child_id=None,
                        leaf_value=float(leaf_val),
                        default_left=True,
                        depth=depth
                    )
                else:
                    split = splits[level]
                    feat_idx = split['float_feature_index']
                    max_feature_id = max(max_feature_id, feat_idx)

                    left_id = build_symmetric(level - 1, leaf_idx_offset, depth + 1)
                    right_id = build_symmetric(level - 1, leaf_idx_offset + (1 << level), depth + 1)

                    node = TreeNode(
                        node_id=node_id,
                        feature_index=feat_idx,
                        threshold=float(split['border']),
                        left_child_id=left_id,
                        right_child_id=right_id,
                        leaf_value=None,
                        default_left=True,
                        depth=depth
                    )

                nodes[node_id] = node
                return node_id

            root_id = build_symmetric(len(splits) - 1, 0, 0)
            model_ir_trees.append(TreeIR(
                tree_id=tree_id,
                nodes=nodes,
                root_id=root_id,
                max_depth=max_depth
            ))

        logger.info(f"Parsed CatBoost model: {len(model_ir_trees)} trees, {max_feature_id + 1} features")

        return ModelIR(
            model_type="catboost",
            trees=model_ir_trees,
            num_features=max_feature_id + 1,
            base_score=0.0
        )


@lru_cache(maxsize=None)
def get_parser(library_type: str) -> Optional[BaseParser]:
    """
    Get the appropriate parser for a library type.

    Parser instances are cached since they are stateless.

    Args:
        library_type: One of 'xgboost', 'lightgbm', 'catboost',
                       'logistic_regression', 'random_forest',
                       'decision_tree', 'glm'

    Returns:
        Parser instance or None if not supported
    """
    # Lazy imports to avoid circular dependencies
    from .sklearn_parser import (
        ScikitLearnLogisticRegressionParser,
        ScikitLearnRandomForestParser,
        ScikitLearnDecisionTreeParser,
    )
    from .glm_parser import StatsmodelsGLMParser

    key = library_type.lower()
    parsers = {
        "xgboost": XGBoostParser,
        "lightgbm": LightGBMParser,
        "catboost": CatBoostParser,
        "logistic_regression": ScikitLearnLogisticRegressionParser,
        "random_forest": ScikitLearnRandomForestParser,
        "decision_tree": ScikitLearnDecisionTreeParser,
        "glm": StatsmodelsGLMParser,
    }
    parser_cls = parsers.get(key)
    return parser_cls() if parser_cls else None


def parse_model(content: bytes, library_type: str) -> ModelIR:
    """
    Parse a model file into ModelIR.

    Args:
        content: Raw model file content
        library_type: One of 'xgboost', 'lightgbm', 'catboost'

    Returns:
        Parsed ModelIR

    Raises:
        ValueError: If library type is not supported or content is invalid
    """
    parser = get_parser(library_type)
    if parser is None:
        raise ValueError(f"Unsupported library type: {library_type}")
    return parser.parse(content)
