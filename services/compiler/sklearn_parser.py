"""
Scikit-learn Model Parsers

Parsers for scikit-learn models exported as JSON:
- LogisticRegression
- RandomForestClassifier / RandomForestRegressor
- DecisionTreeClassifier / DecisionTreeRegressor
"""

import json
import logging
from typing import Dict, List, Optional

from .ir import (
    TreeIR, TreeNode, ModelIR, ModelFamily, LinkFunction,
    Aggregation, LinearCoefficients, PreprocessingStep,
)
from .parser import BaseParser

logger = logging.getLogger(__name__)


class ScikitLearnLogisticRegressionParser(BaseParser):
    """Parser for scikit-learn LogisticRegression exported via model_export.py."""

    def parse(self, content: bytes) -> ModelIR:
        data = self._validate_content(content)

        if data.get("model_type") != "logistic_regression":
            raise ValueError(
                f"Expected model_type 'logistic_regression', got '{data.get('model_type')}'"
            )

        coefs = data["coefficients"]
        # sklearn stores shape (n_classes, n_features); for binary, take first row
        if isinstance(coefs[0], list):
            weights = [float(c) for c in coefs[0]]
        else:
            weights = [float(c) for c in coefs]

        intercept_val = data["intercept"]
        if isinstance(intercept_val, list):
            intercept_val = float(intercept_val[0])
        else:
            intercept_val = float(intercept_val)

        num_features = len(weights)
        feature_names = data.get("feature_names")
        class_labels = data.get("classes")

        # Build preprocessing steps from export
        preprocessing = []
        if "preprocessing" in data:
            for step in data["preprocessing"]:
                preprocessing.append(PreprocessingStep(
                    step_type=step["type"],
                    params=step.get("params", {}),
                    feature_indices=step.get("feature_indices"),
                ))

        logger.info(
            f"Parsed LogisticRegression: {num_features} features, "
            f"intercept={intercept_val:.4f}"
        )

        return ModelIR(
            model_type="logistic_regression",
            trees=[],
            num_features=num_features,
            base_score=0.0,
            model_family=ModelFamily.LINEAR,
            coefficients=LinearCoefficients(weights=weights, intercept=intercept_val),
            link_function=LinkFunction.LOGIT,
            glm_family="binomial",
            aggregation=Aggregation.NONE,
            feature_names=feature_names,
            preprocessing=preprocessing,
            class_labels=[str(c) for c in class_labels] if class_labels else None,
        )


class ScikitLearnRandomForestParser(BaseParser):
    """Parser for scikit-learn RandomForest exported via model_export.py."""

    def parse(self, content: bytes) -> ModelIR:
        data = self._validate_content(content)

        if data.get("model_type") != "random_forest":
            raise ValueError(
                f"Expected model_type 'random_forest', got '{data.get('model_type')}'"
            )

        trees_data = data["trees"]
        model_ir_trees = []
        max_feature_id = 0

        for tree_id, tree_data in enumerate(trees_data):
            nodes_data = tree_data["nodes"]
            nodes: Dict[int, TreeNode] = {}
            max_depth = 0

            for node_data in nodes_data:
                nid = node_data["node_id"]
                depth = node_data.get("depth", 0)
                max_depth = max(max_depth, depth)

                is_leaf = node_data.get("is_leaf", False)

                if is_leaf:
                    # For classification, leaf_value is the majority class probability
                    leaf_val = node_data.get("value", 0.0)
                    if isinstance(leaf_val, list):
                        # For classifier: value is class distribution, take class 1 prob
                        if len(leaf_val) > 0 and isinstance(leaf_val[0], list):
                            counts = leaf_val[0]
                            total = sum(counts)
                            leaf_val = counts[-1] / total if total > 0 else 0.0
                        else:
                            leaf_val = float(leaf_val[0]) if leaf_val else 0.0

                    nodes[nid] = TreeNode(
                        node_id=nid,
                        leaf_value=float(leaf_val),
                        depth=depth,
                    )
                else:
                    feat_idx = node_data["feature_index"]
                    max_feature_id = max(max_feature_id, feat_idx)
                    nodes[nid] = TreeNode(
                        node_id=nid,
                        feature_index=feat_idx,
                        threshold=float(node_data["threshold"]),
                        left_child_id=node_data.get("left_child"),
                        right_child_id=node_data.get("right_child"),
                        depth=depth,
                    )

            model_ir_trees.append(TreeIR(
                tree_id=tree_id,
                nodes=nodes,
                root_id=0,
                max_depth=max_depth,
            ))

        num_features = data.get("num_features", max_feature_id + 1)
        feature_names = data.get("feature_names")
        is_classifier = data.get("is_classifier", True)

        logger.info(
            f"Parsed RandomForest: {len(model_ir_trees)} trees, "
            f"{num_features} features, classifier={is_classifier}"
        )

        return ModelIR(
            model_type="random_forest",
            trees=model_ir_trees,
            num_features=num_features,
            base_score=0.0,
            model_family=ModelFamily.RANDOM_FOREST,
            link_function=LinkFunction.IDENTITY,
            aggregation=Aggregation.MEAN,
            feature_names=feature_names,
            class_labels=data.get("classes"),
            metadata={"is_classifier": is_classifier},
        )


class ScikitLearnDecisionTreeParser(BaseParser):
    """Parser for scikit-learn DecisionTree exported via model_export.py."""

    def parse(self, content: bytes) -> ModelIR:
        data = self._validate_content(content)

        if data.get("model_type") != "decision_tree":
            raise ValueError(
                f"Expected model_type 'decision_tree', got '{data.get('model_type')}'"
            )

        nodes_data = data["nodes"]
        nodes: Dict[int, TreeNode] = {}
        max_depth = 0
        max_feature_id = 0

        for node_data in nodes_data:
            nid = node_data["node_id"]
            depth = node_data.get("depth", 0)
            max_depth = max(max_depth, depth)
            is_leaf = node_data.get("is_leaf", False)

            if is_leaf:
                leaf_val = node_data.get("value", 0.0)
                if isinstance(leaf_val, list):
                    if len(leaf_val) > 0 and isinstance(leaf_val[0], list):
                        counts = leaf_val[0]
                        total = sum(counts)
                        leaf_val = counts[-1] / total if total > 0 else 0.0
                    else:
                        leaf_val = float(leaf_val[0]) if leaf_val else 0.0

                nodes[nid] = TreeNode(
                    node_id=nid,
                    leaf_value=float(leaf_val),
                    depth=depth,
                )
            else:
                feat_idx = node_data["feature_index"]
                max_feature_id = max(max_feature_id, feat_idx)
                nodes[nid] = TreeNode(
                    node_id=nid,
                    feature_index=feat_idx,
                    threshold=float(node_data["threshold"]),
                    left_child_id=node_data.get("left_child"),
                    right_child_id=node_data.get("right_child"),
                    depth=depth,
                )

        tree_ir = TreeIR(
            tree_id=0,
            nodes=nodes,
            root_id=0,
            max_depth=max_depth,
        )

        num_features = data.get("num_features", max_feature_id + 1)
        is_classifier = data.get("is_classifier", True)

        # Extract reason codes (adverse action) from feature names
        feature_names = data.get("feature_names")

        logger.info(
            f"Parsed DecisionTree: {len(nodes)} nodes, depth={max_depth}, "
            f"{num_features} features, classifier={is_classifier}"
        )

        return ModelIR(
            model_type="decision_tree",
            trees=[tree_ir],
            num_features=num_features,
            base_score=0.0,
            model_family=ModelFamily.SINGLE_TREE,
            link_function=LinkFunction.IDENTITY,
            aggregation=Aggregation.NONE,
            feature_names=feature_names,
            class_labels=data.get("classes"),
            metadata={
                "is_classifier": is_classifier,
                "supports_reason_codes": True,
            },
        )
