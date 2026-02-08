"""
Model Export Utilities

Helpers for customers to export trained models into the FHE-GBDT platform's
JSON format. Supports scikit-learn and statsmodels models.

Usage:
    from fhe_gbdt_sdk.model_export import export_model
    json_bytes = export_model(trained_model, feature_names=["age", "income"])
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def export_model(
    model: Any,
    feature_names: Optional[List[str]] = None,
    preprocessing: Optional[List[Dict]] = None,
) -> bytes:
    """
    Export a trained model to the platform's JSON format.

    Supported model types:
    - sklearn.linear_model.LogisticRegression
    - sklearn.ensemble.RandomForestClassifier / RandomForestRegressor
    - sklearn.tree.DecisionTreeClassifier / DecisionTreeRegressor
    - statsmodels.genmod.generalized_linear_model.GLMResultsWrapper

    Args:
        model: Trained model object
        feature_names: Optional list of feature names
        preprocessing: Optional preprocessing steps

    Returns:
        JSON bytes ready for the compiler

    Raises:
        ValueError: If model type is not supported
    """
    model_class = type(model).__name__
    module = type(model).__module__ or ""

    if "LogisticRegression" in model_class:
        return _export_logistic_regression(model, feature_names, preprocessing)
    elif "RandomForest" in model_class:
        return _export_random_forest(model, feature_names, preprocessing)
    elif "DecisionTree" in model_class:
        return _export_decision_tree(model, feature_names, preprocessing)
    elif "GLM" in model_class or "glm" in module:
        return _export_statsmodels_glm(model, feature_names, preprocessing)
    else:
        raise ValueError(
            f"Unsupported model type: {model_class}. "
            f"Supported: LogisticRegression, RandomForest*, DecisionTree*, GLM"
        )


def _export_logistic_regression(
    model: Any,
    feature_names: Optional[List[str]],
    preprocessing: Optional[List[Dict]],
) -> bytes:
    """Export scikit-learn LogisticRegression."""
    data = {
        "model_type": "logistic_regression",
        "library": "scikit-learn",
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist() if hasattr(model, "classes_") else None,
        "num_features": model.coef_.shape[1],
        "feature_names": feature_names,
        "preprocessing": preprocessing or [],
        "regularization": {
            "C": float(model.C) if hasattr(model, "C") else None,
            "penalty": model.penalty if hasattr(model, "penalty") else None,
        },
    }
    return json.dumps(data, indent=2).encode("utf-8")


def _export_random_forest(
    model: Any,
    feature_names: Optional[List[str]],
    preprocessing: Optional[List[Dict]],
) -> bytes:
    """Export scikit-learn RandomForest."""
    is_classifier = hasattr(model, "classes_")
    trees = []

    for tree_idx, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        nodes = []
        for i in range(tree.node_count):
            is_leaf = tree.children_left[i] == -1
            node_data = {
                "node_id": i,
                "depth": _compute_depth(tree, i),
                "is_leaf": is_leaf,
            }
            if is_leaf:
                if is_classifier:
                    node_data["value"] = tree.value[i].tolist()
                else:
                    node_data["value"] = float(tree.value[i].flat[0])
            else:
                node_data["feature_index"] = int(tree.feature[i])
                node_data["threshold"] = float(tree.threshold[i])
                node_data["left_child"] = int(tree.children_left[i])
                node_data["right_child"] = int(tree.children_right[i])
            nodes.append(node_data)

        trees.append({"tree_id": tree_idx, "nodes": nodes})

    data = {
        "model_type": "random_forest",
        "library": "scikit-learn",
        "trees": trees,
        "num_features": model.n_features_in_,
        "n_estimators": len(model.estimators_),
        "is_classifier": is_classifier,
        "classes": model.classes_.tolist() if is_classifier else None,
        "feature_names": feature_names,
        "preprocessing": preprocessing or [],
    }
    return json.dumps(data, indent=2).encode("utf-8")


def _export_decision_tree(
    model: Any,
    feature_names: Optional[List[str]],
    preprocessing: Optional[List[Dict]],
) -> bytes:
    """Export scikit-learn DecisionTree."""
    is_classifier = hasattr(model, "classes_")
    tree = model.tree_
    nodes = []

    for i in range(tree.node_count):
        is_leaf = tree.children_left[i] == -1
        node_data = {
            "node_id": i,
            "depth": _compute_depth(tree, i),
            "is_leaf": is_leaf,
        }
        if is_leaf:
            if is_classifier:
                node_data["value"] = tree.value[i].tolist()
            else:
                node_data["value"] = float(tree.value[i].flat[0])
        else:
            node_data["feature_index"] = int(tree.feature[i])
            node_data["threshold"] = float(tree.threshold[i])
            node_data["left_child"] = int(tree.children_left[i])
            node_data["right_child"] = int(tree.children_right[i])
        nodes.append(node_data)

    data = {
        "model_type": "decision_tree",
        "library": "scikit-learn",
        "nodes": nodes,
        "num_features": model.n_features_in_,
        "max_depth": int(model.tree_.max_depth),
        "is_classifier": is_classifier,
        "classes": model.classes_.tolist() if is_classifier else None,
        "feature_names": feature_names,
        "preprocessing": preprocessing or [],
    }
    return json.dumps(data, indent=2).encode("utf-8")


def _export_statsmodels_glm(
    model: Any,
    feature_names: Optional[List[str]],
    preprocessing: Optional[List[Dict]],
) -> bytes:
    """Export statsmodels GLM results."""
    # Determine family
    family_name = "gaussian"
    if hasattr(model, "family"):
        family_class = type(model.family).__name__.lower()
        for name in ("binomial", "poisson", "gamma", "tweedie", "gaussian", "inverse_gaussian"):
            if name in family_class:
                family_name = name
                break

    # Determine link
    link_name = None
    if hasattr(model, "family") and hasattr(model.family, "link"):
        link_class = type(model.family.link).__name__.lower()
        link_map = {"logit": "logit", "log": "log", "identity": "identity", "inverse": "reciprocal"}
        for k, v in link_map.items():
            if k in link_class:
                link_name = v
                break

    params = model.params.tolist() if hasattr(model.params, "tolist") else list(model.params)
    param_names = feature_names
    if hasattr(model, "model") and hasattr(model.model, "exog_names"):
        param_names = model.model.exog_names

    data = {
        "model_type": "glm",
        "library": "statsmodels",
        "params": params,
        "param_names": param_names,
        "family": family_name,
        "link": link_name,
        "num_features": len(params) - (1 if param_names and param_names[0] in ("const", "Intercept") else 0),
        "feature_names": feature_names,
        "preprocessing": preprocessing or [],
        "aic": float(model.aic) if hasattr(model, "aic") else None,
        "bic": float(model.bic) if hasattr(model, "bic") else None,
        "deviance": float(model.deviance) if hasattr(model, "deviance") else None,
    }
    return json.dumps(data, indent=2).encode("utf-8")


def _compute_depth(tree: Any, node_id: int) -> int:
    """Compute depth of a node in sklearn's tree structure."""
    depth = 0
    # Walk up from node to root by searching
    stack = [(0, 0)]  # (node_id, depth)
    depths = {}
    while stack:
        nid, d = stack.pop()
        depths[nid] = d
        if tree.children_left[nid] != -1:
            stack.append((tree.children_left[nid], d + 1))
            stack.append((tree.children_right[nid], d + 1))
    return depths.get(node_id, 0)


def create_manual_logistic_regression(
    weights: List[float],
    intercept: float = 0.0,
    feature_names: Optional[List[str]] = None,
    preprocessing: Optional[List[Dict]] = None,
) -> bytes:
    """
    Create a logistic regression model export manually (without sklearn).

    Useful for regulatory models (credit scorecards) where coefficients
    are manually specified by risk teams.

    Args:
        weights: Per-feature coefficients
        intercept: Bias term
        feature_names: Feature names
        preprocessing: Preprocessing steps (e.g., WoE binning)

    Returns:
        JSON bytes for the compiler
    """
    data = {
        "model_type": "logistic_regression",
        "library": "manual",
        "coefficients": [weights],
        "intercept": [intercept],
        "classes": [0, 1],
        "num_features": len(weights),
        "feature_names": feature_names,
        "preprocessing": preprocessing or [],
    }
    return json.dumps(data, indent=2).encode("utf-8")


def create_manual_glm(
    weights: List[float],
    intercept: float = 0.0,
    family: str = "gaussian",
    link: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    preprocessing: Optional[List[Dict]] = None,
) -> bytes:
    """
    Create a GLM export manually (without statsmodels).

    Args:
        weights: Per-feature coefficients
        intercept: Bias term
        family: GLM family name
        link: Link function override
        feature_names: Feature names
        preprocessing: Preprocessing steps

    Returns:
        JSON bytes for the compiler
    """
    params = [intercept] + weights
    param_names = ["const"] + (feature_names or [f"x{i}" for i in range(len(weights))])

    data = {
        "model_type": "glm",
        "library": "manual",
        "params": params,
        "param_names": param_names,
        "family": family,
        "link": link,
        "num_features": len(weights),
        "feature_names": feature_names,
        "preprocessing": preprocessing or [],
    }
    return json.dumps(data, indent=2).encode("utf-8")
