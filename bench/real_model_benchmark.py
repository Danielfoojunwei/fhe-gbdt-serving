#!/usr/bin/env python3
"""
Real Model Benchmark for FHE-GBDT Innovations

Uses REAL trained XGBoost and LightGBM models on REAL sklearn datasets.
No mocks, no random trees -- genuine empirical evaluation.

Datasets:
  1. Breast Cancer Wisconsin (classification, 30 features, 569 samples)
  2. California Housing (regression, 8 features, 2000 subsample)
  3. Iris (binary classification, 4 features, 150 samples)
  4. Diabetes (regression, 10 features, 442 samples)

Model Configurations:
  1. XGBoost (n_estimators=50, max_depth=5)
  2. LightGBM (n_estimators=50, max_depth=5)
  3. XGBoost many-shallow (n_estimators=100, max_depth=3)
  4. XGBoost few-deep (n_estimators=10, max_depth=8)

Innovations Benchmarked:
  1. Leaf-Centric Encoding
  2. Gradient-Informed Noise Allocation
  3. Homomorphic Ensemble Pruning
  4. Polynomial Leaf Functions
  5. MOAI-Native Tree Conversion
  6. Bootstrap-Aligned Architecture
  7. Unified Engine (all innovations combined)
"""

import sys
import os
import time
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Suppress noisy warnings from sklearn/xgboost/lightgbm
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.compiler.ir import ModelIR, TreeIR, TreeNode
from services.innovations.leaf_centric import LeafCentricEncoder
from services.innovations.gradient_noise import (
    GradientAwareNoiseAllocator, AdaptivePrecisionEncoder, FeatureImportanceAnalyzer
)
from services.innovations.homomorphic_pruning import (
    HomomorphicEnsemblePruner, PruningConfig
)
from services.innovations.polynomial_leaves import (
    PolynomialLeafGBDT, PolynomialLeafTrainer, PolynomialLeafConfig
)
from services.innovations.moai_native import (
    RotationOptimalConverter, ConversionConfig, ObliviousTreeSynthesizer
)
from services.innovations.bootstrap_aligned import (
    BootstrapAwareTreeBuilder, create_bootstrap_aligned_forest,
    BootstrapInterleavedEnsemble
)
from services.innovations.unified_architecture import (
    NovelFHEGBDTEngine, InnovationConfig, optimize_model_for_fhe
)

import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import (
    load_breast_cancer, fetch_california_housing, load_iris, load_diabetes
)
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SEED = 42


# =============================================================================
# XGBoost/LightGBM to ModelIR Conversion
# =============================================================================

def xgboost_to_model_ir(model, num_features, is_classifier=False):
    """
    Convert a trained XGBoost model to ModelIR.

    Parses the JSON tree dump from get_booster().get_dump(dump_format='json').
    XGBoost convention: 'yes' branch = feature < split_condition (left child).
    """
    booster = model.get_booster()

    # Extract base_score from internal config
    config = json.loads(booster.save_config())
    base_score_raw = float(
        config['learner']['learner_model_param']['base_score'].strip('[]')
    )

    if is_classifier:
        # Config stores probability; convert to logit for raw score space
        bs_clipped = np.clip(base_score_raw, 1e-7, 1 - 1e-7)
        base_score = float(np.log(bs_clipped / (1 - bs_clipped)))
    else:
        base_score = base_score_raw

    # Parse each tree from JSON dump
    dump = booster.get_dump(dump_format='json')
    trees = []
    for tree_id, tree_json_str in enumerate(dump):
        tree_json = json.loads(tree_json_str)
        tree_ir = _parse_xgb_tree(tree_json, tree_id)
        trees.append(tree_ir)

    return ModelIR(
        model_type="xgboost",
        trees=trees,
        num_features=num_features,
        base_score=base_score
    )


def _parse_xgb_tree(tree_json, tree_id):
    """
    Parse a single XGBoost JSON tree into TreeIR.

    XGBoost JSON node fields:
      Internal: nodeid, depth, split ("f0"), split_condition, yes, no, missing, children
      Leaf: nodeid, leaf
    """
    nodes = {}
    max_node_depth = [0]

    def parse_node(node_json, depth=0):
        nid = node_json['nodeid']
        max_node_depth[0] = max(max_node_depth[0], depth)

        if 'leaf' in node_json:
            # Leaf node
            nodes[nid] = TreeNode(
                node_id=nid,
                leaf_value=float(node_json['leaf']),
                depth=depth
            )
        else:
            # Split node
            feat_str = node_json['split']  # e.g., "f0", "f12"
            feat_idx = int(feat_str[1:])
            threshold = float(node_json['split_condition'])
            yes_id = node_json['yes']   # left: feature < threshold
            no_id = node_json['no']     # right: feature >= threshold

            nodes[nid] = TreeNode(
                node_id=nid,
                feature_index=feat_idx,
                threshold=threshold,
                left_child_id=yes_id,
                right_child_id=no_id,
                default_left=(node_json.get('missing', no_id) == yes_id),
                depth=depth
            )

            # Recurse into children
            for child in node_json.get('children', []):
                parse_node(child, depth + 1)

    parse_node(tree_json)
    return TreeIR(
        tree_id=tree_id,
        nodes=nodes,
        root_id=0,
        max_depth=max_node_depth[0] + 1
    )


def lightgbm_to_model_ir(model, num_features, is_classifier=False):
    """
    Convert a trained LightGBM model to ModelIR.

    Parses the model dump from booster_.dump_model()['tree_info'].
    LightGBM convention: left_child when feature <= threshold.
    We use '<' in predict_standard; for float64 continuous data the
    difference at exact threshold values is negligible (measure zero).
    """
    dump = model.booster_.dump_model()
    trees_info = dump['tree_info']

    trees = []
    for tree_id, tree_info in enumerate(trees_info):
        tree_ir = _parse_lgb_tree(tree_info['tree_structure'], tree_id)
        trees.append(tree_ir)

    # LightGBM absorbs any base score into the tree leaf values
    return ModelIR(
        model_type="lightgbm",
        trees=trees,
        num_features=num_features,
        base_score=0.0
    )


def _parse_lgb_tree(tree_structure, tree_id):
    """
    Parse a single LightGBM tree structure into TreeIR.

    LightGBM node fields:
      Internal: split_index, split_feature, threshold, decision_type,
                default_left, left_child, right_child
      Leaf: leaf_index, leaf_value
    """
    nodes = {}
    counter = [0]

    def parse_node(node, depth=0):
        nid = counter[0]
        counter[0] += 1

        if 'leaf_value' in node:
            nodes[nid] = TreeNode(
                node_id=nid,
                leaf_value=float(node['leaf_value']),
                depth=depth
            )
            return nid

        my_nid = nid
        feat_idx = int(node['split_feature'])
        threshold = float(node['threshold'])

        # Recurse: left child first, then right
        left_id = parse_node(node['left_child'], depth + 1)
        right_id = parse_node(node['right_child'], depth + 1)

        nodes[my_nid] = TreeNode(
            node_id=my_nid,
            feature_index=feat_idx,
            threshold=threshold,
            left_child_id=left_id,
            right_child_id=right_id,
            default_left=node.get('default_left', True),
            depth=depth
        )
        return my_nid

    root_id = parse_node(tree_structure)
    max_depth = max(n.depth for n in nodes.values()) + 1
    return TreeIR(
        tree_id=tree_id,
        nodes=nodes,
        root_id=root_id,
        max_depth=max_depth
    )


def get_original_raw_predictions(model, X, is_classifier, framework):
    """
    Get raw-score predictions directly from the original trained model.

    For classifiers: returns logits (pre-sigmoid scores).
    For regressors: returns predicted values.
    """
    if framework == 'xgboost':
        if is_classifier:
            dmat = xgb.DMatrix(X)
            return model.get_booster().predict(dmat, output_margin=True)
        else:
            return model.predict(X)
    elif framework == 'lightgbm':
        if is_classifier:
            raw = model.predict_proba(X, raw_score=True)
            if raw.ndim == 2:
                return raw[:, 1]
            return raw
        else:
            return model.predict(X)
    else:
        raise ValueError(f"Unknown framework: {framework}")


# =============================================================================
# Standard tree traversal (from accuracy_benchmark.py)
# =============================================================================

def predict_standard(model_ir, X, use_leq=False):
    """
    Standard tree traversal prediction (baseline).

    Args:
        model_ir: ModelIR with trees to traverse.
        X: Feature matrix (n_samples, n_features).
        use_leq: If True, use '<=' for left branch (LightGBM convention).
                 If False, use '<' for left branch (XGBoost convention).
                 Default False for consistency with accuracy_benchmark.py.
    """
    predictions = np.full(X.shape[0], model_ir.base_score)
    for tree in model_ir.trees:
        for i in range(X.shape[0]):
            node = tree.nodes.get(tree.root_id)
            while node is not None:
                if node.leaf_value is not None:
                    predictions[i] += node.leaf_value
                    break
                feat_val = X[i, node.feature_index]
                if use_leq:
                    go_left = feat_val <= node.threshold
                else:
                    go_left = feat_val < node.threshold
                if go_left:
                    node = tree.nodes.get(node.left_child_id)
                else:
                    node = tree.nodes.get(node.right_child_id)
    return predictions


# =============================================================================
# Metrics
# =============================================================================

def compute_mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_accuracy(y_true, raw_scores):
    """Classification accuracy from raw logit scores."""
    probs = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -500, 500)))
    pred_labels = (probs > 0.5).astype(float)
    return float(np.mean(pred_labels == y_true))


def compute_auc(y_true, raw_scores):
    """AUC from raw logit scores."""
    probs = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -500, 500)))
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return 0.0


# =============================================================================
# Dataset Loading
# =============================================================================

def load_datasets():
    """Load all four real sklearn datasets with 70/30 splits."""
    datasets = []

    # 1. Breast Cancer Wisconsin (binary classification)
    bc = load_breast_cancer()
    X_bc, y_bc = bc.data.astype(np.float64), bc.target.astype(np.float64)
    datasets.append({
        'name': 'Breast Cancer',
        'task': 'classification',
        'X': X_bc,
        'y': y_bc,
        'num_features': X_bc.shape[1],
        'n_samples': len(y_bc),
    })

    # 2. California Housing (regression, subsample to 2000)
    cal = fetch_california_housing()
    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(cal.target), size=2000, replace=False)
    X_cal = cal.data[indices].astype(np.float64)
    y_cal = cal.target[indices].astype(np.float64)
    datasets.append({
        'name': 'California Housing',
        'task': 'regression',
        'X': X_cal,
        'y': y_cal,
        'num_features': X_cal.shape[1],
        'n_samples': len(y_cal),
    })

    # 3. Iris (binarized: class 0 vs rest, keeps all 150 samples)
    iris = load_iris()
    X_iris = iris.data.astype(np.float64)
    y_iris = (iris.target == 0).astype(np.float64)
    datasets.append({
        'name': 'Iris (binary)',
        'task': 'classification',
        'X': X_iris,
        'y': y_iris,
        'num_features': X_iris.shape[1],
        'n_samples': len(y_iris),
    })

    # 4. Diabetes (regression)
    diab = load_diabetes()
    X_diab, y_diab = diab.data.astype(np.float64), diab.target.astype(np.float64)
    datasets.append({
        'name': 'Diabetes',
        'task': 'regression',
        'X': X_diab,
        'y': y_diab,
        'num_features': X_diab.shape[1],
        'n_samples': len(y_diab),
    })

    return datasets


# =============================================================================
# Model Training and Conversion
# =============================================================================

def train_and_convert(X_train, y_train, num_features, task):
    """
    Train 4 model configurations and convert each to ModelIR.

    Returns list of (name, original_model, model_ir, is_classifier, framework).
    """
    is_clf = (task == 'classification')
    models = []

    # 1. XGBoost standard (50 trees, depth 5)
    if is_clf:
        m1 = xgb.XGBClassifier(
            n_estimators=50, max_depth=5, learning_rate=0.1,
            random_state=SEED, eval_metric='logloss',
            verbosity=0
        )
    else:
        m1 = xgb.XGBRegressor(
            n_estimators=50, max_depth=5, learning_rate=0.1,
            random_state=SEED, verbosity=0
        )
    m1.fit(X_train, y_train)
    ir1 = xgboost_to_model_ir(m1, num_features, is_classifier=is_clf)
    models.append(('XGB-50t-d5', m1, ir1, is_clf, 'xgboost'))

    # 2. LightGBM standard (50 trees, depth 5)
    if is_clf:
        m2 = lgb.LGBMClassifier(
            n_estimators=50, max_depth=5, learning_rate=0.1,
            random_state=SEED, verbose=-1, num_leaves=31
        )
    else:
        m2 = lgb.LGBMRegressor(
            n_estimators=50, max_depth=5, learning_rate=0.1,
            random_state=SEED, verbose=-1, num_leaves=31
        )
    m2.fit(X_train, y_train)
    ir2 = lightgbm_to_model_ir(m2, num_features, is_classifier=is_clf)
    models.append(('LGB-50t-d5', m2, ir2, is_clf, 'lightgbm'))

    # 3. XGBoost many shallow trees (100 trees, depth 3)
    if is_clf:
        m3 = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=SEED, eval_metric='logloss',
            verbosity=0
        )
    else:
        m3 = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=SEED, verbosity=0
        )
    m3.fit(X_train, y_train)
    ir3 = xgboost_to_model_ir(m3, num_features, is_classifier=is_clf)
    models.append(('XGB-100t-d3', m3, ir3, is_clf, 'xgboost'))

    # 4. XGBoost few deep trees (10 trees, depth 8)
    if is_clf:
        m4 = xgb.XGBClassifier(
            n_estimators=10, max_depth=8, learning_rate=0.3,
            random_state=SEED, eval_metric='logloss',
            verbosity=0
        )
    else:
        m4 = xgb.XGBRegressor(
            n_estimators=10, max_depth=8, learning_rate=0.3,
            random_state=SEED, verbosity=0
        )
    m4.fit(X_train, y_train)
    ir4 = xgboost_to_model_ir(m4, num_features, is_classifier=is_clf)
    models.append(('XGB-10t-d8', m4, ir4, is_clf, 'xgboost'))

    return models


# =============================================================================
# Per-Innovation Benchmark Functions
# =============================================================================

def benchmark_leaf_centric(model_ir, X_train, y_train, X_test, y_test):
    """Benchmark leaf-centric encoding accuracy."""
    start = time.time()
    encoder = LeafCentricEncoder()
    plan = encoder.encode_model(model_ir)
    preds = encoder.evaluate_plaintext(plan, X_test, model_ir.base_score)
    elapsed = (time.time() - start) * 1000

    baseline = predict_standard(model_ir, X_test)
    return preds, baseline, elapsed, {
        "num_trees": plan.num_trees,
        "max_depth": plan.max_depth,
        "total_leaves": plan.total_leaves,
    }


def benchmark_gradient_noise(model_ir, X_train, y_train, X_test, y_test):
    """Benchmark gradient-aware noise allocation accuracy."""
    start = time.time()
    allocator = GradientAwareNoiseAllocator()
    allocations = allocator.allocate(model_ir, model_ir.num_features)
    encoder = AdaptivePrecisionEncoder(allocations)

    # Encode then decode (simulating FHE quantization noise)
    encoded, scales = encoder.encode(X_test)
    decoded = encoder.decode(encoded, scales)

    # Predict with quantized features
    preds = predict_standard(model_ir, decoded)
    elapsed = (time.time() - start) * 1000

    baseline = predict_standard(model_ir, X_test)

    precision_bits = {k: v.precision_bits for k, v in allocations.items()}
    avg_bits = np.mean(list(precision_bits.values())) if precision_bits else 0

    return preds, baseline, elapsed, {
        "avg_precision_bits": round(float(avg_bits), 2),
        "min_precision_bits": int(min(precision_bits.values())) if precision_bits else 0,
        "max_precision_bits": int(max(precision_bits.values())) if precision_bits else 0,
        "encode_decode_mae": float(np.mean(np.abs(X_test - decoded))),
    }


def benchmark_homomorphic_pruning(model_ir, X_train, y_train, X_test, y_test):
    """Benchmark homomorphic pruning accuracy."""
    start = time.time()
    config = PruningConfig(
        significance_threshold=0.1,
        soft_pruning=True,
        min_trees=max(1, len(model_ir.trees) * 3 // 4),
        max_prune_fraction=0.3,
    )
    pruner = HomomorphicEnsemblePruner(config)

    # Compute per-tree outputs for all test samples
    tree_outputs = np.zeros((X_test.shape[0], len(model_ir.trees)))
    for tree_idx, tree in enumerate(model_ir.trees):
        for i in range(X_test.shape[0]):
            node = tree.nodes.get(tree.root_id)
            while node is not None:
                if node.leaf_value is not None:
                    tree_outputs[i, tree_idx] = node.leaf_value
                    break
                if X_test[i, node.feature_index] < node.threshold:
                    node = tree.nodes.get(node.left_child_id)
                else:
                    node = tree.nodes.get(node.right_child_id)

    aggregated, metadata = pruner.prune_plaintext(tree_outputs, preserve_accuracy=False)
    preds = aggregated + model_ir.base_score
    elapsed = (time.time() - start) * 1000

    baseline = predict_standard(model_ir, X_test)
    return preds, baseline, elapsed, {
        "num_active_trees": metadata.get("num_active_trees", len(model_ir.trees)),
        "pruning_ratio": round(metadata.get("pruning_ratio", 0.0), 4),
        "total_trees": len(model_ir.trees),
        "computation_saved_pct": round(metadata.get("pruning_ratio", 0.0) * 100, 2),
    }


def benchmark_polynomial_leaves(model_ir, X_train, y_train, X_test, y_test):
    """Benchmark polynomial leaf functions accuracy."""
    start = time.time()
    config = PolynomialLeafConfig(
        max_degree=2, min_samples_for_poly=5, r2_threshold=0.05
    )
    poly_model = PolynomialLeafGBDT(model_ir, config=config)
    poly_model.fit_polynomials(X_train, y_train)
    preds = poly_model.predict(X_test)
    elapsed = (time.time() - start) * 1000

    baseline = predict_standard(model_ir, X_test)
    stats = poly_model.get_statistics()
    return preds, baseline, elapsed, {
        "num_polynomial_leaves": stats.get("num_polynomial_leaves", 0),
        "coverage_pct": round(stats.get("coverage", 0) * 100, 2),
        "avg_degree": round(stats.get("avg_degree", 0), 2),
        "avg_leaf_r2": round(stats.get("avg_r2", 0), 4),
    }


def benchmark_moai_conversion(model_ir, X_train, y_train, X_test, y_test):
    """Benchmark MOAI-native tree conversion with accuracy-aware retuning."""
    start = time.time()
    config = ConversionConfig(
        retune_leaves=True,
        max_accuracy_loss=0.05,
    )
    converter = RotationOptimalConverter(config)
    # Limit validation data to prevent overfitting during retuning
    val_size = min(100, len(X_train))
    result = converter.convert_model(model_ir, X_train[:val_size], y_train[:val_size])

    def predict_oblivious(X, base_score, trees):
        p = np.full(X.shape[0], base_score)
        for tree in trees:
            for i in range(X.shape[0]):
                leaf_idx = 0
                for depth, level in enumerate(tree.levels):
                    if X[i, level.feature_idx] >= level.threshold:
                        leaf_idx |= (1 << depth)
                if leaf_idx < len(tree.leaf_values):
                    p[i] += tree.leaf_values[leaf_idx]
        return p

    preds = predict_oblivious(X_test, model_ir.base_score, result.oblivious_trees)
    elapsed = (time.time() - start) * 1000

    baseline = predict_standard(model_ir, X_test)
    return preds, baseline, elapsed, {
        "rotation_savings_pct": result.rotation_savings.get("savings_percent", 0),
        "speedup_factor": result.rotation_savings.get("speedup_factor", 1.0),
        "original_rotations": result.rotation_savings.get("original_rotations", 0),
        "oblivious_rotations": result.rotation_savings.get("oblivious_rotations", 0),
        "accuracy_loss_on_val": round(result.accuracy_loss, 4),
    }


def benchmark_bootstrap_aligned(model_ir, X_train, y_train, X_test, y_test):
    """Benchmark bootstrap-aligned architecture accuracy."""
    start = time.time()
    forest = create_bootstrap_aligned_forest(model_ir)
    ensemble = BootstrapInterleavedEnsemble(forest)
    preds = ensemble.evaluate_plaintext(X_test, model_ir.base_score)
    elapsed = (time.time() - start) * 1000

    baseline = predict_standard(model_ir, X_test)

    builder = BootstrapAwareTreeBuilder()
    analysis = builder.analyze_noise_consumption(model_ir)

    return preds, baseline, elapsed, {
        "num_chunks": len(forest.chunks),
        "total_trees": forest.total_trees,
        "needs_bootstrap": analysis.get("needs_bootstrap", False),
        "estimated_noise_bits": round(analysis.get("total_estimated_noise", 0), 2),
    }


def benchmark_unified_engine(model_ir, X_train, y_train, X_test, y_test):
    """Benchmark unified architecture with all innovations enabled."""
    start = time.time()
    engine, plan = optimize_model_for_fhe(model_ir, X_train, y_train)
    preds = engine.predict(X_test)
    elapsed = (time.time() - start) * 1000

    baseline = predict_standard(model_ir, X_test)
    report = engine.get_optimization_report()

    return preds, baseline, elapsed, {
        "innovations_enabled": report.get("innovations_enabled", []),
        "rotation_savings_pct": round(plan.rotation_savings_percent, 2),
        "estimated_latency_ms": round(plan.estimated_latency_ms, 2),
    }


INNOVATIONS = [
    ("Leaf-Centric Encoding", benchmark_leaf_centric),
    ("Gradient-Aware Noise", benchmark_gradient_noise),
    ("Homomorphic Pruning", benchmark_homomorphic_pruning),
    ("Polynomial Leaves", benchmark_polynomial_leaves),
    ("MOAI-Native Conversion", benchmark_moai_conversion),
    ("Bootstrap-Aligned", benchmark_bootstrap_aligned),
    ("Unified Engine (All)", benchmark_unified_engine),
]


# =============================================================================
# Result Data Structure
# =============================================================================

@dataclass
class BenchmarkResult:
    dataset: str
    task: str
    model_name: str
    framework: str
    num_trees: int
    innovation: str
    # Original model quality (direct XGB/LGB predict)
    original_mse_vs_truth: float
    original_metric: float           # accuracy or R2
    original_metric_name: str        # "accuracy" or "R2"
    # Baseline ModelIR quality
    baseline_mse_vs_truth: float
    baseline_metric: float
    # Conversion fidelity
    conversion_mse: float            # ModelIR vs original model
    # Innovation quality
    innovation_mse_vs_truth: float
    innovation_metric: float
    # Innovation vs baseline comparison
    innovation_vs_baseline_mse: float
    accuracy_preserved_pct: float
    # Classification-specific
    baseline_auc: Optional[float] = None
    innovation_auc: Optional[float] = None
    original_auc: Optional[float] = None
    # Extra innovation-specific metrics
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    time_ms: float = 0.0


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_real_model_benchmark():
    print("=" * 100)
    print("  REAL MODEL BENCHMARK FOR FHE-GBDT INNOVATIONS")
    print("  Using real XGBoost/LightGBM models on real sklearn datasets")
    print("=" * 100)

    datasets = load_datasets()
    all_results: List[BenchmarkResult] = []
    errors = []

    for ds in datasets:
        ds_name = ds['name']
        task = ds['task']
        X, y = ds['X'], ds['y']
        n_feat = ds['num_features']
        is_clf = (task == 'classification')

        # 70/30 train-test split with fixed seed
        split_idx = int(0.7 * len(X))
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(X))
        X_shuffled, y_shuffled = X[perm], y[perm]
        X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
        y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]

        print(f"\n{'=' * 100}")
        print(f"  DATASET: {ds_name}  |  Task: {task}  |  "
              f"N={len(X)}, D={n_feat}, Train={len(X_train)}, Test={len(X_test)}")
        print(f"{'=' * 100}")

        # Train all model configurations
        trained_models = train_and_convert(X_train, y_train, n_feat, task)

        for model_name, orig_model, model_ir, is_classifier, framework in trained_models:
            n_trees = len(model_ir.trees)

            # --- Verify conversion fidelity ---
            # XGBoost internally uses float32 for split comparisons, so
            # we must cast features to float32 for apples-to-apples verification.
            # LightGBM uses float64 thresholds, but its <= convention vs our <
            # can cause negligible differences at exact boundary values.
            original_raw = get_original_raw_predictions(
                orig_model, X_test, is_classifier, framework
            )
            if framework == 'xgboost':
                # XGBoost uses float32 for split comparisons internally
                X_verify = X_test.astype(np.float32)
                use_leq_verify = False
            else:
                # LightGBM uses float64 thresholds with <= convention
                X_verify = X_test
                use_leq_verify = True
            modelir_verify = predict_standard(model_ir, X_verify, use_leq=use_leq_verify)
            conversion_mse = compute_mse(
                np.asarray(original_raw, dtype=np.float64),
                np.asarray(modelir_verify, dtype=np.float64)
            )

            # Baseline uses float64 with '<' consistently (same convention as
            # innovations, which matches accuracy_benchmark.py predict_standard)
            modelir_raw = predict_standard(model_ir, X_test)

            # Compute original model quality metrics
            if is_clf:
                orig_metric = compute_accuracy(y_test, original_raw)
                orig_metric_name = "accuracy"
                orig_auc = compute_auc(y_test, original_raw)
            else:
                orig_metric = compute_r2(y_test, original_raw)
                orig_metric_name = "R2"
                orig_auc = None
            orig_mse = compute_mse(y_test, original_raw)

            # Baseline ModelIR quality
            baseline_preds = modelir_raw  # predict_standard result
            if is_clf:
                base_metric = compute_accuracy(y_test, baseline_preds)
                base_auc = compute_auc(y_test, baseline_preds)
            else:
                base_metric = compute_r2(y_test, baseline_preds)
                base_auc = None
            base_mse = compute_mse(y_test, baseline_preds)

            # Print model header
            conv_status = "OK" if conversion_mse < 1e-6 else f"WARN ({conversion_mse:.2e})"
            print(f"\n  Model: {model_name} ({framework}, {n_trees} trees)")
            print(f"  Conversion fidelity MSE: {conversion_mse:.2e}  [{conv_status}]")
            print(f"  Original {orig_metric_name}: {orig_metric:.4f}  |  "
                  f"Baseline {orig_metric_name}: {base_metric:.4f}  |  "
                  f"Baseline MSE: {base_mse:.4f}")
            if is_clf:
                print(f"  Original AUC: {orig_auc:.4f}  |  Baseline AUC: {base_auc:.4f}")

            assert conversion_mse < 1e-6, (
                f"ModelIR conversion failed for {model_name}: "
                f"MSE={conversion_mse:.2e} exceeds 1e-6"
            )

            # Print innovation table header
            if is_clf:
                print(f"  {'Innovation':<28} {'Acc':>7} {'AUC':>7} "
                      f"{'MSEvTruth':>10} {'MSEvBase':>10} "
                      f"{'Preserved%':>10} {'Time(ms)':>10}")
                print(f"  {'_'*28} {'_'*7} {'_'*7} "
                      f"{'_'*10} {'_'*10} "
                      f"{'_'*10} {'_'*10}")
            else:
                print(f"  {'Innovation':<28} {'R2':>8} "
                      f"{'MSEvTruth':>10} {'MSEvBase':>10} "
                      f"{'Preserved%':>10} {'Time(ms)':>10}")
                print(f"  {'_'*28} {'_'*8} "
                      f"{'_'*10} {'_'*10} "
                      f"{'_'*10} {'_'*10}")

            # --- Benchmark each innovation ---
            for innov_name, innov_fn in INNOVATIONS:
                try:
                    preds, _, elapsed, extra = innov_fn(
                        model_ir, X_train, y_train, X_test, y_test
                    )

                    # Innovation quality vs ground truth
                    innov_mse_truth = compute_mse(y_test, preds)
                    if is_clf:
                        innov_metric = compute_accuracy(y_test, preds)
                        innov_auc = compute_auc(y_test, preds)
                    else:
                        innov_metric = compute_r2(y_test, preds)
                        innov_auc = None

                    # Innovation vs baseline comparison
                    innov_vs_base_mse = compute_mse(baseline_preds, preds)

                    # Accuracy preservation: penalize only degradation
                    if base_mse > 1e-10:
                        degradation = max(0, innov_mse_truth - base_mse) / base_mse
                        preserved = max(0.0, (1 - degradation)) * 100
                    else:
                        preserved = 100.0 if innov_mse_truth < 1e-10 else 0.0

                    result = BenchmarkResult(
                        dataset=ds_name,
                        task=task,
                        model_name=model_name,
                        framework=framework,
                        num_trees=n_trees,
                        innovation=innov_name,
                        original_mse_vs_truth=round(orig_mse, 6),
                        original_metric=round(orig_metric, 6),
                        original_metric_name=orig_metric_name,
                        baseline_mse_vs_truth=round(base_mse, 6),
                        baseline_metric=round(base_metric, 6),
                        conversion_mse=round(conversion_mse, 10),
                        innovation_mse_vs_truth=round(innov_mse_truth, 6),
                        innovation_metric=round(innov_metric, 6),
                        innovation_vs_baseline_mse=round(innov_vs_base_mse, 6),
                        accuracy_preserved_pct=round(preserved, 2),
                        baseline_auc=round(base_auc, 4) if base_auc is not None else None,
                        innovation_auc=round(innov_auc, 4) if innov_auc is not None else None,
                        original_auc=round(orig_auc, 4) if orig_auc is not None else None,
                        extra_metrics=extra,
                        time_ms=round(elapsed, 2),
                    )
                    all_results.append(result)

                    # Print row
                    if is_clf:
                        print(
                            f"  {innov_name:<28} {innov_metric:>7.4f} "
                            f"{innov_auc:>7.4f} "
                            f"{innov_mse_truth:>10.4f} {innov_vs_base_mse:>10.6f} "
                            f"{preserved:>9.1f}% {elapsed:>9.1f}ms"
                        )
                    else:
                        print(
                            f"  {innov_name:<28} {innov_metric:>8.4f} "
                            f"{innov_mse_truth:>10.4f} {innov_vs_base_mse:>10.6f} "
                            f"{preserved:>9.1f}% {elapsed:>9.1f}ms"
                        )

                except Exception as e:
                    err_msg = str(e)[:80]
                    print(f"  {innov_name:<28} ERROR: {err_msg}")
                    errors.append({
                        'dataset': ds_name,
                        'model': model_name,
                        'innovation': innov_name,
                        'error': str(e),
                    })

    # Generate summary and save
    print_summary(all_results, errors)
    save_results(all_results, errors)
    return all_results


# =============================================================================
# Summary Reporting
# =============================================================================

def print_summary(results: List[BenchmarkResult], errors: List[Dict]):
    print("\n\n")
    print("=" * 100)
    print("  RESULTS SUMMARY")
    print("=" * 100)

    if not results:
        print("  No successful results to summarize.")
        return

    # --------------------------------------------------
    # 1. Per-Innovation Aggregate
    # --------------------------------------------------
    print("\n  1. PER-INNOVATION ACCURACY (averaged across all models and datasets)")
    print(f"  {'_' * 96}")
    print(f"  {'Innovation':<28} {'Avg Metric':>10} {'Avg MSE':>10} "
          f"{'Avg MSEvBase':>12} {'Preserved%':>10} {'Avg ms':>8} {'Status':>8}")
    print(f"  {'_'*28} {'_'*10} {'_'*10} {'_'*12} {'_'*10} {'_'*8} {'_'*8}")

    innovations_seen = sorted(set(r.innovation for r in results))
    for innov in innovations_seen:
        ir = [r for r in results if r.innovation == innov]
        avg_metric = np.mean([r.innovation_metric for r in ir])
        avg_mse = np.mean([r.innovation_mse_vs_truth for r in ir])
        avg_mse_base = np.mean([r.innovation_vs_baseline_mse for r in ir])
        avg_preserved = np.mean([r.accuracy_preserved_pct for r in ir])
        avg_time = np.mean([r.time_ms for r in ir])
        status = "PASS" if avg_preserved > 80 else "WARN" if avg_preserved > 50 else "FAIL"
        print(
            f"  {innov:<28} {avg_metric:>10.4f} {avg_mse:>10.4f} "
            f"{avg_mse_base:>12.6f} {avg_preserved:>9.1f}% {avg_time:>7.1f}ms "
            f"  {status:>6}"
        )

    # --------------------------------------------------
    # 2. Per-Dataset Aggregate
    # --------------------------------------------------
    print(f"\n  2. PER-DATASET ACCURACY (averaged across all models and innovations)")
    print(f"  {'_' * 96}")
    print(f"  {'Dataset':<24} {'Task':<15} {'Avg Orig':>10} {'Avg Baseline':>12} "
          f"{'Avg Innov':>10} {'Preserved%':>10}")
    print(f"  {'_'*24} {'_'*15} {'_'*10} {'_'*12} {'_'*10} {'_'*10}")

    datasets_seen = sorted(set(r.dataset for r in results))
    for ds in datasets_seen:
        dr = [r for r in results if r.dataset == ds]
        task = dr[0].task
        avg_orig = np.mean([r.original_metric for r in dr])
        avg_base = np.mean([r.baseline_metric for r in dr])
        avg_innov = np.mean([r.innovation_metric for r in dr])
        avg_pres = np.mean([r.accuracy_preserved_pct for r in dr])
        print(
            f"  {ds:<24} {task:<15} {avg_orig:>10.4f} {avg_base:>12.4f} "
            f"{avg_innov:>10.4f} {avg_pres:>9.1f}%"
        )

    # --------------------------------------------------
    # 3. Per-Model Aggregate
    # --------------------------------------------------
    print(f"\n  3. PER-MODEL ACCURACY (averaged across all datasets and innovations)")
    print(f"  {'_' * 96}")
    print(f"  {'Model':<20} {'Framework':<10} {'Trees':>6} {'Avg Orig':>10} "
          f"{'Avg Baseline':>12} {'Avg Innov':>10} {'Preserved%':>10}")
    print(f"  {'_'*20} {'_'*10} {'_'*6} {'_'*10} {'_'*12} {'_'*10} {'_'*10}")

    models_seen = sorted(set(r.model_name for r in results))
    for model in models_seen:
        mr = [r for r in results if r.model_name == model]
        fw = mr[0].framework
        nt = mr[0].num_trees
        avg_orig = np.mean([r.original_metric for r in mr])
        avg_base = np.mean([r.baseline_metric for r in mr])
        avg_innov = np.mean([r.innovation_metric for r in mr])
        avg_pres = np.mean([r.accuracy_preserved_pct for r in mr])
        print(
            f"  {model:<20} {fw:<10} {nt:>6} {avg_orig:>10.4f} "
            f"{avg_base:>12.4f} {avg_innov:>10.4f} {avg_pres:>9.1f}%"
        )

    # --------------------------------------------------
    # 4. Innovation-Specific Highlights
    # --------------------------------------------------
    print(f"\n  4. INNOVATION-SPECIFIC METRICS")
    print(f"  {'_' * 96}")

    # MOAI rotation savings
    moai = [r for r in results if r.innovation == "MOAI-Native Conversion"]
    if moai:
        avg_sav = np.mean([r.extra_metrics.get("rotation_savings_pct", 0) for r in moai])
        avg_spd = np.mean([r.extra_metrics.get("speedup_factor", 1) for r in moai])
        print(f"  MOAI Conversion: Avg rotation savings = {avg_sav:.1f}%, "
              f"Avg speedup = {avg_spd:.1f}x")

    # Pruning
    prune = [r for r in results if r.innovation == "Homomorphic Pruning"]
    if prune:
        avg_pr = np.mean([r.extra_metrics.get("computation_saved_pct", 0) for r in prune])
        avg_act = np.mean([r.extra_metrics.get("num_active_trees", 0) for r in prune])
        print(f"  Homomorphic Pruning: Avg computation saved = {avg_pr:.1f}%, "
              f"Avg active trees = {avg_act:.0f}")

    # Polynomial
    poly = [r for r in results if r.innovation == "Polynomial Leaves"]
    if poly:
        avg_cov = np.mean([r.extra_metrics.get("coverage_pct", 0) for r in poly])
        avg_lr2 = np.mean([r.extra_metrics.get("avg_leaf_r2", 0) for r in poly])
        print(f"  Polynomial Leaves: Avg leaf coverage = {avg_cov:.1f}%, "
              f"Avg leaf R2 = {avg_lr2:.4f}")

    # Gradient noise
    noise = [r for r in results if r.innovation == "Gradient-Aware Noise"]
    if noise:
        avg_bits = np.mean([r.extra_metrics.get("avg_precision_bits", 0) for r in noise])
        avg_mae = np.mean([r.extra_metrics.get("encode_decode_mae", 0) for r in noise])
        print(f"  Gradient-Aware Noise: Avg precision = {avg_bits:.1f} bits, "
              f"Avg encode/decode MAE = {avg_mae:.6f}")

    # Bootstrap
    boot = [r for r in results if r.innovation == "Bootstrap-Aligned"]
    if boot:
        avg_ch = np.mean([r.extra_metrics.get("num_chunks", 0) for r in boot])
        n_boot = sum(1 for r in boot if r.extra_metrics.get("needs_bootstrap", False))
        print(f"  Bootstrap-Aligned: Avg chunks = {avg_ch:.1f}, "
              f"Models needing bootstrap = {n_boot}/{len(boot)}")

    # --------------------------------------------------
    # 5. Conversion Fidelity Summary
    # --------------------------------------------------
    print(f"\n  5. CONVERSION FIDELITY (ModelIR vs original model)")
    print(f"  {'_' * 96}")
    # Group by model to show one conversion MSE per model-dataset pair
    seen = set()
    for r in results:
        key = (r.dataset, r.model_name)
        if key not in seen:
            seen.add(key)
            status = "EXACT" if r.conversion_mse < 1e-10 else "OK" if r.conversion_mse < 1e-6 else "WARN"
            print(f"  {r.dataset:<24} {r.model_name:<20} "
                  f"MSE={r.conversion_mse:.2e}  [{status}]")

    # --------------------------------------------------
    # 6. Classification AUC Summary
    # --------------------------------------------------
    clf_results = [r for r in results if r.task == 'classification']
    if clf_results:
        print(f"\n  6. CLASSIFICATION AUC SUMMARY")
        print(f"  {'_' * 96}")
        print(f"  {'Dataset':<20} {'Model':<20} {'Innovation':<28} "
              f"{'Orig AUC':>9} {'Base AUC':>9} {'Innov AUC':>9}")
        print(f"  {'_'*20} {'_'*20} {'_'*28} {'_'*9} {'_'*9} {'_'*9}")
        for r in clf_results:
            if r.original_auc is not None:
                print(
                    f"  {r.dataset:<20} {r.model_name:<20} {r.innovation:<28} "
                    f"{r.original_auc:>9.4f} {r.baseline_auc:>9.4f} "
                    f"{r.innovation_auc:>9.4f}"
                )

    # --------------------------------------------------
    # Overall
    # --------------------------------------------------
    if results:
        overall_preserved = np.mean([r.accuracy_preserved_pct for r in results])
        print(f"\n  OVERALL: Avg accuracy preserved = {overall_preserved:.1f}%")
        print(f"  Total benchmarks: {len(results)} successful, {len(errors)} errors")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    {e['dataset']} / {e['model']} / {e['innovation']}: "
                  f"{e['error'][:80]}")


# =============================================================================
# Save Results
# =============================================================================

def save_results(results: List[BenchmarkResult], errors: List[Dict]):
    report_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(report_dir, exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, set):
                return list(obj)
            return super().default(obj)

    output = {
        "metadata": {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "seed": SEED,
            "train_ratio": 0.7,
            "num_datasets": len(set(r.dataset for r in results)),
            "num_models": len(set(r.model_name for r in results)),
            "num_innovations": len(set(r.innovation for r in results)),
            "total_benchmarks": len(results),
            "total_errors": len(errors),
        },
        "results": [asdict(r) for r in results],
        "errors": errors,
        "summary": {
            "overall_accuracy_preserved_pct": round(
                float(np.mean([r.accuracy_preserved_pct for r in results])), 2
            ) if results else 0.0,
            "per_innovation": {},
            "per_dataset": {},
            "per_model": {},
        }
    }

    # Fill in summary aggregates
    for innov in sorted(set(r.innovation for r in results)):
        ir = [r for r in results if r.innovation == innov]
        output["summary"]["per_innovation"][innov] = {
            "avg_preserved_pct": round(float(np.mean([r.accuracy_preserved_pct for r in ir])), 2),
            "avg_innovation_metric": round(float(np.mean([r.innovation_metric for r in ir])), 4),
            "avg_time_ms": round(float(np.mean([r.time_ms for r in ir])), 2),
            "count": len(ir),
        }

    for ds in sorted(set(r.dataset for r in results)):
        dr = [r for r in results if r.dataset == ds]
        output["summary"]["per_dataset"][ds] = {
            "avg_preserved_pct": round(float(np.mean([r.accuracy_preserved_pct for r in dr])), 2),
            "avg_baseline_metric": round(float(np.mean([r.baseline_metric for r in dr])), 4),
            "avg_innovation_metric": round(float(np.mean([r.innovation_metric for r in dr])), 4),
            "count": len(dr),
        }

    for model in sorted(set(r.model_name for r in results)):
        mr = [r for r in results if r.model_name == model]
        output["summary"]["per_model"][model] = {
            "avg_preserved_pct": round(float(np.mean([r.accuracy_preserved_pct for r in mr])), 2),
            "avg_innovation_metric": round(float(np.mean([r.innovation_metric for r in mr])), 4),
            "framework": mr[0].framework,
            "num_trees": mr[0].num_trees,
            "count": len(mr),
        }

    json_path = os.path.join(report_dir, 'real_model_benchmark.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\n  Results saved to: {json_path}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_real_model_benchmark()
