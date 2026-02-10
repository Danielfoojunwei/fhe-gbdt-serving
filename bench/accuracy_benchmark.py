#!/usr/bin/env python3
"""
Comprehensive Accuracy Benchmark for Novel FHE-GBDT Innovations

Benchmarks 6 model configurations x 3 datasets across all innovations:

Model Configurations:
  1. XGBoost-style (non-oblivious, irregular splits)
  2. LightGBM-style (leaf-wise, deeper trees)
  3. CatBoost-style (oblivious trees, symmetric splits)
  4. Deep Ensemble (many shallow trees)
  5. Wide Ensemble (few deep trees)
  6. Mixed Ensemble (varied depth/feature usage)

Datasets:
  1. Classification (binary, 30 features — breast cancer-like)
  2. Regression (10 features — diabetes-like)
  3. High-dimensional (50 features — fraud detection-like)

Innovations benchmarked:
  1. Leaf-Centric Encoding
  2. Gradient-Informed Noise Allocation
  3. Homomorphic Ensemble Pruning
  4. Polynomial Leaf Functions
  5. MOAI-Native Tree Conversion (with accuracy-aware retuning)
  6. Bootstrap-Aligned Architecture
  7. Unified Architecture (all innovations combined)
"""

import sys
import os
import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.compiler.ir import ModelIR, TreeIR, TreeNode
from services.innovations.leaf_centric import LeafCentricEncoder, LeafIndicatorComputer
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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Generation
# =============================================================================

def generate_classification_dataset(n_samples=500, n_features=30, seed=42):
    """Binary classification dataset (breast cancer-like)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Non-linear decision boundary
    score = (
        0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.8 * X[:, 2]
        + 0.2 * X[:, 0] * X[:, 1]
        - 0.4 * X[:, 3] ** 2
        + 0.1 * rng.randn(n_samples)
    )
    y = (score > 0).astype(np.float64)
    return X, y, "classification_30f"


def generate_regression_dataset(n_samples=500, n_features=10, seed=42):
    """Regression dataset (diabetes-like)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = (
        2.0 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2]
        + 0.5 * X[:, 3] * X[:, 4]
        + 0.3 * np.sin(X[:, 5] * 2)
        + 0.2 * rng.randn(n_samples)
    )
    return X, y, "regression_10f"


def generate_highdim_dataset(n_samples=500, n_features=50, seed=42):
    """High-dimensional dataset (fraud detection-like)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Sparse signal — only first 8 features matter
    score = (
        1.0 * X[:, 0] - 0.7 * X[:, 1] + 0.5 * X[:, 2]
        + 0.3 * X[:, 3] + 0.2 * X[:, 4]
        - 0.1 * X[:, 5] + 0.4 * X[:, 6] - 0.6 * X[:, 7]
        + 0.15 * rng.randn(n_samples)
    )
    y = score
    return X, y, "highdim_50f"


# =============================================================================
# Model Generation
# =============================================================================

def _build_tree(tree_id, depth, num_features, rng, oblivious=False):
    """Build a single tree with given properties."""
    nodes = {}
    node_id = 0

    # For oblivious trees: one feature per level
    level_features = []
    level_thresholds = []
    if oblivious:
        for d in range(depth):
            level_features.append(rng.randint(0, num_features))
            level_thresholds.append(float(rng.uniform(-1, 1)))

    for d in range(depth):
        num_nodes_at_depth = 2 ** d
        for i in range(num_nodes_at_depth):
            if d < depth - 1:
                if oblivious:
                    feat = level_features[d]
                    thresh = level_thresholds[d]
                else:
                    feat = rng.randint(0, num_features)
                    thresh = float(rng.uniform(-1.5, 1.5))
                nodes[node_id] = TreeNode(
                    node_id=node_id,
                    feature_index=feat,
                    threshold=thresh,
                    left_child_id=node_id * 2 + 1,
                    right_child_id=node_id * 2 + 2,
                    depth=d
                )
            else:
                nodes[node_id] = TreeNode(
                    node_id=node_id,
                    leaf_value=float(rng.uniform(-0.5, 0.5)),
                    depth=d
                )
            node_id += 1

    return TreeIR(tree_id=tree_id, nodes=nodes, root_id=0, max_depth=depth)


def generate_xgboost_model(num_features, seed=42):
    """XGBoost-style: non-oblivious, depth 4-6, 20 trees."""
    rng = np.random.RandomState(seed)
    trees = []
    for i in range(20):
        depth = rng.choice([4, 5, 6])
        trees.append(_build_tree(i, depth, num_features, rng, oblivious=False))
    return ModelIR(model_type="xgboost", trees=trees, num_features=num_features, base_score=0.5)


def generate_lightgbm_model(num_features, seed=42):
    """LightGBM-style: leaf-wise deeper trees, 15 trees."""
    rng = np.random.RandomState(seed)
    trees = []
    for i in range(15):
        depth = rng.choice([5, 6, 7])
        trees.append(_build_tree(i, depth, num_features, rng, oblivious=False))
    return ModelIR(model_type="lightgbm", trees=trees, num_features=num_features, base_score=0.0)


def generate_catboost_model(num_features, seed=42):
    """CatBoost-style: oblivious trees, depth 4-6, 25 trees."""
    rng = np.random.RandomState(seed)
    trees = []
    for i in range(25):
        depth = rng.choice([4, 5, 6])
        trees.append(_build_tree(i, depth, num_features, rng, oblivious=True))
    return ModelIR(model_type="catboost", trees=trees, num_features=num_features, base_score=0.0)


def generate_deep_ensemble(num_features, seed=42):
    """Deep ensemble: 50 shallow trees (depth 3)."""
    rng = np.random.RandomState(seed)
    trees = [_build_tree(i, 3, num_features, rng) for i in range(50)]
    return ModelIR(model_type="xgboost", trees=trees, num_features=num_features, base_score=0.5)


def generate_wide_ensemble(num_features, seed=42):
    """Wide ensemble: 8 deep trees (depth 8)."""
    rng = np.random.RandomState(seed)
    trees = [_build_tree(i, 8, num_features, rng) for i in range(8)]
    return ModelIR(model_type="xgboost", trees=trees, num_features=num_features, base_score=0.5)


def generate_mixed_ensemble(num_features, seed=42):
    """Mixed ensemble: varied tree depths and configurations."""
    rng = np.random.RandomState(seed)
    trees = []
    for i in range(30):
        depth = rng.choice([3, 4, 5, 6, 7])
        oblivious = rng.random() > 0.5
        trees.append(_build_tree(i, depth, num_features, rng, oblivious=oblivious))
    return ModelIR(model_type="xgboost", trees=trees, num_features=num_features, base_score=0.5)


# =============================================================================
# Metrics
# =============================================================================

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def compute_mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_accuracy(y_true, y_pred, threshold=0.5):
    """Classification accuracy using threshold."""
    pred_labels = (y_pred > threshold).astype(float)
    return float(np.mean(pred_labels == y_true))


def predict_standard(model_ir, X):
    """Standard tree traversal prediction (baseline)."""
    predictions = np.full(X.shape[0], model_ir.base_score)
    for tree in model_ir.trees:
        for i in range(X.shape[0]):
            node = tree.nodes.get(tree.root_id)
            while node is not None:
                if node.leaf_value is not None:
                    predictions[i] += node.leaf_value
                    break
                if X[i, node.feature_index] < node.threshold:
                    node = tree.nodes.get(node.left_child_id)
                else:
                    node = tree.nodes.get(node.right_child_id)
    return predictions


# =============================================================================
# Per-Innovation Accuracy Benchmarks
# =============================================================================

@dataclass
class InnovationResult:
    innovation: str
    model_name: str
    dataset_name: str
    baseline_r2: float
    innovation_r2: float
    r2_delta: float
    baseline_mse: float
    innovation_mse: float
    mse_delta_pct: float
    accuracy_preserved_pct: float
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    time_ms: float = 0.0


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
    avg_bits = np.mean(list(precision_bits.values()))

    return preds, baseline, elapsed, {
        "avg_precision_bits": round(avg_bits, 2),
        "min_precision_bits": min(precision_bits.values()),
        "max_precision_bits": max(precision_bits.values()),
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

    # Compute per-tree outputs
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
    config = PolynomialLeafConfig(max_degree=2, min_samples_for_poly=5, r2_threshold=0.05)
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
    # Use limited validation data to prevent overfitting during retuning
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

    # Predict with oblivious trees on test set
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


# =============================================================================
# Main Benchmark Runner
# =============================================================================

INNOVATIONS = [
    ("Leaf-Centric Encoding", benchmark_leaf_centric),
    ("Gradient-Aware Noise", benchmark_gradient_noise),
    ("Homomorphic Pruning", benchmark_homomorphic_pruning),
    ("Polynomial Leaves", benchmark_polynomial_leaves),
    ("MOAI-Native Conversion", benchmark_moai_conversion),
    ("Bootstrap-Aligned", benchmark_bootstrap_aligned),
    ("Unified Engine (All)", benchmark_unified_engine),
]

MODEL_GENERATORS = [
    ("XGBoost-style", generate_xgboost_model),
    ("LightGBM-style", generate_lightgbm_model),
    ("CatBoost-style", generate_catboost_model),
    ("Deep-Ensemble", generate_deep_ensemble),
    ("Wide-Ensemble", generate_wide_ensemble),
    ("Mixed-Ensemble", generate_mixed_ensemble),
]

DATASET_GENERATORS = [
    generate_classification_dataset,
    generate_regression_dataset,
    generate_highdim_dataset,
]


def run_accuracy_benchmarks():
    print("=" * 80)
    print("COMPREHENSIVE ACCURACY BENCHMARK FOR NOVEL FHE-GBDT INNOVATIONS")
    print("=" * 80)
    print()

    all_results: List[InnovationResult] = []

    for ds_gen in DATASET_GENERATORS:
        X, y, ds_name = ds_gen()
        n_features = X.shape[1]

        # Train/test split
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"\n{'─' * 80}")
        print(f"DATASET: {ds_name}  (N={len(X)}, D={n_features}, train={split}, test={len(X)-split})")
        print(f"{'─' * 80}")

        for model_name, model_gen in MODEL_GENERATORS:
            model_ir = model_gen(n_features)

            baseline_preds = predict_standard(model_ir, X_test)
            baseline_r2 = compute_r2(y_test, baseline_preds)
            baseline_mse = compute_mse(y_test, baseline_preds)

            print(f"\n  Model: {model_name} ({len(model_ir.trees)} trees)")
            print(f"  Baseline R²={baseline_r2:.4f}  MSE={baseline_mse:.4f}")
            print(f"  {'Innovation':<28} {'R²':>8} {'ΔR²':>8} {'MSE':>10} {'ΔMSE%':>8} {'Preserved%':>10} {'Time(ms)':>10}")
            print(f"  {'─'*28} {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*10} {'─'*10}")

            for innov_name, innov_fn in INNOVATIONS:
                try:
                    preds, _, elapsed, extra = innov_fn(
                        model_ir, X_train, y_train, X_test, y_test
                    )
                    innov_r2 = compute_r2(y_test, preds)
                    innov_mse = compute_mse(y_test, preds)

                    r2_delta = innov_r2 - baseline_r2
                    mse_delta_pct = ((innov_mse - baseline_mse) / max(abs(baseline_mse), 1e-10)) * 100

                    # Accuracy preservation: only penalize degradation (higher MSE)
                    # If innovation has lower MSE than baseline, it's at least 100% preserved
                    if baseline_mse > 1e-10:
                        degradation = max(0, innov_mse - baseline_mse) / baseline_mse
                        preserved = max(0, (1 - degradation)) * 100
                    else:
                        preserved = 100.0 if innov_mse < 1e-10 else 0.0

                    result = InnovationResult(
                        innovation=innov_name,
                        model_name=model_name,
                        dataset_name=ds_name,
                        baseline_r2=round(baseline_r2, 6),
                        innovation_r2=round(innov_r2, 6),
                        r2_delta=round(r2_delta, 6),
                        baseline_mse=round(baseline_mse, 6),
                        innovation_mse=round(innov_mse, 6),
                        mse_delta_pct=round(mse_delta_pct, 4),
                        accuracy_preserved_pct=round(preserved, 2),
                        extra_metrics=extra,
                        time_ms=round(elapsed, 2),
                    )
                    all_results.append(result)

                    print(
                        f"  {innov_name:<28} {innov_r2:>8.4f} {r2_delta:>+8.4f} "
                        f"{innov_mse:>10.4f} {mse_delta_pct:>+8.2f} "
                        f"{preserved:>9.1f}% {elapsed:>9.1f}ms"
                    )

                except Exception as e:
                    print(f"  {innov_name:<28} {'ERROR':>8}  {str(e)[:50]}")
                    all_results.append(InnovationResult(
                        innovation=innov_name,
                        model_name=model_name,
                        dataset_name=ds_name,
                        baseline_r2=round(baseline_r2, 6),
                        innovation_r2=0.0,
                        r2_delta=0.0,
                        baseline_mse=round(baseline_mse, 6),
                        innovation_mse=0.0,
                        mse_delta_pct=0.0,
                        accuracy_preserved_pct=0.0,
                        extra_metrics={"error": str(e)},
                    ))

    # Generate summary
    print_summary(all_results)
    save_results(all_results)
    return all_results


def print_summary(results: List[InnovationResult]):
    print("\n")
    print("=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    # 1. Per-Innovation aggregate across all models/datasets
    print("\n1. ACCURACY BY INNOVATION (averaged across 6 models x 3 datasets)")
    print(f"{'─' * 100}")
    print(
        f"  {'Innovation':<28} {'Avg R²':>8} {'Avg ΔR²':>9} {'Avg MSE':>10} "
        f"{'Avg ΔMSE%':>10} {'Preserved%':>10} {'Avg ms':>8} {'Status':>8}"
    )
    print(f"  {'─'*28} {'─'*8} {'─'*9} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

    innovations = sorted(set(r.innovation for r in results))
    for innov in innovations:
        innov_results = [r for r in results if r.innovation == innov and "error" not in r.extra_metrics]
        if not innov_results:
            print(f"  {innov:<28} {'ERROR':>8}")
            continue
        avg_r2 = np.mean([r.innovation_r2 for r in innov_results])
        avg_delta_r2 = np.mean([r.r2_delta for r in innov_results])
        avg_mse = np.mean([r.innovation_mse for r in innov_results])
        avg_delta_mse = np.mean([r.mse_delta_pct for r in innov_results])
        avg_preserved = np.mean([r.accuracy_preserved_pct for r in innov_results])
        avg_time = np.mean([r.time_ms for r in innov_results])
        status = "PASS" if avg_preserved > 80 else "WARN" if avg_preserved > 50 else "FAIL"
        print(
            f"  {innov:<28} {avg_r2:>8.4f} {avg_delta_r2:>+9.4f} {avg_mse:>10.4f} "
            f"{avg_delta_mse:>+10.2f} {avg_preserved:>9.1f}% {avg_time:>7.1f}ms "
            f"{'  ' + status:>8}"
        )

    # 2. Per-Dataset aggregate
    print(f"\n2. ACCURACY BY DATASET (averaged across 6 models x 7 innovations)")
    print(f"{'─' * 100}")
    print(
        f"  {'Dataset':<24} {'Avg Baseline R²':>15} {'Avg Innovation R²':>18} "
        f"{'Avg ΔR²':>9} {'Avg Preserved%':>14}"
    )
    print(f"  {'─'*24} {'─'*15} {'─'*18} {'─'*9} {'─'*14}")

    datasets = sorted(set(r.dataset_name for r in results))
    for ds in datasets:
        ds_results = [r for r in results if r.dataset_name == ds and "error" not in r.extra_metrics]
        if not ds_results:
            continue
        avg_base_r2 = np.mean([r.baseline_r2 for r in ds_results])
        avg_innov_r2 = np.mean([r.innovation_r2 for r in ds_results])
        avg_delta = np.mean([r.r2_delta for r in ds_results])
        avg_preserved = np.mean([r.accuracy_preserved_pct for r in ds_results])
        print(
            f"  {ds:<24} {avg_base_r2:>15.4f} {avg_innov_r2:>18.4f} "
            f"{avg_delta:>+9.4f} {avg_preserved:>13.1f}%"
        )

    # 3. Per-Model aggregate
    print(f"\n3. ACCURACY BY MODEL TYPE (averaged across 3 datasets x 7 innovations)")
    print(f"{'─' * 100}")
    print(
        f"  {'Model':<20} {'Trees':>6} {'Avg Baseline R²':>15} {'Avg Innovation R²':>18} "
        f"{'Avg ΔR²':>9} {'Avg Preserved%':>14}"
    )
    print(f"  {'─'*20} {'─'*6} {'─'*15} {'─'*18} {'─'*9} {'─'*14}")

    models = sorted(set(r.model_name for r in results))
    for model in models:
        model_results = [r for r in results if r.model_name == model and "error" not in r.extra_metrics]
        if not model_results:
            continue
        avg_base_r2 = np.mean([r.baseline_r2 for r in model_results])
        avg_innov_r2 = np.mean([r.innovation_r2 for r in model_results])
        avg_delta = np.mean([r.r2_delta for r in model_results])
        avg_preserved = np.mean([r.accuracy_preserved_pct for r in model_results])
        # Get tree count
        sample_r = model_results[0]
        print(
            f"  {model:<20} {'':>6} {avg_base_r2:>15.4f} {avg_innov_r2:>18.4f} "
            f"{avg_delta:>+9.4f} {avg_preserved:>13.1f}%"
        )

    # 4. Innovation-specific highlights
    print(f"\n4. INNOVATION-SPECIFIC METRICS")
    print(f"{'─' * 100}")

    # MOAI Conversion rotation savings
    moai_results = [r for r in results if r.innovation == "MOAI-Native Conversion" and "error" not in r.extra_metrics]
    if moai_results:
        avg_savings = np.mean([r.extra_metrics.get("rotation_savings_pct", 0) for r in moai_results])
        avg_speedup = np.mean([r.extra_metrics.get("speedup_factor", 1) for r in moai_results])
        print(f"  MOAI Conversion: Avg rotation savings = {avg_savings:.1f}%, Avg speedup = {avg_speedup:.1f}x")

    # Pruning savings
    prune_results = [r for r in results if r.innovation == "Homomorphic Pruning" and "error" not in r.extra_metrics]
    if prune_results:
        avg_prune = np.mean([r.extra_metrics.get("computation_saved_pct", 0) for r in prune_results])
        avg_active = np.mean([r.extra_metrics.get("num_active_trees", 0) for r in prune_results])
        print(f"  Homomorphic Pruning: Avg computation saved = {avg_prune:.1f}%, Avg active trees = {avg_active:.0f}")

    # Polynomial coverage
    poly_results = [r for r in results if r.innovation == "Polynomial Leaves" and "error" not in r.extra_metrics]
    if poly_results:
        avg_coverage = np.mean([r.extra_metrics.get("coverage_pct", 0) for r in poly_results])
        avg_leaf_r2 = np.mean([r.extra_metrics.get("avg_leaf_r2", 0) for r in poly_results])
        print(f"  Polynomial Leaves: Avg leaf coverage = {avg_coverage:.1f}%, Avg leaf R² = {avg_leaf_r2:.4f}")

    # Gradient noise precision
    noise_results = [r for r in results if r.innovation == "Gradient-Aware Noise" and "error" not in r.extra_metrics]
    if noise_results:
        avg_bits = np.mean([r.extra_metrics.get("avg_precision_bits", 0) for r in noise_results])
        avg_enc_mae = np.mean([r.extra_metrics.get("encode_decode_mae", 0) for r in noise_results])
        print(f"  Gradient-Aware Noise: Avg precision = {avg_bits:.1f} bits, Avg encode/decode MAE = {avg_enc_mae:.6f}")

    # Bootstrap analysis
    boot_results = [r for r in results if r.innovation == "Bootstrap-Aligned" and "error" not in r.extra_metrics]
    if boot_results:
        avg_chunks = np.mean([r.extra_metrics.get("num_chunks", 0) for r in boot_results])
        needs_boot = sum(1 for r in boot_results if r.extra_metrics.get("needs_bootstrap", False))
        print(f"  Bootstrap-Aligned: Avg chunks = {avg_chunks:.1f}, Models needing bootstrap = {needs_boot}/{len(boot_results)}")

    # Overall summary
    valid_results = [r for r in results if "error" not in r.extra_metrics]
    if valid_results:
        overall_preserved = np.mean([r.accuracy_preserved_pct for r in valid_results])
        overall_r2_delta = np.mean([r.r2_delta for r in valid_results])
        print(f"\n  OVERALL: Avg accuracy preserved = {overall_preserved:.1f}%, Avg R² delta = {overall_r2_delta:+.4f}")
        print(f"  Total benchmarks run: {len(valid_results)} ({len(results) - len(valid_results)} errors)")


def save_results(results: List[InnovationResult]):
    os.makedirs("bench/reports", exist_ok=True)

    # JSON (handle numpy types)
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
            return super().default(obj)

    json_path = "bench/reports/accuracy_benchmark.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2, cls=NumpyEncoder)

    # Markdown
    md_path = "bench/reports/accuracy_benchmark.md"
    with open(md_path, 'w') as f:
        f.write("# Accuracy Benchmark Results\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total benchmarks**: {len(results)}\n\n")

        f.write("## Per-Innovation Summary\n\n")
        f.write("| Innovation | Avg R² | Avg ΔR² | Avg Preserved% | Status |\n")
        f.write("|-----------|--------|---------|---------------|--------|\n")

        innovations = sorted(set(r.innovation for r in results))
        for innov in innovations:
            innov_results = [r for r in results if r.innovation == innov and "error" not in r.extra_metrics]
            if not innov_results:
                continue
            avg_r2 = np.mean([r.innovation_r2 for r in innov_results])
            avg_delta = np.mean([r.r2_delta for r in innov_results])
            avg_preserved = np.mean([r.accuracy_preserved_pct for r in innov_results])
            status = "PASS" if avg_preserved > 80 else "WARN" if avg_preserved > 50 else "FAIL"
            f.write(f"| {innov} | {avg_r2:.4f} | {avg_delta:+.4f} | {avg_preserved:.1f}% | {status} |\n")

        f.write("\n## Per-Dataset Summary\n\n")
        f.write("| Dataset | Avg Baseline R² | Avg Innovation R² | Avg Preserved% |\n")
        f.write("|---------|----------------|-------------------|---------------|\n")

        datasets = sorted(set(r.dataset_name for r in results))
        for ds in datasets:
            ds_results = [r for r in results if r.dataset_name == ds and "error" not in r.extra_metrics]
            if not ds_results:
                continue
            avg_base = np.mean([r.baseline_r2 for r in ds_results])
            avg_innov = np.mean([r.innovation_r2 for r in ds_results])
            avg_preserved = np.mean([r.accuracy_preserved_pct for r in ds_results])
            f.write(f"| {ds} | {avg_base:.4f} | {avg_innov:.4f} | {avg_preserved:.1f}% |\n")

    print(f"\nReports saved to:")
    print(f"  {json_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    run_accuracy_benchmarks()
