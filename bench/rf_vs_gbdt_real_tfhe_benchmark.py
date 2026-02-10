#!/usr/bin/env python3
"""
CANONICAL BENCHMARK: RF vs GBDT under Real TFHE Encryption

Validates C3 (Independent Noise Channels for Random Forests) end-to-end
using Concrete ML's actual TFHE encryption. NO simulation, NO mocks.

Theory being validated:
  RF noise scales as O(D + log T) because trees are independent.
  GBDT noise scales as O(T × D) because trees are sequentially dependent.
  Therefore RF should maintain accuracy better under FHE quantization,
  especially as tree count increases.

Methodology:
  1. Train both RandomForestClassifier and XGBClassifier via Concrete ML
  2. Compare plain, simulated-FHE, and real-FHE accuracy
  3. Sweep tree counts (10, 25, 50, 100) and bit widths (3, 4, 6)
  4. Measure accuracy degradation: (plain_acc - fhe_acc) / plain_acc
  5. Compare RF vs GBDT degradation to validate noise theory

Requires: pip install concrete-ml scikit-learn xgboost numpy
"""
import sys
import os
import time
import json
import warnings
import traceback
import numpy as np

from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# Concrete ML - required (not optional)
from concrete.ml.sklearn import (
    RandomForestClassifier as FHERFClassifier,
    XGBClassifier as FHEXGBClassifier,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.innovations.model_aware_fhe import IndependentNoiseOptimizer


# =============================================================================
# Dataset Loading
# =============================================================================

def _binarize_iris():
    from sklearn.datasets import load_iris
    data = load_iris()
    data.target = (data.target == 0).astype(int)
    return data


def _binarize_wine():
    from sklearn.datasets import load_wine
    data = load_wine()
    data.target = (data.target == 0).astype(int)
    return data


DATASETS = {
    "breast_cancer": {
        "loader": load_breast_cancer,
        "description": "569 samples, 30 features, binary",
    },
    "iris_binary": {
        "loader": _binarize_iris,
        "description": "150 samples, 4 features, binary (class 0 vs rest)",
    },
    "wine_binary": {
        "loader": _binarize_wine,
        "description": "178 samples, 13 features, binary (class 0 vs rest)",
    },
}


def load_dataset(name):
    info = DATASETS[name]
    data = info["loader"]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


# =============================================================================
# Core benchmark: RF vs GBDT under real TFHE
# =============================================================================

def benchmark_rf_vs_gbdt_fhe(
    n_fhe_samples=10,
    datasets=None,
    tree_counts=None,
    bit_widths=None,
    max_depth=4,
):
    """
    Head-to-head comparison of RF vs XGBoost under real TFHE encryption.

    For each (dataset, n_trees, n_bits) configuration:
      1. Train FHE-RF and FHE-XGB with same hyperparameters
      2. Measure plain accuracy (no encryption)
      3. Measure simulated FHE accuracy (quantization only, no crypto noise)
      4. Measure real FHE accuracy (actual TFHE encrypt → evaluate → decrypt)
      5. Compare accuracy degradation

    The RF noise theory predicts:
      - RF should have LESS accuracy degradation under FHE
      - The advantage should GROW with more trees (because GBDT noise is O(T*D))
      - The advantage should be especially visible at lower bit widths
    """
    if datasets is None:
        datasets = ["breast_cancer", "iris_binary", "wine_binary"]
    if tree_counts is None:
        tree_counts = [10, 25, 50, 100]
    if bit_widths is None:
        bit_widths = [3, 4, 6]

    all_results = []
    noise_optimizer = IndependentNoiseOptimizer()

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name} ({DATASETS[dataset_name]['description']})")
        print(f"{'='*70}")

        X_train, X_test, y_train, y_test = load_dataset(dataset_name)

        for n_bits in bit_widths:
            print(f"\n  Bit width: {n_bits}")
            print(f"  {'Model':<8} {'Trees':<7} {'Plain':>8} {'SimFHE':>8} "
                  f"{'RealFHE':>8} {'Degrad%':>8} {'FHE_ms':>8}")
            print(f"  {'-'*55}")

            for n_trees in tree_counts:
                entry = {
                    "dataset": dataset_name,
                    "n_trees": n_trees,
                    "n_bits": n_bits,
                    "max_depth": max_depth,
                }

                # --- Random Forest ---
                try:
                    rf = FHERFClassifier(
                        n_estimators=n_trees, max_depth=max_depth,
                        n_bits=n_bits, random_state=42,
                    )
                    rf.fit(X_train, y_train)

                    rf_plain_preds = rf.predict(X_test)
                    rf_plain_acc = accuracy_score(y_test, rf_plain_preds)

                    rf.compile(X_train)

                    rf_sim_preds = rf.predict(X_test, fhe="simulate")
                    rf_sim_acc = accuracy_score(y_test, rf_sim_preds)

                    X_fhe = X_test[:n_fhe_samples]
                    y_fhe = y_test[:n_fhe_samples]

                    t0 = time.time()
                    rf_real_preds = rf.predict(X_fhe, fhe="execute")
                    rf_fhe_ms = (time.time() - t0) / n_fhe_samples * 1000

                    rf_real_acc = accuracy_score(y_fhe, rf_real_preds)
                    rf_degradation = (rf_plain_acc - rf_real_acc) / max(rf_plain_acc, 1e-10) * 100

                    entry["rf_plain_acc"] = round(rf_plain_acc, 4)
                    entry["rf_sim_acc"] = round(rf_sim_acc, 4)
                    entry["rf_real_acc"] = round(rf_real_acc, 4)
                    entry["rf_degradation_pct"] = round(rf_degradation, 2)
                    entry["rf_fhe_ms"] = round(rf_fhe_ms, 1)

                    print(f"  {'RF':<8} {n_trees:<7} {rf_plain_acc:>8.4f} {rf_sim_acc:>8.4f} "
                          f"{rf_real_acc:>8.4f} {rf_degradation:>7.2f}% {rf_fhe_ms:>7.0f}ms")

                except Exception as e:
                    print(f"  {'RF':<8} {n_trees:<7} ERROR: {e}")
                    entry["rf_error"] = str(e)
                    traceback.print_exc()

                # --- XGBoost (GBDT) ---
                try:
                    xgb = FHEXGBClassifier(
                        n_estimators=n_trees, max_depth=max_depth,
                        n_bits=n_bits, random_state=42,
                    )
                    xgb.fit(X_train, y_train)

                    xgb_plain_preds = xgb.predict(X_test)
                    xgb_plain_acc = accuracy_score(y_test, xgb_plain_preds)

                    xgb.compile(X_train)

                    xgb_sim_preds = xgb.predict(X_test, fhe="simulate")
                    xgb_sim_acc = accuracy_score(y_test, xgb_sim_preds)

                    X_fhe = X_test[:n_fhe_samples]
                    y_fhe = y_test[:n_fhe_samples]

                    t0 = time.time()
                    xgb_real_preds = xgb.predict(X_fhe, fhe="execute")
                    xgb_fhe_ms = (time.time() - t0) / n_fhe_samples * 1000

                    xgb_real_acc = accuracy_score(y_fhe, xgb_real_preds)
                    xgb_degradation = (xgb_plain_acc - xgb_real_acc) / max(xgb_plain_acc, 1e-10) * 100

                    entry["xgb_plain_acc"] = round(xgb_plain_acc, 4)
                    entry["xgb_sim_acc"] = round(xgb_sim_acc, 4)
                    entry["xgb_real_acc"] = round(xgb_real_acc, 4)
                    entry["xgb_degradation_pct"] = round(xgb_degradation, 2)
                    entry["xgb_fhe_ms"] = round(xgb_fhe_ms, 1)

                    print(f"  {'XGB':<8} {n_trees:<7} {xgb_plain_acc:>8.4f} {xgb_sim_acc:>8.4f} "
                          f"{xgb_real_acc:>8.4f} {xgb_degradation:>7.2f}% {xgb_fhe_ms:>7.0f}ms")

                except Exception as e:
                    print(f"  {'XGB':<8} {n_trees:<7} ERROR: {e}")
                    entry["xgb_error"] = str(e)
                    traceback.print_exc()

                # --- Theoretical noise comparison ---
                theory = noise_optimizer.get_theoretical_comparison(n_trees, max_depth)
                entry["theory_noise_reduction"] = round(theory["noise_reduction_factor"], 1)
                entry["theory_gbdt_bits"] = round(theory["gbdt_total_noise_bits"], 1)
                entry["theory_rf_bits"] = round(theory["rf_total_noise_bits"], 1)

                # --- Delta: RF advantage ---
                if "rf_degradation_pct" in entry and "xgb_degradation_pct" in entry:
                    entry["rf_advantage_pct"] = round(
                        entry["xgb_degradation_pct"] - entry["rf_degradation_pct"], 2
                    )

                all_results.append(entry)

    return all_results


# =============================================================================
# Scaling analysis: Does RF advantage grow with tree count?
# =============================================================================

def analyze_scaling(results):
    """
    The core theoretical prediction:
    As T (tree count) increases, GBDT noise grows as O(T*D) while RF grows as O(D + log T).
    Therefore the RF advantage (xgb_degradation - rf_degradation) should INCREASE with T.
    """
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS: Does RF advantage grow with tree count?")
    print("Theory predicts: RF advantage increases with more trees")
    print("=" * 70)

    # Group by (dataset, n_bits) and look at tree count trend
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if "rf_advantage_pct" in r:
            key = (r["dataset"], r["n_bits"])
            groups[key].append(r)

    scaling_results = []
    for (dataset, n_bits), entries in sorted(groups.items()):
        entries.sort(key=lambda x: x["n_trees"])
        print(f"\n  {dataset}, {n_bits}-bit:")
        print(f"    {'Trees':<7} {'RF_deg%':>8} {'XGB_deg%':>8} {'RF_adv%':>8} "
              f"{'Theory_ratio':>12}")

        for e in entries:
            print(f"    {e['n_trees']:<7} {e.get('rf_degradation_pct', 'N/A'):>8} "
                  f"{e.get('xgb_degradation_pct', 'N/A'):>8} "
                  f"{e.get('rf_advantage_pct', 'N/A'):>8} "
                  f"{e.get('theory_noise_reduction', 'N/A'):>12}x")

        # Check if RF advantage is monotonically increasing with tree count
        advantages = [e.get("rf_advantage_pct", 0) for e in entries]
        tree_counts = [e["n_trees"] for e in entries]
        if len(advantages) >= 2:
            increasing = all(
                advantages[i] <= advantages[i + 1]
                for i in range(len(advantages) - 1)
            )
            trend = "INCREASING (matches theory)" if increasing else "NON-MONOTONIC"
            print(f"    Trend: {trend}")

            scaling_results.append({
                "dataset": dataset,
                "n_bits": n_bits,
                "tree_counts": tree_counts,
                "rf_advantages": advantages,
                "monotonically_increasing": increasing,
            })

    return scaling_results


# =============================================================================
# Latency analysis: RF vs GBDT FHE inference time
# =============================================================================

def analyze_latency(results):
    """Compare FHE inference latency between RF and GBDT."""
    print("\n" + "=" * 70)
    print("LATENCY ANALYSIS: RF vs GBDT FHE inference time")
    print("=" * 70)

    for r in results:
        if "rf_fhe_ms" in r and "xgb_fhe_ms" in r:
            ratio = r["rf_fhe_ms"] / max(r["xgb_fhe_ms"], 0.1)
            print(f"  {r['dataset']:<15} {r['n_trees']}T/{r['n_bits']}b: "
                  f"RF={r['rf_fhe_ms']:.0f}ms, XGB={r['xgb_fhe_ms']:.0f}ms "
                  f"(ratio={ratio:.2f}x)")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("CANONICAL BENCHMARK: RF vs GBDT under Real TFHE Encryption")
    print("Validating C3: Independent Noise Channels for Random Forests")
    print("NO simulation. NO mocks. Real Concrete ML TFHE encryption.")
    print("=" * 70)
    print(f"NumPy version: {np.__version__}")

    # Real TFHE is expensive: ~1-3s per sample per model
    # 3 datasets × 3 tree counts × 3 bit widths × 2 models × 5 samples = ~270 FHE inferences
    N_FHE_SAMPLES = 5

    t0 = time.time()
    results = benchmark_rf_vs_gbdt_fhe(
        n_fhe_samples=N_FHE_SAMPLES,
        datasets=["breast_cancer", "iris_binary", "wine_binary"],
        tree_counts=[10, 25, 50],
        bit_widths=[3, 4, 6],
        max_depth=4,
    )
    benchmark_time = time.time() - t0

    # Analysis
    scaling_results = analyze_scaling(results)
    analyze_latency(results)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    rf_wins = sum(1 for r in results if r.get("rf_advantage_pct", 0) > 0)
    xgb_wins = sum(1 for r in results if r.get("rf_advantage_pct", 0) < 0)
    ties = sum(1 for r in results if r.get("rf_advantage_pct", 0) == 0)
    total = rf_wins + xgb_wins + ties

    print(f"  RF better under FHE: {rf_wins}/{total} configs ({rf_wins/max(total,1)*100:.0f}%)")
    print(f"  XGB better under FHE: {xgb_wins}/{total} configs ({xgb_wins/max(total,1)*100:.0f}%)")
    print(f"  Tied: {ties}/{total} configs")

    avg_rf_adv = np.mean([r.get("rf_advantage_pct", 0) for r in results if "rf_advantage_pct" in r])
    print(f"  Average RF advantage: {avg_rf_adv:.2f}% less accuracy degradation")

    mono_count = sum(1 for s in scaling_results if s.get("monotonically_increasing"))
    print(f"  Scaling trend matches theory: {mono_count}/{len(scaling_results)} groups")

    print(f"\n  Total benchmark time: {benchmark_time:.0f}s")

    # Save results
    output = {
        "benchmark": "rf_vs_gbdt_real_tfhe",
        "methodology": "Concrete ML real TFHE encryption, no simulation",
        "n_fhe_samples": N_FHE_SAMPLES,
        "max_depth": 4,
        "results": results,
        "scaling_analysis": scaling_results,
        "summary": {
            "rf_wins": rf_wins,
            "xgb_wins": xgb_wins,
            "ties": ties,
            "total_configs": total,
            "avg_rf_advantage_pct": round(avg_rf_adv, 2),
            "scaling_matches_theory": mono_count,
            "scaling_total_groups": len(scaling_results),
            "benchmark_time_seconds": round(benchmark_time, 1),
        },
    }

    output_path = os.path.join(
        os.path.dirname(__file__), 'reports', 'rf_vs_gbdt_real_tfhe.json'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    return output


if __name__ == "__main__":
    main()
