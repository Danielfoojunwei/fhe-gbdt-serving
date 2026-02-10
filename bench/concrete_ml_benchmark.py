#!/usr/bin/env python3
"""
Real FHE Benchmark using Concrete ML (Zama).

This script runs ACTUAL encrypted inference using TFHE via Concrete ML,
providing ground-truth FHE performance numbers for comparison.

Requires: pip install concrete-ml onnx scikit-learn
"""
import time
import json
import os
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from concrete.ml.sklearn import XGBClassifier as FHEXGBClassifier

N_FHE_SAMPLES = 5  # Number of samples for actual FHE execution (expensive)


def benchmark_classification(dataset_name, X, y, n_estimators, max_depth, n_bits_list):
    """Benchmark classification with Concrete ML."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    results = []

    for n_bits in n_bits_list:
        print(f"\n  Concrete ML XGBClassifier: n_bits={n_bits}, "
              f"n_estimators={n_estimators}, max_depth={max_depth}")

        model = FHEXGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_bits=n_bits,
            random_state=42,
        )

        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Plaintext (quantized, no encryption)
        y_pred_plain = model.predict(X_test)
        plain_acc = accuracy_score(y_test, y_pred_plain)

        # Compile FHE circuit
        t0 = time.time()
        try:
            model.compile(X_train)
            compile_time = time.time() - t0
            compiled = True
        except Exception as e:
            compile_time = 0
            compiled = False
            print(f"    Compilation failed: {e}")

        sim_acc = 0.0
        fhe_acc = 0.0
        fhe_per_sample = 0.0

        if compiled:
            # FHE simulation (bit-accurate, no crypto overhead)
            y_pred_sim = model.predict(X_test, fhe="simulate")
            sim_acc = accuracy_score(y_test, y_pred_sim)

            # REAL FHE execution
            n_samples = min(N_FHE_SAMPLES, len(X_test))
            t0 = time.time()
            try:
                y_pred_fhe = model.predict(X_test[:n_samples], fhe="execute")
                fhe_time = time.time() - t0
                fhe_acc = accuracy_score(y_test[:n_samples], y_pred_fhe)
                fhe_per_sample = fhe_time / n_samples * 1000
            except Exception as e:
                print(f"    FHE execution failed: {e}")

        print(f"    Plain: {plain_acc:.4f} | FHE-Sim: {sim_acc:.4f} | "
              f"REAL FHE: {fhe_acc:.4f} ({fhe_per_sample:.0f}ms/sample)")

        results.append({
            "dataset": dataset_name,
            "task": "classification",
            "n_bits": n_bits,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "plain_accuracy": round(float(plain_acc), 4),
            "fhe_sim_accuracy": round(float(sim_acc), 4),
            "fhe_real_accuracy": round(float(fhe_acc), 4),
            "fhe_compiled": compiled,
            "compile_time_s": round(compile_time, 2),
            "train_time_s": round(train_time, 2),
            "fhe_ms_per_sample": round(fhe_per_sample, 1),
            "fhe_samples_tested": min(N_FHE_SAMPLES, len(X_test)) if compiled else 0,
        })

    return results


def main():
    print("=" * 80)
    print("  CONCRETE ML REAL FHE BENCHMARK")
    print("  Actual encrypted inference using TFHE")
    print("=" * 80)

    all_results = []

    # Breast Cancer
    print(f"\n{'─' * 80}")
    print("DATASET: Breast Cancer Wisconsin (30 features, 569 samples)")
    print(f"{'─' * 80}")
    X, y = load_breast_cancer(return_X_y=True)
    all_results.extend(benchmark_classification(
        "breast_cancer", X, y, n_estimators=50, max_depth=5, n_bits_list=[3, 5, 7]
    ))

    # Iris
    print(f"\n{'─' * 80}")
    print("DATASET: Iris binary (4 features, 150 samples)")
    print(f"{'─' * 80}")
    X, y = load_iris(return_X_y=True)
    y = (y == 0).astype(int)
    all_results.extend(benchmark_classification(
        "iris_binary", X, y, n_estimators=20, max_depth=3, n_bits_list=[3, 5, 7]
    ))

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  {'Dataset':<20} {'n_bits':>6} {'Plain':>8} {'FHE-Sim':>8} "
          f"{'REAL FHE':>9} {'ms/sample':>10}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*8} {'─'*9} {'─'*10}")
    for r in all_results:
        print(f"  {r['dataset']:<20} {r['n_bits']:>6} "
              f"{r['plain_accuracy']:>8.4f} {r['fhe_sim_accuracy']:>8.4f} "
              f"{r['fhe_real_accuracy']:>9.4f} {r['fhe_ms_per_sample']:>9.0f}ms")

    # Save
    os.makedirs("bench/reports", exist_ok=True)
    path = "bench/reports/concrete_ml_fhe_benchmark.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
