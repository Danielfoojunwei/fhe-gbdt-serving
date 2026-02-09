# Accuracy Benchmark Results

**Date**: 2026-02-09
**Total benchmarks**: 126 (7 innovations x 6 models x 3 datasets)
**Errors**: 0
**Methodology**: No fallbacks. Raw innovation vs baseline comparison.

## Per-Innovation Summary

| Innovation | Avg R² | Avg ΔR² | Avg Preserved% | Avg Time (ms) | Status |
|---|---|---|---|---|---|
| Bootstrap-Aligned | -2.9132 | +0.0000 | 100.0% | 2.9 | PASS |
| Gradient-Aware Noise | -2.9134 | -0.0001 | 100.0% | 3.6 | PASS |
| Leaf-Centric Encoding | -2.1835 | +0.7297 | 100.0% | 206.6 | PASS |
| Polynomial Leaves | -2.8445 | +0.0687 | 100.0% | 310.3 | PASS |
| Unified Engine (All) | -2.8566 | +0.0566 | 100.0% | 319.7 | PASS |
| Homomorphic Pruning | -3.0228 | -0.1096 | 95.7% | 2.8 | PASS |
| MOAI-Native Conversion | -3.1241 | -0.2109 | 92.4% | 139.0 | PASS |

## Per-Dataset Summary

| Dataset | Features | Avg Baseline R² | Avg Innovation R² | Avg ΔR² | Avg Preserved% |
|---|---|---|---|---|---|
| classification_30f | 30 | -7.0969 | -6.9618 | +0.1351 | 98.0% |
| regression_10f | 10 | -0.3022 | -0.2483 | +0.0539 | 99.5% |
| highdim_50f | 50 | -1.3406 | -1.3006 | +0.0400 | 97.5% |

## Per-Model Summary

| Model | Trees | Avg Baseline R² | Avg Innovation R² | Avg ΔR² | Avg Preserved% |
|---|---|---|---|---|---|
| XGBoost-style | 20 | -2.5435 | -2.3758 | +0.1677 | 99.3% |
| LightGBM-style | 15 | -2.3783 | -2.3191 | +0.0591 | 98.9% |
| CatBoost-style | 25 | -3.1542 | -3.0988 | +0.0555 | 97.6% |
| Deep-Ensemble | 50 | -3.9680 | -3.9124 | +0.0556 | 97.6% |
| Wide-Ensemble | 8 | -1.1974 | -1.1405 | +0.0569 | 99.0% |
| Mixed-Ensemble | 30 | -4.2380 | -4.1747 | +0.0634 | 97.4% |

## Innovation-Specific Metrics

| Metric | Value |
|---|---|
| MOAI Avg Rotation Savings | 99.4% |
| MOAI Avg Speedup | 268.9x |
| Pruning Avg Computation Saved | 22.0% |
| Polynomial Avg Leaf Coverage | 5.6% |
| Polynomial Avg Leaf R² | 0.1570 |
| Gradient Noise Avg Precision | 14.7 bits |
| Gradient Noise Encode/Decode MAE | 0.000016 |
| Bootstrap Models Needing Bootstrap | 15/18 |

## Worst Cases (Preserved < 95%)

| Innovation | Model | Dataset | Preserved% |
|---|---|---|---|
| MOAI Conversion | Deep-Ensemble | classification_30f | 69.8% |
| MOAI Conversion | Mixed-Ensemble | highdim_50f | 70.0% |
| MOAI Conversion | CatBoost-style | highdim_50f | 79.3% |
| Pruning | Deep-Ensemble | highdim_50f | 82.6% |
| MOAI Conversion | Mixed-Ensemble | classification_30f | 83.2% |
| Pruning | CatBoost-style | highdim_50f | 83.6% |
| MOAI Conversion | Wide-Ensemble | highdim_50f | 84.4% |
| MOAI Conversion | LightGBM-style | classification_30f | 84.8% |
| Pruning | XGBoost-style | regression_10f | 90.1% |
| Pruning | CatBoost-style | classification_30f | 91.9% |
| Pruning | Mixed-Ensemble | regression_10f | 93.2% |
| Pruning | LightGBM-style | highdim_50f | 94.2% |
| MOAI Conversion | CatBoost-style | classification_30f | 94.5% |

## Full Results

See `RESULTS.md` for complete per-innovation x per-model x per-dataset breakdowns with root cause analysis.
See `accuracy_benchmark.json` for raw data (126 entries).
