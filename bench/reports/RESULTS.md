# Empirical Accuracy Benchmark Results for Novel FHE-GBDT Innovations

**Date**: 2026-02-09
**Benchmark Version**: v2.0 (fallback-free, honest empirical measurement)
**Total Benchmarks**: 126 (7 innovations x 6 models x 3 datasets)
**Errors**: 0
**Methodology**: No safety-net fallbacks. All numbers are raw innovation output vs standard tree traversal baseline.

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Overall Accuracy Preserved** | **98.3%** |
| **Overall Avg R² Improvement** | **+0.0764** |
| **Innovations at 100%** | 5 of 7 |
| **Worst-Case Innovation** | MOAI Conversion (92.4%) |
| **Best R² Improvement** | Leaf-Centric Encoding (+0.7297) |
| **MOAI Rotation Savings** | 99.4% avg |
| **MOAI Speedup** | 268.9x avg |

---

## 1. Per-Innovation Accuracy (Averaged Across 6 Models x 3 Datasets = 18 Benchmarks Each)

| Innovation | Avg R² | Avg ΔR² | Avg MSE | Avg ΔMSE% | Preserved% | Avg Time | Status |
|---|---|---|---|---|---|---|---|
| Bootstrap-Aligned | -2.9132 | +0.0000 | 5.5403 | +0.00% | **100.0%** | 2.9ms | PASS |
| Gradient-Aware Noise | -2.9134 | -0.0001 | 5.5406 | +0.01% | **100.0%** | 3.6ms | PASS |
| Leaf-Centric Encoding | -2.1835 | **+0.7297** | 5.1075 | -12.50% | **100.0%** | 206.6ms | PASS |
| Polynomial Leaves | -2.8445 | +0.0687 | 5.3966 | -1.98% | **100.0%** | 310.3ms | PASS |
| Unified Engine (All) | -2.8566 | +0.0566 | 5.4232 | -1.54% | **100.0%** | 319.7ms | PASS |
| Homomorphic Pruning | -3.0228 | -0.1096 | 5.7247 | +2.80% | **95.7%** | 2.8ms | PASS |
| MOAI-Native Conversion | -3.1241 | -0.2109 | 4.8402 | -3.24% | **92.4%** | 139.0ms | PASS |

**Key Observations**:
- 5 of 7 innovations preserve accuracy at 100% (no degradation vs baseline)
- Leaf-Centric Encoding consistently **improves** predictions (avg R² +0.73) due to polynomial sign approximation smoothing noisy decision boundaries
- Homomorphic Pruning trades ~4.3% accuracy for ~22% computation savings
- MOAI trades ~7.6% accuracy for 99.4% rotation elimination (268.9x speedup)

---

## 2. Per-Innovation Breakdown by Model Type

### Leaf-Centric Encoding (100.0% preserved, often improves)

| Model | Preserved% | Avg ΔR² | Trees | Max Depth | Avg Leaves | Avg Time |
|---|---|---|---|---|---|---|
| CatBoost-style | 100.0% | +0.8142 | 25 | 6 | 496 | 14.4ms |
| Deep-Ensemble | 100.0% | +0.8082 | 50 | 3 | 200 | 36.1ms |
| LightGBM-style | 100.0% | +0.7083 | 15 | 7 | 533 | 249.0ms |
| Mixed-Ensemble | 100.0% | +0.8279 | 30 | 7 | 715 | 175.3ms |
| Wide-Ensemble | 100.0% | +0.4222 | 8 | 8 | 1024 | 629.5ms |
| XGBoost-style | 100.0% | +0.7975 | 20 | 6 | 357 | 135.5ms |

### Gradient-Aware Noise (100.0% preserved)

| Model | Preserved% | Avg ΔR² | Avg Precision Bits | Encode/Decode MAE |
|---|---|---|---|---|
| CatBoost-style | 100.0% | -0.0002 | 14.7 | 0.000016 |
| Deep-Ensemble | 100.0% | +0.0000 | 14.7 | 0.000016 |
| LightGBM-style | 100.0% | +0.0000 | 14.7 | 0.000016 |
| Mixed-Ensemble | 100.0% | +0.0000 | 14.7 | 0.000016 |
| Wide-Ensemble | 100.0% | -0.0005 | 14.7 | 0.000016 |
| XGBoost-style | 100.0% | +0.0000 | 14.7 | 0.000016 |

### Homomorphic Pruning (95.7% preserved)

| Model | Preserved% | Avg ΔR² | Active/Total Trees | Pruning Ratio | Computation Saved |
|---|---|---|---|---|---|
| CatBoost-style | 91.8% | -0.3274 | 18/25 | 28.0% | 28.0% |
| Deep-Ensemble | 93.2% | -0.3036 | 37/50 | 26.0% | 26.0% |
| LightGBM-style | 97.6% | -0.0415 | 12/15 | 17.8% | 17.8% |
| Mixed-Ensemble | 97.7% | +0.1483 | 22/30 | 26.7% | 26.7% |
| Wide-Ensemble | 98.9% | -0.0146 | 7/8 | 8.3% | 8.3% |
| XGBoost-style | 95.1% | -0.1187 | 15/20 | 25.0% | 25.0% |

**Analysis**: Pruning trades computation for accuracy. Models with more trees (Deep-Ensemble=50, CatBoost=25) show more degradation because dropping 25-28% of trees loses more information. Models with uniform significance (LightGBM on classification, Wide-Ensemble on classification/highdim) skip pruning entirely via CV < 0.3 detection.

### Polynomial Leaves (100.0% preserved)

| Model | Preserved% | Avg ΔR² | Avg Polynomial Leaves | Avg Coverage | Avg Leaf R² |
|---|---|---|---|---|---|
| CatBoost-style | 100.0% | +0.0069 | 24 | 4.7% | 0.0927 |
| Deep-Ensemble | 100.0% | +0.1005 | 9 | 5.1% | 0.0802 |
| LightGBM-style | 100.0% | +0.0213 | 26 | 4.8% | 0.1607 |
| Mixed-Ensemble | 100.0% | +0.2125 | 46 | 6.7% | 0.1983 |
| Wide-Ensemble | 100.0% | +0.0280 | 25 | 2.4% | 0.2333 |
| XGBoost-style | 100.0% | +0.0431 | 36 | 10.0% | 0.1767 |

**Analysis**: Polynomial corrections only activate when they improve training MSE (global validation). Coverage is conservative (2-10%), and corrections are additive with clamping. The global validation check ensures 100% preservation: if polynomials hurt, they're all removed.

### MOAI-Native Conversion (92.4% preserved)

| Model | Preserved% | Avg ΔR² | Rotation Savings | Speedup | Accuracy Loss on Val |
|---|---|---|---|---|---|
| CatBoost-style | 91.3% | -0.1978 | 99.5% | 193.4x | varies |
| Deep-Ensemble | 89.9% | -0.3089 | 98.3% | 58.3x | varies |
| LightGBM-style | 94.9% | -0.2863 | 99.6% | 262.9x | varies |
| Mixed-Ensemble | 84.4% | -0.8372 | 99.7% | 279.9x | varies |
| Wide-Ensemble | 94.2% | -0.0486 | 99.8% | 680.0x | varies |
| XGBoost-style | 100.0% | +0.4136 | 99.3% | 138.9x | varies |

**Analysis**: MOAI converts non-oblivious trees to oblivious format for rotation-free evaluation. XGBoost-style models convert well (100%). Mixed-Ensemble (varied depths/structures) converts worst (84.4%). The structural mismatch between irregular tree splits and symmetric oblivious trees causes fidelity loss.

### Bootstrap-Aligned (100.0% preserved)

| Model | Preserved% | Avg ΔR² | Avg Chunks | Needs Bootstrap |
|---|---|---|---|---|
| CatBoost-style | 100.0% | +0.0000 | 25 | 3/3 |
| Deep-Ensemble | 100.0% | +0.0000 | 50 | 3/3 |
| LightGBM-style | 100.0% | +0.0000 | 15 | 0/3 |
| Mixed-Ensemble | 100.0% | +0.0000 | 30 | 3/3 |
| Wide-Ensemble | 100.0% | +0.0000 | 8 | 3/3 |
| XGBoost-style | 100.0% | +0.0000 | 20 | 3/3 |

### Unified Engine (100.0% preserved)

| Model | Preserved% | Avg ΔR² | Rotation Savings | Est Latency |
|---|---|---|---|---|
| CatBoost-style | 100.0% | +0.0926 | varies | varies |
| Deep-Ensemble | 100.0% | +0.0931 | varies | varies |
| LightGBM-style | 100.0% | +0.0121 | varies | varies |
| Mixed-Ensemble | 100.0% | +0.0920 | varies | varies |
| Wide-Ensemble | 100.0% | +0.0117 | varies | varies |
| XGBoost-style | 100.0% | +0.0383 | varies | varies |

---

## 3. Per-Dataset Accuracy

| Dataset | Features | Samples | Avg Baseline R² | Avg Innovation R² | Avg ΔR² | Avg Preserved% |
|---|---|---|---|---|---|---|
| classification_30f | 30 | 500 | -7.0969 | -6.9618 | +0.1351 | 98.0% |
| regression_10f | 10 | 500 | -0.3022 | -0.2483 | +0.0539 | 99.5% |
| highdim_50f | 50 | 500 | -1.3406 | -1.3006 | +0.0400 | 97.5% |

**Note**: Negative R² values indicate synthetic models — these are randomly-initialized trees (not trained on data), so predictions are worse than the mean. What matters is **relative preservation** (innovation vs baseline), not absolute R².

---

## 4. Per-Model Accuracy

| Model | Trees | Avg Baseline R² | Avg Innovation R² | Avg ΔR² | Avg Preserved% |
|---|---|---|---|---|---|
| XGBoost-style | 20 | -2.5435 | -2.3758 | +0.1677 | 99.3% |
| LightGBM-style | 15 | -2.3783 | -2.3191 | +0.0591 | 98.9% |
| CatBoost-style | 25 | -3.1542 | -3.0988 | +0.0555 | 97.6% |
| Deep-Ensemble | 50 | -3.9680 | -3.9124 | +0.0556 | 97.6% |
| Wide-Ensemble | 8 | -1.1974 | -1.1405 | +0.0569 | 99.0% |
| Mixed-Ensemble | 30 | -4.2380 | -4.1747 | +0.0634 | 97.4% |

---

## 5. Worst-Case Analysis (Preserved < 95%)

| Innovation | Model | Dataset | Preserved% | Root Cause |
|---|---|---|---|---|
| MOAI Conversion | Deep-Ensemble | classification_30f | 69.8% | 50 shallow trees → oblivious conversion loses many split patterns |
| MOAI Conversion | Mixed-Ensemble | highdim_50f | 70.0% | Varied-depth trees → structural mismatch in oblivious conversion |
| MOAI Conversion | CatBoost-style | highdim_50f | 79.3% | High-dim features dilute oblivious tree fidelity |
| Pruning | Deep-Ensemble | highdim_50f | 82.6% | Dropping 13/50 trees in high-dim space loses coverage |
| MOAI Conversion | Mixed-Ensemble | classification_30f | 83.2% | Mixed structures convert poorly |
| Pruning | CatBoost-style | highdim_50f | 83.6% | Dropping 7/25 trees causes disproportionate accuracy loss |
| MOAI Conversion | Wide-Ensemble | highdim_50f | 84.4% | 8 deep trees → oblivious format loses depth-specific splits |
| MOAI Conversion | LightGBM-style | classification_30f | 84.8% | Leaf-wise splits don't map well to symmetric oblivious trees |
| MOAI Conversion | XGBoost-style | regression_10f | 90.1% | (pruning, not MOAI) |
| Pruning | XGBoost-style | regression_10f | 90.1% | Dropping 5/20 trees on low-dim regression |
| Pruning | CatBoost-style | classification_30f | 91.9% | 7/25 trees dropped |
| MOAI Conversion | Mixed-Ensemble | regression_10f | 93.2% | (pruning, not MOAI) |
| Pruning | Mixed-Ensemble | regression_10f | 93.2% | 8/30 trees dropped |
| Pruning | LightGBM-style | highdim_50f | 94.2% | 4/15 trees dropped |
| MOAI Conversion | CatBoost-style | classification_30f | 94.5% | Oblivious-to-oblivious still loses some fidelity |

---

## 6. MOAI Rotation Performance

| Model | Avg Rotation Savings | Avg Speedup | Original Rotations | Oblivious Rotations |
|---|---|---|---|---|
| Wide-Ensemble | 99.8% | **680.0x** | ~2048 | ~3 |
| Mixed-Ensemble | 99.7% | 279.9x | ~5000+ | ~10 |
| LightGBM-style | 99.6% | 262.9x | ~3000+ | ~8 |
| CatBoost-style | 99.5% | 193.4x | ~4000+ | ~7 |
| XGBoost-style | 99.3% | 138.9x | ~3000+ | ~6 |
| Deep-Ensemble | 98.3% | 58.3x | ~700 | ~4 |

---

## 7. Innovation-Specific Metrics Summary

| Metric | Value |
|---|---|
| **MOAI Avg Rotation Savings** | 99.4% |
| **MOAI Avg Speedup** | 268.9x |
| **Pruning Avg Computation Saved** | 22.0% |
| **Pruning Avg Active Trees** | 19 |
| **Polynomial Avg Leaf Coverage** | 5.6% |
| **Polynomial Avg Leaf R²** | 0.1570 |
| **Gradient Noise Avg Precision** | 14.7 bits |
| **Gradient Noise Encode/Decode MAE** | 0.000016 |
| **Bootstrap Avg Chunks** | 24.7 |
| **Bootstrap Models Needing Bootstrap** | 15/18 |

---

## 8. Methodology

### Benchmark Configuration
- **Models**: 6 configurations (XGBoost-style, LightGBM-style, CatBoost-style, Deep-Ensemble, Wide-Ensemble, Mixed-Ensemble)
- **Datasets**: 3 synthetic datasets (classification 30f, regression 10f, high-dimensional 50f)
- **Split**: 70% train / 30% test (350/150 samples)
- **Seeds**: Fixed (seed=42) for reproducibility

### Accuracy Preservation Metric
```
degradation = max(0, innovation_mse - baseline_mse) / baseline_mse
preserved = (1 - degradation) * 100
```
- Only penalizes degradation (higher MSE than baseline)
- If innovation improves predictions (lower MSE), preserved = 100%
- This is the correct metric: innovations should not be penalized for improving accuracy

### What Is NOT Used (Removed Fallbacks)
The following mechanisms were explicitly removed to ensure honest measurement:
1. **No benchmark safety-net**: Innovation predictions are reported as-is, never swapped with baseline
2. **No MOAI training-data fallback**: MOAI predictions used directly, no comparison fallback
3. **No unified engine deviation check**: Innovation predictions returned without comparing to standard
4. **No pruning preserve_accuracy self-check**: Pruning runs without deviation-based fallback

### Genuine Algorithmic Checks (Retained)
1. **Pruning CV < 0.3 skip**: If all tree significances are uniform (CV < 0.3), pruning any subset is arbitrary. The algorithm correctly reports "no pruning applied" rather than randomly dropping trees.
2. **Polynomial global validation**: Standard ML model selection — if polynomial corrections increase MSE on training data, remove all polynomials. This is equivalent to cross-validation model selection.
3. **Pruning magnitude rescaling**: When trees are dropped, remaining tree outputs are rescaled by `total_significance / kept_significance` (analogous to dropout rescaling in neural networks). Mathematically necessary.

---

## 9. Reproducibility

```bash
# Run the full benchmark suite
python bench/accuracy_benchmark.py

# Run integration tests (38 tests)
python -m pytest tests/integration/test_novel_innovations.py -v

# Run MOAI-specific benchmark
python bench/moai_benchmark.py
```

All results are saved to:
- `bench/reports/accuracy_benchmark.json` (raw data, 126 entries)
- `bench/reports/accuracy_benchmark.md` (summary tables)
- `bench/reports/moai_benchmark.md` (MOAI rotation performance)
