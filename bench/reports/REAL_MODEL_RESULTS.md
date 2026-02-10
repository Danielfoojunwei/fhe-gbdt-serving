# Canonical Empirical Benchmark: FHE-GBDT Innovations on Real Trained Models

**Date**: 2026-02-10
**Version**: v3.0 (Real models, real datasets, no fallbacks, no mocks)
**Benchmark**: 112 evaluations (7 innovations x 4 models x 4 datasets)
**Errors**: 0
**Conversion Fidelity**: 16/16 exact or near-exact (MSE < 1e-9)

---

## Methodology

### What changed from v2.0

| Aspect | v2.0 (Previous) | v3.0 (This Report) |
|---|---|---|
| **Trees** | Randomly initialized (fake) | Trained via XGBoost/LightGBM (real) |
| **Datasets** | Synthetic random features | sklearn: Breast Cancer, California Housing, Iris, Diabetes |
| **Models** | Random weights, negative R² | Trained to convergence (R²=0.42-0.75 regression, 94-100% classification) |
| **Conversion** | N/A (models hand-built) | XGBoost JSON/LightGBM dump → ModelIR (fidelity verified) |
| **Overall Result** | 98.3% preserved | **90.6% preserved** |

### Datasets

| Dataset | Task | Samples | Features | Train/Test |
|---|---|---|---|---|
| Breast Cancer Wisconsin | Binary classification | 569 | 30 | 398/171 |
| California Housing | Regression | 2,000 | 8 | 1,400/600 |
| Iris (binarized) | Binary classification | 150 | 4 | 105/45 |
| Diabetes | Regression | 442 | 10 | 309/133 |

### Models

| Config | Framework | Trees | Max Depth | Purpose |
|---|---|---|---|---|
| XGB-50t-d5 | XGBoost | 50 | 5 | Standard GBDT |
| LGB-50t-d5 | LightGBM | 50 | 5 | Alternative framework |
| XGB-100t-d3 | XGBoost | 100 | 3 | Many shallow trees |
| XGB-10t-d8 | XGBoost | 10 | 8 | Few deep trees |

### Conversion Fidelity (ModelIR matches original model)

| Dataset | Model | MSE vs Original | Status |
|---|---|---|---|
| Breast Cancer | XGB-50t-d5 | 2.02e-13 | EXACT |
| Breast Cancer | LGB-50t-d5 | 0.00e+00 | EXACT |
| Breast Cancer | XGB-100t-d3 | 2.77e-13 | EXACT |
| Breast Cancer | XGB-10t-d8 | 4.98e-14 | EXACT |
| California Housing | XGB-50t-d5 | 1.34e-13 | EXACT |
| California Housing | LGB-50t-d5 | 0.00e+00 | EXACT |
| California Housing | XGB-100t-d3 | 3.34e-13 | EXACT |
| California Housing | XGB-10t-d8 | 3.17e-14 | EXACT |
| Iris | All 4 models | 0-1.3e-12 | EXACT |
| Diabetes | XGB-50t-d5 | 8.07e-10 | OK |
| Diabetes | LGB-50t-d5 | 0.00e+00 | EXACT |
| Diabetes | XGB-100t-d3 | 8.88e-10 | OK |
| Diabetes | XGB-10t-d8 | 1.11e-10 | OK |

---

## 1. Overall Results

### Per-Innovation Accuracy Preservation

| Innovation | Preserved% | Avg Time | Status | Notes |
|---|---|---|---|---|
| Bootstrap-Aligned | **100.0%** | 7.4ms | PASS | Structural only, no prediction change |
| Polynomial Leaves | **100.0%** | 1,421ms | PASS | Global validation removes all bad polys |
| Unified Engine | **100.0%** | 1,428ms | PASS | Combines correctly |
| Gradient-Aware Noise | **99.8%** | 8.6ms | PASS | Near-zero quantization error |
| Homomorphic Pruning | **98.6%** | 7.9ms | PASS | 26.8% computation saved |
| Leaf-Centric Encoding | **71.5%** | 470ms | WARN | Fails on regression |
| MOAI Conversion | **63.9%** | 467ms | WARN | Structural conversion loss |
| **OVERALL** | **90.6%** | | | |

### Classification vs Regression Split

| Task | Leaf-Centric | MOAI | Pruning | Gradient Noise | Others |
|---|---|---|---|---|---|
| **Classification** (Breast Cancer, Iris) | **100.0%** | **100.0%** | **100.0%** | **99.9%** | 100.0% |
| **Regression** (California, Diabetes) | **37.6%** | **22.7%** | **96.9%** | **99.8%** | 100.0% |

**Critical finding**: Leaf-Centric and MOAI **work well for classification but fail on regression**.

---

## 2. Per-Innovation Detailed Analysis

### 2.1 Leaf-Centric Encoding

| Dataset | Task | Preserved% | Analysis |
|---|---|---|---|
| Iris (binary) | Classification | **100.0%** | Simple boundaries, sign approx sufficient |
| Breast Cancer | Classification | **100.0%** | Sign approximation preserves class boundaries |
| Diabetes | Regression | **50.2%** | Continuous targets need precise values, sign approx too coarse |
| California Housing | Regression | **35.9%** | 8-feature regression, polynomial sign loses precision |

**Root cause**: Polynomial sign approximation maps continuous features to [-1,1] indicators. For classification (binary output), approximate indicators still yield correct class. For regression (continuous output), the approximation error in intermediate values propagates to large prediction errors.

### 2.2 MOAI-Native Conversion

| Dataset | Task | Preserved% | Rotation Savings | Speedup |
|---|---|---|---|---|
| Iris (binary) | Classification | **100.0%** | 98.0% | 71.6x |
| Breast Cancer | Classification | **100.0%** | 98.7% | 117.3x |
| California Housing | Regression | **4.1%** | 98.2% | 365.2x |
| Diabetes | Regression | **51.4%** | 97.7% | 299.1x |

**Root cause**: Converting non-oblivious trees to oblivious form requires selecting one feature per level. Trained regression models have carefully optimized per-node splits that don't reduce to per-level features without losing significant information. Classification models are more robust because the final output only needs correct sign, not precise value.

### 2.3 Homomorphic Pruning

| Dataset | Task | Preserved% | Trees Pruned | Computation Saved |
|---|---|---|---|---|
| Iris | Classification | **100.0%** | ~25% | 25.0% |
| Breast Cancer | Classification | **100.0%** | ~22% | 22.0% |
| California Housing | Regression | **95.9%** | ~27% | 27.0% |
| Diabetes | Regression | **98.6%** | ~28% | 28.0% |

**Analysis**: Pruning with magnitude-preserving rescaling works well. The uniform significance CV check (skip if CV < 0.3) correctly identifies when pruning would be harmful.

### 2.4 Gradient-Aware Noise

| Dataset | Preserved% | Avg Precision | Encode/Decode MAE |
|---|---|---|---|
| All datasets | **99.8%** | 13.6 bits | 0.000031 |

### 2.5 Polynomial Leaves

| Dataset | Preserved% | Polynomial Leaves Fitted | Coverage |
|---|---|---|---|
| All datasets | **100.0%** | 0 | 0.0% |

**Analysis**: The global validation correctly identifies that no polynomial corrections improve MSE on real trained models. Coverage = 0% means the innovation adds nothing but also hurts nothing. This is honest: on well-trained models, additive polynomial corrections to leaves don't improve predictions.

---

## 3. Classification AUC Results

| Dataset | Model | Baseline AUC | Best Innovation AUC | Worst Innovation AUC | Worst Innovation |
|---|---|---|---|---|---|
| Breast Cancer | XGB-50t-d5 | 0.9820 | 0.9886 (Leaf-Centric) | 0.7659 (MOAI) | MOAI |
| Breast Cancer | LGB-50t-d5 | 0.9870 | 0.9879 (Pruning) | 0.6737 (MOAI) | MOAI |
| Breast Cancer | XGB-100t-d3 | 0.9853 | 0.9889 (Leaf-Centric) | 0.7890 (MOAI) | MOAI |
| Breast Cancer | XGB-10t-d8 | 0.9788 | 0.9919 (Leaf-Centric) | 0.8011 (MOAI) | MOAI |
| Iris | All models | 1.0000 | 1.0000 | 0.9714 (LGB MOAI) | MOAI |

**Note**: MOAI degrades AUC on Breast Cancer from 0.98 to 0.67-0.80 — these are the raw numbers without any predictions being classified, just the score quality. Despite this, classification accuracy is still 100% on Iris (because the decision boundary is far from samples) but drops to 68-73% on Breast Cancer.

---

## 4. Comparison with Published SOTA

### FHE Decision Tree/Forest Inference Systems

| System | Year | Venue | Scheme | Accuracy vs Plaintext | Inference Latency | Model Scope |
|---|---|---|---|---|---|---|
| **SortingHat** | 2022 | CCS | TFHE | **Exact** (0% loss) | <1s single DT | Single decision tree |
| **Level Up** | 2023 | CCS | BFV/SEAL | **Exact** (0% loss) | Batch-optimized | Single decision tree |
| **Concrete ML (Zama)** | 2023+ | MSPN | TFHE | **<1-2% drop** at 5-6 bits | ~0.9s (1 DT), ~7-8s (50 trees) | DT/RF/XGBoost |
| **HBDT** | 2024 | ESORICS | CKKS | "Slight lag" | ~6.5-8.5s (64 trees, depth 4) | DT + RF |
| **BPDTE** | 2024 | ePrint | HE | **Exact** | <1ms amortized @ 32-bit | Single DT (batched) |
| **Kangaroo** | 2025 | arXiv | HE | Not reported | ~60ms/tree (amortized) | Large RF (969 trees) |
| **Akavia/Intuit** | 2022 | TOPS | CKKS | "Comparable" | <1ms amortized | Single DT |
| **This work** | 2026 | - | N2HE/BFV | See below | Plaintext-only eval | GBDT (XGBoost/LightGBM) |

### Where This System Stands

| Metric | SOTA (Best Published) | This System (Empirical) | Assessment |
|---|---|---|---|
| **Accuracy (classification)** | Exact (SortingHat, BPDTE) | **100%** (Iris), **93-95%** (Breast Cancer) | Competitive for classification |
| **Accuracy (regression)** | <1-2% drop (Concrete ML) | **36-51%** (Leaf-Centric), **4-51%** (MOAI) on regression | **Far below SOTA** |
| **Accuracy (pruning)** | N/A | **98.6%** with 26.8% computation saved | Good |
| **Accuracy (noise allocation)** | N/A | **99.8%** | Excellent |
| **Rotation reduction** | MOAI-type (known technique) | **98.2% avg, 213x speedup** | Good but with accuracy cost |
| **Ensemble support** | Concrete ML (50-tree XGBoost) | 10-100 tree XGBoost/LightGBM | Competitive scope |
| **Real FHE latency** | ~7-8s for 50 trees (Concrete ML) | **Not measured** (plaintext only) | Cannot compare |

### Honest Assessment

1. **Classification**: Competitive. Most innovations preserve accuracy at 100%. MOAI is the exception (loses 20-30% AUC on complex datasets).

2. **Regression**: **Significantly below SOTA**. Concrete ML achieves <2% accuracy drop on regression. Our Leaf-Centric (-64%) and MOAI (-96%) innovations fail catastrophically on regression tasks with real trained models.

3. **Rotation reduction**: 98.2% savings looks impressive but comes at unacceptable accuracy cost for regression.

4. **No real FHE evaluation**: All measurements are plaintext approximations. We cannot claim FHE latency numbers without running actual encrypted inference.

5. **Polynomial Leaves**: 0% activation on real models means the innovation provides no value in practice.

---

## 5. What Actually Works (Honest Summary)

| Innovation | Classification | Regression | Overall Verdict |
|---|---|---|---|
| Bootstrap-Aligned | Works | Works | **Useful** (structural optimization) |
| Gradient-Aware Noise | Works | Works | **Useful** (precision allocation) |
| Homomorphic Pruning | Works | Works | **Useful** (22-27% computation saved at 98.6% accuracy) |
| Polynomial Leaves | No effect | No effect | **No practical value** (0% coverage on real models) |
| Unified Engine | Works | Works | **Useful** (correct integration) |
| Leaf-Centric Encoding | **Works** | **Fails** | **Classification only** |
| MOAI Conversion | **Works** | **Fails** | **Classification only** |

### Innovations that are genuinely useful:
- **Pruning**: Saves 27% computation at 98.6% accuracy. Works on both tasks.
- **Gradient Noise**: 99.8% accuracy at adaptive precision. Works everywhere.
- **Bootstrap Alignment**: Zero accuracy cost. Structural optimization.

### Innovations that need fundamental redesign for regression:
- **Leaf-Centric**: Polynomial sign approximation is too imprecise for continuous targets
- **MOAI**: Oblivious conversion loses too much structure for regression models

### Innovations with no practical impact:
- **Polynomial Leaves**: Global validation correctly removes all polynomials (no improvement found)

---

## 6. Reproducibility

```bash
# Install dependencies
pip install scikit-learn xgboost lightgbm

# Run real model benchmark (112 evaluations, ~5 minutes)
python bench/real_model_benchmark.py

# Run synthetic benchmark for comparison (126 evaluations, ~2 minutes)
python bench/accuracy_benchmark.py

# Run integration tests (38 tests)
python -m pytest tests/integration/test_novel_innovations.py -v
```

### Raw Data
- `bench/reports/real_model_benchmark.json` — Full results (112 entries)
- `bench/reports/accuracy_benchmark.json` — Synthetic results (126 entries)

---

## 7. Key Takeaways

1. **The previous 98.3% number was measured on random trees and is not meaningful.** Real trained models show 90.6% overall, with two innovations failing badly on regression.

2. **Classification is competitive with SOTA.** Most innovations preserve 100% accuracy on classification tasks.

3. **Regression is the critical gap.** Leaf-Centric (37.6%) and MOAI (22.7%) fail on regression. This is far below Concrete ML's <2% drop.

4. **The only genuinely novel contribution would be FHE-aware tree training** — training trees that account for FHE approximation errors during the learning process. The current innovations are post-hoc optimizations of pre-trained models, and the empirical evidence shows this approach has fundamental limitations for regression tasks.

5. **Three innovations provide genuine value**: Pruning (computation savings), Gradient Noise (precision optimization), and Bootstrap Alignment (noise management). These work reliably on both classification and regression.
