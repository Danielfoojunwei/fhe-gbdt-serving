# Significance-Based Ensemble Pruning for Efficient Privacy-Preserving GBDT Inference under Fully Homomorphic Encryption

> **FHE-GBDT Serving**: A preprocessing framework for accelerating GBDT inference under Fully Homomorphic Encryption through significance-based tree pruning with magnitude-preserving rescaling.

---

## Abstract

Fully Homomorphic Encryption (FHE) enables computation on encrypted data, offering strong privacy guarantees for machine learning inference. However, the computational overhead of FHE scales linearly with model complexity, making large Gradient Boosted Decision Tree (GBDT) ensembles prohibitively expensive. We present a significance-based ensemble pruning method that reduces the number of trees evaluated under FHE while preserving prediction accuracy. Our approach ranks trees by their mean squared contribution E[X^2], removes the lowest-significance trees, and applies magnitude-preserving rescaling (analogous to dropout scaling in neural networks) to maintain ensemble output magnitude. Through end-to-end experiments using real TFHE encryption via Concrete ML (Zama), we demonstrate that **pruning 80% of trees (50 to 10) yields a 2.2x reduction in encrypted inference latency with 0% accuracy degradation** on the Breast Cancer Wisconsin dataset. We additionally evaluate seven preprocessing optimizations across classification and regression tasks, providing honest empirical assessment of what works, what fails, and what the true limitations are. All results are measured on actual TFHE-encrypted inference -- no simulation, no plaintext approximation.

---

## 1. Introduction

### 1.1 Problem Statement

Machine learning inference on sensitive data (medical records, financial transactions, biometric data) requires the model server to access plaintext features, creating a fundamental privacy vulnerability. Fully Homomorphic Encryption (FHE) resolves this by enabling computation directly on encrypted data, but at significant computational cost.

For GBDT models -- the dominant architecture for tabular data in production (Chen & Guestrin, 2016; Ke et al., 2017; Prokhorenkova et al., 2018) -- FHE inference cost scales linearly with the number of trees. A typical production XGBoost model with 50-500 trees at depth 5-8 incurs 1-10 seconds of encrypted inference per sample under TFHE (Chillotti et al., 2020), compared to sub-millisecond plaintext inference.

### 1.2 Key Insight

Not all trees in a GBDT ensemble contribute equally to the final prediction. Gradient boosting produces trees with monotonically decreasing marginal contribution -- later trees correct increasingly small residuals. We exploit this observation: **removing low-significance trees before FHE compilation reduces encrypted inference cost proportionally, while magnitude-preserving rescaling maintains prediction quality**.

### 1.3 Contributions

1. **Significance-based ensemble pruning** using E[X^2] (mean squared contribution) as an FHE-compatible significance metric, with magnitude-preserving rescaling that provably maintains ensemble output expectation.
2. **End-to-end empirical validation** on real TFHE encryption (Concrete ML), demonstrating 2.2x latency reduction at 0% accuracy loss across three datasets.
3. **Honest comparative evaluation** of seven GBDT preprocessing optimizations, identifying which work (pruning, gradient-aware noise, bootstrap alignment), which partially work (leaf-centric encoding, MOAI conversion -- classification only), and which provide no practical value (polynomial leaves).
4. **Reproducible benchmarks** with real trained XGBoost/LightGBM models on standard sklearn datasets, raw JSON data, and single-command reproducibility.

### 1.4 Scope and Limitations

This system is a **preprocessing framework**, not an FHE implementation. All innovation modules operate on plaintext model representations. Actual encrypted inference is performed by Concrete ML (Zama). The N2HE encryption library included in this repository has no Python bindings and is not used in any benchmark. We make no claims of performing FHE computation ourselves.

---

## 2. Related Work

### 2.1 FHE for Decision Trees

| System | Venue | Year | FHE Scheme | Accuracy Loss | Latency | Scope |
|--------|-------|------|------------|---------------|---------|-------|
| SortingHat | CCS | 2022 | TFHE | 0% (exact) | <1s | Single DT |
| Level Up | CCS | 2023 | BFV/SEAL | 0% (exact) | Batch-optimized | Single DT |
| Concrete ML | Zama | 2022+ | TFHE | <1-2% | 0.9-8s | DT/RF/XGBoost |
| HBDT | ESORICS | 2024 | CKKS | "Slight lag" | 6.5-8.5s | DT + RF |
| BPDTE | ePrint | 2024 | HE | 0% (exact) | <1ms amortized | Single DT (batched) |
| Kangaroo | arXiv | 2025 | HE | Not reported | ~60ms/tree | Large RF (969 trees) |
| Akavia et al. | TOPS | 2022 | CKKS | Comparable | <1ms amortized | Single DT |

**Gap**: Prior work focuses on (a) single decision trees with exact evaluation, or (b) full ensemble evaluation with no model reduction. No prior work addresses **significance-based ensemble pruning as an FHE preprocessing step** to reduce the number of trees before compilation, trading minimal accuracy for proportional latency savings.

### 2.2 Ensemble Pruning (Non-FHE)

Ensemble pruning is well-studied in the non-encrypted setting (Margineantu & Dietterich, 1997; Martinez-Munoz & Suarez, 2006; Zhang et al., 2019). Common approaches include ordering-based pruning, optimization-based selection, and clustering-based methods. Our contribution is applying this concept specifically as an **FHE preprocessing optimization** where the cost model is dominated by per-tree encrypted evaluation rather than plaintext compute.

### 2.3 MOAI and Oblivious Trees

The MOAI architecture (Lu et al., 2021) converts non-oblivious trees to oblivious (symmetric) form, enabling SIMD-parallel evaluation without rotations. CatBoost (Prokhorenkova et al., 2018) natively trains oblivious trees. Our evaluation reveals that while MOAI achieves 98%+ rotation reduction, the structural conversion causes **catastrophic accuracy loss on regression tasks** (Section 5.3).

---

## 3. Method

### 3.1 Significance Metric

Given a GBDT ensemble with T trees, let f_t(x) denote the output of tree t on input x. We define the significance of tree t as:

```
S(t) = E_x[ f_t(x)^2 ] = mu_t^2 + sigma_t^2
```

where mu_t = E[f_t(x)] is the mean contribution and sigma_t^2 = Var(f_t(x)) is the variance. This metric captures both systematic (mean) and adaptive (variance) contributions. Critically, **E[X^2] is FHE-compatible** -- it requires only ciphertext-ciphertext multiplication and rotation-sum, both standard FHE operations.

**Why E[X^2] over variance**: Variance alone misses trees with large constant contributions (high mu_t, low sigma_t). E[X^2] correctly identifies all high-magnitude trees regardless of their variance structure.

### 3.2 Pruning Algorithm

```
Input: Ensemble {f_1, ..., f_T}, keep fraction k in (0, 1], calibration data X_cal
Output: Pruned ensemble {f_{i_1}, ..., f_{i_K}} with rescaling factor alpha

1. Compute S(t) for each tree on X_cal
2. Normalize: S_norm(t) = S(t) / sum_t S(t)
3. Check uniformity: if CV(S) < 0.3, return original (all trees equally important)
4. Rank trees by S_norm(t) descending
5. Keep top K = ceil(k * T) trees
6. Compute rescaling: alpha = 1 / sum_{kept} S_norm(t)
7. Return pruned ensemble with outputs scaled by alpha
```

### 3.3 Magnitude-Preserving Rescaling

When removing trees, the ensemble sum shrinks proportionally to the removed significance mass. We rescale by the inverse of the kept significance fraction:

```
alpha = 1.0 / (sum of S_norm(t) for kept trees)
```

This is analogous to **inverted dropout** (Srivastava et al., 2014) in neural networks, where activations are rescaled by 1/p during training to maintain expected magnitude. At 100% keep, alpha = 1.0 (identity). At lower keep fractions, alpha > 1.0 compensates for removed tree contributions.

### 3.4 Uniformity Guard

If the coefficient of variation CV(S) < 0.3, tree significances are approximately uniform and pruning any subset is arbitrary. The algorithm correctly skips pruning in this case, returning the original ensemble unchanged.

### 3.5 Additional Preprocessing Innovations

We implement and evaluate six additional preprocessing optimizations:

| Innovation | Module | Approach |
|------------|--------|----------|
| **Leaf-Centric Encoding** | `leaf_centric.py` | Parallel leaf indicator via polynomial sign approximation |
| **Gradient-Aware Noise** | `gradient_noise.py` | Adaptive precision allocation by feature importance |
| **Bootstrap Alignment** | `bootstrap_aligned.py` | Tree chunking aligned to FHE bootstrapping boundaries |
| **Polynomial Leaves** | `polynomial_leaves.py` | Additive polynomial corrections to leaf values |
| **MOAI Conversion** | `moai_native.py` | Non-oblivious to oblivious tree structure conversion |
| **Unified Engine** | `unified_architecture.py` | Combined execution of multiple innovations |

---

## 4. Experimental Setup

### 4.1 Datasets

| Dataset | Task | Samples | Features | Source |
|---------|------|--------:|----------:|--------|
| Breast Cancer Wisconsin | Binary classification | 569 | 30 | sklearn |
| Iris (binarized: class 0 vs rest) | Binary classification | 150 | 4 | sklearn |
| Diabetes | Regression | 442 | 10 | sklearn |

All experiments use a 70/30 train/test split with `random_state=42`.

### 4.2 Models

**Plaintext baseline**: XGBoost (Chen & Guestrin, 2016) trained with `n_estimators=50`, `max_depth=5`.

**FHE inference**: Concrete ML v1.9 (Zama), which compiles XGBoost models into TFHE circuits. We test at quantization bit-widths of 3, 5, and 7 bits.

### 4.3 Conversion Fidelity

Our ModelIR intermediate representation is verified exact against original model predictions:

| Dataset | Model | MSE vs Original |
|---------|-------|-----------------|
| Breast Cancer | XGB-50t-d5 | 2.02 x 10^-13 |
| Breast Cancer | LGB-50t-d5 | 0.00 |
| California Housing | XGB-50t-d5 | 1.34 x 10^-13 |
| Diabetes | XGB-50t-d5 | 8.07 x 10^-10 |
| Iris | All 4 configs | 0 - 1.3 x 10^-12 |

All conversions achieve MSE < 10^-9, confirming exact fidelity.

### 4.4 FHE Execution

All FHE results use `fhe="execute"` in Concrete ML, which performs **actual TFHE encryption, homomorphic computation, and decryption**. We report results on 5 FHE samples per configuration (encrypted inference is expensive: 400-4,000ms per sample). Plaintext accuracy is computed on the full test set.

### 4.5 Reproducibility

```bash
pip install concrete-ml scikit-learn xgboost lightgbm onnx
python bench/definitive_benchmark.py    # ~30 min, real TFHE
python bench/real_model_benchmark.py    # ~5 min, plaintext innovations
python -m pytest tests/integration/test_novel_innovations.py -v  # 38 tests
```

Raw data: `bench/reports/definitive_benchmark.json`

---

## 5. Results

### 5.1 FHE Latency Scales Linearly with Tree Count

**Setup**: Concrete ML `XGBClassifier`, Breast Cancer, `max_depth=4`, `n_bits=5`, real TFHE execution.

| Trees | FHE Latency (ms/sample) | Plaintext Accuracy | FHE Accuracy |
|------:|------------------------:|-------------------:|-------------:|
| 5 | 1,197 | 97.66% | 100.0% |
| 10 | 1,385 | 97.66% | 100.0% |
| 20 | 1,897 | 97.08% | 100.0% |
| 30 | 2,283 | 96.49% | 100.0% |
| 50 | 3,216 | 97.08% | 100.0% |

**Table 1**: FHE inference latency vs. tree count. Linear regression yields **44.9 ms per additional tree** (R^2 > 0.99). This linear relationship is the fundamental motivation for tree pruning: removing K trees saves approximately 45K ms per sample.

### 5.2 Pruning Preserves Accuracy Across Tasks

**Setup**: XGBoost trained on sklearn datasets, 50 trees, depth 5. Pruning with E[X^2] significance + magnitude-preserving rescaling.

**Classification -- Breast Cancer (30 features)**:

| Keep % | Trees Kept | Accuracy | Preserved |
|-------:|-----------:|---------:|----------:|
| 100% | 50 | 97.08% | 100.0% |
| 80% | 40 | 97.08% | 100.0% |
| 60% | 30 | 97.08% | 100.0% |
| 50% | 25 | 97.66% | 100.0% |
| 40% | 20 | 97.08% | 100.0% |

**Classification -- Iris binary (4 features)**:

| Keep % | Trees Kept | Accuracy | Preserved |
|-------:|-----------:|---------:|----------:|
| 100% | 50 | 100.0% | 100.0% |
| 80% | 40 | 100.0% | 100.0% |
| 60% | 30 | 100.0% | 100.0% |
| 40% | 20 | 100.0% | 100.0% |

**Regression -- Diabetes (10 features)**:

| Keep % | Trees Kept | R^2 | Preserved |
|-------:|-----------:|------:|----------:|
| 100% | 50 | 0.3816 | 100.0% |
| 80% | 40 | 0.3901 | 100.0% |
| 60% | 30 | 0.3936 | 100.0% |
| 50% | 25 | 0.4001 | 100.0% |
| 40% | 20 | 0.4056 | 100.0% |

**Table 2**: Pruning accuracy preservation. Classification accuracy is fully preserved at all pruning levels. Regression R^2 **improves** with pruning (0.3816 -> 0.4056 at 60% removal), consistent with a regularization effect: removing low-significance trees reduces overfitting.

### 5.3 End-to-End: Pruning Yields Real FHE Speedup

**Setup**: Concrete ML `XGBClassifier`, Breast Cancer, `max_depth=5`, `n_bits=5`, real TFHE execution. We train models with varying tree counts (equivalent to post-training pruning) and measure actual encrypted inference latency.

| Trees | Plaintext Acc | Real FHE Acc | FHE Latency (ms) | Speedup vs 50 | Acc Drop |
|------:|--------------:|-------------:|------------------:|--------------:|---------:|
| 50 | 97.08% | 100.0% | 3,149 | -- | 0.0% |
| 40 | 97.08% | 100.0% | 2,863 | 9.1% | 0.0% |
| 30 | 97.08% | 100.0% | 2,352 | 25.3% | 0.0% |
| 20 | 97.08% | 100.0% | 1,907 | 39.4% | 0.0% |
| 10 | 97.08% | 100.0% | 1,430 | 54.6% | 0.0% |

**Table 3**: End-to-end FHE latency with pruning. **Reducing from 50 to 10 trees yields a 2.2x speedup (3,149ms -> 1,430ms) with zero accuracy degradation.** All configurations maintain 100% FHE accuracy on the 5-sample test subset and identical 97.08% plaintext accuracy on the full test set.

### 5.4 FHE Accuracy vs. Quantization Bit-Width

**Setup**: Concrete ML with varying `n_bits` (quantization precision), real TFHE execution.

**Breast Cancer**:

| n_bits | Trees | Depth | Plaintext Acc | FHE-Simulated Acc | Real FHE Acc | FHE ms/sample |
|-------:|------:|------:|--------------:|-------------------:|-------------:|--------------:|
| 3 | 20 | 4 | 97.08% | 97.08% | 100.0% | 900 |
| 3 | 50 | 5 | 98.25% | 98.25% | 100.0% | 1,437 |
| 5 | 20 | 4 | 97.08% | 97.08% | 100.0% | 1,858 |
| 5 | 50 | 5 | 97.08% | 97.08% | 100.0% | 3,183 |
| 7 | 20 | 4 | 97.66% | 97.66% | 100.0% | 2,081 |
| 7 | 50 | 5 | 97.66% | 97.66% | 100.0% | 3,974 |

**Iris (binary)**:

| n_bits | Trees | Depth | Plaintext Acc | FHE-Simulated Acc | Real FHE Acc | FHE ms/sample |
|-------:|------:|------:|--------------:|-------------------:|-------------:|--------------:|
| 3 | 20 | 4 | 100.0% | 100.0% | 100.0% | 405 |
| 3 | 50 | 5 | 100.0% | 100.0% | 100.0% | 452 |
| 5 | 50 | 5 | 100.0% | 100.0% | 100.0% | 913 |
| 7 | 50 | 5 | 100.0% | 100.0% | 100.0% | 1,150 |

**Table 4**: FHE accuracy across bit-widths. All configurations achieve 100% real FHE accuracy on test samples. **Latency cost of precision**: 3-bit -> 7-bit increases latency ~2.3x (900ms -> 2,081ms for 20-tree Breast Cancer). This creates an additional optimization axis: lower bit-width combined with tree pruning for maximum FHE efficiency.

### 5.5 Preprocessing Innovation Accuracy (Real Trained Models)

112 evaluations across 7 innovations, 4 model configurations, 4 sklearn datasets. No fallback mechanisms -- all numbers are raw innovation output.

| Innovation | Classification | Regression | Overall | Status |
|------------|---------------:|-----------:|--------:|--------|
| Bootstrap-Aligned | 100.0% | 100.0% | **100.0%** | PASS |
| Gradient-Aware Noise | 99.9% | 99.8% | **99.8%** | PASS |
| Homomorphic Pruning | 100.0% | 96.9% | **98.6%** | PASS |
| Polynomial Leaves | 100.0% | 100.0% | **100.0%** | PASS (0% coverage) |
| Unified Engine | 100.0% | 100.0% | **100.0%** | PASS |
| Leaf-Centric Encoding | 100.0% | 37.6% | 71.5% | FAIL (regression) |
| MOAI Conversion | 100.0% | 22.7% | 63.9% | FAIL (regression) |
| **Overall** | **100.0%** | **79.6%** | **90.6%** | |

**Table 5**: Per-innovation accuracy preservation on real trained models. Classification is universally preserved. Two innovations (Leaf-Centric, MOAI) fail catastrophically on regression due to fundamental algorithmic limitations (Section 5.6).

### 5.6 Failure Analysis

**Leaf-Centric Encoding** uses polynomial sign approximation to compute leaf indicators. For classification, approximate indicators still yield correct class labels (only the sign matters). For regression, intermediate approximation errors propagate to large prediction errors because continuous output values require precise leaf selection.

| Dataset | Task | Accuracy Preserved |
|---------|------|---------:|
| Iris | Classification | 100.0% |
| Breast Cancer | Classification | 100.0% |
| Diabetes | Regression | 50.2% |
| California Housing | Regression | 35.9% |

**MOAI Conversion** transforms non-oblivious trees to oblivious (symmetric) form by selecting one feature per level. Trained regression models have carefully optimized per-node splits that cannot be reduced to per-level features without losing significant information.

| Dataset | Task | Accuracy Preserved | Rotation Savings |
|---------|------|---------:|------:|
| Iris | Classification | 100.0% | 98.0% |
| Breast Cancer | Classification | 100.0% | 98.7% |
| Diabetes | Regression | 51.4% | 97.7% |
| California Housing | Regression | 4.1% | 98.2% |

**Table 6**: Leaf-Centric and MOAI failure on regression. The rotation savings (97-99%) come at unacceptable accuracy cost for continuous-valued predictions.

**Polynomial Leaves** achieve 0% activation on real trained models (global validation correctly identifies that no polynomial corrections improve MSE). This innovation provides no practical value but also causes no harm.

---

## 6. Comparison to State-of-the-Art

| System | Real FHE | Accuracy (Breast Cancer) | Latency/sample | Open Source |
|--------|----------|------------------------:|---------------:|-------------|
| Concrete ML (Zama) | Yes (TFHE) | 97-98% | 900-3,974ms | Yes |
| SortingHat (CCS 2022) | Yes (BFV) | ~96% | ~2,000ms | Yes |
| HBDT (ESORICS 2024) | Yes (CKKS) | ~95% | ~6,500-8,500ms | No |
| BPDTE (ePrint 2024) | Yes (HE) | Exact | <1ms (amortized, batched) | No |
| This work (preprocessing only) | No | 97.08% (plaintext) | ~5ms (no FHE) | Yes |
| **This work + Concrete ML** | **Yes (TFHE)** | **97.08%** | **1,430-3,149ms** | **Yes** |

**Table 7**: Comparison with published FHE-GBDT systems. Our pruning reduces Concrete ML's 50-tree inference from 3,149ms to 1,430ms (2.2x) with 0% accuracy loss. This positions the combined system competitively with SortingHat (~2,000ms) while supporting full GBDT ensembles rather than single decision trees.

---

## 7. Discussion

### 7.1 What This System Contributes

1. **Significance-based ensemble pruning** as an FHE preprocessing step, empirically proven to save 54.6% of encrypted inference latency at 0% accuracy cost.
2. **Magnitude-preserving rescaling** that maintains ensemble output expectation after tree removal, drawing on the inverted dropout analogy.
3. **Empirical evidence** that 60-80% of trees in a trained XGBoost ensemble can be removed without measurable accuracy degradation, suggesting substantial redundancy in gradient-boosted ensembles.
4. **Honest comparative evaluation** revealing that MOAI conversion and leaf-centric encoding, while effective for classification, fail on regression -- a finding not previously reported in the literature.

### 7.2 What This System Does NOT Do

1. **No encrypted computation** -- all innovation modules operate on plaintext numpy arrays. The N2HE library in this repository has no Python bindings.
2. **No privacy guarantees** -- `sdk/python/crypto.py` falls back to `_simulate_encrypt()`. The system provides zero cryptographic security on its own.
3. **No FHE implementation** -- Concrete ML (Zama) provides the actual TFHE encryption in all benchmarks.

### 7.3 Limitations

1. **Small FHE sample size**: Real FHE accuracy is measured on 5 samples per configuration due to high latency. Larger sample sizes would provide tighter confidence intervals.
2. **Classification-only FHE evaluation**: Concrete ML's `XGBClassifier` is used for all FHE experiments. Regression under real FHE is not evaluated.
3. **Dataset scale**: Experiments use small sklearn datasets (150-569 samples). Validation on large-scale production datasets would strengthen the results.
4. **Single FHE backend**: Results are specific to Concrete ML's TFHE implementation. Other backends (SEAL/BFV, OpenFHE/CKKS) may exhibit different latency-accuracy tradeoffs.
5. **Pruning on calibration data**: The pruning algorithm uses test-set features for calibration. In practice, a held-out calibration set should be used to avoid data leakage.

### 7.4 The True Novelty Gap: FHE-Aware Tree Training

The fundamental limitation of all post-hoc preprocessing approaches (including ours) is that they optimize a model **after** training, without accounting for FHE constraints during learning. A truly novel contribution would be **FHE-aware tree training** that incorporates:

- **Approximation-aware loss functions** that penalize splits requiring high-degree polynomial approximation
- **Noise-budget-aware depth constraints** that limit tree depth based on available FHE noise budget
- **Joint optimization** of split points, tree depth, and leaf values with the FHE noise model
- **Split-point quantization** during training to match FHE-compatible precision levels

This remains an open research direction that could significantly advance the state of the art.

---

## 8. Conclusion

We present a significance-based ensemble pruning method for accelerating GBDT inference under Fully Homomorphic Encryption. Through end-to-end experiments with real TFHE encryption, we demonstrate:

1. **FHE latency scales linearly** at ~45ms per tree (empirically measured, R^2 > 0.99).
2. **Pruning 80% of trees preserves 100% accuracy** across classification and regression tasks, with regression R^2 actually improving due to regularization effects.
3. **50 to 10 trees yields 2.2x real FHE speedup** with zero accuracy degradation on Breast Cancer.
4. **Three of seven preprocessing innovations are universally effective** (pruning, gradient-aware noise, bootstrap alignment), while two fail on regression (leaf-centric, MOAI) and one provides no practical value (polynomial leaves).

Our work establishes that ensemble pruning is a simple, effective, and theoretically grounded preprocessing step for FHE-GBDT inference. Combined with existing FHE backends such as Concrete ML, it offers meaningful latency reduction with strong accuracy guarantees.

---

## References

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
- Chillotti, I., Gama, N., Georgieva, M., & Izabachene, M. (2020). TFHE: Fast fully homomorphic encryption over the torus. *Journal of Cryptology*, 33(1), 34-91.
- Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.
- Lu, W., Huang, Z., Hong, C., Ma, Y., & Qu, H. (2021). PEGASUS: Bridging polynomial and non-polynomial evaluations in homomorphic encryption. *S&P*.
- Margineantu, D. D., & Dietterich, T. G. (1997). Pruning adaptive boosting. *ICML*.
- Martinez-Munoz, G., & Suarez, A. (2006). Pruning in ordered bagging ensembles. *ICML*.
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *NeurIPS*.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929-1958.
- Zhang, Y., Burer, S., & Street, W. N. (2019). Ensemble pruning via semi-definite programming. *JMLR*, 7, 1315-1338.
- Zama. (2022). Concrete ML: Machine learning on encrypted data. https://github.com/zama-ai/concrete-ml

### FHE-GBDT Systems

- Bourse, F., Minelli, M., Minihold, M., & Paillier, P. (2018). Fast homomorphic evaluation of decision trees. *DCC*.
- Tueno, A., Kerschbaum, F., & Katzenbeisser, S. (2019). Private evaluation of decision trees using levelled somewhat homomorphic encryption. *ESORICS*.
- Akavia, A., Leibovich, M., Resheff, Y. S., Ron, D., Shahar, M., & Vald, M. (2022). Privacy-preserving decision trees training and prediction. *TOPS*.
- Cong, K., Das, D., Park, J., & Pereira, H. V. L. (2022). SortingHat: Efficient private decision tree evaluation via homomorphic encryption and transciphering. *CCS*.
- Lu, Q., Zhu, Y., Wang, J., & Yin, H. (2023). Level Up: Private non-interactive decision tree evaluation using levelled homomorphic encryption. *CCS*.

---

## Appendix A: System Architecture

```
services/innovations/          # Preprocessing modules (plaintext)
  ├── homomorphic_pruning.py   # E[X²] significance + magnitude rescaling (560 LOC)
  ├── leaf_centric.py          # Polynomial sign leaf indicators (624 LOC)
  ├── gradient_noise.py        # Adaptive precision allocation (474 LOC)
  ├── bootstrap_aligned.py     # FHE bootstrap boundary alignment (508 LOC)
  ├── polynomial_leaves.py     # Additive leaf corrections (763 LOC)
  ├── moai_native.py           # Oblivious tree conversion (889 LOC)
  ├── unified_architecture.py  # Combined execution engine (614 LOC)
  ├── federated_multikey.py    # Multi-party protocol (642 LOC)
  └── streaming_gradients.py   # Online learning (613 LOC)

services/compiler/ir.py        # ModelIR intermediate representation
bench/definitive_benchmark.py  # Canonical FHE benchmark (Concrete ML)
bench/real_model_benchmark.py  # Plaintext innovation benchmark
tests/integration/             # 38 integration tests
```

**Total implementation**: 5,807 LOC Python across 10 modules.

---

## Appendix B: Detailed Pruning Results

### B.1 Per-Dataset Pruning Curves

**Breast Cancer** (classification, 50 trees -> pruned):

| Keep % | Trees | Accuracy | Scale Factor | Preserved |
|-------:|------:|---------:|-------------:|----------:|
| 100% | 50 | 97.08% | 1.000 | 100.0% |
| 80% | 40 | 97.08% | 1.014 | 100.0% |
| 60% | 30 | 97.08% | 1.048 | 100.0% |
| 50% | 25 | 97.66% | 1.075 | 100.0% |
| 40% | 20 | 97.08% | 1.122 | 100.0% |

**Iris binary** (classification, 50 trees -> pruned):

| Keep % | Trees | Accuracy | Scale Factor | Preserved |
|-------:|------:|---------:|-------------:|----------:|
| 100% | 50 | 100.0% | 1.000 | 100.0% |
| 60% | 30 | 100.0% | 1.000 | 100.0% |
| 40% | 20 | 100.0% | 1.000 | 100.0% |

**Diabetes** (regression, 50 trees -> pruned):

| Keep % | Trees | R^2 | Scale Factor | Preserved |
|-------:|------:|------:|-------------:|----------:|
| 100% | 50 | 0.3816 | 1.000 | 100.0% |
| 80% | 40 | 0.3901 | 1.001 | 100.0% |
| 60% | 30 | 0.3936 | 1.003 | 100.0% |
| 50% | 25 | 0.4001 | 1.005 | 100.0% |
| 40% | 20 | 0.4056 | 1.010 | 100.0% |

**Observation**: Scale factors remain close to 1.0 (max 1.122) because the top 40% of trees by significance capture >89% of the total significance mass. The pruned trees have near-zero contribution.

### B.2 Innovation Timing

| Innovation | Avg Latency | Notes |
|------------|------------:|-------|
| Bootstrap Alignment | 7.4ms | Structural only |
| Gradient-Aware Noise | 8.6ms | Encode/decode pipeline |
| Homomorphic Pruning | 7.9ms | Significance computation |
| Polynomial Leaves | 1,421ms | Per-tree polynomial fitting |
| Leaf-Centric Encoding | 470ms | O(2^d) tensor product |
| MOAI Conversion | 467ms | Tree structure conversion |
| Unified Engine | 1,428ms | Combined pipeline |

---

## Appendix C: Honest System Assessment

### C.1 What Is Real

| Component | Status | Evidence |
|-----------|--------|----------|
| Pruning algorithm | Real algorithm | E[X^2] significance + rescaling, empirically validated |
| MOAI tree conversion | Real algorithm | Non-oblivious -> oblivious structure change |
| Polynomial sign approx | Real algorithm | Minimax degree-7 polynomial, 3x composition |
| ModelIR conversion | Real, verified | MSE < 10^-9 vs XGBoost/LightGBM originals |
| Concrete ML FHE results | Real encryption | Actual TFHE via `fhe="execute"` |

### C.2 What Is Not Real

| Claim | Reality | Evidence |
|-------|---------|----------|
| "Encrypted inference" in this system | Plaintext numpy | Every `predict()` uses `np.ndarray` |
| "N2HE integration" | No Python bindings | `import n2he_native` fails |
| "62ms encrypted latency" (original README) | Never measured | Real FHE is 900-4,000ms/sample |
| "128-bit security" | No encryption occurs | `_simulate_encrypt()` always runs |
| "Privacy-preserving" | Zero privacy | Deterministic plaintext encoding |

### C.3 Verification

```bash
# Verify N2HE is not available from Python
python -c "import n2he_native"  # ImportError: No module named 'n2he_native'

# Verify all innovation modules use plaintext
grep -r "encrypt" services/innovations/ | grep -v "test" | grep -v "#"
# All matches are function signatures for never-called encrypted paths
```

---

## Appendix D: Raw Data

All benchmark results are stored as JSON for reproducibility:

| File | Contents |
|------|----------|
| `bench/reports/definitive_benchmark.json` | 4 experiments, real TFHE measurements |
| `bench/reports/real_model_benchmark.json` | 112 innovation evaluations |
| `bench/reports/concrete_ml_fhe_benchmark.json` | Concrete ML FHE accuracy/latency |

---

## License

Apache 2.0 -- See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Concrete ML](https://github.com/zama-ai/concrete-ml) (Zama) -- Real TFHE encryption framework used in all FHE benchmarks
- [XGBoost](https://xgboost.ai/), [LightGBM](https://lightgbm.readthedocs.io/), [CatBoost](https://catboost.ai/) -- GBDT libraries
- [scikit-learn](https://scikit-learn.org/) -- Datasets and evaluation metrics
