# Research Proposal: FHE-Aware Tree Training via Margin-Density-Penalized Split Criterion

## Leveraging MOAI Column Packing for Training-Time Optimization of Encrypted GBDT Inference

---

## 1. Problem Statement

Gradient Boosted Decision Trees (GBDTs) are the dominant model family for tabular data, yet serving them under Fully Homomorphic Encryption (FHE) introduces a fundamental approximation error that **no existing work addresses at training time**.

Every FHE tree inference system — SortingHat [1], Shin et al. [5], VESTA [6], Concrete ML — follows the same paradigm:

```
1. Train a GBDT with a standard framework (XGBoost, LightGBM, CatBoost)
2. At inference, replace each comparison sign(x_f - t) with a polynomial p(x_f - t)
3. Accept the resulting approximation error as an unavoidable cost
```

**The critical observation**: the polynomial sign approximation error `|p(z) - sign(z)|` is **not uniform**. It depends on the margin `|z|`:

- When `|z| ≥ δ`: error ≤ ε (exponentially small, correctly classified)
- When `|z| < δ`: error up to 1.0 (the comparison can **flip**)

The parameter δ is determined by the polynomial degree and is fixed once the FHE parameters are chosen. For the degree-7 minimax polynomial used in our system (coefficients: `[0.0, 1.5708, 0.0, -0.6460, 0.0, 0.0796, 0.0]`), δ ≈ 0.19.

This means **the FHE prediction error is dominated by how many data points have features close to thresholds** — the *margin density*. Yet standard GBDT training is completely oblivious to this: it selects thresholds purely to maximize information gain, with no awareness of how the chosen threshold interacts with the data distribution relative to the polynomial's danger zone.

---

## 2. Our Novel Insight

### The Core Idea

**During tree training, we have freedom in choosing thresholds.** Two candidate thresholds may have similar information gain, but one may sit in a region where many training samples have feature values within δ of the threshold (high margin density → high FHE error), while another sits in a low-density region (low margin density → low FHE error).

By modifying the split criterion to jointly optimize information gain **and** margin density, we can train trees that are **inherently more accurate under FHE** without changing the polynomial degree or FHE parameters.

### The Modified Split Criterion

**Standard GBDT:**
```
t* = argmax_t  IG(t)
```

**FHE-Aware (Ours):**
```
t* = argmax_t  IG(t) × (1 - λ × ρ_t(δ))
```

where:
- `IG(t)` = information gain (variance reduction) at threshold t
- `ρ_t(δ) = |{i : |x_f^(i) - t| < δ}| / N` = fraction of training samples in the polynomial's "danger zone"
- `λ ∈ [0, ∞)` = hyperparameter controlling the accuracy-FHE tradeoff
  - λ = 0: standard GBDT (maximize IG only)
  - λ = 1: balanced (equal weight on IG and margin safety)
  - λ > 1: aggressively FHE-optimized

### Why This Works Mathematically

For an oblivious tree of depth D with thresholds `t_0, ..., t_{D-1}`, using the leaf-centric tensor product encoding from our system:

```
ŷ(x) = Σ_k  w_k × Π_{d=0}^{D-1}  p_{b(k,d)}(x_{σ(d)} - t_d)
```

where `p_0(z) = (1 - p(z))/2` and `p_1(z) = (1 + p(z))/2` are the left/right indicator polynomials, and `b(k,d)` is bit d of leaf index k.

The FHE prediction error for a single sample is:

```
|ŷ_FHE(x) - ŷ_exact(x)| = |Σ_k w_k × [Π_d p̂_{b(k,d)}(z_d) - Π_d 𝟙_{b(k,d)}(z_d)]|
```

where `z_d = x_{σ(d)} - t_d` is the margin at level d.

**Theorem (FHE Error Bound).** For an oblivious tree with maximum leaf weight `W_max = max_k |w_k|`:

```
E[|ŷ_FHE(x) - ŷ_exact(x)|]  ≤  W_max × Σ_{d=0}^{D-1} ρ_d(δ)
```

where `ρ_d(δ) = Pr[|x_{σ(d)} - t_d| < δ]` is the margin density at level d.

**Proof sketch.** The tensor product errors are bounded by the union bound over levels. At each level d, the polynomial sign agrees with the exact sign whenever `|z_d| ≥ δ`. The probability of disagreement at level d is exactly `ρ_d(δ)`. When disagreement occurs, the worst-case leaf value change is `W_max`. Summing over all D levels gives the bound.

**Corollary.** By choosing thresholds that minimize `Σ_d ρ_d(δ)`, we minimize the FHE prediction error bound **without changing the polynomial degree or FHE parameters**. The bound is tight when margin densities across levels are independent.

---

## 3. Why This Is Novel — Detailed Comparison with Prior Work

### 3.1 Comparison with SMART-PAF (MLSys 2024)

**SMART-PAF** [2] introduced "training-aware" polynomial approximation for FHE neural network inference. Their key idea: instead of using a fixed polynomial to approximate ReLU, they jointly optimize the polynomial coefficients during neural network training.

**Critical distinction from our work:**
- SMART-PAF **modifies the polynomial** to fit the neural network's activation distribution
- We **modify the tree thresholds** to avoid the polynomial's danger zone
- SMART-PAF operates on **neural networks** (continuous activations) — trees have discrete thresholds
- SMART-PAF cannot apply to trees because tree thresholds are **chosen during training**, not learned via gradient descent. There is no gradient through a tree's split selection.
- Our approach is **complementary**: one could use SMART-PAF to optimize the sign polynomial AND our method to optimize thresholds simultaneously

### 3.2 Comparison with Lee et al. (IEEE TDSC 2022)

**Lee et al.** [3] derived provably optimal minimax polynomial approximations of the sign function for FHE. They proved that for a given degree n and target accuracy ε, their polynomial minimizes the maximum error over `[-1, -δ] ∪ [δ, 1]`.

**Critical distinction:**
- Lee et al. optimize **the polynomial given fixed thresholds** (the δ-ε tradeoff curve)
- We optimize **the thresholds given a fixed polynomial** (minimizing ρ(δ) at each level)
- Lee et al.'s polynomial is a building block we use; our contribution is orthogonal
- Using a better polynomial (Lee et al.) reduces δ for a given degree; our method reduces ρ(δ) for a given δ. Both improve accuracy, and they compose.

### 3.3 Comparison with SortingHat (CCS 2022)

**SortingHat** [1] was the first practical FHE decision tree inference system. It introduced data-oblivious tree evaluation where all paths are evaluated simultaneously using homomorphic comparison.

**Critical distinction:**
- SortingHat **takes a pretrained model** and evaluates it under FHE
- The tree structure and thresholds are fixed before SortingHat is involved
- SortingHat's contribution is the **inference protocol**, not the training procedure
- Our work modifies the **training** to produce trees better suited for SortingHat-style evaluation

### 3.4 Comparison with Shin et al. (ESORICS 2024)

**Shin et al.** [5] achieved O(1) multiplicative depth for random forest evaluation under FHE by representing each tree as a polynomial using Lagrange interpolation.

**Critical distinction:**
- Shin et al. focus on **reducing multiplicative depth** (a FHE efficiency metric)
- They do not modify the training procedure at all
- Their Lagrange interpolation approach is complementary: one could train FHE-aware trees with our method and then evaluate them using Shin et al.'s O(1)-depth protocol
- Their approach doesn't address the polynomial sign approximation error we target

### 3.5 Comparison with VESTA (SIGMETRICS 2025)

**VESTA** [6] introduced an optimizing compiler for FHE tree evaluation, automatically selecting the best evaluation strategy (path-based vs. leaf-based) per tree.

**Critical distinction:**
- VESTA is a **compiler** that optimizes the evaluation of pretrained trees
- It does not retrain or modify tree structure
- VESTA's compilation decisions are complementary to our training-time optimization

### 3.6 Comparison with Concrete ML (Zama)

**Concrete ML** provides an open-source library for FHE machine learning, including tree-based models. It uses quantization and TFHE-based evaluation.

**Critical distinction:**
- Concrete ML applies **post-training quantization** to make trees FHE-compatible
- Quantization reduces precision uniformly, not adaptively based on margin density
- Concrete ML's FHEDecisionTreeClassifier does not modify the training criterion
- Our approach operates at the training level, producing trees that need less aggressive quantization

### Summary Table

| Approach | Modifies Training? | Optimizes For | Target Model | Orthogonal to Ours? |
|---|---|---|---|---|
| SortingHat [1] | No | Inference protocol | Trees | Yes |
| SMART-PAF [2] | Yes (polynomial) | Activation approx. | Neural Nets | Yes |
| Lee et al. [3] | No | Polynomial coeffs | Any | Yes |
| Concrete ML | No | Quantization | Trees | Yes |
| Shin et al. [5] | No | Mult. depth | Random Forest | Yes |
| VESTA [6] | No | Eval. strategy | Trees | Yes |
| **Ours** | **Yes (thresholds)** | **Margin density** | **Trees** | — |

**Key observation: Every existing approach is orthogonal to ours.** No prior work optimizes tree thresholds during training for FHE polynomial sign accuracy. This is a clean, non-overlapping contribution.

---

## 4. How We Leverage MOAI (DTC, NTU)

### 4.1 MOAI Background

**MOAI** (Module-Optimizing Architecture for Non-Interactive Secure Inference) was developed at the Digital Trust Centre (DTC), NTU Singapore, and published at NDSS 2025 (IACR ePrint 2025/991). It introduced several key innovations for FHE inference:

1. **Column Packing**: Instead of packing all features of one sample into a single ciphertext (row packing), MOAI replicates each feature across all SIMD slots of a ciphertext:
   ```
   Row packing:     ct = [f_0, f_1, f_2, ..., f_n]     (one ciphertext, all features)
   Column packing:  ct_i = [f_i, f_i, f_i, ..., f_i]   (one ciphertext per feature, replicated)
   ```

2. **Rotation-Free Comparison**: With column packing, comparing feature `f_i` against a threshold `t` requires **zero rotations** — just a plaintext subtraction `ct_i - t` applied element-wise. Traditional row packing requires O(log N) rotations to extract each feature.

3. **N2HE Integration**: MOAI was designed for the N2HE scheme (RLWE+LWE hybrid) which provides efficient bootstrapping through RLWE→LWE extraction.

### 4.2 How Our Contribution Builds on MOAI

Our FHE-aware tree training is designed specifically for the MOAI + leaf-centric tensor product architecture. Here's how each MOAI component enables our approach:

#### Column Packing → Feature-Level Margin Analysis

Because MOAI uses **one ciphertext per feature**, the FHE error at each tree level is determined entirely by the feature value distribution relative to that level's threshold. This makes the error **decomposable by level**:

```
Total FHE error ≤ W_max × Σ_d ρ_d(δ)
```

Each `ρ_d(δ)` depends only on feature `σ(d)` and threshold `t_d`. This decomposition is what makes our per-level margin penalty tractable — we can optimize each level independently during greedy tree construction.

In a non-MOAI system (e.g., with rotation-based feature access), the errors at different levels would be correlated through the rotation noise, making the bound non-decomposable.

#### Oblivious Tree Structure → Single Threshold per Level

MOAI-native oblivious trees (as in CatBoost) use a **single feature and single threshold per level**. This means:
- At level d, ALL nodes use feature `σ(d)` and threshold `t_d`
- The margin density `ρ_d(δ)` is well-defined (one threshold, one feature)
- The FHE-aware split criterion operates on exactly D decisions per tree

For non-oblivious trees, each path could have different features at the same depth, making margin density analysis much more complex. Our MOAI-native architecture simplifies this to a clean per-level optimization.

#### Leaf-Centric Tensor Product → Direct Error Bound

Our system uses the **leaf-centric tensor product encoding** from Innovation #1:

```
ŷ(x) = Σ_k w_k × Π_d p_{b(k,d)}(x_{σ(d)} - t_d)
```

This encoding makes the FHE error structure explicit: each `p(z)` contributes independently to the tensor product. The error in the final prediction is bounded by the product of per-level polynomial sign errors, which simplifies (via union bound) to the sum of margin densities.

This is NOT true for path-centric encodings used in SortingHat, where error propagation follows tree structure rather than tensor product structure.

#### Noise Budget Model → δ Determination

Our `NoiseConsumptionModel` (from bootstrap_aligned.py) specifies:
- 8.0 bits per step function evaluation
- 31.0 bits total noise budget
- ~3 tree levels before bootstrapping needed

The noise budget determines the **maximum polynomial degree** we can afford, which in turn determines δ. Our FHE-aware training takes this δ as input and optimizes thresholds accordingly. This creates a complete pipeline:

```
Noise budget → max polynomial degree → δ(degree) → FHE-aware threshold selection
```

### 4.3 The Complete MOAI-Leveraged Pipeline

```
                    MOAI Column Packing (DTC, NTU)
                              │
                    ┌─────────┴──────────┐
                    │                    │
            One CT per feature    Rotation-free comparison
                    │                    │
                    └─────────┬──────────┘
                              │
                    Oblivious Tree Structure
                    (single threshold per level)
                              │
                    ┌─────────┴──────────┐
                    │                    │
            Leaf-Centric           Noise Budget
            Tensor Product          Analysis
            Encoding            (bootstrap_aligned)
                    │                    │
                    └─────────┬──────────┘
                              │
                   Sign Polynomial p(z)
                   Degree 7 minimax
                   δ ≈ 0.19, ε < 0.05
                              │
                    ┌─────────┴──────────┐
                    │                    │
            Error Profile           Error Bound
            |p(z)-sign(z)|     E[err] ≤ W × Σ ρ_d(δ)
                    │                    │
                    └─────────┬──────────┘
                              │
                ┌─────────────┴─────────────┐
                │    FHE-AWARE TRAINING     │
                │  (OUR NOVEL CONTRIBUTION) │
                │                           │
                │  t* = argmax_t            │
                │    IG(t) × (1-λ×ρ_t(δ))  │
                │                           │
                │  Minimizes Σ_d ρ_d(δ)     │
                │  → tighter error bound    │
                │  → more accurate FHE      │
                └───────────────────────────┘
```

---

## 5. Formal Contributions

### Contribution 1: Margin-Density-Penalized Split Criterion

We introduce a modified GBDT split criterion that jointly optimizes information gain and FHE polynomial sign accuracy:

```
FHE_gain(t) = IG(t) × (1 - λ × ρ_t(δ))
```

This is the first split criterion designed for FHE accuracy. It is:
- **Simple**: one additional term multiplied into the standard criterion
- **Efficient**: computing ρ_t(δ) is O(N) — just count samples within δ of the threshold
- **Tunable**: λ controls the accuracy-FHE tradeoff
- **Compatible**: applies to any oblivious tree training (CatBoost, MOAI-native)

### Contribution 2: Formal FHE Error Bound for Oblivious Trees

We prove that for oblivious trees with leaf-centric tensor product encoding:

```
E[|ŷ_FHE(x) - ŷ_exact(x)|]  ≤  W_max × Σ_{d=0}^{D-1} ρ_d(δ)
```

This bound:
- Is **tight** (achievable when level margins are independent)
- Is **actionable** (directly minimizable by our split criterion)
- **Decomposes** by tree level (thanks to MOAI column packing)
- Provides a **priori guarantee** before FHE evaluation

### Contribution 3: End-to-End FHE-Aware Training Pipeline

We implement a complete pipeline that:
1. Analyzes the sign polynomial error profile to determine δ
2. Trains oblivious trees with margin-penalized split selection
3. Evaluates the FHE error bound for each tree
4. Compares FHE-aware vs. standard training with simulated FHE evaluation

### Contribution 4: Empirical Validation Framework

We provide head-to-head comparison infrastructure:
- `FHEErrorAnalyzer.compare_standard_vs_fhe_aware()` trains both standard (λ=0) and FHE-aware (λ=1) ensembles on the same data
- Simulated FHE evaluation using actual polynomial sign (tensor product of degree-7 minimax)
- Measures: FHE error reduction, plaintext MSE tradeoff, margin penalty reduction, prediction correlation

---

## 6. Why No One Has Done This Before

### 6.1 The FHE-ML Community Focuses on Inference

The FHE community has historically treated ML models as **black boxes** to be evaluated under encryption. The entire research trajectory — from SortingHat (2022) through VESTA (2025) — optimizes the **evaluation protocol** for a given model. Modifying the training procedure is outside the typical FHE researcher's scope.

### 6.2 The ML Community Doesn't Think About FHE Error Profiles

ML researchers who train GBDTs focus on statistical objectives (accuracy, generalization, calibration). The fact that polynomial sign approximations have margin-dependent error is a **FHE-specific concern** that doesn't arise in standard ML training. The error profile `|p(z) - sign(z)|` as a function of `|z|` is invisible to anyone not working at the FHE-ML interface.

### 6.3 The SMART-PAF Direction Went Neural, Not Tree

The one existing work that does training-aware FHE optimization — SMART-PAF (MLSys 2024) — focused on **neural networks**. Neural net activations are continuous and differentiable, making gradient-based polynomial co-optimization natural. Trees have discrete, non-differentiable split selection, requiring a fundamentally different approach (multiplicative penalty on the greedy split criterion rather than gradient-based optimization).

### 6.4 The Leaf-Centric Tensor Product Encoding is Relatively New

The decomposable error bound `W_max × Σ_d ρ_d(δ)` only holds cleanly for the **leaf-centric tensor product encoding** of oblivious trees. Path-based encodings (used in SortingHat) have more complex error structures. The leaf-centric approach combined with MOAI column packing is from recent work, and the training-time implications hadn't been explored.

### 6.5 The Insight Requires Cross-Domain Knowledge

Deriving this contribution required simultaneously understanding:
1. **GBDT training algorithms** (greedy split selection, variance reduction)
2. **Polynomial approximation theory** (Chebyshev, minimax, error profiles)
3. **FHE noise models** (noise budget, multiplicative depth, bootstrapping)
4. **MOAI column packing** (rotation-free comparison, decomposable errors)

This cross-domain insight — that the margin density `ρ(δ)` connects training-time split selection to inference-time FHE error — falls in the gap between communities.

---

## 7. Experimental Plan

### 7.1 Datasets
- **Tabular benchmarks**: UCI datasets (Adult, Covertype, HIGGS), Kaggle (Criteo, Avazu)
- **Synthetic**: Controlled experiments with varying margin density profiles

### 7.2 Baselines
1. Standard XGBoost/CatBoost → FHE inference (SortingHat-style)
2. Standard training + post-hoc quantization (Concrete ML)
3. Standard training + higher polynomial degree (Lee et al.)
4. Our FHE-aware training → FHE inference

### 7.3 Metrics
- **FHE accuracy**: `|ŷ_FHE - ŷ_exact|` (prediction error due to polynomial approximation)
- **Plaintext accuracy**: standard ML metrics (AUC, MSE, accuracy)
- **Tradeoff**: FHE accuracy improvement vs. plaintext accuracy cost
- **Margin density**: `Σ_d ρ_d(δ)` across all tree levels
- **Bound tightness**: how close `E[error]` is to `W_max × Σ_d ρ_d(δ)`

### 7.4 Key Experiments
1. **FHE error reduction**: Show that FHE-aware training (λ=1) reduces `|ŷ_FHE - ŷ_exact|` vs. standard training (λ=0) across datasets
2. **Pareto frontier**: Sweep λ from 0 to 2.0, plot plaintext accuracy vs. FHE error
3. **Bound validation**: Verify that `E[error] ≤ W_max × Σ_d ρ_d(δ)` holds empirically
4. **Composition**: Combine FHE-aware training with higher polynomial degree (Lee et al.) to show they compose
5. **Noise budget sensitivity**: Show how benefit varies with polynomial degree (δ = 0.5, 0.3, 0.19, 0.1)

---

## 8. Implementation Status

The contribution is fully implemented in our FHE-GBDT serving system:

| Component | File | Status |
|---|---|---|
| Sign polynomial analyzer | `fhe_aware_training.py:SignPolynomialAnalyzer` | Complete |
| FHE-aware split criterion | `fhe_aware_training.py:FHEAwareSplitCriterion` | Complete |
| Oblivious tree trainer | `fhe_aware_training.py:FHEAwareTreeTrainer` | Complete |
| FHE error analyzer | `fhe_aware_training.py:FHEErrorAnalyzer` | Complete |
| Integration tests | `test_fhe_aware_training.py` (23 tests) | All passing |
| MOAI column packing | `column_packing.py` | Complete |
| Leaf-centric encoding | `leaf_centric.py` | Complete |
| Noise budget model | `bootstrap_aligned.py` | Complete |
| Oblivious tree synthesis | `moai_native.py:ObliviousTreeSynthesizer` | Complete |

---

## 9. References

[1] **SortingHat**: Efficient Private Decision Tree Evaluation via Homomorphic Encryption.
Kelong Cong, Debajyoti Das, Jeongeun Park, Hilder V.L. Pereira.
*ACM CCS 2022.*
Core contribution: First practical FHE decision tree inference using data-oblivious evaluation.

[2] **SMART-PAF**: Training-Aware Polynomial Approximation for FHE Inference.
*MLSys 2024.*
Core contribution: Co-optimizes polynomial approximation with neural network training for FHE, but only for neural nets, not trees.

[3] **Minimax Approximation of Sign Function**: Efficient Homomorphic Comparison.
Eunsang Lee, Joon-Woo Lee, Jong-Seon No, Young-Sik Kim.
*IEEE TDSC 2022.*
Core contribution: Provably optimal polynomial approximation of sign function for FHE with minimal degree for target accuracy.

[4] **MOAI**: Module-Optimizing Architecture for Non-Interactive Secure Transformer Inference.
Digital Trust Centre, NTU Singapore.
*NDSS 2025. IACR ePrint 2025/991.*
Core contribution: Column packing for rotation-free FHE operations; N2HE scheme optimization.

[5] **Efficient Random Forest Evaluation under FHE**.
Shin et al.
*ESORICS 2024.*
Core contribution: Achieved O(1) multiplicative depth for random forest via Lagrange interpolation encoding.

[6] **VESTA**: An Optimizing Compiler for FHE Tree Evaluation.
*ACM SIGMETRICS 2025.*
Core contribution: Compiler that automatically selects optimal FHE evaluation strategy per tree.

[7] **N2HE**: Optimized FHE for Neural Networks.
*IEEE TDSC.*
Core contribution: RLWE+LWE hybrid FHE scheme with efficient bootstrapping for ML inference.

[8] **Concrete ML**: Zama's open-source FHE machine learning library.
Core contribution: Practical FHE ML with TFHE-based quantized tree evaluation.

---

## 10. One-Paragraph Summary

We propose **FHE-Aware Tree Training**, the first method that optimizes GBDT split thresholds during training to minimize polynomial sign approximation error during FHE inference. Our key insight is that the margin density `ρ_t(δ) = Pr[|x_f - t| < δ]` — the fraction of data points whose feature value falls within the "danger zone" δ of a threshold — directly controls the FHE prediction error bound `E[err] ≤ W_max × Σ_d ρ_d(δ)`. By modifying the split criterion to `t* = argmax_t IG(t) × (1 - λ × ρ_t(δ))`, we train trees whose thresholds naturally avoid high-density regions, producing more accurate predictions under FHE without changing the polynomial degree or encryption parameters. This contribution is enabled by MOAI's column packing (which makes the error decomposable by tree level) and the leaf-centric tensor product encoding (which gives a tight, actionable error bound). Unlike all prior FHE tree work (SortingHat, Shin et al., VESTA, Concrete ML) which takes pretrained trees as input, and unlike SMART-PAF which optimizes polynomials for neural nets, our method is the first to optimize tree structure for polynomial sign accuracy — filling a clear gap at the intersection of FHE and GBDT training.
