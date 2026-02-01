# MOAI Implementation Plan for FHE-GBDT Serving

## Reference Paper

**MOAI: Module-Optimizing Architecture for Non-Interactive Secure Transformer Inference**
- Authors: Linru Zhang, Xiangning Wang, Jun Jie Sim, Zhicong Huang, Jiahao Zhong, Huaxiong Wang, Pu Duan, Kwok Yan Lam
- Affiliation: Digital Trust Centre (DTC), Nanyang Technological University (NTU), Singapore
- Publication: IACR ePrint 2025/991, NDSS 2025
- GitHub: https://github.com/dtc2025ag/MOAI_GPU

---

## Executive Summary

This document outlines how to adopt key techniques from the MOAI paper to optimize FHE-GBDT inference. While MOAI targets Transformers, its core innovations—**rotation minimization**, **consistent packing**, and **rotation-free non-linear operations**—directly apply to decision tree evaluation.

### Expected Improvements

| Metric | Current | After MOAI Adoption | Improvement |
|--------|---------|---------------------|-------------|
| HE Rotations per inference | O(nodes × depth) | O(unique_features) | 10-50x fewer |
| Step function cost | 1 LUT per comparison | Batched rotation-free | 5-10x faster |
| Format conversions | Per-level | None | Eliminated |
| GPU utilization | ~30% | ~80% | 2.5x better |

---

## Phase 1: Column Packing for Features

### MOAI Concept
The paper uses **column packing** for weight matrices to enable rotation-free plaintext-ciphertext multiplication. We adapt this for feature vectors.

### Current Implementation
```python
# optimizer.py - Feature-major packing
feature_map[fid] = idx  # Simple slot assignment
```

### MOAI-Style Implementation

```python
# New: Column-packed feature layout
class ColumnPackedLayout:
    """
    Pack features in columns to enable rotation-free threshold comparison.

    Traditional:  ct[0] = [f0, f1, f2, ..., f_n]  (row packing)
    Column:       ct[0] = [f0, f0, f0, ..., f0]   (replicated for batch)
                  ct[1] = [f1, f1, f1, ..., f1]

    Benefit: Comparing f0 against threshold requires NO rotation.
    """

    def __init__(self, num_features: int, batch_size: int):
        self.num_features = num_features
        self.batch_size = batch_size
        # One ciphertext per feature (or group of features)
        self.num_ciphertexts = (num_features + batch_size - 1) // batch_size

    def pack_features(self, features: List[float]) -> List[Ciphertext]:
        """Pack features into column-format ciphertexts."""
        cts = []
        for i in range(self.num_ciphertexts):
            # Replicate feature across all slots
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_features)

            slots = []
            for j in range(start, end):
                # Replicate feature j across batch dimension
                slots.extend([features[j]] * self.batch_size)

            cts.append(encrypt(slots))
        return cts
```

### File Changes

| File | Change |
|------|--------|
| `services/compiler/ir.py` | Add `ColumnPackedLayout` class |
| `services/compiler/optimizer.py` | Option to use column packing |
| `sdk/python/crypto.py` | Column-packed encryption |

---

## Phase 2: Rotation-Free Step Function

### MOAI Concept
MOAI eliminates rotations in Softmax/LayerNorm using algebraic transformations. We apply similar techniques to the Step (comparison) function.

### Current Implementation
```cpp
// step_bundle.cpp - Requires rotation for each comparison
Ciphertext Step(delta, strict) {
    // Each comparison may need rotation to align slots
}
```

### MOAI-Style Rotation-Free Comparison

The key insight: with column packing, thresholds can be encoded as **plaintext polynomials** matching the ciphertext structure, eliminating rotation.

```cpp
// New: Rotation-free Step using column packing
class RotationFreeStep {
public:
    /**
     * Rotation-free comparison using MOAI's algebraic approach.
     *
     * Given: ct = Enc([f, f, f, ...])  (column-packed feature)
     *        threshold = [t, t, t, ...]  (replicated plaintext)
     *
     * Compute: delta = ct - threshold (NO rotation needed)
     *          step = Sign(delta)      (LUT or polynomial approx)
     */
    std::shared_ptr<Ciphertext> Compare(
        const Ciphertext& feature_ct,      // Column-packed feature
        const std::vector<float>& thresholds,  // Per-slot thresholds
        bool strict) {

        // Encode thresholds as plaintext polynomial
        auto threshold_pt = EncodePlaintext(thresholds);

        // Subtract: No rotation needed due to column packing!
        auto delta = ctx_->SubPlain(feature_ct, threshold_pt);

        // Apply sign extraction (rotation-free LUT)
        return ApplySignLUT(delta, strict);
    }

private:
    /**
     * MOAI-style rotation-free sign extraction.
     * Uses polynomial approximation instead of slot-wise LUT.
     */
    std::shared_ptr<Ciphertext> ApplySignLUT(
        const Ciphertext& delta, bool strict) {

        // Polynomial approximation of sign function
        // sign(x) ≈ x * (c0 + c1*x^2 + c2*x^4 + ...)
        // Coefficients from Chebyshev or minimax approximation

        constexpr int POLY_DEGREE = 7;
        std::array<double, POLY_DEGREE> coeffs = {
            1.0, -0.1667, 0.0083, -0.0002, ...  // Precomputed
        };

        auto result = ctx_->MulPlain(delta, coeffs[0]);
        auto delta_sq = ctx_->Mul(delta, delta);
        auto power = delta_sq;

        for (int i = 1; i < POLY_DEGREE; i++) {
            auto term = ctx_->MulPlain(power, coeffs[i]);
            result = ctx_->Add(result, term);
            power = ctx_->Mul(power, delta_sq);
        }

        return result;
    }
};
```

### Rotation Count Analysis

| Operation | Current | MOAI-Style | Savings |
|-----------|---------|------------|---------|
| Feature access | 1 rotation | 0 rotations | 100% |
| Threshold comparison | 1 rotation | 0 rotations | 100% |
| Step function | ~3 rotations (LUT) | 0 rotations (polynomial) | 100% |
| **Per node total** | **5 rotations** | **0 rotations** | **100%** |

For a 100-tree ensemble with depth 6:
- Current: 100 × 63 × 5 = **31,500 rotations**
- MOAI-style: **0 rotations**

---

## Phase 3: Consistent Packing Across Levels

### MOAI Concept
MOAI maintains consistent packing format across all Transformer layers, eliminating format conversion overhead.

### Current Implementation
```cpp
// executor.cpp - May change packing between levels
for (const auto& block : plan.schedule) {
    ExecuteBlock(block, ...);  // Packing might change
}
```

### MOAI-Style Consistent Packing

```cpp
// New: Maintain column packing throughout execution
class ConsistentPackingExecutor {
public:
    std::shared_ptr<Ciphertext> Execute(
        const ParsedPlan& plan,
        const std::vector<Ciphertext>& column_packed_inputs) {

        // All operations preserve column packing format
        std::vector<std::shared_ptr<Ciphertext>> tree_accumulators(plan.num_trees);

        // Initialize accumulators (already in column format)
        for (int t = 0; t < plan.num_trees; t++) {
            tree_accumulators[t] = CreateZeroCiphertext();
        }

        // Levelized execution - NO format changes
        for (const auto& block : plan.schedule) {
            ExecuteLevel(block, column_packed_inputs, tree_accumulators);
            // Packing format preserved!
        }

        // Final aggregation (still column format)
        return AggregateTreesColumnFormat(tree_accumulators);
    }

private:
    void ExecuteLevel(
        const ScheduleBlock& block,
        const std::vector<Ciphertext>& inputs,
        std::vector<std::shared_ptr<Ciphertext>>& accumulators) {

        // Group comparisons by feature (exploit column packing)
        std::unordered_map<int, std::vector<ComparisonOp>> by_feature;

        for (const auto& op : block.ops) {
            by_feature[op.feature_idx].push_back(op);
        }

        // Process each feature group (rotation-free within group)
        for (const auto& [feat_idx, ops] : by_feature) {
            // Get column-packed feature ciphertext
            const auto& feat_ct = inputs[feat_idx];

            // Batch all thresholds for this feature
            std::vector<float> thresholds;
            for (const auto& op : ops) {
                thresholds.push_back(op.threshold);
            }

            // Rotation-free batched comparison!
            auto step_results = rotation_free_step_->Compare(
                feat_ct, thresholds, false);

            // Update accumulators
            for (size_t i = 0; i < ops.size(); i++) {
                int tree_idx = ops[i].tree_idx;
                accumulators[tree_idx] = ctx_->Add(
                    *accumulators[tree_idx], *step_results);
            }
        }
    }
};
```

---

## Phase 4: Phantom-FHE Integration

### MOAI's GPU Backend
MOAI uses Phantom-FHE for GPU acceleration. We should integrate it alongside N2HE.

### Integration Plan

```cpp
// New: Phantom-FHE backend option
// services/runtime/kernel/phantom_backend.h

#pragma once
#include "backend.h"
#include <phantom.h>  // Phantom-FHE header

namespace fhe_gbdt::kernel {

class PhantomBackend : public Backend {
public:
    PhantomBackend(const PhantomParams& params) {
        // Initialize Phantom context
        context_ = phantom::Context::Create(params);
        encoder_ = phantom::CKKSEncoder(context_);
        evaluator_ = phantom::Evaluator(context_);
    }

    std::shared_ptr<Ciphertext> Add(
        const Ciphertext& a, const Ciphertext& b) override {
        // Use Phantom's GPU-accelerated addition
        phantom::Ciphertext result;
        evaluator_.add(ToPhantom(a), ToPhantom(b), result);
        return FromPhantom(result);
    }

    // Rotation-free operations leverage Phantom's optimized kernels
    std::shared_ptr<Ciphertext> SubPlain(
        const Ciphertext& ct, const Plaintext& pt) override {
        phantom::Ciphertext result;
        evaluator_.sub_plain(ToPhantom(ct), ToPhantom(pt), result);
        return FromPhantom(result);
    }

private:
    phantom::Context context_;
    phantom::CKKSEncoder encoder_;
    phantom::Evaluator evaluator_;
};

} // namespace fhe_gbdt::kernel
```

### CMake Integration

```cmake
# CMakeLists.txt addition
option(USE_PHANTOM_FHE "Use Phantom-FHE for GPU acceleration" OFF)

if(USE_PHANTOM_FHE)
    add_subdirectory(third_party/phantom-fhe)
    target_link_libraries(fhe_runtime PRIVATE phantom::phantom)
    target_compile_definitions(fhe_runtime PRIVATE USE_PHANTOM_FHE)
endif()
```

---

## Phase 5: Interleaved Batching for Aggregation

### MOAI Concept
MOAI uses interleaved batching to reduce rotations in ciphertext-ciphertext operations.

### Application to Tree Aggregation

```cpp
// New: Interleaved batching for tree sum
class InterleavedAggregator {
public:
    /**
     * Aggregate tree outputs using interleaved batching.
     *
     * Instead of: result = t0 + t1 + t2 + ... + t_{n-1}  (n-1 additions)
     *
     * Use interleaved:
     *   Pack trees in interleaved format
     *   Use rotation + add pattern to sum in log(n) steps
     */
    std::shared_ptr<Ciphertext> AggregateInterleaved(
        const std::vector<std::shared_ptr<Ciphertext>>& tree_outputs,
        int batch_size) {

        int num_trees = tree_outputs.size();

        // Pack into interleaved format
        // [t0_s0, t1_s0, t2_s0, ..., t0_s1, t1_s1, ...]
        auto packed = PackInterleaved(tree_outputs, batch_size);

        // Log-reduction with minimal rotations
        int stride = 1;
        while (stride < num_trees) {
            // Rotate by stride and add
            auto rotated = ctx_->Rotate(packed, stride);
            packed = ctx_->Add(packed, rotated);
            stride *= 2;
        }

        // Extract final sum (in slot 0 of each batch element)
        return packed;
    }

private:
    std::shared_ptr<Ciphertext> PackInterleaved(
        const std::vector<std::shared_ptr<Ciphertext>>& trees,
        int batch_size) {

        // Interleave tree outputs for efficient reduction
        std::vector<int64_t> slots(batch_size * trees.size());

        for (int s = 0; s < batch_size; s++) {
            for (size_t t = 0; t < trees.size(); t++) {
                slots[s * trees.size() + t] =
                    trees[t]->rlwe_data[0][s];
            }
        }

        return Encrypt(slots);
    }
};
```

### Rotation Analysis for Aggregation

| Method | Rotations | Additions |
|--------|-----------|-----------|
| Sequential | 0 | n-1 |
| Interleaved | log(n) | log(n) |

For 100 trees:
- Sequential: 0 rotations, 99 additions
- Interleaved: 7 rotations, 7 additions (14x fewer additions)

---

## Implementation Roadmap

### Week 1-2: Column Packing Foundation

| Task | File | Priority |
|------|------|----------|
| Add `ColumnPackedLayout` class | `services/compiler/ir.py` | P0 |
| Update optimizer for column packing | `services/compiler/optimizer.py` | P0 |
| Modify SDK encryption | `sdk/python/crypto.py` | P0 |
| Add column-packed test cases | `tests/unit/test_column_packing.py` | P1 |

### Week 3-4: Rotation-Free Step

| Task | File | Priority |
|------|------|----------|
| Implement `RotationFreeStep` class | `services/runtime/kernel/rotation_free_step.h` | P0 |
| Polynomial sign approximation | `services/runtime/kernel/polynomial_approx.h` | P0 |
| Integrate with executor | `services/runtime/engine/executor.cpp` | P0 |
| Benchmark rotation counts | `bench/rotation_benchmark.py` | P1 |

### Week 5-6: Phantom-FHE Integration

| Task | File | Priority |
|------|------|----------|
| Add Phantom-FHE submodule | `third_party/phantom-fhe/` | P0 |
| Implement PhantomBackend | `services/runtime/kernel/phantom_backend.h` | P0 |
| CMake integration | `services/runtime/CMakeLists.txt` | P0 |
| Backend selection logic | `services/runtime/kernel/crypto_context.cpp` | P1 |

### Week 7-8: Consistent Packing & Aggregation

| Task | File | Priority |
|------|------|----------|
| Implement consistent packing executor | `services/runtime/engine/consistent_executor.cpp` | P0 |
| Interleaved aggregation | `services/runtime/engine/interleaved_agg.h` | P1 |
| End-to-end benchmarks | `bench/e2e_moai_benchmark.py` | P0 |
| Documentation | `docs/MOAI_USAGE.md` | P2 |

---

## Performance Targets

Based on MOAI paper results scaled to GBDT:

| Metric | Current Estimate | MOAI Target |
|--------|------------------|-------------|
| Rotations (100 trees, depth 6) | 31,500 | <100 |
| Inference latency (batch=256) | ~500ms | <100ms |
| GPU utilization | ~30% | >75% |
| Memory bandwidth efficiency | ~20% | >60% |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Polynomial approximation accuracy | Medium | High | Use high-degree Chebyshev, validate numerically |
| Phantom-FHE compatibility | Low | Medium | Maintain N2HE fallback |
| Column packing memory overhead | Medium | Low | Lazy materialization, streaming |
| Performance regression on small models | Low | Low | Adaptive strategy selection |

---

## Conclusion

Adopting MOAI techniques can reduce HE rotations by **99%+** and improve inference latency by **5-10x**. The key changes are:

1. **Column packing** - Replicate features for rotation-free access
2. **Rotation-free Step** - Polynomial approximation of sign function
3. **Consistent packing** - No format conversions between levels
4. **Phantom-FHE** - GPU-optimized HE operations
5. **Interleaved aggregation** - Log-reduction for tree summation

This plan provides a path to production-ready FHE-GBDT inference with performance comparable to the MOAI paper's Transformer results.

---

*Plan created: 2026-02-01*
*Reference: MOAI (IACR ePrint 2025/991)*
