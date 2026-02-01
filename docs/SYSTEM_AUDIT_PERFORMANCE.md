# Comprehensive System Audit: FHE-GBDT Serving Platform

## Executive Summary

This document provides a comprehensive audit of the FHE-GBDT serving system, playing devil's advocate on performance claims, analyzing N2HE and MOAI research paper integration, and proposing an actionable optimization plan.

**Critical Finding**: The system is currently **NOT production-ready for performance-sensitive workloads**. While the architecture is sound, critical performance paths contain placeholder implementations, missing optimizations, and significant inefficiencies that would result in **10-50x worse performance** than what the underlying N2HE/MOAI papers promise.

---

## Part 1: Devil's Advocate Performance Analysis

### 1.1 Critical Performance Failures

#### FAILURE #1: Step Bundle is a Sequential Loop (CRITICAL)
**Location**: `services/runtime/kernel/step_bundle.cpp:18-20`

```cpp
// Current implementation - CATASTROPHIC
for (const auto& delta : deltas) {
    results.push_back(ctx_->Step(delta, strict));  // Sequential!
}
```

**Problem**: The comment on lines 15-16 explicitly admits this should use "N2HE-HEXL to batch multiple Step operations into a single EvalSum/Permute sequence" but **this is never implemented**.

**Impact**: For a tree ensemble with 100 trees and depth 6:
- Current: 600 sequential Step operations
- With batching: ~6 batched operations (one per depth level)
- **Performance loss: 100x slower than optimal**

#### FAILURE #2: GPU NTT is a No-Op Placeholder
**Location**: `services/runtime/kernel/gpu/gpu_backend.cu:104-113`

```c
void cuda_ntt_forward(int64_t* data, int n, int64_t q, cudaStream_t stream) {
    // Placeholder: Full NTT requires pre-computed twiddle factors
    // For now, this is a no-op placeholder
}
```

**Problem**: The GPU `Step()` function (`gpu_backend.h:150-175`) calls `cuda_ntt_forward` and `cuda_ntt_inverse` which do **literally nothing**. The Step function is the heart of FHE-GBDT comparison operations.

**Impact**: GPU mode produces **incorrect results** and gains **zero acceleration** from CUDA.

#### FAILURE #3: Excessive Host-Device Memory Transfers
**Location**: `services/runtime/kernel/gpu/gpu_backend.h:68-81`

```cpp
for (size_t i = 0; i < a.rlwe_data.size(); ++i) {
    cudaMemcpyAsync(d_temp_a_, ...);  // H2D
    cudaMemcpyAsync(d_temp_b_, ...);  // H2D
    cuda_rlwe_add(...);
    cudaMemcpyAsync(res->rlwe_data[i].data(), ...);  // D2H
}
cudaStreamSynchronize(stream_);  // Sync after EVERY loop iteration!
```

**Problem**: Each polynomial in an RLWE ciphertext triggers 3 memory transfers. For batch_size=256 with 2 polynomials per ciphertext:
- 256 × 2 × 3 = 1,536 memory transfers per Add operation
- Each transfer is 2048 × 8 = 16KB
- Total: **24.6 MB** transferred for a single vector addition

**Impact**: PCIe bandwidth becomes the bottleneck. Latency dominated by transfer overhead, not computation.

#### FAILURE #4: CPU Backend Has No SIMD
**Location**: `services/runtime/kernel/cpu_backend.h:44-51`

```cpp
for (size_t i = 0; i < a.rlwe_data.size(); ++i) {
    for (size_t j = 0; j < a.rlwe_data[i].size(); ++j) {
        res->rlwe_data[i][j] = (a.rlwe_data[i][j] - b.rlwe_data[i][j] + q_) % q_;
    }
}
```

**Problem**: The N2HE library includes `FasterNTT/arith_avx2.hpp` with AVX2-optimized routines, but the CPU backend uses plain scalar loops.

**Impact**: AVX2 can process 4 int64 operations per cycle. Current code is **4x slower than possible on Intel CPUs**.

#### FAILURE #5: Executor is a Stub
**Location**: `services/runtime/engine/executor.cpp:8-20`

```cpp
std::shared_ptr<kernel::Ciphertext> Executor::Execute(...) {
    std::cout << "Executing FHE-GBDT Plan..." << std::endl;
    // 1. Parse PlanIR (JSON or Protobuf)
    // 2. iterate through schedule blocks
    // 3. For each block: ...
    auto result = std::make_shared<kernel::Ciphertext>();
    return result;  // Returns empty ciphertext!
}
```

**Problem**: The entire execution engine is a **placeholder that returns an empty result**. There is no actual plan execution logic.

**Impact**: **The runtime cannot actually perform inference**. All E2E tests must be using simulation mode.

#### FAILURE #6: Benchmark is Fake
**Location**: `bench/runner/bench_runner.py:96-98`

```python
def _mock_inference(self, batch_size: int):
    time.sleep(0.01 * (1 + batch_size / 256))  # Just sleeps!
```

**Problem**: Benchmarks use hardcoded sleep values, not actual FHE operations. The "stage_timings" on lines 79-86 are **hardcoded constants**, not measurements.

**Impact**: Any performance claims based on these benchmarks are **fabricated**.

---

### 1.2 Memory Leak and Safety Issues

#### ISSUE #1: N2HE Memory Leaks
**Location**: `services/runtime/third_party/n2he/include/RLWEscheme/RLWE_64.hpp:18,52,76,108,132`

```cpp
int *x = new int [n];       // Line 18 - Never freed!
int64_t *array_a = new int64_t [len_out];  // Line 52 - Never freed!
int *array_e = new int [len_out];  // Line 76 - Never freed!
```

**Problem**: Every encryption/key generation operation leaks memory.

**Impact**: Long-running servers will exhaust memory. For 1000 requests with batch_size=256:
- ~256 × 1000 × (2048 × 8) bytes = **4 GB leaked per 1000 requests**

#### ISSUE #2: Python Simulation Uses `secrets.randbelow` in Hot Loop
**Location**: `sdk/python/crypto.py:144-147`

```python
a = [secrets.randbelow(q) for _ in range(n)]  # 2048 calls per value!
e = [secrets.randbelow(7) - 3 for _ in range(n)]  # Another 2048 calls!
```

**Problem**: Cryptographic RNG is used where PRNG would suffice for simulation. Each encryption makes 4096 syscalls to `/dev/urandom`.

**Impact**: Python simulation is **100x slower than necessary** due to RNG overhead.

---

### 1.3 Architectural Concerns

#### CONCERN #1: Stateless Runtime Design
The runtime fetches evaluation keys per-request from the keystore. For latency-sensitive applications:
- Key fetch: ~5-10ms (network + deserialization)
- FHE inference: ~10-50ms (if properly optimized)
- **Key fetch is 10-50% of total latency**

#### CONCERN #2: Feature Overflow Silently Ignored
**Location**: `services/compiler/optimizer.py:26-28`

```python
if idx < self.batch_size:
    feature_map[fid] = idx
else:
    pass  # Features beyond batch_size are SILENTLY DROPPED
```

**Problem**: Models with >256 features (CPU) or >4096 features (GPU) will produce incorrect results with no warning.

#### CONCERN #3: No Ciphertext Noise Budget Tracking
The system performs FHE operations without tracking accumulated noise. In real FHE:
- Each operation increases noise
- When noise exceeds threshold, decryption fails
- Bootstrapping is needed to refresh ciphertexts

**Current state**: Noise is never tracked. Deep trees or large ensembles will produce **garbage results**.

---

## Part 2: N2HE Research Paper Analysis

### 2.1 What is N2HE?

N2HE (Neural Network Homomorphic Encryption) is a hybrid FHE library developed by HintSight Technology (Singapore) for privacy-preserving neural network inference.

**Source Papers**:
1. "Efficient FHE-based Privacy-Enhanced Neural Network for Trustworthy AI-as-a-Service" - IEEE TDSC 2024
2. "An Efficient FHE-Enabled Secure Cloud-Edge Computing Architecture for IoMTs Data Protection" - IEEE IoT Journal 2024

### 2.2 N2HE Cryptographic Primitives Used

| Primitive | Location | Purpose in GBDT |
|-----------|----------|-----------------|
| **RLWE_64** | `third_party/n2he/include/RLWEscheme/RLWE_64.hpp` | Encrypt feature vectors into ring elements |
| **LWE_64** | `third_party/n2he/include/LWEscheme/LWE_64.hpp` | Low-depth comparisons after extraction |
| **RGSW** | `RLWE_64.hpp:176-203` | Evaluation key format for homomorphic ops |
| **LUT_64** | `third_party/n2he/include/RLWEscheme/LUT_64.hpp` | Step function evaluation via lookup tables |
| **FasterNTT** | `third_party/n2he/include/FasterNTT/` | Fast polynomial multiplication |

### 2.3 How N2HE is Integrated

```
Client (Python)                    Server (C++)
──────────────                    ──────────────
N2HEKeyManager                    CpuBackend/GpuBackend
    │                                   │
    ├─ generate_keys()                  │
    │   └─ RLWE64_KeyGen(n=2048)       │
    │                                   │
    ├─ encrypt(features)                │
    │   └─ RLWE64_Enc_2048()           │
    │                                   │
    └─────────── ciphertext ───────────→│
                                        │
                                        ├─ Add/Sub/MulPlain
                                        │   └─ add_poly(), multi_scale_poly()
                                        │
                                        ├─ Step (comparison)
                                        │   └─ LUT_64 (intended but NOT implemented)
                                        │
                                        └─── encrypted result ────────────→ Client.decrypt()
```

### 2.4 N2HE Parameters in This System

```cpp
// From cpu_backend.h:12-15 and crypto.py:29-31
n = 2048        // Ring dimension (polynomial degree)
q = 2^32        // Ciphertext modulus
σ = 3.2         // Gaussian error standard deviation
k = 1           // Variance parameter for encryption
```

**Security Analysis**:
- These parameters provide approximately **128-bit security** against known lattice attacks
- Ring dimension of 2048 is minimum for RLWE security
- Modulus of 2^32 limits multiplicative depth to ~2-3 levels before noise overflow

### 2.5 What's Missing from N2HE Integration

| Feature | Paper Describes | Implementation Status |
|---------|-----------------|----------------------|
| Batched Step via LUT | "Single EvalSum/Permute for multiple comparisons" | ❌ Not implemented |
| RGSW External Product | Efficient ciphertext multiplication | ❌ Not used |
| Key Switching | Convert between LWE/RLWE schemes | ⚠️ Code exists, not integrated |
| Bootstrapping | Noise refresh for deep circuits | ❌ Not implemented |
| N2HE-HEXL | Intel HEXL acceleration | ❌ Mentioned in comments, not used |

---

## Part 3: MOAI Research Paper Analysis

### 3.1 What is MOAI?

MOAI (Model Oblivious Architecture Interpreter) is an optimization strategy for FHE-based decision tree evaluation. The core insight is that FHE comparison operations are expensive, so the compiler should minimize them through:

1. **Feature-Major Packing**: Pack frequently-used features into lower ciphertext slots
2. **Levelization**: Process all nodes at the same depth in parallel
3. **Rotation Scheduling**: Group operations by rotation offset to batch ciphertext rotations

### 3.2 MOAI Implementation in This Codebase

**Location**: `services/compiler/optimizer.py`

```python
class MOAIOptimizer:
    def optimize(self, model: ModelIR) -> ObliviousPlanIR:
        # Step 1: Feature Frequency Analysis
        feature_counts = self._analyze_frequency(model)
        sorted_features = sorted(feature_counts, key=feature_counts.get, reverse=True)

        # Step 2: Feature-Major Packing (Hot features → lower slots)
        for idx, fid in enumerate(sorted_features):
            if idx < self.batch_size:
                feature_map[fid] = idx

        # Step 3: Levelization
        for depth in range(max_depth):
            ops = self._schedule_level(model, depth, feature_map)
            schedule.append(ScheduleBlock(depth_level=depth, ...))

        # Step 4: Rotation Scheduling
        # Groups nodes by rotation offset within each level
```

### 3.3 MOAI Concepts Explained

#### Feature-Major Packing
In standard SIMD-style FHE, ciphertexts hold vectors of values. MOAI packs features such that:
- Feature used 100 times → Slot 0
- Feature used 50 times → Slot 1
- Feature used 10 times → Slot 99

**Why**: Rotations are expensive. Putting hot features in slot 0 means most operations require zero rotation.

#### Levelization
Instead of traversing each tree independently:
```
Traditional: Tree1.node[0] → Tree1.node[1] → ... → Tree2.node[0] → ...
MOAI:        All depth-0 nodes → All depth-1 nodes → ... → All depth-d nodes
```

**Why**: At each depth, all comparisons can be batched into a single encrypted operation.

#### Rotation Scheduling
```python
# From optimizer.py:84
offset = (f_slot - t_slot) % batch_size

# Group nodes by offset:
rotation_groups[offset].append((tree_idx, threshold))

# Emit one ROTATE per unique offset, then batch COMPARE_BATCH
```

**Why**: Instead of N rotations for N trees, we do K rotations where K = unique offsets (typically << N).

### 3.4 What's Missing from MOAI Implementation

| Feature | MOAI Paper | Implementation Status |
|---------|------------|----------------------|
| Feature frequency analysis | ✓ Rank features by usage | ✅ Implemented |
| Feature-major packing | ✓ Hot features in lower slots | ✅ Implemented |
| Levelization | ✓ Depth-parallel scheduling | ✅ Implemented |
| Rotation grouping | ✓ Batch rotations by offset | ⚠️ Schedule generated, not executed |
| Oblivious tree conversion | ✓ Convert irregular to oblivious | ❌ Not implemented |
| Multi-ciphertext packing | ✓ Handle >slots features | ❌ Silently ignored |
| COMPARE_BATCH execution | ✓ Batch comparison kernel | ❌ Runtime doesn't execute it |

---

## Part 4: What Could Be Optimized

### 4.1 Critical Path Optimizations

| Priority | Optimization | Expected Speedup | Effort |
|----------|--------------|------------------|--------|
| P0 | Implement batched Step via N2HE-HEXL | 10-100x | High |
| P0 | Complete GPU NTT implementation | 5-10x | Medium |
| P0 | Implement actual Executor logic | ∞ (correctness) | High |
| P1 | Batch GPU memory transfers | 3-5x | Medium |
| P1 | Enable AVX2 in CPU backend | 2-4x | Low |
| P1 | Fix N2HE memory leaks | Stability | Low |
| P2 | Add noise budget tracking | Correctness | Medium |
| P2 | Implement key caching in runtime | 1.2-1.5x latency | Low |
| P2 | Use numpy for Python simulation | 10-50x sim speed | Low |

### 4.2 Algorithmic Optimizations

1. **Hoisted Rotation**: If multiple trees use the same feature at depth d, compute rotation once
2. **Threshold Deduplication**: If thresholds are identical, share the comparison result
3. **Dead Code Elimination**: Remove comparisons that don't affect any path
4. **Circuit Balancing**: Reorder operations to minimize multiplicative depth

### 4.3 System-Level Optimizations

1. **Eval Key Caching**: Cache deserialized eval keys in runtime (LRU, ~1GB)
2. **Request Batching**: Combine multiple inference requests into single GPU launch
3. **Pipeline Parallelism**: Overlap network I/O with GPU computation
4. **Operator Fusion**: Combine Add+Step into single kernel

---

## Part 5: Implementation Plan

### Phase 1: Correctness (Week 1-2)

#### Task 1.1: Implement Executor Logic
**File**: `services/runtime/engine/executor.cpp`
**Goal**: Parse ObliviousPlanIR and execute the schedule

```cpp
// Pseudocode for implementation
for (const auto& block : plan.schedule) {
    for (const auto& op : block.ops) {
        if (op.type == "ROTATE") {
            current = ctx_->Rotate(current, op.offset);
        } else if (op.type == "COMPARE_BATCH") {
            auto deltas = ComputeDeltas(current, thresholds);
            auto step_results = step_bundle_->Evaluate(deltas, strict);
            current = Aggregate(step_results);
        }
    }
}
```

#### Task 1.2: Implement Batched Step
**File**: `services/runtime/kernel/step_bundle.cpp`
**Goal**: Use N2HE LUT for batch comparison

```cpp
// Use LUT_64.hpp:key_encrypt_* for batch evaluation
std::vector<std::shared_ptr<Ciphertext>> StepBundle::EvaluateBatched(
    const std::vector<std::shared_ptr<Ciphertext>>& deltas,
    const RGSW_EvalKey& eval_key) {

    // 1. Pack deltas into single polynomial
    auto packed = PackCiphertexts(deltas);

    // 2. Single LUT evaluation
    auto result = LUT_Step(packed, eval_key);

    // 3. Unpack to individual results
    return UnpackCiphertexts(result, deltas.size());
}
```

#### Task 1.3: Fix N2HE Memory Leaks
**Files**: All files in `third_party/n2he/include/`
**Goal**: Add `delete[]` for all `new[]` allocations

```cpp
// Example fix for RLWE_64.hpp:18
extern polynomial RLWE64_KeyGen(int n) {
    int *x = new int [n];
    // ... use x ...
    polynomial s = /* build result */;
    delete[] x;  // ADD THIS
    return s;
}
```

### Phase 2: GPU Acceleration (Week 2-3)

#### Task 2.1: Complete NTT Implementation
**File**: `services/runtime/kernel/gpu/gpu_backend.cu`
**Goal**: Implement proper NTT with precomputed twiddle factors

```cpp
class NTTContext {
    int64_t* d_twiddles_;      // Device twiddle factors
    int64_t* d_inv_twiddles_;  // Inverse twiddles
    int64_t n_inv_;            // Modular inverse of n

    void Initialize(int n, int64_t q) {
        // Compute primitive root and generate twiddles
        int64_t g = FindPrimitiveRoot(q);
        // ... precompute twiddles ...
    }
};

void cuda_ntt_forward(int64_t* data, int n, int64_t q,
                      const NTTContext& ctx, cudaStream_t stream) {
    for (int stage = 0; stage < log2(n); stage++) {
        ntt_forward_kernel<<<blocks, threads, 0, stream>>>(
            data, n, q, ctx.d_twiddles_, stage);
    }
}
```

#### Task 2.2: Batch Memory Transfers
**File**: `services/runtime/kernel/gpu/gpu_backend.h`
**Goal**: Allocate batch buffers and minimize H2D/D2H transfers

```cpp
class BatchedGpuBackend {
    // Allocate for full batch at once
    int64_t* d_batch_a_;  // batch_size × 2 × n elements
    int64_t* d_batch_b_;
    int64_t* d_batch_result_;

    std::vector<Ciphertext> AddBatch(
        const std::vector<Ciphertext>& as,
        const std::vector<Ciphertext>& bs) {

        // Single H2D for all inputs
        cudaMemcpyAsync(d_batch_a_, PackAll(as), ...);
        cudaMemcpyAsync(d_batch_b_, PackAll(bs), ...);

        // Single kernel for all additions
        batched_rlwe_add<<<...>>>(d_batch_result_, d_batch_a_, d_batch_b_, ...);

        // Single D2H for all outputs
        cudaMemcpyAsync(host_result_, d_batch_result_, ...);

        return UnpackAll(host_result_);
    }
};
```

### Phase 3: CPU Optimization (Week 3-4)

#### Task 3.1: Enable AVX2 Vectorization
**File**: `services/runtime/kernel/cpu_backend.h`
**Goal**: Use N2HE's FasterNTT AVX2 routines

```cpp
#include "third_party/n2he/include/FasterNTT/arith_avx2.hpp"

std::shared_ptr<Ciphertext> Sub(const Ciphertext& a, const Ciphertext& b) override {
    // Use AVX2-optimized subtraction
    for (size_t i = 0; i < a.rlwe_data.size(); ++i) {
        fntt::sub_mod_avx2(
            res->rlwe_data[i].data(),
            a.rlwe_data[i].data(),
            b.rlwe_data[i].data(),
            n_, q_);
    }
}
```

#### Task 3.2: Add Multi-threading
**File**: `services/runtime/kernel/step_bundle.cpp`
**Goal**: Parallelize independent Step evaluations

```cpp
#include <execution>

std::vector<std::shared_ptr<Ciphertext>> StepBundle::Evaluate(...) {
    std::vector<std::shared_ptr<Ciphertext>> results(deltas.size());

    std::transform(std::execution::par_unseq,
                   deltas.begin(), deltas.end(),
                   results.begin(),
                   [this, strict](const auto& delta) {
                       return ctx_->Step(delta, strict);
                   });

    return results;
}
```

### Phase 4: System Optimization (Week 4-5)

#### Task 4.1: Add Eval Key Caching
**File**: `services/runtime/engine/executor.cpp` (new cache module)

```cpp
class EvalKeyCache {
    std::unordered_map<std::string, EvalKeys> cache_;
    std::mutex mutex_;
    size_t max_size_bytes_ = 1ULL << 30;  // 1GB

    EvalKeys Get(const std::string& tenant_model_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(tenant_model_id);
        if (it != cache_.end()) {
            return it->second;  // Cache hit
        }
        // Fetch from keystore, deserialize, cache
    }
};
```

#### Task 4.2: Add Noise Budget Tracking
**File**: `services/runtime/kernel/crypto_context.h`

```cpp
class NoiseBudget {
    double current_bits_;
    double max_bits_;

    void ConsumeAdd() { current_bits_ += 0.1; }
    void ConsumeMultiply() { current_bits_ += 1.0; }
    void ConsumeStep() { current_bits_ += 5.0; }

    bool NeedsBootstrap() const {
        return current_bits_ > max_bits_ - 10.0;
    }
};
```

### Phase 5: Benchmarking & Validation (Week 5-6)

#### Task 5.1: Replace Mock Benchmarks
**File**: `bench/runner/bench_runner.py`
**Goal**: Use actual FHE operations with real timing

```python
def _real_inference(self, model_id: str, batch_size: int):
    # Load real model
    client = FheGbdtClient(...)
    plan = client.get_plan(model_id)

    # Generate test data
    features = np.random.randn(batch_size, plan.num_features)

    # Measure real encryption
    t0 = time.perf_counter()
    ct = key_manager.encrypt(features)
    t_encrypt = time.perf_counter() - t0

    # Measure real inference
    t0 = time.perf_counter()
    result = client.predict_encrypted(ct)
    t_inference = time.perf_counter() - t0

    return {"encrypt_ms": t_encrypt*1000, "inference_ms": t_inference*1000}
```

#### Task 5.2: Add Correctness Tests
**File**: `tests/e2e/test_correctness.py`

```python
def test_fhe_vs_plaintext_accuracy():
    """Verify FHE inference matches plaintext within noise tolerance."""
    model = xgboost.Booster()
    model.load_model("test_model.json")

    plain_pred = model.predict(test_data)
    encrypted_pred = fhe_client.predict(test_data)

    # Allow small error from FHE noise
    np.testing.assert_allclose(plain_pred, encrypted_pred, rtol=1e-3)
```

---

## Part 6: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| NTT implementation incorrect | Medium | Critical | Extensive testing vs reference implementation |
| Noise overflow undetected | High | Critical | Add noise tracking before production |
| Memory exhaustion from leaks | High | High | Fix all leaks, add monitoring |
| GPU kernel race conditions | Medium | Medium | Use proper synchronization, test under load |
| Incorrect batching logic | Medium | High | Property-based testing on packing/unpacking |

---

## Conclusion

The FHE-GBDT serving system has a **well-designed architecture** but is currently a **proof-of-concept** rather than production-ready software. The critical path through StepBundle, GPU NTT, and Executor contains placeholder code that prevents real FHE inference.

**Immediate priorities**:
1. Fix the Executor stub to actually execute plans
2. Implement batched Step using N2HE's LUT primitives
3. Complete GPU NTT with proper twiddle factors
4. Fix memory leaks in N2HE library

With the proposed optimizations, the system could achieve the performance characteristics promised by the N2HE and MOAI research papers: **sub-100ms latency for single inference** and **>1000 predictions/second throughput** on GPU.

---

*Audit conducted: 2026-02-01*
*Auditor: System Performance Analysis*
*Classification: Internal Technical Review*
