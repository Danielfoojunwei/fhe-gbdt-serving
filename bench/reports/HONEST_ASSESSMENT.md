# Honest System Assessment: What Is Real and What Is Not

**Date**: 2026-02-10
**Auditor**: Automated code audit + empirical verification
**Verdict**: This system is a **plaintext simulation framework**. It performs ZERO encrypted computation.

---

## Executive Summary

This system claims to perform privacy-preserving GBDT inference using Fully Homomorphic Encryption. **It does not.** Every computation runs on plaintext numpy arrays. The N2HE encryption library exists in C++ but has no Python bindings — all Python code falls back to `_simulate_encrypt()` which provides zero cryptographic security.

---

## 1. What Is Real

| Component | Status | Evidence |
|---|---|---|
| Tree structure conversion (MOAI) | **Real algorithm** | Converts non-oblivious → oblivious trees. Plaintext only. |
| Polynomial sign approximation | **Real algorithm** | Minimax polynomial for step function. Plaintext only. |
| Ensemble pruning with rescaling | **Real algorithm** | E[X²] significance metric, magnitude-preserving. Plaintext only. |
| Gradient-aware noise allocation | **Real algorithm** | Feature importance → precision bits. Plaintext only. |
| Bootstrap alignment | **Real algorithm** | Noise budget partitioning. Plaintext only. |
| XGBoost/LightGBM → ModelIR conversion | **Real, verified** | MSE < 1e-9 vs original model predictions |
| C++ N2HE library | **Real code exists** | `services/runtime/third_party/n2he/` has real RLWE crypto |

## 2. What Is Fake

| Claim | Reality | Evidence |
|---|---|---|
| "Encrypted inference" | **Plaintext numpy operations** | Every `predict()` call uses `np.ndarray`, never ciphertext |
| "Homomorphic pruning" | **`prune_plaintext()` always called** | `unified_architecture.py:452` calls plaintext version |
| "Privacy-preserving" | **Zero privacy** | `sdk/python/crypto.py` falls back to `_simulate_encrypt()` |
| "N2HE integration" | **No Python bindings** | `import n2he_native` fails, simulation fallback always used |
| "Encrypted tree traversal" | **Dead code** | `prune_encrypted()` exists but is never called |
| "FHE latency benchmarks" | **Never measured** | All timing is plaintext numpy, not ciphertext operations |
| "128-bit security" | **No encryption occurs** | Fake ciphertext = deterministic encoding of plaintext |

## 3. How The Simulation Works

```python
# sdk/python/crypto.py - What actually happens when you "encrypt"
def encrypt(self, values):
    if self._use_native:
        # NEVER EXECUTES - n2he_native doesn't exist
        ciphertext = n2he_native.encrypt(...)
    else:
        # ALWAYS EXECUTES
        return self._simulate_encrypt(values)  # Deterministic plaintext encoding

# What _simulate_encrypt does:
# 1. Takes plaintext value: 0.5
# 2. Scales to integer: int(0.5 * 2^30)
# 3. Packs into bytes with random padding
# 4. Returns "ciphertext" blob that trivially decodes back to 0.5
# 5. Zero cryptographic operations
```

## 4. Real FHE Comparison: Concrete ML

We installed Concrete ML (Zama) — a real FHE framework that actually encrypts data and computes on ciphertexts. Here are **real FHE numbers** measured on the same machine:

### Breast Cancer (30 features, 50 trees, depth 5)

| System | Accuracy | Latency per sample | Encryption | Privacy |
|---|---|---|---|---|
| **Concrete ML (3-bit FHE)** | **98.25%** | **1,523ms** | REAL (TFHE) | YES |
| **Concrete ML (5-bit FHE)** | **97.08%** | **3,543ms** | REAL (TFHE) | YES |
| **Concrete ML (7-bit FHE)** | **97.66%** | **4,101ms** | REAL (TFHE) | YES |
| This system (plaintext) | 94.15% | ~5ms | NONE | NO |
| This system + Leaf-Centric | 95.32% | ~130ms | NONE | NO |
| This system + MOAI | 68.42% | ~370ms | NONE | NO |

### Iris Binary (4 features, 20 trees, depth 3)

| System | Accuracy | Latency per sample | Encryption | Privacy |
|---|---|---|---|---|
| **Concrete ML (3-bit FHE)** | **100.0%** | **402ms** | REAL (TFHE) | YES |
| **Concrete ML (5-bit FHE)** | **100.0%** | **768ms** | REAL (TFHE) | YES |
| **Concrete ML (7-bit FHE)** | **100.0%** | **1,002ms** | REAL (TFHE) | YES |
| This system (plaintext) | 100.0% | ~1ms | NONE | NO |

### Key Observations

1. **Concrete ML achieves 97-100% accuracy with REAL encryption** (actual TFHE ciphertexts, actual homomorphic operations, actual cryptographic keys)

2. **This system achieves 68-95% accuracy with NO encryption** (plaintext numpy arrays with structural tree conversions)

3. **Concrete ML latency is 400-4,100ms per sample** — this is the REAL cost of FHE. Our system's ~5ms "latency" is meaningless because it's not doing FHE.

4. **Concrete ML provides actual privacy guarantees**. This system provides zero privacy.

---

## 5. What The Innovations Actually Measure

Since no FHE computation occurs, what the benchmarks actually measure is:

| Innovation | What it claims to measure | What it actually measures |
|---|---|---|
| Leaf-Centric | "FHE leaf indicator evaluation" | Polynomial sign approximation error on plaintext |
| MOAI | "Rotation-free FHE evaluation" | Accuracy loss from tree structure conversion |
| Pruning | "Encrypted ensemble pruning" | Effect of dropping trees on prediction quality |
| Gradient Noise | "FHE noise budget optimization" | Quantization error from reduced precision |
| Polynomial Leaves | "FHE polynomial evaluation" | Whether additive corrections help predictions |
| Bootstrap Alignment | "FHE noise management" | Tree partitioning (no accuracy impact) |

These are **valid algorithmic studies** — understanding whether these transformations preserve accuracy is a necessary precondition for building a real FHE system. But they are NOT FHE benchmarks and should not be presented as such.

---

## 6. What Would Be Needed For Real FHE

To make this system perform actual FHE inference:

1. **Compile N2HE with Python bindings** (pybind11) — the C++ code exists but is disconnected
2. **Or integrate Concrete ML** — already works, provides real TFHE operations
3. **Or integrate OpenFHE/SEAL** — established C++ FHE libraries with Python bindings
4. **Replace all `_plaintext()` calls** with actual ciphertext operations
5. **Measure real FHE latency** — expect 100-5,000ms per inference, not 5ms
6. **Validate accuracy under real noise** — polynomial approximation error compounds with FHE noise

---

## 7. Files Confirming This Assessment

| File | Line | Evidence |
|---|---|---|
| `sdk/python/crypto.py` | 53 | `"WARN: N2HE native bindings not available, using simulation"` |
| `sdk/python/client.py` | 60-65 | `_simulate_backend_processing()` — sleeps and returns plaintext |
| `services/innovations/unified_architecture.py` | 452 | Calls `prune_plaintext()`, never `prune_encrypted()` |
| `services/innovations/homomorphic_pruning.py` | 430-470 | `prune_encrypted()` is dead code |
| `services/runtime/CMakeLists.txt` | 42-44 | Falls back to `NO_N2HE` if library not found |

---

## 8. Reproducibility

```bash
# Install Concrete ML for real FHE comparison
pip install concrete-ml onnx

# Run REAL FHE benchmark (actual encrypted inference)
python bench/concrete_ml_benchmark.py

# Run this system's plaintext benchmark
python bench/real_model_benchmark.py

# Verify N2HE is not available from Python
python -c "import n2he_native"  # Will fail: No module named 'n2he_native'
```
