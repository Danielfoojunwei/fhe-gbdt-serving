#pragma once
#include "backend.h"
#include "crypto_context.h"
#include "third_party/n2he/include/include.hpp"
#include <iostream>
#include <cstring>

// AVX2 support detection
#if defined(__AVX2__)
#include <immintrin.h>
#define HAS_AVX2 1
#else
#define HAS_AVX2 0
#endif

namespace fhe_gbdt::kernel {

/**
 * CpuBackend: Optimized CPU implementation of FHE operations
 *
 * Optimizations:
 * - AVX2 SIMD for polynomial arithmetic (4x int64 per cycle)
 * - Cache-friendly memory access patterns
 * - Lazy modular reduction
 */
class CpuBackend : public Backend {
public:
    CpuBackend(const std::string& scheme_id) : scheme_id_(scheme_id) {
        // N2HE parameters initialization (e.g., q, n, k)
        q_ = 1ULL << 32; // Default for RLWE64
        n_ = 2048;
        k_ = 1; // variance param

#if HAS_AVX2
        std::cout << "CPU Backend initialized with AVX2 support" << std::endl;
#else
        std::cout << "CPU Backend initialized (scalar mode)" << std::endl;
#endif
    }

    std::shared_ptr<Ciphertext> Add(const Ciphertext& a, const Ciphertext& b) override {
        auto res = std::make_shared<Ciphertext>();
        res->scheme_id = scheme_id_;
        res->is_rlwe = a.is_rlwe;

        if (a.is_rlwe) {
            res->rlwe_data = a.rlwe_data;
            for (size_t i = 0; i < a.rlwe_data.size(); ++i) {
#if HAS_AVX2
                add_poly_avx2(res->rlwe_data[i].data(), b.rlwe_data[i].data(), n_, q_);
#else
                add_poly(res->rlwe_data[i], b.rlwe_data[i], q_);
#endif
            }
        } else {
            // LWE addition
            res->lwe_data = a.lwe_data;
            for (size_t i = 0; i < a.lwe_data.size(); ++i) {
                res->lwe_data[i] = (res->lwe_data[i] + b.lwe_data[i]) % q_;
            }
        }
        return res;
    }

    std::shared_ptr<Ciphertext> Sub(const Ciphertext& a, const Ciphertext& b) override {
        auto res = std::make_shared<Ciphertext>();
        res->scheme_id = scheme_id_;
        res->is_rlwe = a.is_rlwe;

        if (a.is_rlwe) {
            res->rlwe_data.resize(a.rlwe_data.size());
            for (size_t i = 0; i < a.rlwe_data.size(); ++i) {
                res->rlwe_data[i].resize(n_);
#if HAS_AVX2
                sub_poly_avx2(res->rlwe_data[i].data(),
                              a.rlwe_data[i].data(),
                              b.rlwe_data[i].data(), n_, q_);
#else
                for (size_t j = 0; j < a.rlwe_data[i].size(); ++j) {
                    res->rlwe_data[i][j] = (a.rlwe_data[i][j] - b.rlwe_data[i][j] + q_) % q_;
                }
#endif
            }
        } else {
            // LWE subtraction
            res->lwe_data = a.lwe_data;
            for (size_t i = 0; i < a.lwe_data.size(); ++i) {
                res->lwe_data[i] = (a.lwe_data[i] - b.lwe_data[i] + q_) % q_;
            }
        }
        return res;
    }

    std::shared_ptr<Ciphertext> MulPlain(const Ciphertext& a, float constant) override {
        auto res = std::make_shared<Ciphertext>();
        res->scheme_id = scheme_id_;
        res->is_rlwe = a.is_rlwe;

        int64_t scalar = static_cast<int64_t>(constant);

        if (a.is_rlwe) {
            res->rlwe_data = a.rlwe_data;
            for (auto& poly : res->rlwe_data) {
#if HAS_AVX2
                mul_scalar_avx2(poly.data(), scalar, n_, q_);
#else
                multi_scale_poly(scalar, poly, q_);
#endif
            }
        }
        return res;
    }

    std::shared_ptr<Ciphertext> Step(const Ciphertext& delta, bool strict) override {
        // Step function: sign extraction for comparison
        // In real implementation, this would use N2HE LUT or bootstrapping
        auto res = std::make_shared<Ciphertext>();
        res->scheme_id = scheme_id_;
        res->is_rlwe = delta.is_rlwe;

        if (delta.is_rlwe && !delta.rlwe_data.empty()) {
            res->rlwe_data.resize(delta.rlwe_data.size());
            for (size_t i = 0; i < delta.rlwe_data.size(); ++i) {
                res->rlwe_data[i].resize(n_);
                // Sign extraction approximation
                for (size_t j = 0; j < delta.rlwe_data[i].size(); ++j) {
                    int64_t val = delta.rlwe_data[i][j];
                    // If val < q/2, it's "positive" -> output 1 (encoded)
                    // Otherwise -> output 0
                    int64_t threshold = q_ / 2;
                    res->rlwe_data[i][j] = (val < threshold) ? (q_ / 4) : 0;
                }
            }
        }

        return res;
    }

private:
    std::string scheme_id_;
    int64_t q_;
    int n_;
    int k_;

#if HAS_AVX2
    // AVX2-optimized polynomial addition: result[i] = (a[i] + b[i]) mod q
    void add_poly_avx2(int64_t* a, const int64_t* b, size_t n, int64_t q) {
        __m256i vq = _mm256_set1_epi64x(q);
        size_t i = 0;

        // Process 4 elements at a time
        for (; i + 4 <= n; i += 4) {
            __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
            __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));

            // Add
            __m256i vsum = _mm256_add_epi64(va, vb);

            // Conditional modular reduction: if sum >= q, subtract q
            __m256i vcmp = _mm256_cmpgt_epi64(vsum, _mm256_sub_epi64(vq, _mm256_set1_epi64x(1)));
            __m256i vred = _mm256_and_si256(vcmp, vq);
            vsum = _mm256_sub_epi64(vsum, vred);

            _mm256_storeu_si256((__m256i*)(a + i), vsum);
        }

        // Handle remainder
        for (; i < n; ++i) {
            int64_t sum = a[i] + b[i];
            a[i] = sum >= q ? sum - q : sum;
        }
    }

    // AVX2-optimized polynomial subtraction: result[i] = (a[i] - b[i] + q) mod q
    void sub_poly_avx2(int64_t* result, const int64_t* a, const int64_t* b, size_t n, int64_t q) {
        __m256i vq = _mm256_set1_epi64x(q);
        __m256i vzero = _mm256_setzero_si256();
        size_t i = 0;

        for (; i + 4 <= n; i += 4) {
            __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
            __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));

            // Subtract
            __m256i vdiff = _mm256_sub_epi64(va, vb);

            // If diff < 0, add q
            __m256i vneg = _mm256_cmpgt_epi64(vzero, vdiff);
            __m256i vadd = _mm256_and_si256(vneg, vq);
            vdiff = _mm256_add_epi64(vdiff, vadd);

            _mm256_storeu_si256((__m256i*)(result + i), vdiff);
        }

        for (; i < n; ++i) {
            int64_t diff = a[i] - b[i];
            result[i] = diff < 0 ? diff + q : diff;
        }
    }

    // AVX2-optimized scalar multiplication
    void mul_scalar_avx2(int64_t* data, int64_t scalar, size_t n, int64_t q) {
        // For scalar multiplication, we need 128-bit arithmetic
        // AVX2 doesn't have native 64x64->128 multiply, so we use scalar loop
        // but with prefetching for better cache performance

        for (size_t i = 0; i < n; i += 4) {
            // Prefetch next cache line
            if (i + 8 < n) {
                _mm_prefetch((char*)(data + i + 8), _MM_HINT_T0);
            }

            for (size_t j = i; j < std::min(i + 4, n); ++j) {
                __int128 prod = (__int128)data[j] * scalar;
                data[j] = (int64_t)(prod % q);
            }
        }
    }
#endif
};

} // namespace fhe_gbdt::kernel
