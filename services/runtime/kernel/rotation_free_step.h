#pragma once
#include "crypto_context.h"
#include <vector>
#include <memory>
#include <cmath>
#include <array>
#include <iostream>

namespace fhe_gbdt::kernel {

/**
 * MOAI-Style Rotation-Free Step Function
 *
 * Based on: "MOAI: Module-Optimizing Architecture for Non-Interactive Secure
 * Transformer Inference" by Digital Trust Centre, NTU Singapore.
 *
 * Key insight: With column packing, thresholds can be encoded as plaintext
 * polynomials matching the ciphertext structure, eliminating ALL rotations.
 *
 * The Step function (sign extraction) is implemented via polynomial
 * approximation instead of rotation-heavy LUT evaluation.
 */
class RotationFreeStep {
public:
    // Polynomial approximation degree for sign function
    static constexpr int POLY_DEGREE = 7;

    // Chebyshev coefficients for sign(x) approximation on [-1, 1]
    // sign(x) ≈ sum_{i=0}^{n} c_i * T_i(x) where T_i is Chebyshev polynomial
    static constexpr std::array<double, POLY_DEGREE> SIGN_COEFFS = {
        0.0,           // c0 (constant term)
        1.1963,        // c1 * x
        0.0,           // c2 * x^2
        -0.2849,       // c3 * x^3
        0.0,           // c4 * x^4
        0.0951,        // c5 * x^5
        0.0,           // c6 * x^6
    };

    // Minimax coefficients (alternative, potentially more accurate)
    static constexpr std::array<double, POLY_DEGREE> MINIMAX_COEFFS = {
        0.0,
        1.5708,        // π/2
        0.0,
        -0.6460,       // -π/2 * (1/6)
        0.0,
        0.0796,        // π/2 * (1/120)
        0.0,
    };

    explicit RotationFreeStep(
        std::shared_ptr<CryptoContext> ctx,
        bool use_minimax = true)
        : ctx_(ctx)
        , use_minimax_(use_minimax)
        , rotation_count_(0)
        , comparison_count_(0) {

        std::cout << "RotationFreeStep initialized with "
                  << (use_minimax_ ? "minimax" : "Chebyshev")
                  << " polynomial approximation" << std::endl;
    }

    /**
     * Rotation-free comparison using MOAI's column packing approach.
     *
     * Given: ct = Enc([f, f, f, ...])     (column-packed feature)
     *        threshold_pt = [t, t, t, ...]  (replicated plaintext threshold)
     *
     * Compute: delta = ct - threshold_pt   (NO rotation needed!)
     *          step = PolynomialSign(delta)
     */
    std::shared_ptr<Ciphertext> Compare(
        const Ciphertext& feature_ct,
        float threshold,
        bool strict = false) {

        comparison_count_++;

        // Step 1: Compute delta = feature - threshold (rotation-free!)
        auto delta = ComputeDelta(feature_ct, threshold);

        // Step 2: Apply polynomial approximation of sign function
        auto result = ApplyPolynomialSign(*delta, strict);

        return result;
    }

    /**
     * Batch comparison - compare one feature against multiple thresholds.
     * All comparisons are rotation-free due to column packing.
     */
    std::vector<std::shared_ptr<Ciphertext>> CompareBatch(
        const Ciphertext& feature_ct,
        const std::vector<float>& thresholds,
        bool strict = false) {

        std::vector<std::shared_ptr<Ciphertext>> results;
        results.reserve(thresholds.size());

        for (float threshold : thresholds) {
            results.push_back(Compare(feature_ct, threshold, strict));
        }

        return results;
    }

    /**
     * Compare with pre-encoded plaintext threshold (most efficient).
     */
    std::shared_ptr<Ciphertext> CompareWithPlaintext(
        const Ciphertext& feature_ct,
        const std::vector<int64_t>& threshold_slots,
        bool strict = false) {

        comparison_count_++;

        // Direct subtraction with plaintext polynomial
        auto delta = SubtractPlaintext(feature_ct, threshold_slots);

        return ApplyPolynomialSign(*delta, strict);
    }

    // Statistics
    size_t get_rotation_count() const { return rotation_count_; }
    size_t get_comparison_count() const { return comparison_count_; }

    void reset_stats() {
        rotation_count_ = 0;
        comparison_count_ = 0;
    }

    void print_stats() const {
        std::cout << "RotationFreeStep Stats:" << std::endl
                  << "  Comparisons: " << comparison_count_ << std::endl
                  << "  Rotations: " << rotation_count_ << std::endl
                  << "  Rotations per comparison: "
                  << (comparison_count_ > 0 ? (double)rotation_count_ / comparison_count_ : 0)
                  << std::endl;
    }

private:
    std::shared_ptr<CryptoContext> ctx_;
    bool use_minimax_;
    size_t rotation_count_;
    size_t comparison_count_;

    /**
     * Compute delta = feature - threshold (rotation-free).
     */
    std::shared_ptr<Ciphertext> ComputeDelta(
        const Ciphertext& feature_ct,
        float threshold) {

        // Encode threshold as plaintext polynomial
        // All slots have the same value due to column packing
        auto result = std::make_shared<Ciphertext>();
        result->scheme_id = feature_ct.scheme_id;
        result->is_rlwe = feature_ct.is_rlwe;
        result->batch_size = feature_ct.batch_size;

        if (feature_ct.is_rlwe && !feature_ct.rlwe_data.empty()) {
            result->rlwe_data.resize(feature_ct.rlwe_data.size());
            size_t n = feature_ct.rlwe_data[0].size();
            int64_t q = 1ULL << 32;

            // Encode threshold with proper scaling
            int64_t encoded_threshold = EncodeFloat(threshold, q);

            for (size_t poly_idx = 0; poly_idx < feature_ct.rlwe_data.size(); ++poly_idx) {
                result->rlwe_data[poly_idx].resize(n);
                for (size_t i = 0; i < n; ++i) {
                    int64_t val = feature_ct.rlwe_data[poly_idx][i];
                    // Subtract threshold from the b polynomial (message carrier)
                    if (poly_idx == 1) {
                        val = (val - encoded_threshold + q) % q;
                    }
                    result->rlwe_data[poly_idx][i] = val;
                }
            }
        }

        // NO rotation needed! This is the key MOAI insight.
        return result;
    }

    /**
     * Subtract pre-encoded plaintext from ciphertext.
     */
    std::shared_ptr<Ciphertext> SubtractPlaintext(
        const Ciphertext& ct,
        const std::vector<int64_t>& plaintext_slots) {

        auto result = std::make_shared<Ciphertext>();
        result->scheme_id = ct.scheme_id;
        result->is_rlwe = ct.is_rlwe;
        result->batch_size = ct.batch_size;

        if (ct.is_rlwe && !ct.rlwe_data.empty()) {
            result->rlwe_data.resize(ct.rlwe_data.size());
            size_t n = ct.rlwe_data[0].size();
            int64_t q = 1ULL << 32;

            for (size_t poly_idx = 0; poly_idx < ct.rlwe_data.size(); ++poly_idx) {
                result->rlwe_data[poly_idx].resize(n);
                for (size_t i = 0; i < n; ++i) {
                    int64_t val = ct.rlwe_data[poly_idx][i];
                    if (poly_idx == 1 && i < plaintext_slots.size()) {
                        val = (val - plaintext_slots[i] + q) % q;
                    }
                    result->rlwe_data[poly_idx][i] = val;
                }
            }
        }

        return result;
    }

    /**
     * Apply polynomial approximation of sign function.
     *
     * sign(x) ≈ c1*x + c3*x³ + c5*x⁵ + ...
     *
     * This is rotation-free because polynomial evaluation only uses:
     * - Ciphertext-ciphertext multiplication
     * - Plaintext-ciphertext multiplication
     * - Addition
     */
    std::shared_ptr<Ciphertext> ApplyPolynomialSign(
        const Ciphertext& delta,
        bool strict) {

        const auto& coeffs = use_minimax_ ? MINIMAX_COEFFS : SIGN_COEFFS;

        // Normalize delta to [-1, 1] range
        // Assuming input is already scaled appropriately

        // Start with c1 * delta
        auto result = ctx_->MulPlain(delta, static_cast<float>(coeffs[1]));

        // Compute delta² for higher-order terms
        auto delta_sq = ctx_->Mul(delta, delta);

        // Current power of delta (starts at delta²)
        auto delta_power = delta_sq;

        // Add odd-degree terms: c3*x³, c5*x⁵, c7*x⁷, ...
        for (int deg = 3; deg < POLY_DEGREE; deg += 2) {
            if (std::abs(coeffs[deg]) > 1e-10) {
                // term = c_deg * delta^deg = c_deg * delta * delta^(deg-1)
                auto delta_deg = ctx_->Mul(delta, *delta_power);
                auto term = ctx_->MulPlain(*delta_deg, static_cast<float>(coeffs[deg]));
                result = ctx_->Add(*result, *term);
            }

            // Update power: delta^(deg+1) = delta^(deg-1) * delta²
            if (deg + 2 < POLY_DEGREE) {
                delta_power = ctx_->Mul(*delta_power, *delta_sq);
            }
        }

        // Map from [-1, 1] to [0, 1] for step function output
        // step(x) = (sign(x) + 1) / 2
        result = ctx_->Add(*result, *CreateConstantCiphertext(1.0, delta));
        result = ctx_->MulPlain(*result, 0.5f);

        return result;
    }

    /**
     * Create a ciphertext with constant value in all slots.
     */
    std::shared_ptr<Ciphertext> CreateConstantCiphertext(
        double value,
        const Ciphertext& template_ct) {

        auto result = std::make_shared<Ciphertext>();
        result->scheme_id = template_ct.scheme_id;
        result->is_rlwe = template_ct.is_rlwe;
        result->batch_size = template_ct.batch_size;

        if (template_ct.is_rlwe && !template_ct.rlwe_data.empty()) {
            size_t n = template_ct.rlwe_data[0].size();
            int64_t q = 1ULL << 32;
            int64_t encoded = EncodeFloat(value, q);

            result->rlwe_data.resize(2);
            result->rlwe_data[0].resize(n, 0);  // a = 0 (no randomness for constant)
            result->rlwe_data[1].resize(n);
            for (size_t i = 0; i < n; ++i) {
                result->rlwe_data[1][i] = encoded;
            }
        }

        return result;
    }

    /**
     * Encode floating-point value to integer for FHE.
     */
    static int64_t EncodeFloat(double value, int64_t q) {
        // Scale to use ~1/4 of modulus range for message
        double scale = static_cast<double>(q) / 4.0;
        int64_t encoded = static_cast<int64_t>(value * scale);
        return ((encoded % q) + q) % q;
    }
};

} // namespace fhe_gbdt::kernel
