#pragma once
/**
 * MOAI Rotation-Free Step Function
 *
 * Production-hardened implementation based on:
 * "MOAI: Module-Optimizing Architecture for Non-Interactive Secure
 * Transformer Inference" by Digital Trust Centre, NTU Singapore.
 * (IACR ePrint 2025/991, NDSS 2025)
 *
 * Key insight: Column packing enables rotation-free comparisons via
 * polynomial approximation of the sign function.
 */

#include "crypto_context.h"
#include <vector>
#include <memory>
#include <cmath>
#include <array>
#include <iostream>
#include <stdexcept>
#include <atomic>
#include <mutex>

namespace fhe_gbdt::kernel {

/**
 * RotationFreeStep - Polynomial approximation of sign function
 *
 * With column packing, thresholds are encoded as plaintext polynomials
 * matching ciphertext structure, eliminating ALL rotations.
 *
 * Thread-safe implementation with input validation.
 */
class RotationFreeStep {
public:
    static constexpr int POLY_DEGREE = 7;

    // Chebyshev coefficients for sign(x) on [-1, 1]
    static constexpr std::array<double, POLY_DEGREE> SIGN_COEFFS = {
        0.0, 1.1963, 0.0, -0.2849, 0.0, 0.0951, 0.0
    };

    // Minimax coefficients (higher accuracy)
    static constexpr std::array<double, POLY_DEGREE> MINIMAX_COEFFS = {
        0.0, 1.5708, 0.0, -0.6460, 0.0, 0.0796, 0.0
    };

    explicit RotationFreeStep(
        std::shared_ptr<CryptoContext> ctx,
        bool use_minimax = true)
        : ctx_(ctx)
        , use_minimax_(use_minimax)
        , rotation_count_(0)
        , comparison_count_(0) {

        if (!ctx_) {
            throw std::invalid_argument("RotationFreeStep: ctx cannot be null");
        }

        std::cout << "RotationFreeStep initialized ("
                  << (use_minimax_ ? "minimax" : "Chebyshev") << ")" << std::endl;
    }

    RotationFreeStep(const RotationFreeStep&) = delete;
    RotationFreeStep& operator=(const RotationFreeStep&) = delete;

    /**
     * Rotation-free comparison using column packing.
     *
     * @param feature_ct Column-packed feature ciphertext
     * @param threshold Comparison threshold
     * @param strict Use strict inequality (< vs <=)
     * @return Step function result ciphertext
     * @throws std::invalid_argument if inputs are invalid
     */
    std::shared_ptr<Ciphertext> Compare(
        const Ciphertext& feature_ct,
        float threshold,
        bool strict = false) {

        std::lock_guard<std::mutex> lock(mutex_);

        if (!ValidateCiphertext(feature_ct)) {
            throw std::invalid_argument("RotationFreeStep: invalid input ciphertext");
        }

        comparison_count_++;

        auto delta = ComputeDelta(feature_ct, threshold);
        return ApplyPolynomialSign(*delta, strict);
    }

    /**
     * Batch comparison - compare one feature against multiple thresholds.
     *
     * @param feature_ct Column-packed feature ciphertext
     * @param thresholds Vector of thresholds to compare against
     * @param strict Use strict inequality
     * @return Vector of step function results
     */
    std::vector<std::shared_ptr<Ciphertext>> CompareBatch(
        const Ciphertext& feature_ct,
        const std::vector<float>& thresholds,
        bool strict = false) {

        if (thresholds.empty()) {
            return {};
        }

        std::vector<std::shared_ptr<Ciphertext>> results;
        results.reserve(thresholds.size());

        for (float threshold : thresholds) {
            results.push_back(Compare(feature_ct, threshold, strict));
        }

        return results;
    }

    /**
     * Compare with pre-encoded plaintext threshold.
     */
    std::shared_ptr<Ciphertext> CompareWithPlaintext(
        const Ciphertext& feature_ct,
        const std::vector<int64_t>& threshold_slots,
        bool strict = false) {

        std::lock_guard<std::mutex> lock(mutex_);

        if (!ValidateCiphertext(feature_ct)) {
            throw std::invalid_argument("RotationFreeStep: invalid input ciphertext");
        }

        if (threshold_slots.empty()) {
            throw std::invalid_argument("RotationFreeStep: threshold_slots cannot be empty");
        }

        comparison_count_++;

        auto delta = SubtractPlaintext(feature_ct, threshold_slots);
        return ApplyPolynomialSign(*delta, strict);
    }

    // Thread-safe statistics
    size_t get_rotation_count() const { return rotation_count_.load(); }
    size_t get_comparison_count() const { return comparison_count_.load(); }

    void reset_stats() {
        rotation_count_ = 0;
        comparison_count_ = 0;
    }

    void print_stats() const {
        size_t comps = comparison_count_.load();
        size_t rots = rotation_count_.load();
        std::cout << "RotationFreeStep Stats:" << std::endl
                  << "  Comparisons: " << comps << std::endl
                  << "  Rotations: " << rots << std::endl
                  << "  Rotations/comparison: "
                  << (comps > 0 ? static_cast<double>(rots) / comps : 0) << std::endl;
    }

private:
    std::shared_ptr<CryptoContext> ctx_;
    bool use_minimax_;
    std::atomic<size_t> rotation_count_;
    std::atomic<size_t> comparison_count_;
    mutable std::mutex mutex_;

    bool ValidateCiphertext(const Ciphertext& ct) const {
        if (!ct.is_rlwe) return true;  // Non-RLWE is okay
        if (ct.rlwe_data.empty()) return false;
        if (ct.rlwe_data[0].empty()) return false;
        return true;
    }

    std::shared_ptr<Ciphertext> ComputeDelta(
        const Ciphertext& feature_ct,
        float threshold) {

        auto result = std::make_shared<Ciphertext>();
        result->scheme_id = feature_ct.scheme_id;
        result->is_rlwe = feature_ct.is_rlwe;
        result->batch_size = feature_ct.batch_size;

        if (feature_ct.is_rlwe && !feature_ct.rlwe_data.empty()) {
            result->rlwe_data.resize(feature_ct.rlwe_data.size());
            size_t n = feature_ct.rlwe_data[0].size();
            int64_t q = 1ULL << 32;
            int64_t encoded_threshold = EncodeFloat(threshold, q);

            for (size_t poly_idx = 0; poly_idx < feature_ct.rlwe_data.size(); ++poly_idx) {
                result->rlwe_data[poly_idx].resize(n);
                for (size_t i = 0; i < n; ++i) {
                    int64_t val = feature_ct.rlwe_data[poly_idx][i];
                    if (poly_idx == 1) {
                        val = (val - encoded_threshold + q) % q;
                    }
                    result->rlwe_data[poly_idx][i] = val;
                }
            }
        }

        return result;
    }

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

    std::shared_ptr<Ciphertext> ApplyPolynomialSign(
        const Ciphertext& delta,
        bool strict) {

        const auto& coeffs = use_minimax_ ? MINIMAX_COEFFS : SIGN_COEFFS;

        auto result = ctx_->MulPlain(delta, static_cast<float>(coeffs[1]));
        auto delta_sq = ctx_->Mul(delta, delta);
        auto delta_power = delta_sq;

        for (int deg = 3; deg < POLY_DEGREE; deg += 2) {
            if (std::abs(coeffs[deg]) > 1e-10) {
                auto delta_deg = ctx_->Mul(delta, *delta_power);
                auto term = ctx_->MulPlain(*delta_deg, static_cast<float>(coeffs[deg]));
                result = ctx_->Add(*result, *term);
            }
            if (deg + 2 < POLY_DEGREE) {
                delta_power = ctx_->Mul(*delta_power, *delta_sq);
            }
        }

        result = ctx_->Add(*result, *CreateConstantCiphertext(1.0, delta));
        result = ctx_->MulPlain(*result, 0.5f);

        return result;
    }

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
            result->rlwe_data[0].resize(n, 0);
            result->rlwe_data[1].resize(n, encoded);
        }

        return result;
    }

    static int64_t EncodeFloat(double value, int64_t q) {
        double scale = static_cast<double>(q) / 4.0;
        int64_t encoded = static_cast<int64_t>(value * scale);
        return ((encoded % q) + q) % q;
    }
};

} // namespace fhe_gbdt::kernel
