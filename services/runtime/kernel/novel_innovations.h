/**
 * Novel FHE-GBDT Kernel Innovations
 *
 * High-performance C++ implementations of novel FHE operations:
 * 1. Leaf-Centric Tensor Product - Direct leaf indicator computation
 * 2. Adaptive Precision Encoding - Gradient-informed encoding
 * 3. Homomorphic Pruning Primitives - Encrypted significance and gating
 * 4. Bootstrap-Aligned Execution - Noise-aware chunk scheduling
 * 5. Polynomial Leaf Evaluation - Horner's method in FHE
 *
 * These primitives leverage:
 * - MOAI column packing for rotation-free access
 * - N2HE weighted sum for O(1) aggregation
 * - Intel HEXL for NTT acceleration
 */

#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>

#include "noise_budget.h"

namespace fhe_gbdt::kernel::innovations {

/**
 * Configuration for novel innovations
 */
struct InnovationConfig {
    // Leaf-centric configuration
    int max_tree_depth = 10;
    int max_leaves = 1024;  // 2^10

    // Adaptive precision
    int min_precision_bits = 8;
    int max_precision_bits = 16;
    int base_precision_bits = 12;

    // Pruning configuration
    double significance_threshold = 0.1;
    bool soft_pruning = true;

    // Polynomial configuration
    int max_poly_degree = 3;

    // Bootstrap alignment
    double noise_budget_bits = 31.0;
    double step_function_bits = 8.0;
    double bootstrap_margin_bits = 5.0;
};

// =============================================================================
// Innovation #1: Leaf-Centric Tensor Product
// =============================================================================

/**
 * LeafCentricComputer: Computes leaf indicators via tensor product
 *
 * Key insight: For oblivious trees, leaf indicator is product of d sign values
 * leaf_k = Π_{d=0}^{D-1} (sign_d if bit(k,d)==1 else (1-sign_d))
 *
 * This maps perfectly to SIMD operations and column-packed FHE.
 */
class LeafCentricComputer {
public:
    explicit LeafCentricComputer(int max_depth = 10)
        : max_depth_(max_depth)
        , max_leaves_(1 << max_depth) {
        precompute_patterns();
    }

    /**
     * Compute all 2^d leaf indicators from d sign values.
     *
     * @param signs Array of sign values [s_0, s_1, ..., s_{d-1}] in [0,1]
     * @param depth Tree depth
     * @param out_indicators Output array of size 2^depth
     */
    void compute_leaf_indicators(
        const double* signs,
        int depth,
        double* out_indicators
    ) const {
        int num_leaves = 1 << depth;

        // Initialize all indicators to 1
        for (int i = 0; i < num_leaves; ++i) {
            out_indicators[i] = 1.0;
        }

        // Tensor product across all levels
        for (int d = 0; d < depth; ++d) {
            double s_d = signs[d];
            double s_d_complement = 1.0 - s_d;

            for (int leaf_idx = 0; leaf_idx < num_leaves; ++leaf_idx) {
                // Check if bit d is set in leaf_idx
                bool bit_set = (leaf_idx >> d) & 1;
                out_indicators[leaf_idx] *= bit_set ? s_d : s_d_complement;
            }
        }
    }

    /**
     * Batch compute leaf indicators for multiple samples.
     *
     * @param signs Shape [batch_size, depth] sign values
     * @param batch_size Number of samples
     * @param depth Tree depth
     * @param out_indicators Shape [batch_size, 2^depth] indicators
     */
    void compute_batch(
        const double* signs,
        int batch_size,
        int depth,
        double* out_indicators
    ) const {
        int num_leaves = 1 << depth;

        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            compute_leaf_indicators(
                signs + b * depth,
                depth,
                out_indicators + b * num_leaves
            );
        }
    }

    /**
     * Weighted sum of leaf values using indicators.
     * Result = Σ_k (indicator_k × leaf_value_k)
     *
     * This is O(1) in N2HE due to native weighted sum support!
     */
    double weighted_leaf_sum(
        const double* indicators,
        const double* leaf_values,
        int num_leaves
    ) const {
        double result = 0.0;

        // Vectorizable loop
        #pragma omp simd reduction(+:result)
        for (int k = 0; k < num_leaves; ++k) {
            result += indicators[k] * leaf_values[k];
        }

        return result;
    }

private:
    int max_depth_;
    int max_leaves_;
    std::vector<std::array<int8_t, 16>> patterns_;  // Precomputed bit patterns

    void precompute_patterns() {
        patterns_.resize(max_leaves_);
        for (int leaf_idx = 0; leaf_idx < max_leaves_; ++leaf_idx) {
            for (int d = 0; d < std::min(16, max_depth_); ++d) {
                patterns_[leaf_idx][d] = (leaf_idx >> d) & 1;
            }
        }
    }
};

// =============================================================================
// Innovation #2: Adaptive Precision Encoding
// =============================================================================

/**
 * AdaptivePrecisionEncoder: Per-feature precision based on importance
 */
class AdaptivePrecisionEncoder {
public:
    AdaptivePrecisionEncoder(
        int num_features,
        const double* importance_scores,
        const InnovationConfig& config = InnovationConfig()
    ) : num_features_(num_features)
      , config_(config) {

        allocate_precision(importance_scores);
    }

    /**
     * Encode feature value with its allocated precision.
     */
    int64_t encode(double value, int feature_idx) const {
        if (feature_idx < 0 || feature_idx >= num_features_) {
            throw std::out_of_range("Feature index out of range");
        }

        double scale = scales_[feature_idx];
        return static_cast<int64_t>(std::round(value * scale));
    }

    /**
     * Decode encoded value back to float.
     */
    double decode(int64_t encoded, int feature_idx) const {
        double scale = scales_[feature_idx];
        return static_cast<double>(encoded) / scale;
    }

    /**
     * Encode threshold with matching precision.
     */
    int64_t encode_threshold(double threshold, int feature_idx) const {
        return encode(threshold, feature_idx);
    }

    /**
     * Get precision bits for a feature.
     */
    int get_precision_bits(int feature_idx) const {
        return precision_bits_[feature_idx];
    }

    /**
     * Get total allocated precision (for noise budget estimation).
     */
    int get_total_precision_bits() const {
        int total = 0;
        for (int bits : precision_bits_) {
            total += bits;
        }
        return total;
    }

private:
    int num_features_;
    InnovationConfig config_;
    std::vector<int> precision_bits_;
    std::vector<double> scales_;

    void allocate_precision(const double* importance_scores) {
        precision_bits_.resize(num_features_);
        scales_.resize(num_features_);

        // Normalize importance scores
        double total_importance = 0.0;
        for (int i = 0; i < num_features_; ++i) {
            total_importance += importance_scores[i];
        }
        if (total_importance == 0.0) total_importance = 1.0;

        // Allocate precision proportionally
        int available_bonus = config_.max_precision_bits - config_.base_precision_bits;

        for (int i = 0; i < num_features_; ++i) {
            double norm_importance = importance_scores[i] / total_importance;
            int bonus = static_cast<int>(norm_importance * num_features_ * available_bonus);
            int precision = config_.base_precision_bits + bonus;

            // Clamp to valid range
            precision = std::max(config_.min_precision_bits,
                        std::min(config_.max_precision_bits, precision));

            precision_bits_[i] = precision;
            scales_[i] = static_cast<double>(1ULL << precision);
        }
    }
};

// =============================================================================
// Innovation #3: Homomorphic Pruning Primitives
// =============================================================================

/**
 * HomomorphicPruningGate: Computes significance and gates in encrypted domain
 */
class HomomorphicPruningGate {
public:
    explicit HomomorphicPruningGate(const InnovationConfig& config = InnovationConfig())
        : config_(config) {}

    /**
     * Compute tree significance from outputs (plaintext for validation).
     *
     * Significance = normalized variance contribution
     */
    void compute_significance(
        const double* tree_outputs,  // Shape [batch_size, num_trees]
        int batch_size,
        int num_trees,
        double* out_significance
    ) const {
        // Compute per-tree variance
        std::vector<double> means(num_trees, 0.0);
        std::vector<double> variances(num_trees, 0.0);

        // Compute means
        for (int t = 0; t < num_trees; ++t) {
            for (int b = 0; b < batch_size; ++b) {
                means[t] += tree_outputs[b * num_trees + t];
            }
            means[t] /= batch_size;
        }

        // Compute variances
        for (int t = 0; t < num_trees; ++t) {
            for (int b = 0; b < batch_size; ++b) {
                double diff = tree_outputs[b * num_trees + t] - means[t];
                variances[t] += diff * diff;
            }
            variances[t] /= batch_size;
        }

        // Normalize to [0, 1]
        double max_var = *std::max_element(variances.begin(), variances.end());
        if (max_var > 0) {
            for (int t = 0; t < num_trees; ++t) {
                out_significance[t] = variances[t] / max_var;
            }
        } else {
            for (int t = 0; t < num_trees; ++t) {
                out_significance[t] = 1.0 / num_trees;
            }
        }
    }

    /**
     * Compute gate values from significance.
     *
     * Soft gate: smooth sigmoid transition
     * Hard gate: binary threshold
     */
    void compute_gates(
        const double* significance,
        int num_trees,
        double* out_gates
    ) const {
        double threshold = config_.significance_threshold;

        if (config_.soft_pruning) {
            // Soft gate: sigmoid around threshold
            double steepness = 10.0;
            for (int t = 0; t < num_trees; ++t) {
                double x = (significance[t] - threshold) * steepness;
                out_gates[t] = 1.0 / (1.0 + std::exp(-x));
            }
        } else {
            // Hard gate
            for (int t = 0; t < num_trees; ++t) {
                out_gates[t] = significance[t] >= threshold ? 1.0 : 0.0;
            }
        }
    }

    /**
     * Apply gates to tree outputs.
     */
    void apply_gates(
        const double* tree_outputs,  // Shape [batch_size, num_trees]
        const double* gates,         // Shape [num_trees]
        int batch_size,
        int num_trees,
        double* out_gated            // Shape [batch_size, num_trees]
    ) const {
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < num_trees; ++t) {
                out_gated[b * num_trees + t] =
                    tree_outputs[b * num_trees + t] * gates[t];
            }
        }
    }

private:
    InnovationConfig config_;
};

// =============================================================================
// Innovation #4: Bootstrap-Aligned Execution
// =============================================================================

/**
 * BootstrapAlignedScheduler: Schedules execution to align with noise budget
 */
class BootstrapAlignedScheduler {
public:
    struct ChunkInfo {
        int start_tree;
        int end_tree;
        double estimated_noise_bits;
        bool requires_bootstrap_after;
    };

    explicit BootstrapAlignedScheduler(const InnovationConfig& config = InnovationConfig())
        : config_(config) {}

    /**
     * Compute execution chunks that fit within noise budget.
     */
    std::vector<ChunkInfo> compute_chunks(
        const int* tree_depths,
        int num_trees
    ) const {
        std::vector<ChunkInfo> chunks;

        double available_budget = config_.noise_budget_bits -
                                  config_.bootstrap_margin_bits;
        double initial_noise = 3.2;  // Initial encryption noise

        double current_noise = initial_noise;
        int chunk_start = 0;

        for (int t = 0; t < num_trees; ++t) {
            // Estimate noise for this tree
            double tree_noise = tree_depths[t] * config_.step_function_bits;

            if (current_noise + tree_noise > available_budget && t > chunk_start) {
                // Finalize current chunk
                chunks.push_back({
                    chunk_start,
                    t,
                    current_noise,
                    true  // Needs bootstrap after
                });

                chunk_start = t;
                current_noise = initial_noise + tree_noise;
            } else {
                current_noise += tree_noise;
            }
        }

        // Finalize last chunk
        if (chunk_start < num_trees) {
            chunks.push_back({
                chunk_start,
                num_trees,
                current_noise,
                false  // No bootstrap after last chunk
            });
        }

        return chunks;
    }

    /**
     * Estimate total bootstraps needed.
     */
    int estimate_bootstrap_count(const int* tree_depths, int num_trees) const {
        auto chunks = compute_chunks(tree_depths, num_trees);
        int count = 0;
        for (const auto& chunk : chunks) {
            if (chunk.requires_bootstrap_after) count++;
        }
        return count;
    }

private:
    InnovationConfig config_;
};

// =============================================================================
// Innovation #5: Polynomial Leaf Evaluation
// =============================================================================

/**
 * PolynomialLeafEvaluator: Evaluates polynomial functions at leaves
 *
 * Uses Horner's method for efficient FHE evaluation:
 * p(x) = c_0 + x(c_1 + x(c_2 + x(...)))
 *
 * This minimizes multiplicative depth.
 */
class PolynomialLeafEvaluator {
public:
    explicit PolynomialLeafEvaluator(int max_degree = 3)
        : max_degree_(max_degree) {}

    /**
     * Evaluate polynomial at point using Horner's method.
     *
     * @param coeffs Coefficients [c_0, c_1, ..., c_d] (constant term first)
     * @param num_coeffs Number of coefficients (degree + 1)
     * @param x Point to evaluate at
     */
    double evaluate_horner(
        const double* coeffs,
        int num_coeffs,
        double x
    ) const {
        if (num_coeffs == 0) return 0.0;

        // Start from highest degree coefficient
        double result = coeffs[num_coeffs - 1];

        for (int i = num_coeffs - 2; i >= 0; --i) {
            result = result * x + coeffs[i];
        }

        return result;
    }

    /**
     * Batch evaluate polynomial for multiple samples.
     */
    void evaluate_batch(
        const double* coeffs,
        int num_coeffs,
        const double* x_values,
        int batch_size,
        double* out_values
    ) const {
        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            out_values[b] = evaluate_horner(coeffs, num_coeffs, x_values[b]);
        }
    }

    /**
     * Estimate noise cost for polynomial evaluation.
     *
     * Each multiplication approximately doubles noise.
     * Horner's method: (degree) multiplications.
     */
    double estimate_noise_cost(int degree) const {
        double base_noise = 3.2;
        double mul_noise = 10.0;  // Conservative estimate
        return base_noise + degree * mul_noise;
    }

    /**
     * Compute maximum safe degree given noise budget.
     */
    int max_safe_degree(double available_noise_bits) const {
        double base_noise = 3.2;
        double mul_noise = 10.0;
        int degree = static_cast<int>((available_noise_bits - base_noise) / mul_noise);
        return std::max(0, std::min(degree, max_degree_));
    }

private:
    int max_degree_;
};

// =============================================================================
// Unified Novel Kernel Interface
// =============================================================================

/**
 * NovelKernelInterface: Unified interface for all innovations
 */
class NovelKernelInterface {
public:
    explicit NovelKernelInterface(const InnovationConfig& config = InnovationConfig())
        : config_(config)
        , leaf_computer_(config.max_tree_depth)
        , pruning_gate_(config)
        , bootstrap_scheduler_(config)
        , poly_evaluator_(config.max_poly_degree)
        , noise_tracker_() {}

    // Access individual components
    LeafCentricComputer& leaf_computer() { return leaf_computer_; }
    HomomorphicPruningGate& pruning_gate() { return pruning_gate_; }
    BootstrapAlignedScheduler& bootstrap_scheduler() { return bootstrap_scheduler_; }
    PolynomialLeafEvaluator& poly_evaluator() { return poly_evaluator_; }
    NoiseTracker& noise_tracker() { return noise_tracker_; }

    /**
     * Create adaptive precision encoder for features.
     */
    std::unique_ptr<AdaptivePrecisionEncoder> create_encoder(
        int num_features,
        const double* importance_scores
    ) {
        return std::make_unique<AdaptivePrecisionEncoder>(
            num_features, importance_scores, config_
        );
    }

    /**
     * Get configuration.
     */
    const InnovationConfig& config() const { return config_; }

private:
    InnovationConfig config_;
    LeafCentricComputer leaf_computer_;
    HomomorphicPruningGate pruning_gate_;
    BootstrapAlignedScheduler bootstrap_scheduler_;
    PolynomialLeafEvaluator poly_evaluator_;
    NoiseTracker noise_tracker_;
};

}  // namespace fhe_gbdt::kernel::innovations
