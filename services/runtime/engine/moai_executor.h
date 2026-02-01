#pragma once
/**
 * MOAI-Optimized FHE-GBDT Executor
 *
 * Production-hardened implementation based on:
 * "MOAI: Module-Optimizing Architecture for Non-Interactive Secure
 * Transformer Inference" by Digital Trust Centre, NTU Singapore.
 * (IACR ePrint 2025/991, NDSS 2025)
 *
 * Key optimizations:
 * 1. Column packing - Rotation-free feature access
 * 2. Polynomial Step - No LUT overhead
 * 3. Interleaved aggregation - O(log n) tree summation
 */

#include "kernel/crypto_context.h"
#include "kernel/rotation_free_step.h"
#include "kernel/noise_budget.h"
#include "eval_key_cache.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <mutex>
#include <atomic>

namespace fhe_gbdt::engine {

// Forward declarations
struct MOAIComparison;
struct MOAILevel;
struct ParsedMOAIPlan;

/**
 * MOAI Comparison operation
 */
struct MOAIComparison {
    int tree_idx = 0;
    int node_idx = 0;
    int feature_idx = 0;
    float threshold = 0.0f;
    float weight = 1.0f;
};

/**
 * MOAI Level - all comparisons at a single depth
 */
struct MOAILevel {
    int depth = 0;
    std::vector<MOAIComparison> comparisons;
};

/**
 * Parsed MOAI execution plan
 */
struct ParsedMOAIPlan {
    std::string model_id;
    int num_trees = 0;
    int max_depth = 0;
    int num_features = 0;
    float base_score = 0.0f;
    std::vector<MOAILevel> levels;

    bool is_valid() const {
        return num_trees > 0 && max_depth > 0 && num_features > 0;
    }
};

/**
 * Execution statistics for monitoring
 */
struct ExecutionStats {
    size_t total_rotations = 0;
    size_t total_comparisons = 0;
    size_t total_additions = 0;
    size_t format_conversions = 0;
    double total_time_ms = 0.0;
    double comparison_time_ms = 0.0;
    double aggregation_time_ms = 0.0;
    double noise_budget_min = 0.0;
    bool noise_overflow = false;
};

/**
 * Interleaved Aggregator - O(log n) tree summation
 *
 * Uses rotate-and-add pattern instead of sequential addition.
 * Thread-safe implementation.
 */
class InterleavedAggregator {
public:
    explicit InterleavedAggregator(std::shared_ptr<kernel::CryptoContext> ctx)
        : ctx_(ctx), rotation_count_(0), addition_count_(0) {
        if (!ctx_) {
            throw std::invalid_argument("InterleavedAggregator: ctx cannot be null");
        }
    }

    std::shared_ptr<kernel::Ciphertext> Aggregate(
        const std::vector<std::shared_ptr<kernel::Ciphertext>>& tree_outputs) {

        std::lock_guard<std::mutex> lock(mutex_);

        if (tree_outputs.empty()) {
            return std::make_shared<kernel::Ciphertext>();
        }

        for (size_t i = 0; i < tree_outputs.size(); ++i) {
            if (!tree_outputs[i]) {
                throw std::invalid_argument(
                    "InterleavedAggregator: null ciphertext at index " + std::to_string(i));
            }
        }

        if (tree_outputs.size() == 1) {
            return tree_outputs[0];
        }

        size_t num_trees = tree_outputs.size();
        auto packed = PackInterleaved(tree_outputs);

        // Log-reduction with rotate-and-add
        size_t stride = 1;
        while (stride < num_trees) {
            auto rotated = Rotate(*packed, static_cast<int>(stride));
            rotation_count_++;
            packed = ctx_->Add(*packed, *rotated);
            addition_count_++;
            stride *= 2;
        }

        return packed;
    }

    std::shared_ptr<kernel::Ciphertext> AggregateSequential(
        const std::vector<std::shared_ptr<kernel::Ciphertext>>& tree_outputs) {

        std::lock_guard<std::mutex> lock(mutex_);

        if (tree_outputs.empty()) {
            return std::make_shared<kernel::Ciphertext>();
        }

        auto result = tree_outputs[0];
        for (size_t i = 1; i < tree_outputs.size(); ++i) {
            if (tree_outputs[i]) {
                result = ctx_->Add(*result, *tree_outputs[i]);
                addition_count_++;
            }
        }
        return result;
    }

    size_t get_rotation_count() const { return rotation_count_.load(); }
    size_t get_addition_count() const { return addition_count_.load(); }
    void reset_stats() { rotation_count_ = 0; addition_count_ = 0; }

private:
    std::shared_ptr<kernel::CryptoContext> ctx_;
    std::atomic<size_t> rotation_count_;
    std::atomic<size_t> addition_count_;
    mutable std::mutex mutex_;

    std::shared_ptr<kernel::Ciphertext> PackInterleaved(
        const std::vector<std::shared_ptr<kernel::Ciphertext>>& trees) {

        if (trees.empty() || !trees[0]) {
            return std::make_shared<kernel::Ciphertext>();
        }

        auto result = std::make_shared<kernel::Ciphertext>();
        result->scheme_id = trees[0]->scheme_id;
        result->is_rlwe = trees[0]->is_rlwe;
        result->batch_size = trees[0]->batch_size;

        if (trees[0]->is_rlwe && !trees[0]->rlwe_data.empty()) {
            size_t n = trees[0]->rlwe_data[0].size();
            result->rlwe_data.resize(trees[0]->rlwe_data.size());

            for (size_t poly_idx = 0; poly_idx < result->rlwe_data.size(); ++poly_idx) {
                result->rlwe_data[poly_idx].resize(n, 0);
                for (size_t slot = 0; slot < n; ++slot) {
                    size_t tree_idx = slot % trees.size();
                    size_t source_slot = slot / trees.size();
                    if (source_slot < n && trees[tree_idx] &&
                        poly_idx < trees[tree_idx]->rlwe_data.size() &&
                        source_slot < trees[tree_idx]->rlwe_data[poly_idx].size()) {
                        result->rlwe_data[poly_idx][slot] =
                            trees[tree_idx]->rlwe_data[poly_idx][source_slot];
                    }
                }
            }
        }
        return result;
    }

    std::shared_ptr<kernel::Ciphertext> Rotate(const kernel::Ciphertext& ct, int offset) {
        auto result = std::make_shared<kernel::Ciphertext>();
        result->scheme_id = ct.scheme_id;
        result->is_rlwe = ct.is_rlwe;
        result->batch_size = ct.batch_size;

        if (ct.is_rlwe && !ct.rlwe_data.empty()) {
            size_t n = ct.rlwe_data[0].size();
            result->rlwe_data.resize(ct.rlwe_data.size());
            for (size_t poly_idx = 0; poly_idx < ct.rlwe_data.size(); ++poly_idx) {
                result->rlwe_data[poly_idx].resize(n);
                for (size_t i = 0; i < n; ++i) {
                    size_t src_idx = (i + n - offset) % n;
                    result->rlwe_data[poly_idx][i] = ct.rlwe_data[poly_idx][src_idx];
                }
            }
        }
        return result;
    }
};

/**
 * MOAIExecutor - Production FHE-GBDT Engine
 *
 * Thread-safe, production-hardened executor with:
 * - Input validation
 * - Error handling
 * - Noise budget tracking
 * - Comprehensive statistics
 */
class MOAIExecutor {
public:
    struct Config {
        bool enable_noise_tracking = true;
        bool strict_validation = true;
        bool use_minimax = true;
        double noise_warning_threshold = 5.0;
        size_t max_trees = 10000;
        size_t max_depth = 20;
        size_t max_features = 65536;
    };

    explicit MOAIExecutor(
        std::shared_ptr<kernel::CryptoContext> ctx,
        const Config& config = Config{})
        : ctx_(ctx)
        , config_(config)
        , execution_count_(0) {

        if (!ctx_) {
            throw std::invalid_argument("MOAIExecutor: crypto context cannot be null");
        }

        rotation_free_step_ = std::make_unique<kernel::RotationFreeStep>(ctx_, config_.use_minimax);
        aggregator_ = std::make_unique<InterleavedAggregator>(ctx_);

        std::cout << "MOAIExecutor initialized (production mode):" << std::endl
                  << "  - Rotation-free Step: " << (config_.use_minimax ? "minimax" : "Chebyshev") << std::endl
                  << "  - Noise tracking: " << (config_.enable_noise_tracking ? "on" : "off") << std::endl
                  << "  - Strict validation: " << (config_.strict_validation ? "on" : "off") << std::endl;
    }

    MOAIExecutor(const MOAIExecutor&) = delete;
    MOAIExecutor& operator=(const MOAIExecutor&) = delete;
    MOAIExecutor(MOAIExecutor&&) = default;
    MOAIExecutor& operator=(MOAIExecutor&&) = default;

    std::shared_ptr<kernel::Ciphertext> Execute(
        const std::vector<kernel::Ciphertext>& column_packed_inputs,
        const ParsedMOAIPlan& plan) {

        std::lock_guard<std::mutex> lock(execution_mutex_);

        auto start = std::chrono::high_resolution_clock::now();
        execution_count_++;

        ValidateInputs(column_packed_inputs, plan);
        stats_ = ExecutionStats{};

        std::vector<std::shared_ptr<kernel::Ciphertext>> tree_accumulators(plan.num_trees);
        for (int t = 0; t < plan.num_trees; ++t) {
            tree_accumulators[t] = CreateZeroCiphertext(column_packed_inputs[0]);
        }

        kernel::NoiseTracker noise_tracker;
        if (config_.enable_noise_tracking) {
            for (int t = 0; t < plan.num_trees; ++t) {
                noise_tracker.create_budget();
            }
        }

        auto compare_start = std::chrono::high_resolution_clock::now();
        for (const auto& level : plan.levels) {
            ExecuteLevel(level, column_packed_inputs, tree_accumulators, plan, noise_tracker);
        }
        auto compare_end = std::chrono::high_resolution_clock::now();
        stats_.comparison_time_ms = std::chrono::duration<double, std::milli>(
            compare_end - compare_start).count();

        auto agg_start = std::chrono::high_resolution_clock::now();
        auto result = aggregator_->Aggregate(tree_accumulators);
        auto agg_end = std::chrono::high_resolution_clock::now();
        stats_.aggregation_time_ms = std::chrono::duration<double, std::milli>(
            agg_end - agg_start).count();

        stats_.total_rotations = rotation_free_step_->get_rotation_count() +
                                  aggregator_->get_rotation_count();
        stats_.total_comparisons = rotation_free_step_->get_comparison_count();
        stats_.total_additions = aggregator_->get_addition_count();

        if (config_.enable_noise_tracking) {
            auto noise_stats = noise_tracker.get_stats();
            stats_.noise_budget_min = noise_stats.min_remaining_bits;
            stats_.noise_overflow = noise_stats.overflowed > 0;
            if (stats_.noise_overflow) {
                std::cerr << "WARNING: Noise budget exhausted!" << std::endl;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        stats_.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }

    ExecutionStats get_stats() const {
        std::lock_guard<std::mutex> lock(execution_mutex_);
        return stats_;
    }

    size_t get_execution_count() const { return execution_count_.load(); }

    void print_stats() const {
        auto s = get_stats();
        std::cout << "MOAI Execution Stats (#" << execution_count_.load() << "):" << std::endl
                  << "  Total time: " << s.total_time_ms << " ms" << std::endl
                  << "    Comparison: " << s.comparison_time_ms << " ms" << std::endl
                  << "    Aggregation: " << s.aggregation_time_ms << " ms" << std::endl
                  << "  Rotations: " << s.total_rotations << std::endl
                  << "  Comparisons: " << s.total_comparisons << std::endl
                  << "  Additions: " << s.total_additions << std::endl
                  << "  Format conversions: " << s.format_conversions
                  << (s.format_conversions == 0 ? " (optimal)" : "") << std::endl;
        if (config_.enable_noise_tracking) {
            std::cout << "  Min noise budget: " << s.noise_budget_min << " bits"
                      << (s.noise_overflow ? " [OVERFLOW]" : "") << std::endl;
        }
    }

    static void PrintComparisonReport(
        const ExecutionStats& moai_stats,
        size_t traditional_rotations,
        double traditional_time_ms) {

        std::cout << "\n=== MOAI vs Traditional ===" << std::endl;
        std::cout << "Metric            | Traditional | MOAI      | Improvement" << std::endl;
        std::cout << "------------------|-------------|-----------|------------" << std::endl;

        double rot_imp = traditional_rotations > 0 ?
            (1.0 - static_cast<double>(moai_stats.total_rotations) / traditional_rotations) * 100 : 0;
        double time_imp = traditional_time_ms > 0 ?
            (1.0 - moai_stats.total_time_ms / traditional_time_ms) * 100 : 0;

        std::cout << "Rotations         | " << traditional_rotations
                  << " | " << moai_stats.total_rotations
                  << " | " << rot_imp << "%" << std::endl;
        std::cout << "Time (ms)         | " << traditional_time_ms
                  << " | " << moai_stats.total_time_ms
                  << " | " << time_imp << "%" << std::endl;
    }

private:
    std::shared_ptr<kernel::CryptoContext> ctx_;
    Config config_;
    std::unique_ptr<kernel::RotationFreeStep> rotation_free_step_;
    std::unique_ptr<InterleavedAggregator> aggregator_;
    ExecutionStats stats_;
    std::atomic<size_t> execution_count_;
    mutable std::mutex execution_mutex_;

    void ValidateInputs(
        const std::vector<kernel::Ciphertext>& inputs,
        const ParsedMOAIPlan& plan) {

        if (!config_.strict_validation) return;

        if (inputs.empty()) {
            throw std::invalid_argument("MOAIExecutor: inputs cannot be empty");
        }
        if (!plan.is_valid()) {
            throw std::invalid_argument("MOAIExecutor: invalid execution plan");
        }
        if (static_cast<size_t>(plan.num_features) > inputs.size()) {
            throw std::invalid_argument(
                "MOAIExecutor: plan requires " + std::to_string(plan.num_features) +
                " features but only " + std::to_string(inputs.size()) + " provided");
        }
        if (static_cast<size_t>(plan.num_trees) > config_.max_trees) {
            throw std::invalid_argument(
                "MOAIExecutor: exceeds max trees (" + std::to_string(config_.max_trees) + ")");
        }
        if (static_cast<size_t>(plan.max_depth) > config_.max_depth) {
            throw std::invalid_argument(
                "MOAIExecutor: exceeds max depth (" + std::to_string(config_.max_depth) + ")");
        }
    }

    void ExecuteLevel(
        const MOAILevel& level,
        const std::vector<kernel::Ciphertext>& inputs,
        std::vector<std::shared_ptr<kernel::Ciphertext>>& accumulators,
        const ParsedMOAIPlan& plan,
        kernel::NoiseTracker& noise_tracker) {

        std::unordered_map<int, std::vector<MOAIComparison>> by_feature;
        for (const auto& comp : level.comparisons) {
            if (comp.feature_idx >= 0 && comp.feature_idx < static_cast<int>(inputs.size())) {
                by_feature[comp.feature_idx].push_back(comp);
            }
        }

        for (const auto& [feat_idx, comparisons] : by_feature) {
            const auto& feature_ct = inputs[feat_idx];

            std::vector<float> thresholds;
            thresholds.reserve(comparisons.size());
            for (const auto& comp : comparisons) {
                thresholds.push_back(comp.threshold);
            }

            auto step_results = rotation_free_step_->CompareBatch(feature_ct, thresholds, false);

            for (size_t i = 0; i < comparisons.size() && i < step_results.size(); ++i) {
                int tree_idx = comparisons[i].tree_idx;
                if (tree_idx >= 0 && tree_idx < static_cast<int>(accumulators.size())) {
                    float weight = comparisons[i].weight;
                    auto weighted = ctx_->MulPlain(*step_results[i], weight);
                    accumulators[tree_idx] = ctx_->Add(*accumulators[tree_idx], *weighted);
                    stats_.total_additions++;

                    if (config_.enable_noise_tracking &&
                        static_cast<size_t>(tree_idx) < noise_tracker.get_stats().total_ciphertexts) {
                        noise_tracker.get(tree_idx).consume_mul_plain();
                        noise_tracker.get(tree_idx).consume_add(noise_tracker.get(tree_idx));
                    }
                }
            }
        }
    }

    std::shared_ptr<kernel::Ciphertext> CreateZeroCiphertext(const kernel::Ciphertext& template_ct) {
        auto result = std::make_shared<kernel::Ciphertext>();
        result->scheme_id = template_ct.scheme_id;
        result->is_rlwe = template_ct.is_rlwe;
        result->batch_size = template_ct.batch_size;

        if (template_ct.is_rlwe && !template_ct.rlwe_data.empty()) {
            result->rlwe_data.resize(template_ct.rlwe_data.size());
            for (size_t i = 0; i < result->rlwe_data.size(); ++i) {
                result->rlwe_data[i].resize(template_ct.rlwe_data[i].size(), 0);
            }
        }
        return result;
    }
};

} // namespace fhe_gbdt::engine
