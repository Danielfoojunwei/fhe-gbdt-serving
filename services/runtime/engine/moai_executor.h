#pragma once
#include "kernel/crypto_context.h"
#include "kernel/rotation_free_step.h"
#include "kernel/step_bundle.h"
#include "eval_key_cache.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <cmath>

namespace fhe_gbdt::engine {

/**
 * MOAI-Style Executor with Consistent Packing
 *
 * Based on: "MOAI: Module-Optimizing Architecture for Non-Interactive Secure
 * Transformer Inference" by Digital Trust Centre, NTU Singapore.
 *
 * Key features:
 * 1. Column packing - Features replicated for rotation-free access
 * 2. Rotation-free Step - Polynomial approximation of sign function
 * 3. Consistent packing - No format conversions between levels
 * 4. Interleaved aggregation - Log-reduction for tree summation
 */

/**
 * Interleaved Aggregator for efficient tree output summation.
 *
 * Instead of sequential: result = t0 + t1 + t2 + ... + t_{n-1} (n-1 additions)
 * Use log-reduction:     log(n) additions with interleaved packing
 */
class InterleavedAggregator {
public:
    explicit InterleavedAggregator(std::shared_ptr<kernel::CryptoContext> ctx)
        : ctx_(ctx), rotation_count_(0), addition_count_(0) {}

    /**
     * Aggregate tree outputs using interleaved batching.
     *
     * Trees are packed interleaved:
     *   [t0_s0, t1_s0, t2_s0, ..., t0_s1, t1_s1, t2_s1, ...]
     *
     * Then reduced with rotate-and-add pattern in log(n) steps.
     */
    std::shared_ptr<kernel::Ciphertext> Aggregate(
        const std::vector<std::shared_ptr<kernel::Ciphertext>>& tree_outputs) {

        if (tree_outputs.empty()) {
            return std::make_shared<kernel::Ciphertext>();
        }

        if (tree_outputs.size() == 1) {
            return tree_outputs[0];
        }

        size_t num_trees = tree_outputs.size();

        // Pack into interleaved format
        auto packed = PackInterleaved(tree_outputs);

        // Log-reduction
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

    /**
     * Sequential aggregation (for comparison).
     */
    std::shared_ptr<kernel::Ciphertext> AggregateSequential(
        const std::vector<std::shared_ptr<kernel::Ciphertext>>& tree_outputs) {

        if (tree_outputs.empty()) {
            return std::make_shared<kernel::Ciphertext>();
        }

        auto result = tree_outputs[0];
        for (size_t i = 1; i < tree_outputs.size(); ++i) {
            result = ctx_->Add(*result, *tree_outputs[i]);
            addition_count_++;
        }

        return result;
    }

    size_t get_rotation_count() const { return rotation_count_; }
    size_t get_addition_count() const { return addition_count_; }

    void reset_stats() {
        rotation_count_ = 0;
        addition_count_ = 0;
    }

private:
    std::shared_ptr<kernel::CryptoContext> ctx_;
    size_t rotation_count_;
    size_t addition_count_;

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

                // Interleave tree outputs
                for (size_t slot = 0; slot < n; ++slot) {
                    size_t tree_idx = slot % trees.size();
                    size_t source_slot = slot / trees.size();

                    if (source_slot < n && trees[tree_idx]) {
                        result->rlwe_data[poly_idx][slot] =
                            trees[tree_idx]->rlwe_data[poly_idx][source_slot];
                    }
                }
            }
        }

        return result;
    }

    std::shared_ptr<kernel::Ciphertext> Rotate(
        const kernel::Ciphertext& ct,
        int offset) {

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
 * MOAI Executor - Main execution engine with consistent column packing.
 */
class MOAIExecutor {
public:
    struct ExecutionStats {
        size_t total_rotations = 0;
        size_t total_comparisons = 0;
        size_t total_additions = 0;
        size_t format_conversions = 0;  // Should be 0 with MOAI!
        double total_time_ms = 0.0;
        double comparison_time_ms = 0.0;
        double aggregation_time_ms = 0.0;
    };

    MOAIExecutor(std::shared_ptr<kernel::CryptoContext> ctx)
        : ctx_(ctx)
        , rotation_free_step_(std::make_unique<kernel::RotationFreeStep>(ctx))
        , aggregator_(std::make_unique<InterleavedAggregator>(ctx)) {

        std::cout << "MOAIExecutor initialized with:" << std::endl
                  << "  - Rotation-free Step function" << std::endl
                  << "  - Consistent column packing" << std::endl
                  << "  - Interleaved aggregation" << std::endl;
    }

    /**
     * Execute GBDT inference with MOAI optimizations.
     *
     * @param column_packed_inputs Features in column format (one CT per feature)
     * @param plan Execution plan with tree structure
     * @return Encrypted prediction
     */
    std::shared_ptr<kernel::Ciphertext> Execute(
        const std::vector<kernel::Ciphertext>& column_packed_inputs,
        const ParsedMOAIPlan& plan) {

        auto start = std::chrono::high_resolution_clock::now();

        stats_ = ExecutionStats{};  // Reset stats

        // Initialize tree accumulators (all in column format)
        std::vector<std::shared_ptr<kernel::Ciphertext>> tree_accumulators(plan.num_trees);
        for (int t = 0; t < plan.num_trees; ++t) {
            tree_accumulators[t] = CreateZeroCiphertext(column_packed_inputs[0]);
        }

        // Process each depth level (levelized execution)
        auto compare_start = std::chrono::high_resolution_clock::now();

        for (const auto& level : plan.levels) {
            ExecuteLevel(level, column_packed_inputs, tree_accumulators, plan);
            // NO format conversion needed - consistent column packing!
        }

        auto compare_end = std::chrono::high_resolution_clock::now();
        stats_.comparison_time_ms = std::chrono::duration<double, std::milli>(
            compare_end - compare_start).count();

        // Aggregate tree outputs using interleaved reduction
        auto agg_start = std::chrono::high_resolution_clock::now();

        auto result = aggregator_->Aggregate(tree_accumulators);

        auto agg_end = std::chrono::high_resolution_clock::now();
        stats_.aggregation_time_ms = std::chrono::duration<double, std::milli>(
            agg_end - agg_start).count();

        // Update stats
        stats_.total_rotations = rotation_free_step_->get_rotation_count() +
                                  aggregator_->get_rotation_count();
        stats_.total_comparisons = rotation_free_step_->get_comparison_count();
        stats_.total_additions = aggregator_->get_addition_count();

        auto end = std::chrono::high_resolution_clock::now();
        stats_.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }

    ExecutionStats get_stats() const { return stats_; }

    void print_stats() const {
        std::cout << "MOAI Execution Stats:" << std::endl
                  << "  Total time: " << stats_.total_time_ms << " ms" << std::endl
                  << "    Comparison: " << stats_.comparison_time_ms << " ms" << std::endl
                  << "    Aggregation: " << stats_.aggregation_time_ms << " ms" << std::endl
                  << "  Rotations: " << stats_.total_rotations << std::endl
                  << "  Comparisons: " << stats_.total_comparisons << std::endl
                  << "  Additions: " << stats_.total_additions << std::endl
                  << "  Format conversions: " << stats_.format_conversions
                  << (stats_.format_conversions == 0 ? " (optimal!)" : " (suboptimal)")
                  << std::endl;
    }

    /**
     * Compare with traditional execution.
     */
    static void PrintComparisonReport(
        const ExecutionStats& moai_stats,
        size_t traditional_rotations,
        double traditional_time_ms) {

        std::cout << "\n=== MOAI vs Traditional Comparison ===" << std::endl;
        std::cout << "Metric            | Traditional | MOAI      | Improvement" << std::endl;
        std::cout << "------------------|-------------|-----------|------------" << std::endl;

        double rotation_improvement = traditional_rotations > 0 ?
            (1.0 - (double)moai_stats.total_rotations / traditional_rotations) * 100 : 0;

        double time_improvement = traditional_time_ms > 0 ?
            (1.0 - moai_stats.total_time_ms / traditional_time_ms) * 100 : 0;

        std::cout << "Rotations         | " << traditional_rotations
                  << " | " << moai_stats.total_rotations
                  << " | " << rotation_improvement << "%" << std::endl;

        std::cout << "Time (ms)         | " << traditional_time_ms
                  << " | " << moai_stats.total_time_ms
                  << " | " << time_improvement << "%" << std::endl;

        std::cout << "Format conversions| N/A         | "
                  << moai_stats.format_conversions << "         | eliminated" << std::endl;
    }

private:
    std::shared_ptr<kernel::CryptoContext> ctx_;
    std::unique_ptr<kernel::RotationFreeStep> rotation_free_step_;
    std::unique_ptr<InterleavedAggregator> aggregator_;
    ExecutionStats stats_;

    /**
     * Execute one depth level with consistent column packing.
     */
    void ExecuteLevel(
        const MOAILevel& level,
        const std::vector<kernel::Ciphertext>& inputs,
        std::vector<std::shared_ptr<kernel::Ciphertext>>& accumulators,
        const ParsedMOAIPlan& plan) {

        // Group comparisons by feature for maximum efficiency
        std::unordered_map<int, std::vector<MOAIComparison>> by_feature;

        for (const auto& comp : level.comparisons) {
            by_feature[comp.feature_idx].push_back(comp);
        }

        // Process each feature group (rotation-free!)
        for (const auto& [feat_idx, comparisons] : by_feature) {
            if (feat_idx >= static_cast<int>(inputs.size())) continue;

            const auto& feature_ct = inputs[feat_idx];

            // Batch all thresholds for this feature
            std::vector<float> thresholds;
            for (const auto& comp : comparisons) {
                thresholds.push_back(comp.threshold);
            }

            // Rotation-free batched comparison!
            auto step_results = rotation_free_step_->CompareBatch(
                feature_ct, thresholds, false);

            // Update tree accumulators
            for (size_t i = 0; i < comparisons.size() && i < step_results.size(); ++i) {
                int tree_idx = comparisons[i].tree_idx;
                if (tree_idx < static_cast<int>(accumulators.size())) {
                    float weight = comparisons[i].weight;
                    auto weighted = ctx_->MulPlain(*step_results[i], weight);
                    accumulators[tree_idx] = ctx_->Add(*accumulators[tree_idx], *weighted);
                    stats_.total_additions++;
                }
            }
        }
    }

    std::shared_ptr<kernel::Ciphertext> CreateZeroCiphertext(
        const kernel::Ciphertext& template_ct) {

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

/**
 * Parsed MOAI execution plan structures.
 */
struct MOAIComparison {
    int tree_idx;
    int node_idx;
    int feature_idx;
    float threshold;
    float weight;  // For weighted leaf values
};

struct MOAILevel {
    int depth;
    std::vector<MOAIComparison> comparisons;
};

struct ParsedMOAIPlan {
    std::string model_id;
    int num_trees;
    int max_depth;
    int num_features;
    float base_score;
    std::vector<MOAILevel> levels;
};

} // namespace fhe_gbdt::engine
