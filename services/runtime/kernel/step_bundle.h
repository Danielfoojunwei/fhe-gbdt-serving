#pragma once
#include "crypto_context.h"
#include <vector>
#include <memory>
#include <thread>
#include <future>

namespace fhe_gbdt::kernel {

/**
 * StepBundle: Batched evaluation of Step (comparison) operations
 *
 * Implements MOAI-style batching where multiple comparisons at the same
 * tree depth are evaluated together, amortizing the cost of scheme switching
 * and LUT operations.
 *
 * Optimizations:
 * - Parallel CPU execution using thread pool
 * - Batch GPU kernel launches
 * - LUT-based step function evaluation
 */
class StepBundle {
public:
    explicit StepBundle(std::shared_ptr<CryptoContext> ctx, size_t num_threads = 0);

    // Single step evaluation (for compatibility)
    std::shared_ptr<Ciphertext> EvaluateSingle(
        const std::shared_ptr<Ciphertext>& delta,
        bool strict = false);

    // Batched evaluation - main entry point for MOAI levelized execution
    // Uses parallel execution on CPU or batched kernel on GPU
    std::vector<std::shared_ptr<Ciphertext>> Evaluate(
        const std::vector<std::shared_ptr<Ciphertext>>& deltas,
        bool strict = false);

    // Batched evaluation with packing - packs multiple deltas into single ciphertext
    // for maximum throughput (requires compatible packing layout)
    std::shared_ptr<Ciphertext> EvaluatePacked(
        const std::shared_ptr<Ciphertext>& packed_deltas,
        size_t num_comparisons,
        bool strict = false);

    // Get statistics for performance monitoring
    struct Stats {
        size_t total_evals = 0;
        size_t batched_calls = 0;
        double total_time_ms = 0.0;
        size_t parallel_threads_used = 0;
    };
    Stats get_stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }

private:
    std::shared_ptr<CryptoContext> ctx_;
    size_t num_threads_;
    Stats stats_;

    // Thread pool for parallel CPU evaluation
    std::vector<std::thread> thread_pool_;

    // Apply step function to single ciphertext (internal)
    std::shared_ptr<Ciphertext> ApplyStep(
        const std::shared_ptr<Ciphertext>& delta,
        bool strict);
};

} // namespace fhe_gbdt::kernel
