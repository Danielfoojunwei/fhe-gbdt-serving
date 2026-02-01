#include "step_bundle.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <execution>
#include <mutex>

namespace fhe_gbdt::kernel {

StepBundle::StepBundle(std::shared_ptr<CryptoContext> ctx, size_t num_threads)
    : ctx_(ctx), num_threads_(num_threads) {

    // Auto-detect thread count if not specified
    if (num_threads_ == 0) {
        num_threads_ = std::max(1u, std::thread::hardware_concurrency());
    }

    std::cout << "StepBundle initialized with " << num_threads_ << " threads" << std::endl;
}

std::shared_ptr<Ciphertext> StepBundle::EvaluateSingle(
    const std::shared_ptr<Ciphertext>& delta,
    bool strict) {

    auto start = std::chrono::high_resolution_clock::now();
    auto result = ApplyStep(delta, strict);
    auto end = std::chrono::high_resolution_clock::now();

    stats_.total_evals++;
    stats_.total_time_ms += std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

std::vector<std::shared_ptr<Ciphertext>> StepBundle::Evaluate(
    const std::vector<std::shared_ptr<Ciphertext>>& deltas,
    bool strict) {

    if (deltas.empty()) {
        return {};
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::shared_ptr<Ciphertext>> results(deltas.size());

    // Parallel execution using C++17 parallel algorithms
    // This is the key optimization: instead of sequential loop, we use
    // parallel_for semantics to distribute work across threads

    if (deltas.size() >= 4 && num_threads_ > 1) {
        // Use parallel execution for larger batches
        std::mutex stats_mutex;

        // Create index range for parallel processing
        std::vector<size_t> indices(deltas.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Execute in parallel using execution policy
        std::for_each(
            std::execution::par_unseq,
            indices.begin(),
            indices.end(),
            [&](size_t i) {
                results[i] = ApplyStep(deltas[i], strict);
            }
        );

        stats_.parallel_threads_used = num_threads_;
    } else {
        // Sequential for small batches (overhead not worth it)
        for (size_t i = 0; i < deltas.size(); ++i) {
            results[i] = ApplyStep(deltas[i], strict);
        }
        stats_.parallel_threads_used = 1;
    }

    auto end = std::chrono::high_resolution_clock::now();

    stats_.total_evals += deltas.size();
    stats_.batched_calls++;
    stats_.total_time_ms += std::chrono::duration<double, std::milli>(end - start).count();

    return results;
}

std::shared_ptr<Ciphertext> StepBundle::EvaluatePacked(
    const std::shared_ptr<Ciphertext>& packed_deltas,
    size_t num_comparisons,
    bool strict) {

    // Packed evaluation: all comparisons are encoded in a single RLWE ciphertext
    // The step function is applied element-wise to all slots in parallel
    //
    // This is the most efficient mode for MOAI-style execution where
    // multiple trees at the same depth share the same rotation offset.
    //
    // Implementation uses N2HE's LUT-based step function which operates
    // on all slots simultaneously.

    auto start = std::chrono::high_resolution_clock::now();

    // Apply step to the packed ciphertext
    // The N2HE Step operation inherently works on all polynomial coefficients
    auto result = ctx_->Step(*packed_deltas, strict);

    auto end = std::chrono::high_resolution_clock::now();

    stats_.total_evals += num_comparisons;
    stats_.batched_calls++;
    stats_.total_time_ms += std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

std::shared_ptr<Ciphertext> StepBundle::ApplyStep(
    const std::shared_ptr<Ciphertext>& delta,
    bool strict) {

    // The Step function implements the comparison operation:
    // Step(x) = 1 if x >= 0, else 0 (or strict: x > 0)
    //
    // In FHE, this is implemented via:
    // 1. Sign extraction using functional bootstrapping (TFHE-style), or
    // 2. Lookup table evaluation (N2HE LUT), or
    // 3. Polynomial approximation (less accurate)
    //
    // N2HE uses LUT-based evaluation which is exact and efficient.

    return ctx_->Step(*delta, strict);
}

} // namespace fhe_gbdt::kernel
