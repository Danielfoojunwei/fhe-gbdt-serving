#pragma once
#include "kernel/crypto_context.h"
#include "kernel/step_bundle.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace fhe_gbdt::engine {

/**
 * Operation types in the MOAI execution schedule
 */
enum class OpType {
    ROTATE,         // Rotate ciphertext slots
    COMPARE_BATCH,  // Batch comparison (Step) operations
    DELTA,          // Compute delta = input - threshold
    AGGREGATE,      // Sum tree outputs
    LOAD_THRESHOLD, // Load threshold constants
    STORE_RESULT    // Store intermediate result
};

/**
 * Single operation in the execution plan
 */
struct Operation {
    OpType type;
    int offset = 0;           // For ROTATE
    size_t batch_size = 0;    // For COMPARE_BATCH
    int feature_idx = 0;      // For DELTA
    float threshold = 0.0f;   // For DELTA
    std::vector<float> weights;  // For AGGREGATE
};

/**
 * Schedule block - operations at a single tree depth
 */
struct ScheduleBlock {
    int depth_level;
    int node_group_id;
    std::vector<Operation> ops;
};

/**
 * Parsed plan IR structure
 */
struct ParsedPlan {
    std::string compiled_model_id;
    std::string crypto_params_id;
    int batch_size;
    float base_score;
    int num_trees;
    std::unordered_map<int, int> feature_to_slot;
    std::vector<ScheduleBlock> schedule;
};

/**
 * Executor: Executes compiled FHE-GBDT plans
 *
 * Implements the MOAI execution strategy:
 * 1. Levelized execution (all depth-d nodes processed together)
 * 2. Rotation scheduling (minimize ciphertext rotations)
 * 3. Batched Step evaluation
 */
class Executor {
public:
    explicit Executor(std::shared_ptr<kernel::CryptoContext> ctx);

    /**
     * Execute a compiled plan on encrypted inputs
     *
     * @param plan_json JSON representation of ObliviousPlanIR
     * @param inputs Encrypted feature ciphertexts
     * @return Encrypted prediction result
     */
    std::shared_ptr<kernel::Ciphertext> Execute(
        const std::string& plan_json,
        const std::vector<kernel::Ciphertext>& inputs);

    /**
     * Parse plan JSON into internal representation
     */
    ParsedPlan ParsePlan(const std::string& plan_json);

    /**
     * Execution statistics
     */
    struct Stats {
        size_t total_rotations = 0;
        size_t total_comparisons = 0;
        size_t total_adds = 0;
        double total_time_ms = 0.0;
    };
    Stats get_stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }

private:
    std::shared_ptr<kernel::CryptoContext> ctx_;
    std::unique_ptr<kernel::StepBundle> step_bundle_;
    Stats stats_;

    // Execute a single schedule block
    void ExecuteBlock(
        const ScheduleBlock& block,
        std::vector<std::shared_ptr<kernel::Ciphertext>>& working_set,
        const std::vector<kernel::Ciphertext>& inputs,
        const ParsedPlan& plan);

    // Ciphertext rotation (slot permutation)
    std::shared_ptr<kernel::Ciphertext> Rotate(
        const kernel::Ciphertext& ct,
        int offset);

    // Compute delta = input[feature] - threshold
    std::shared_ptr<kernel::Ciphertext> ComputeDelta(
        const kernel::Ciphertext& input,
        float threshold);
};

} // namespace fhe_gbdt::engine
