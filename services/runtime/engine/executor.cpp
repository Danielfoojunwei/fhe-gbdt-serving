#include "executor.h"
#include <iostream>
#include <chrono>
#include <sstream>
#include <regex>
#include <cmath>

// Simple JSON parsing (production would use nlohmann/json or similar)
namespace {

std::string extract_string(const std::string& json, const std::string& key) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return match[1].str();
    }
    return "";
}

float extract_float(const std::string& json, const std::string& key) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*([\\d.\\-]+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stof(match[1].str());
    }
    return 0.0f;
}

int extract_int(const std::string& json, const std::string& key) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*(\\d+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return 0;
}

} // anonymous namespace

namespace fhe_gbdt::engine {

Executor::Executor(std::shared_ptr<kernel::CryptoContext> ctx)
    : ctx_(ctx) {
    step_bundle_ = std::make_unique<kernel::StepBundle>(ctx);
    std::cout << "Executor initialized with StepBundle" << std::endl;
}

ParsedPlan Executor::ParsePlan(const std::string& plan_json) {
    ParsedPlan plan;

    // Extract top-level fields
    plan.compiled_model_id = extract_string(plan_json, "compiled_model_id");
    plan.crypto_params_id = extract_string(plan_json, "crypto_params_id");
    plan.base_score = extract_float(plan_json, "base_score");
    plan.num_trees = extract_int(plan_json, "num_trees");

    // Extract packing layout
    std::regex slots_pattern("\"slots\"\\s*:\\s*(\\d+)");
    std::smatch slots_match;
    if (std::regex_search(plan_json, slots_match, slots_pattern)) {
        plan.batch_size = std::stoi(slots_match[1].str());
    } else {
        plan.batch_size = 256; // Default
    }

    // Extract feature_to_ciphertext mapping
    std::regex feature_map_pattern("\"(\\d+)\"\\s*:\\s*(\\d+)");
    std::string::const_iterator search_start(plan_json.cbegin());
    std::smatch fm_match;

    // Find the feature_to_ciphertext section
    size_t ftc_pos = plan_json.find("\"feature_to_ciphertext\"");
    if (ftc_pos != std::string::npos) {
        size_t brace_start = plan_json.find("{", ftc_pos);
        size_t brace_end = plan_json.find("}", brace_start);
        if (brace_start != std::string::npos && brace_end != std::string::npos) {
            std::string ftc_section = plan_json.substr(brace_start, brace_end - brace_start + 1);
            while (std::regex_search(ftc_section, fm_match, feature_map_pattern)) {
                int feature = std::stoi(fm_match[1].str());
                int slot = std::stoi(fm_match[2].str());
                plan.feature_to_slot[feature] = slot;
                ftc_section = fm_match.suffix().str();
            }
        }
    }

    // Extract schedule blocks
    std::regex block_pattern("\"depth_level\"\\s*:\\s*(\\d+)");
    std::regex ops_pattern("(ROTATE|COMPARE_BATCH)\\((\\w+)=(\\d+)\\)");

    size_t schedule_pos = plan_json.find("\"schedule\"");
    if (schedule_pos != std::string::npos) {
        std::string schedule_section = plan_json.substr(schedule_pos);

        // Find all depth levels
        std::string::const_iterator it = schedule_section.cbegin();
        std::smatch block_match;

        while (std::regex_search(it, schedule_section.cend(), block_match, block_pattern)) {
            ScheduleBlock block;
            block.depth_level = std::stoi(block_match[1].str());
            block.node_group_id = 0;

            // Find ops for this block (simplified parsing)
            std::string block_section = block_match.suffix().str();
            size_t ops_end = block_section.find("depth_level");
            if (ops_end == std::string::npos) ops_end = block_section.length();
            std::string ops_section = block_section.substr(0, ops_end);

            std::string::const_iterator ops_it = ops_section.cbegin();
            std::smatch ops_match;

            while (std::regex_search(ops_it, ops_section.cend(), ops_match, ops_pattern)) {
                Operation op;
                std::string op_type = ops_match[1].str();
                std::string param_name = ops_match[2].str();
                int param_value = std::stoi(ops_match[3].str());

                if (op_type == "ROTATE") {
                    op.type = OpType::ROTATE;
                    op.offset = param_value;
                } else if (op_type == "COMPARE_BATCH") {
                    op.type = OpType::COMPARE_BATCH;
                    op.batch_size = param_value;
                }

                block.ops.push_back(op);
                ops_it = ops_match.suffix().first;
            }

            plan.schedule.push_back(block);
            it = block_match.suffix().first;
        }
    }

    std::cout << "Parsed plan: model_id=" << plan.compiled_model_id
              << ", trees=" << plan.num_trees
              << ", schedule_blocks=" << plan.schedule.size() << std::endl;

    return plan;
}

std::shared_ptr<kernel::Ciphertext> Executor::Execute(
    const std::string& plan_json,
    const std::vector<kernel::Ciphertext>& inputs) {

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Executing FHE-GBDT Plan..." << std::endl;

    // 1. Parse the plan
    ParsedPlan plan = ParsePlan(plan_json);

    if (inputs.empty()) {
        std::cerr << "Error: No input ciphertexts provided" << std::endl;
        return std::make_shared<kernel::Ciphertext>();
    }

    // 2. Initialize working set with encrypted zeros (for accumulation)
    std::vector<std::shared_ptr<kernel::Ciphertext>> working_set;
    working_set.reserve(plan.num_trees);

    for (int i = 0; i < plan.num_trees; ++i) {
        auto ct = std::make_shared<kernel::Ciphertext>();
        ct->scheme_id = inputs[0].scheme_id;
        ct->is_rlwe = inputs[0].is_rlwe;
        ct->batch_size = inputs[0].batch_size;
        // Initialize with copy of first input (will be overwritten)
        if (!inputs[0].rlwe_data.empty()) {
            ct->rlwe_data = inputs[0].rlwe_data;
            // Zero out
            for (auto& poly : ct->rlwe_data) {
                std::fill(poly.begin(), poly.end(), 0);
            }
        }
        working_set.push_back(ct);
    }

    // 3. Execute schedule blocks (levelized execution)
    for (const auto& block : plan.schedule) {
        ExecuteBlock(block, working_set, inputs, plan);
    }

    // 4. Aggregate tree outputs
    auto result = std::make_shared<kernel::Ciphertext>();
    result->scheme_id = inputs[0].scheme_id;
    result->is_rlwe = inputs[0].is_rlwe;
    result->batch_size = inputs[0].batch_size;

    if (!working_set.empty() && working_set[0]) {
        // Sum all tree outputs
        result = working_set[0];
        for (size_t i = 1; i < working_set.size(); ++i) {
            if (working_set[i]) {
                result = ctx_->Add(*result, *working_set[i]);
                stats_.total_adds++;
            }
        }
    }

    // 5. Add base score (as plaintext multiplication)
    if (plan.base_score != 0.0f) {
        // Base score would be encoded and added
        // For now, this is a no-op as we'd need plaintext encoding
    }

    auto end = std::chrono::high_resolution_clock::now();
    stats_.total_time_ms += std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Execution complete: rotations=" << stats_.total_rotations
              << ", comparisons=" << stats_.total_comparisons
              << ", adds=" << stats_.total_adds
              << ", time=" << stats_.total_time_ms << "ms" << std::endl;

    return result;
}

void Executor::ExecuteBlock(
    const ScheduleBlock& block,
    std::vector<std::shared_ptr<kernel::Ciphertext>>& working_set,
    const std::vector<kernel::Ciphertext>& inputs,
    const ParsedPlan& plan) {

    std::cout << "  Executing block at depth " << block.depth_level << std::endl;

    std::shared_ptr<kernel::Ciphertext> current_input;
    if (!inputs.empty()) {
        current_input = std::make_shared<kernel::Ciphertext>(inputs[0]);
    }

    std::vector<std::shared_ptr<kernel::Ciphertext>> deltas;

    for (const auto& op : block.ops) {
        switch (op.type) {
            case OpType::ROTATE: {
                if (current_input && op.offset != 0) {
                    current_input = Rotate(*current_input, op.offset);
                    stats_.total_rotations++;
                }
                break;
            }

            case OpType::COMPARE_BATCH: {
                // Collect deltas for batch comparison
                // In full implementation, deltas would be computed from
                // rotated inputs and thresholds

                if (current_input) {
                    // For each comparison in the batch, compute delta and step
                    for (size_t i = 0; i < op.batch_size; ++i) {
                        deltas.push_back(current_input);
                    }
                }

                // Batch evaluate Step functions
                if (!deltas.empty()) {
                    auto step_results = step_bundle_->Evaluate(deltas, false);
                    stats_.total_comparisons += deltas.size();

                    // Update working set with step results
                    for (size_t i = 0; i < step_results.size() && i < working_set.size(); ++i) {
                        if (step_results[i] && working_set[i]) {
                            working_set[i] = ctx_->Add(*working_set[i], *step_results[i]);
                            stats_.total_adds++;
                        }
                    }
                    deltas.clear();
                }
                break;
            }

            case OpType::DELTA: {
                if (current_input) {
                    auto delta = ComputeDelta(*current_input, op.threshold);
                    deltas.push_back(delta);
                }
                break;
            }

            case OpType::AGGREGATE: {
                // Weighted sum of tree outputs
                // Implementation depends on tree structure
                break;
            }

            default:
                break;
        }
    }
}

std::shared_ptr<kernel::Ciphertext> Executor::Rotate(
    const kernel::Ciphertext& ct,
    int offset) {

    // Ciphertext rotation: permute slots by offset
    // In RLWE, this is done via automorphism (Galois elements)
    // For now, implement as coefficient rotation

    auto result = std::make_shared<kernel::Ciphertext>();
    result->scheme_id = ct.scheme_id;
    result->is_rlwe = ct.is_rlwe;
    result->batch_size = ct.batch_size;

    if (ct.is_rlwe && !ct.rlwe_data.empty()) {
        result->rlwe_data.resize(ct.rlwe_data.size());
        size_t n = ct.rlwe_data[0].size();

        for (size_t poly_idx = 0; poly_idx < ct.rlwe_data.size(); ++poly_idx) {
            result->rlwe_data[poly_idx].resize(n);
            for (size_t i = 0; i < n; ++i) {
                // Cyclic rotation with sign flip for negacyclic ring
                size_t src_idx = (i + n - offset) % n;
                int64_t val = ct.rlwe_data[poly_idx][src_idx];
                if (i < static_cast<size_t>(offset)) {
                    val = -val; // Negacyclic property
                }
                result->rlwe_data[poly_idx][i] = val;
            }
        }
    }

    return result;
}

std::shared_ptr<kernel::Ciphertext> Executor::ComputeDelta(
    const kernel::Ciphertext& input,
    float threshold) {

    // Delta = input - threshold
    // Threshold is encoded as plaintext polynomial

    auto result = std::make_shared<kernel::Ciphertext>();
    result->scheme_id = input.scheme_id;
    result->is_rlwe = input.is_rlwe;
    result->batch_size = input.batch_size;

    if (input.is_rlwe && !input.rlwe_data.empty()) {
        result->rlwe_data.resize(input.rlwe_data.size());
        size_t n = input.rlwe_data[0].size();
        int64_t q = 1ULL << 32;
        int64_t encoded_threshold = static_cast<int64_t>(threshold * (q / 4)) % q;

        for (size_t poly_idx = 0; poly_idx < input.rlwe_data.size(); ++poly_idx) {
            result->rlwe_data[poly_idx].resize(n);
            for (size_t i = 0; i < n; ++i) {
                // Subtract threshold from first coefficient only (constant term)
                int64_t val = input.rlwe_data[poly_idx][i];
                if (i == 0 && poly_idx == 1) { // b polynomial, constant term
                    val = (val - encoded_threshold + q) % q;
                }
                result->rlwe_data[poly_idx][i] = val;
            }
        }
    }

    return result;
}

} // namespace fhe_gbdt::engine
