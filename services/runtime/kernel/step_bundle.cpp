#include "step_bundle.h"
#include <iostream>

namespace fhe_gbdt::kernel {

StepBundle::StepBundle(std::shared_ptr<CryptoContext> ctx) : ctx_(ctx) {}

std::vector<std::shared_ptr<Ciphertext>> StepBundle::Evaluate(
    const std::vector<std::shared_ptr<Ciphertext>>& deltas, 
    bool strict) {
    
    std::vector<std::shared_ptr<Ciphertext>> results;
    results.reserve(deltas.size());

    // In a production implementation, this would use N2HE-HEXL 
    // to batch multiple Step operations into a single EvalSum/Permute sequence.
    // For now, we iterate and rely on the CryptoContext's Step primitive.
    for (const auto& delta : deltas) {
        results.push_back(ctx_->Step(delta, strict));
    }

    return results;
}

} // namespace fhe_gbdt::kernel
