#pragma once
#include "crypto_context.h"
#include <vector>
#include <memory>

namespace fhe_gbdt::kernel {

class StepBundle {
public:
    StepBundle(std::shared_ptr<CryptoContext> ctx);
    
    // Evaluate multiple Step operations in a bundle to amortize scheme switch costs
    std::vector<std::shared_ptr<Ciphertext>> Evaluate(const std::vector<std::shared_ptr<Ciphertext>>& deltas, bool strict = false);

private:
    std::shared_ptr<CryptoContext> ctx_;
};

} // namespace fhe_gbdt::kernel
