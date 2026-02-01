#pragma once
#include "kernel/crypto_context.h"
#include <string>
#include <vector>

namespace fhe_gbdt::engine {

class Executor {
public:
    Executor(std::shared_ptr<kernel::CryptoContext> ctx);
    
    // Execute a compiled plan
    std::shared_ptr<kernel::Ciphertext> Execute(const std::string& plan_json, const std::vector<kernel::Ciphertext>& inputs);

private:
    std::shared_ptr<kernel::CryptoContext> ctx_;
};

} // namespace fhe_gbdt::engine
