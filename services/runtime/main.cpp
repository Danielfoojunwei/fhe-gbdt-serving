#include <iostream>
#include "kernel/crypto_context.h"
#include "engine/executor.h"

int main() {
    std::cout << "FHE-GBDT Runtime starting..." << std::endl;
    
    auto ctx = std::make_shared<fhe_gbdt::kernel::CryptoContext>("n2he_default");
    fhe_gbdt::engine::Executor executor(ctx);
    
    std::cout << "Runtime ready for inference." << std::endl;
    return 0;
}
