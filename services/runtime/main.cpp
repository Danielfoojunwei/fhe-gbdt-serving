#include <iostream>
#include <stdexcept>
#include "kernel/crypto_context.h"
#include "engine/moai_executor.h"

/**
 * FHE-GBDT Runtime Entry Point
 *
 * Uses MOAI optimizations for production inference:
 * - Column packing (rotation-free comparisons)
 * - Polynomial Step function (no LUT overhead)
 * - Interleaved aggregation (log-reduction)
 */
int main(int argc, char* argv[]) {
    try {
        std::cout << "FHE-GBDT Runtime starting (MOAI-optimized)..." << std::endl;

        // Initialize crypto context with production parameters
        auto ctx = std::make_shared<fhe_gbdt::kernel::CryptoContext>("n2he_default");

        // Initialize MOAI executor
        fhe_gbdt::engine::MOAIExecutor executor(ctx);

        std::cout << "Runtime ready for inference." << std::endl;
        std::cout << "  - Column packing: enabled" << std::endl;
        std::cout << "  - Rotation-free Step: enabled" << std::endl;
        std::cout << "  - Interleaved aggregation: enabled" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
