#include "crypto_context.h"
#include "cpu_backend.h"
#include "gpu/gpu_backend.h"
#include <iostream>
#include <cstdlib>

namespace fhe_gbdt::kernel {

CryptoContext::CryptoContext(const std::string& scheme_id) : scheme_id_(scheme_id) {
    const char* backend_env = std::getenv("N2HE_BACKEND");
    std::string backend_type = (backend_env) ? std::string(backend_env) : "CPU";

    if (backend_type == "GPU") {
        std::cout << "Using GPU Backend" << std::endl;
        backend_ = std::make_unique<gpu::GpuBackend>(scheme_id);
    } else {
        std::cout << "Using CPU Backend" << std::endl;
        backend_ = std::make_unique<CpuBackend>(scheme_id);
    }
}

std::shared_ptr<Ciphertext> CryptoContext::Add(const Ciphertext& a, const Ciphertext& b) {
    return backend_->Add(a, b);
}

std::shared_ptr<Ciphertext> CryptoContext::Sub(const Ciphertext& a, const Ciphertext& b) {
    return backend_->Sub(a, b);
}

std::shared_ptr<Ciphertext> CryptoContext::MulPlain(const Ciphertext& a, float constant) {
    return backend_->MulPlain(a, constant);
}

std::shared_ptr<Ciphertext> CryptoContext::Step(const Ciphertext& delta, bool strict) {
    return backend_->Step(delta, strict);
}

} // namespace fhe_gbdt::kernel
