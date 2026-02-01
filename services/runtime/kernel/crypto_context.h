#pragma once
#include <vector>
#include <string>
#include <memory>

namespace fhe_gbdt::kernel {

class Ciphertext {
public:
    std::vector<uint8_t> data;
    std::string scheme_id;
    uint32_t batch_size;
};

class CryptoContext {
public:
    CryptoContext(const std::string& scheme_id);
    
    // Primitive ops (Wrappers for N2HE)
    std::shared_ptr<Ciphertext> Add(const Ciphertext& a, const Ciphertext& b);
    std::shared_ptr<Ciphertext> Sub(const Ciphertext& a, const Ciphertext& b);
    std::shared_ptr<Ciphertext> MulPlain(const Ciphertext& a, float constant);
    
    // Hybrid Step function
    std::shared_ptr<Ciphertext> Step(const Ciphertext& delta, bool strict = false);

private:
    std::string scheme_id_;
    // n2he::Context* n2he_ctx_;
};

} // namespace fhe_gbdt::kernel
