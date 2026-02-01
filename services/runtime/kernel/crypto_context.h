#include <vector>
#include <string>
#include <memory>
#include "backend.h"

namespace fhe_gbdt::kernel {

class Ciphertext {
public:
    // N2HE types
    std::vector<int64_t> lwe_data;
    std::vector<std::vector<int64_t>> rlwe_data;
    
    std::string scheme_id;
    uint32_t batch_size;
    bool is_rlwe = false;
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
    std::unique_ptr<Backend> backend_;
};

} // namespace fhe_gbdt::kernel
