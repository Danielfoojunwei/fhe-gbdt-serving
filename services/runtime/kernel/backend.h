#pragma once
#include <memory>
#include <vector>
#include <string>

namespace fhe_gbdt::kernel {

class Ciphertext; // Forward declaration

class Backend {
public:
    virtual ~Backend() = default;

    virtual std::shared_ptr<Ciphertext> Add(const Ciphertext& a, const Ciphertext& b) = 0;
    virtual std::shared_ptr<Ciphertext> Sub(const Ciphertext& a, const Ciphertext& b) = 0;
    virtual std::shared_ptr<Ciphertext> MulPlain(const Ciphertext& a, float constant) = 0;
    virtual std::shared_ptr<Ciphertext> Step(const Ciphertext& delta, bool strict) = 0;
};

} // namespace fhe_gbdt::kernel
