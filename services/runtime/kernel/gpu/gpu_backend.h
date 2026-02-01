#pragma once
#include "../backend.h"
#include <string>

namespace fhe_gbdt::kernel::gpu {

class GpuBackend : public Backend {
public:
    GpuBackend(const std::string& scheme_id);
    ~GpuBackend() override;

    std::shared_ptr<Ciphertext> Add(const Ciphertext& a, const Ciphertext& b) override;
    std::shared_ptr<Ciphertext> Sub(const Ciphertext& a, const Ciphertext& b) override;
    std::shared_ptr<Ciphertext> MulPlain(const Ciphertext& a, float constant) override;
    std::shared_ptr<Ciphertext> Step(const Ciphertext& delta, bool strict) override;

private:
    std::string scheme_id_;
    void* cuda_stream_; // Placeholder for cudaStream_t
};

} // namespace fhe_gbdt::kernel::gpu
