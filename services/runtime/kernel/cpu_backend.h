#pragma once
#include "backend.h"
#include "crypto_context.h"
#include "third_party/n2he/include/include.hpp"
#include <iostream>

namespace fhe_gbdt::kernel {

class CpuBackend : public Backend {
public:
    CpuBackend(const std::string& scheme_id) : scheme_id_(scheme_id) {
        // N2HE parameters initialization (e.g., q, n, k)
        q_ = 1ULL << 32; // Default for RLWE64
        n_ = 2048;
        k_ = 1; // variance param
    }

    std::shared_ptr<Ciphertext> Add(const Ciphertext& a, const Ciphertext& b) override {
        auto res = std::make_shared<Ciphertext>();
        res->scheme_id = scheme_id_;
        res->is_rlwe = a.is_rlwe;

        if (a.is_rlwe) {
            res->rlwe_data = a.rlwe_data;
            // Native N2HE addition for RLWE
            for (size_t i = 0; i < a.rlwe_data.size(); ++i) {
                add_poly(res->rlwe_data[i], b.rlwe_data[i], q_);
            }
        } else {
            // LWE addition
            res->lwe_data = a.lwe_data;
            for (size_t i = 0; i < a.lwe_data.size(); ++i) {
                res->lwe_data[i] = (res->lwe_data[i] + b.lwe_data[i]) % q_;
            }
        }
        return res;
    }

    std::shared_ptr<Ciphertext> Sub(const Ciphertext& a, const Ciphertext& b) override {
        auto res = std::make_shared<Ciphertext>();
        // Similar to Add but with subtraction logic
        return res;
    }

    std::shared_ptr<Ciphertext> MulPlain(const Ciphertext& a, float constant) override {
        auto res = std::make_shared<Ciphertext>();
        res->is_rlwe = a.is_rlwe;
        if (a.is_rlwe) {
            res->rlwe_data = a.rlwe_data;
            for (auto& poly : res->rlwe_data) {
                multi_scale_poly(static_cast<int64_t>(constant), poly, q_);
            }
        }
        return res;
    }

    std::shared_ptr<Ciphertext> Step(const Ciphertext& delta, bool strict) override {
        // Here we'd call the N2HE specific "Step" (likely a LUT or comparison)
        // For now, wrapping as a call to RLWE primitives
        return std::make_shared<Ciphertext>();
    }

private:
    std::string scheme_id_;
    int64_t q_;
    int n_;
    int k_;
};

} // namespace fhe_gbdt::kernel
