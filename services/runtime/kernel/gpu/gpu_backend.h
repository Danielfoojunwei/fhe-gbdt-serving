#pragma once
#include "../backend.h"
#include "../crypto_context.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace fhe_gbdt::kernel::gpu {

// CUDA kernel declarations (implemented in .cu file)
extern "C" {
    void cuda_rlwe_add(int64_t* result, const int64_t* a, const int64_t* b,
                       int n, int64_t q, cudaStream_t stream);
    void cuda_rlwe_sub(int64_t* result, const int64_t* a, const int64_t* b,
                       int n, int64_t q, cudaStream_t stream);
    void cuda_rlwe_mul_plain(int64_t* result, const int64_t* a, int64_t scalar,
                             int n, int64_t q, cudaStream_t stream);
    void cuda_rlwe_add_batch(int64_t* result, const int64_t* a, const int64_t* b,
                              int n, int batch_size, int64_t q, cudaStream_t stream);
    void cuda_ntt_forward(int64_t* data, int n, int64_t q, cudaStream_t stream);
    void cuda_ntt_inverse(int64_t* data, int n, int64_t q, cudaStream_t stream);
    void cuda_ntt_init(int n, int64_t q);
    void cuda_ntt_cleanup();
    void cuda_step(int64_t* result, const int64_t* input, int n, int64_t q, cudaStream_t stream);
}

/**
 * GpuBackend: CUDA-accelerated FHE operations
 *
 * Optimizations implemented:
 * - Batched memory transfers to minimize PCIe overhead
 * - Precomputed NTT twiddle factors
 * - Persistent device memory pools
 * - Async stream execution
 */
class GpuBackend : public Backend {
public:
    static constexpr size_t MAX_BATCH_SIZE = 256;
    static constexpr size_t POLY_DEGREE = 2048;

    GpuBackend(const std::string& scheme_id, int device_id = 0)
        : scheme_id_(scheme_id), device_id_(device_id) {

        // Initialize CUDA
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA device: " +
                                     std::string(cudaGetErrorString(err)));
        }

        // Create CUDA stream for async operations
        err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }

        // N2HE default parameters
        q_ = 1ULL << 32;
        n_ = POLY_DEGREE;

        // Allocate device memory pools for batched operations
        size_t batch_poly_size = MAX_BATCH_SIZE * n_ * 2 * sizeof(int64_t);

        cudaMalloc(&d_batch_a_, batch_poly_size);
        cudaMalloc(&d_batch_b_, batch_poly_size);
        cudaMalloc(&d_batch_result_, batch_poly_size);

        // Single-ciphertext buffers (legacy)
        cudaMalloc(&d_temp_a_, n_ * sizeof(int64_t));
        cudaMalloc(&d_temp_b_, n_ * sizeof(int64_t));
        cudaMalloc(&d_temp_result_, n_ * sizeof(int64_t));

        // Initialize NTT with precomputed twiddles
        cuda_ntt_init(n_, q_);

        std::cout << "GPU Backend initialized: device=" << device_id_
                  << ", n=" << n_ << ", q=" << q_ << std::endl;
    }

    ~GpuBackend() {
        cudaFree(d_batch_a_);
        cudaFree(d_batch_b_);
        cudaFree(d_batch_result_);
        cudaFree(d_temp_a_);
        cudaFree(d_temp_b_);
        cudaFree(d_temp_result_);
        cudaStreamDestroy(stream_);
        cuda_ntt_cleanup();
    }

    std::shared_ptr<Ciphertext> Add(const Ciphertext& a, const Ciphertext& b) override {
        auto res = std::make_shared<Ciphertext>();
        res->scheme_id = scheme_id_;
        res->is_rlwe = a.is_rlwe;

        if (a.is_rlwe) {
            res->rlwe_data.resize(a.rlwe_data.size());
            for (size_t i = 0; i < a.rlwe_data.size(); ++i) {
                res->rlwe_data[i].resize(n_);

                cudaMemcpyAsync(d_temp_a_, a.rlwe_data[i].data(), n_ * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_);
                cudaMemcpyAsync(d_temp_b_, b.rlwe_data[i].data(), n_ * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_);

                cuda_rlwe_add(d_temp_result_, d_temp_a_, d_temp_b_, n_, q_, stream_);

                cudaMemcpyAsync(res->rlwe_data[i].data(), d_temp_result_, n_ * sizeof(int64_t),
                               cudaMemcpyDeviceToHost, stream_);
            }
            cudaStreamSynchronize(stream_);
        } else {
            res->lwe_data.resize(a.lwe_data.size());
            cudaMemcpyAsync(d_temp_a_, a.lwe_data.data(), a.lwe_data.size() * sizeof(int64_t),
                           cudaMemcpyHostToDevice, stream_);
            cudaMemcpyAsync(d_temp_b_, b.lwe_data.data(), b.lwe_data.size() * sizeof(int64_t),
                           cudaMemcpyHostToDevice, stream_);
            cuda_rlwe_add(d_temp_result_, d_temp_a_, d_temp_b_, a.lwe_data.size(), q_, stream_);
            cudaMemcpyAsync(res->lwe_data.data(), d_temp_result_, a.lwe_data.size() * sizeof(int64_t),
                           cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);
        }

        return res;
    }

    // Batched Add: process multiple ciphertexts in single kernel launch
    std::vector<std::shared_ptr<Ciphertext>> AddBatch(
        const std::vector<Ciphertext>& as,
        const std::vector<Ciphertext>& bs) {

        if (as.size() != bs.size() || as.empty()) {
            return {};
        }

        size_t batch_size = std::min(as.size(), MAX_BATCH_SIZE);
        std::vector<std::shared_ptr<Ciphertext>> results(batch_size);

        // Pack all ciphertexts into contiguous buffers
        std::vector<int64_t> h_batch_a(batch_size * n_ * 2);
        std::vector<int64_t> h_batch_b(batch_size * n_ * 2);

        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < 2 && j < as[i].rlwe_data.size(); ++j) {
                std::copy(as[i].rlwe_data[j].begin(), as[i].rlwe_data[j].end(),
                          h_batch_a.begin() + (i * 2 + j) * n_);
                std::copy(bs[i].rlwe_data[j].begin(), bs[i].rlwe_data[j].end(),
                          h_batch_b.begin() + (i * 2 + j) * n_);
            }
        }

        // Single H2D transfer
        cudaMemcpyAsync(d_batch_a_, h_batch_a.data(), h_batch_a.size() * sizeof(int64_t),
                       cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_batch_b_, h_batch_b.data(), h_batch_b.size() * sizeof(int64_t),
                       cudaMemcpyHostToDevice, stream_);

        // Single batched kernel
        cuda_rlwe_add_batch(d_batch_result_, d_batch_a_, d_batch_b_, n_, batch_size * 2, q_, stream_);

        // Single D2H transfer
        std::vector<int64_t> h_batch_result(batch_size * n_ * 2);
        cudaMemcpyAsync(h_batch_result.data(), d_batch_result_,
                       h_batch_result.size() * sizeof(int64_t),
                       cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // Unpack results
        for (size_t i = 0; i < batch_size; ++i) {
            results[i] = std::make_shared<Ciphertext>();
            results[i]->scheme_id = scheme_id_;
            results[i]->is_rlwe = true;
            results[i]->rlwe_data.resize(2);
            for (size_t j = 0; j < 2; ++j) {
                results[i]->rlwe_data[j].resize(n_);
                std::copy(h_batch_result.begin() + (i * 2 + j) * n_,
                          h_batch_result.begin() + (i * 2 + j + 1) * n_,
                          results[i]->rlwe_data[j].begin());
            }
        }

        return results;
    }

    std::shared_ptr<Ciphertext> Sub(const Ciphertext& a, const Ciphertext& b) override {
        auto res = std::make_shared<Ciphertext>();
        res->scheme_id = scheme_id_;
        res->is_rlwe = a.is_rlwe;

        if (a.is_rlwe) {
            res->rlwe_data.resize(a.rlwe_data.size());
            for (size_t i = 0; i < a.rlwe_data.size(); ++i) {
                res->rlwe_data[i].resize(n_);

                cudaMemcpyAsync(d_temp_a_, a.rlwe_data[i].data(), n_ * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_);
                cudaMemcpyAsync(d_temp_b_, b.rlwe_data[i].data(), n_ * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_);

                cuda_rlwe_sub(d_temp_result_, d_temp_a_, d_temp_b_, n_, q_, stream_);

                cudaMemcpyAsync(res->rlwe_data[i].data(), d_temp_result_, n_ * sizeof(int64_t),
                               cudaMemcpyDeviceToHost, stream_);
            }
            cudaStreamSynchronize(stream_);
        }

        return res;
    }

    std::shared_ptr<Ciphertext> MulPlain(const Ciphertext& a, float constant) override {
        auto res = std::make_shared<Ciphertext>();
        res->scheme_id = scheme_id_;
        res->is_rlwe = a.is_rlwe;

        int64_t scalar = static_cast<int64_t>(constant);

        if (a.is_rlwe) {
            res->rlwe_data.resize(a.rlwe_data.size());
            for (size_t i = 0; i < a.rlwe_data.size(); ++i) {
                res->rlwe_data[i].resize(n_);

                cudaMemcpyAsync(d_temp_a_, a.rlwe_data[i].data(), n_ * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_);

                cuda_rlwe_mul_plain(d_temp_result_, d_temp_a_, scalar, n_, q_, stream_);

                cudaMemcpyAsync(res->rlwe_data[i].data(), d_temp_result_, n_ * sizeof(int64_t),
                               cudaMemcpyDeviceToHost, stream_);
            }
            cudaStreamSynchronize(stream_);
        }

        return res;
    }

    std::shared_ptr<Ciphertext> Step(const Ciphertext& delta, bool strict) override {
        auto res = std::make_shared<Ciphertext>();
        res->scheme_id = scheme_id_;
        res->is_rlwe = delta.is_rlwe;

        if (delta.is_rlwe && !delta.rlwe_data.empty()) {
            res->rlwe_data.resize(delta.rlwe_data.size());
            for (size_t i = 0; i < delta.rlwe_data.size(); ++i) {
                res->rlwe_data[i].resize(n_);

                // Transfer to device
                cudaMemcpyAsync(d_temp_a_, delta.rlwe_data[i].data(), n_ * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_);

                // Apply NTT for polynomial multiplication context
                cuda_ntt_forward(d_temp_a_, n_, q_, stream_);

                // Apply step function in NTT domain
                cuda_step(d_temp_result_, d_temp_a_, n_, q_, stream_);

                // Convert back from NTT domain
                cuda_ntt_inverse(d_temp_result_, n_, q_, stream_);

                // Transfer back to host
                cudaMemcpyAsync(res->rlwe_data[i].data(), d_temp_result_, n_ * sizeof(int64_t),
                               cudaMemcpyDeviceToHost, stream_);
            }
            cudaStreamSynchronize(stream_);
        }

        return res;
    }

private:
    std::string scheme_id_;
    int device_id_;
    int64_t q_;
    int n_;
    cudaStream_t stream_;

    // Batched device memory pools
    int64_t* d_batch_a_;
    int64_t* d_batch_b_;
    int64_t* d_batch_result_;

    // Single-ciphertext device buffers (legacy compatibility)
    int64_t* d_temp_a_;
    int64_t* d_temp_b_;
    int64_t* d_temp_result_;
};

} // namespace fhe_gbdt::kernel::gpu
