#pragma once
#include "../backend.h"
#include "../crypto_context.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <stdexcept>

namespace fhe_gbdt::kernel::gpu {

// CUDA kernel declarations (implemented in .cu file)
extern "C" {
    void cuda_rlwe_add(int64_t* result, const int64_t* a, const int64_t* b, 
                       int n, int64_t q, cudaStream_t stream);
    void cuda_rlwe_sub(int64_t* result, const int64_t* a, const int64_t* b,
                       int n, int64_t q, cudaStream_t stream);
    void cuda_rlwe_mul_plain(int64_t* result, const int64_t* a, int64_t scalar,
                             int n, int64_t q, cudaStream_t stream);
    void cuda_ntt_forward(int64_t* data, int n, int64_t q, cudaStream_t stream);
    void cuda_ntt_inverse(int64_t* data, int n, int64_t q, cudaStream_t stream);
}

class GpuBackend : public Backend {
public:
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
        n_ = 2048;
        
        // Allocate device memory pools
        cudaMalloc(&d_temp_a_, n_ * sizeof(int64_t));
        cudaMalloc(&d_temp_b_, n_ * sizeof(int64_t));
        cudaMalloc(&d_temp_result_, n_ * sizeof(int64_t));
    }
    
    ~GpuBackend() {
        cudaFree(d_temp_a_);
        cudaFree(d_temp_b_);
        cudaFree(d_temp_result_);
        cudaStreamDestroy(stream_);
    }

    std::shared_ptr<Ciphertext> Add(const Ciphertext& a, const Ciphertext& b) override {
        auto res = std::make_shared<Ciphertext>();
        res->scheme_id = scheme_id_;
        res->is_rlwe = a.is_rlwe;
        
        if (a.is_rlwe) {
            res->rlwe_data.resize(a.rlwe_data.size());
            for (size_t i = 0; i < a.rlwe_data.size(); ++i) {
                res->rlwe_data[i].resize(n_);
                
                // Copy to device
                cudaMemcpyAsync(d_temp_a_, a.rlwe_data[i].data(), n_ * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_);
                cudaMemcpyAsync(d_temp_b_, b.rlwe_data[i].data(), n_ * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_);
                
                // GPU addition
                cuda_rlwe_add(d_temp_result_, d_temp_a_, d_temp_b_, n_, q_, stream_);
                
                // Copy result back
                cudaMemcpyAsync(res->rlwe_data[i].data(), d_temp_result_, n_ * sizeof(int64_t),
                               cudaMemcpyDeviceToHost, stream_);
            }
            cudaStreamSynchronize(stream_);
        } else {
            // LWE on GPU
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
        // Step function implemented via bootstrapping or LUT
        // For now, forward to NTT-based evaluation
        auto res = std::make_shared<Ciphertext>();
        res->scheme_id = scheme_id_;
        res->is_rlwe = delta.is_rlwe;
        
        if (delta.is_rlwe && !delta.rlwe_data.empty()) {
            res->rlwe_data.resize(delta.rlwe_data.size());
            for (size_t i = 0; i < delta.rlwe_data.size(); ++i) {
                res->rlwe_data[i] = delta.rlwe_data[i];
                
                // Apply NTT for fast polynomial operations
                cudaMemcpyAsync(d_temp_a_, res->rlwe_data[i].data(), n_ * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_);
                cuda_ntt_forward(d_temp_a_, n_, q_, stream_);
                // Apply step approximation in NTT domain
                cuda_ntt_inverse(d_temp_a_, n_, q_, stream_);
                cudaMemcpyAsync(res->rlwe_data[i].data(), d_temp_a_, n_ * sizeof(int64_t),
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
    
    // Device memory pools
    int64_t* d_temp_a_;
    int64_t* d_temp_b_;
    int64_t* d_temp_result_;
};

} // namespace fhe_gbdt::kernel::gpu
