#include <cuda_runtime.h>
#include <cstdint>

// Block and grid configuration
constexpr int BLOCK_SIZE = 256;

// Modular addition kernel
__global__ void rlwe_add_kernel(int64_t* result, const int64_t* a, const int64_t* b, 
                                 int n, int64_t q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t sum = a[idx] + b[idx];
        result[idx] = sum >= q ? sum - q : sum;
    }
}

// Modular subtraction kernel
__global__ void rlwe_sub_kernel(int64_t* result, const int64_t* a, const int64_t* b,
                                 int n, int64_t q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t diff = a[idx] - b[idx];
        result[idx] = diff < 0 ? diff + q : diff;
    }
}

// Scalar multiplication kernel
__global__ void rlwe_mul_plain_kernel(int64_t* result, const int64_t* a, int64_t scalar,
                                       int n, int64_t q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use 128-bit arithmetic for precision
        __int128 prod = (__int128)a[idx] * scalar;
        result[idx] = (int64_t)(prod % q);
    }
}

// NTT butterfly operation
__device__ void butterfly(int64_t& a, int64_t& b, int64_t w, int64_t q) {
    int64_t t = (__int128)b * w % q;
    b = a >= t ? a - t : a - t + q;
    a = a + t >= q ? a + t - q : a + t;
}

// Forward NTT kernel (Cooley-Tukey)
__global__ void ntt_forward_kernel(int64_t* data, int n, int64_t q, 
                                    const int64_t* twiddles, int stage) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_len = 1 << stage;
    int len = half_len << 1;
    int group = idx / half_len;
    int pos = idx % half_len;
    
    if (group * len + pos + half_len < n) {
        int i = group * len + pos;
        int j = i + half_len;
        int64_t w = twiddles[half_len + pos];
        butterfly(data[i], data[j], w, q);
    }
}

// Inverse NTT kernel (Gentleman-Sande)
__global__ void ntt_inverse_kernel(int64_t* data, int n, int64_t q,
                                    const int64_t* inv_twiddles, int stage) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_len = 1 << stage;
    int len = half_len << 1;
    int group = idx / half_len;
    int pos = idx % half_len;
    
    if (group * len + pos + half_len < n) {
        int i = group * len + pos;
        int j = i + half_len;
        int64_t w = inv_twiddles[half_len + pos];
        
        // Inverse butterfly
        int64_t t = data[i] >= data[j] ? data[i] - data[j] : data[i] - data[j] + q;
        data[i] = data[i] + data[j] >= q ? data[i] + data[j] - q : data[i] + data[j];
        data[j] = (__int128)t * w % q;
    }
}

// C interface functions for the GpuBackend class
extern "C" {

void cuda_rlwe_add(int64_t* result, const int64_t* a, const int64_t* b,
                   int n, int64_t q, cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rlwe_add_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(result, a, b, n, q);
}

void cuda_rlwe_sub(int64_t* result, const int64_t* a, const int64_t* b,
                   int n, int64_t q, cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rlwe_sub_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(result, a, b, n, q);
}

void cuda_rlwe_mul_plain(int64_t* result, const int64_t* a, int64_t scalar,
                          int n, int64_t q, cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rlwe_mul_plain_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(result, a, scalar, n, q);
}

void cuda_ntt_forward(int64_t* data, int n, int64_t q, cudaStream_t stream) {
    // Placeholder: Full NTT requires pre-computed twiddle factors
    // In production, twiddles would be generated at initialization
    // For now, this is a no-op placeholder
}

void cuda_ntt_inverse(int64_t* data, int n, int64_t q, cudaStream_t stream) {
    // Placeholder: Full inverse NTT 
    // Similar to forward, requires pre-computed inverse twiddles
}

} // extern "C"
