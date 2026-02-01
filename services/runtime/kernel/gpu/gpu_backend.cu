#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// Block and grid configuration
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_NTT_SIZE = 4096;

// ============================================================================
// NTT Context - stores precomputed twiddle factors on device
// ============================================================================

struct NTTContext {
    int64_t* d_twiddles;        // Forward twiddle factors
    int64_t* d_inv_twiddles;    // Inverse twiddle factors
    int n;                       // Ring dimension
    int64_t q;                   // Modulus
    int64_t n_inv;              // Modular inverse of n
    bool initialized;
};

// Global NTT context (initialized once per device)
__device__ __constant__ int64_t d_const_q;
__device__ __constant__ int64_t d_const_n_inv;

static NTTContext g_ntt_ctx = {nullptr, nullptr, 0, 0, 0, false};

// ============================================================================
// Modular Arithmetic Utilities
// ============================================================================

__device__ __forceinline__ int64_t mod_add(int64_t a, int64_t b, int64_t q) {
    int64_t sum = a + b;
    return sum >= q ? sum - q : sum;
}

__device__ __forceinline__ int64_t mod_sub(int64_t a, int64_t b, int64_t q) {
    int64_t diff = a - b;
    return diff < 0 ? diff + q : diff;
}

__device__ __forceinline__ int64_t mod_mul(int64_t a, int64_t b, int64_t q) {
    // Use 128-bit arithmetic for precision
    __int128 prod = (__int128)a * b;
    return (int64_t)(prod % q);
}

// Extended Euclidean Algorithm for modular inverse
__host__ int64_t mod_inverse(int64_t a, int64_t m) {
    int64_t m0 = m, x0 = 0, x1 = 1;
    if (m == 1) return 0;
    while (a > 1) {
        int64_t q = a / m;
        int64_t t = m;
        m = a % m;
        a = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }
    return x1 < 0 ? x1 + m0 : x1;
}

// Find primitive n-th root of unity mod q
__host__ int64_t find_primitive_root(int n, int64_t q) {
    // For N2HE default q = 2^32, we need to find g such that g^n = 1 mod q
    // and g^k != 1 for k < n

    // For power-of-2 moduli, we can use standard root finding
    // g = 3^((q-1)/n) mod q is often a primitive n-th root

    int64_t exp = (q - 1) / n;
    int64_t g = 3;

    // Compute g^exp mod q using fast exponentiation
    int64_t result = 1;
    int64_t base = g;
    while (exp > 0) {
        if (exp & 1) {
            result = mod_mul(result, base, q);
        }
        base = mod_mul(base, base, q);
        exp >>= 1;
    }

    return result;
}

// ============================================================================
// NTT Initialization
// ============================================================================

extern "C" {

void cuda_ntt_init(int n, int64_t q) {
    if (g_ntt_ctx.initialized && g_ntt_ctx.n == n && g_ntt_ctx.q == q) {
        return; // Already initialized with same parameters
    }

    // Clean up previous context
    if (g_ntt_ctx.d_twiddles) cudaFree(g_ntt_ctx.d_twiddles);
    if (g_ntt_ctx.d_inv_twiddles) cudaFree(g_ntt_ctx.d_inv_twiddles);

    // Allocate device memory for twiddle factors
    cudaMalloc(&g_ntt_ctx.d_twiddles, n * sizeof(int64_t));
    cudaMalloc(&g_ntt_ctx.d_inv_twiddles, n * sizeof(int64_t));

    // Compute twiddle factors on host
    int64_t* h_twiddles = new int64_t[n];
    int64_t* h_inv_twiddles = new int64_t[n];

    // Find primitive 2n-th root of unity (for negacyclic NTT)
    int64_t psi = find_primitive_root(2 * n, q);
    int64_t psi_inv = mod_inverse(psi, q);

    // Compute powers of psi for forward NTT
    h_twiddles[0] = 1;
    for (int i = 1; i < n; i++) {
        h_twiddles[i] = mod_mul(h_twiddles[i-1], psi, q);
    }

    // Compute powers of psi^-1 for inverse NTT
    h_inv_twiddles[0] = 1;
    for (int i = 1; i < n; i++) {
        h_inv_twiddles[i] = mod_mul(h_inv_twiddles[i-1], psi_inv, q);
    }

    // Copy to device
    cudaMemcpy(g_ntt_ctx.d_twiddles, h_twiddles, n * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(g_ntt_ctx.d_inv_twiddles, h_inv_twiddles, n * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Store parameters
    g_ntt_ctx.n = n;
    g_ntt_ctx.q = q;
    g_ntt_ctx.n_inv = mod_inverse(n, q);
    g_ntt_ctx.initialized = true;

    // Copy constants to device
    cudaMemcpyToSymbol(d_const_q, &q, sizeof(int64_t));
    cudaMemcpyToSymbol(d_const_n_inv, &g_ntt_ctx.n_inv, sizeof(int64_t));

    delete[] h_twiddles;
    delete[] h_inv_twiddles;
}

void cuda_ntt_cleanup() {
    if (g_ntt_ctx.d_twiddles) cudaFree(g_ntt_ctx.d_twiddles);
    if (g_ntt_ctx.d_inv_twiddles) cudaFree(g_ntt_ctx.d_inv_twiddles);
    g_ntt_ctx.initialized = false;
}

} // extern "C"

// ============================================================================
// CUDA Kernels
// ============================================================================

// Modular addition kernel
__global__ void rlwe_add_kernel(int64_t* result, const int64_t* a, const int64_t* b,
                                 int n, int64_t q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = mod_add(a[idx], b[idx], q);
    }
}

// Modular subtraction kernel
__global__ void rlwe_sub_kernel(int64_t* result, const int64_t* a, const int64_t* b,
                                 int n, int64_t q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = mod_sub(a[idx], b[idx], q);
    }
}

// Scalar multiplication kernel
__global__ void rlwe_mul_plain_kernel(int64_t* result, const int64_t* a, int64_t scalar,
                                       int n, int64_t q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = mod_mul(a[idx], scalar, q);
    }
}

// Batched modular addition kernel
__global__ void rlwe_add_batch_kernel(int64_t* result, const int64_t* a, const int64_t* b,
                                       int n, int batch_size, int64_t q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * batch_size;
    if (idx < total) {
        result[idx] = mod_add(a[idx], b[idx], q);
    }
}

// NTT butterfly operation (Cooley-Tukey, decimation-in-time)
__device__ void ntt_butterfly(int64_t& a, int64_t& b, int64_t w, int64_t q) {
    int64_t t = mod_mul(b, w, q);
    b = mod_sub(a, t, q);
    a = mod_add(a, t, q);
}

// Inverse NTT butterfly operation (Gentleman-Sande, decimation-in-frequency)
__device__ void intt_butterfly(int64_t& a, int64_t& b, int64_t w, int64_t q) {
    int64_t t = mod_sub(a, b, q);
    a = mod_add(a, b, q);
    b = mod_mul(t, w, q);
}

// Forward NTT kernel (Cooley-Tukey radix-2)
__global__ void ntt_forward_kernel(int64_t* data, int n, int64_t q,
                                    const int64_t* twiddles, int stage) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_len = 1 << stage;
    int len = half_len << 1;
    int num_groups = n / len;

    if (idx < n / 2) {
        int group = idx / half_len;
        int pos = idx % half_len;

        int i = group * len + pos;
        int j = i + half_len;

        // Bit-reversed twiddle factor index
        int twiddle_idx = (n / len) * pos;
        int64_t w = twiddles[twiddle_idx];

        ntt_butterfly(data[i], data[j], w, q);
    }
}

// Inverse NTT kernel (Gentleman-Sande radix-2)
__global__ void ntt_inverse_kernel(int64_t* data, int n, int64_t q,
                                    const int64_t* inv_twiddles, int stage) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_len = 1 << (int)(log2f(n) - 1 - stage);
    int len = half_len << 1;

    if (idx < n / 2) {
        int group = idx / half_len;
        int pos = idx % half_len;

        int i = group * len + pos;
        int j = i + half_len;

        int twiddle_idx = (n / len) * pos;
        int64_t w = inv_twiddles[twiddle_idx];

        intt_butterfly(data[i], data[j], w, q);
    }
}

// Final scaling for inverse NTT
__global__ void ntt_scale_kernel(int64_t* data, int n, int64_t q, int64_t n_inv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = mod_mul(data[idx], n_inv, q);
    }
}

// Step function kernel (sign extraction approximation)
__global__ void step_kernel(int64_t* result, const int64_t* input, int n, int64_t q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Sign extraction: if input < q/2, it's "positive" -> 1, else 0
        // This is a simplified step function for demonstration
        // Real implementation would use LUT or bootstrapping
        int64_t val = input[idx];
        int64_t threshold = q / 2;
        result[idx] = (val < threshold) ? (q / 4) : 0;  // Encoded 1 or 0
    }
}

// ============================================================================
// C Interface Functions
// ============================================================================

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

void cuda_rlwe_add_batch(int64_t* result, const int64_t* a, const int64_t* b,
                          int n, int batch_size, int64_t q, cudaStream_t stream) {
    int total = n * batch_size;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rlwe_add_batch_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(result, a, b, n, batch_size, q);
}

void cuda_ntt_forward(int64_t* data, int n, int64_t q, cudaStream_t stream) {
    // Ensure NTT is initialized
    if (!g_ntt_ctx.initialized || g_ntt_ctx.n != n) {
        cuda_ntt_init(n, q);
    }

    int log_n = (int)log2f(n);
    int blocks = (n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Execute NTT stages
    for (int stage = 0; stage < log_n; stage++) {
        ntt_forward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            data, n, q, g_ntt_ctx.d_twiddles, stage);
        cudaStreamSynchronize(stream);  // Barrier between stages
    }
}

void cuda_ntt_inverse(int64_t* data, int n, int64_t q, cudaStream_t stream) {
    // Ensure NTT is initialized
    if (!g_ntt_ctx.initialized || g_ntt_ctx.n != n) {
        cuda_ntt_init(n, q);
    }

    int log_n = (int)log2f(n);
    int blocks = (n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Execute inverse NTT stages
    for (int stage = 0; stage < log_n; stage++) {
        ntt_inverse_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            data, n, q, g_ntt_ctx.d_inv_twiddles, stage);
        cudaStreamSynchronize(stream);  // Barrier between stages
    }

    // Final scaling by n^-1
    int scale_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ntt_scale_kernel<<<scale_blocks, BLOCK_SIZE, 0, stream>>>(
        data, n, q, g_ntt_ctx.n_inv);
}

void cuda_step(int64_t* result, const int64_t* input, int n, int64_t q, cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    step_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(result, input, n, q);
}

} // extern "C"
