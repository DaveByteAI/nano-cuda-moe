// Phase-1 quantized kernels: correct first. One warp per output row; lane i takes groups i, i+32, ... of the row,
// dequantizes them with dequant8<T> and dots them with x. Speed work (several rows per warp, dp4a, multi-token)
// comes after the forward pass matches the references.
#include <stdexcept>
#include <string>

#include "bl/cuda_quant.h"
#include <cuda_bf16.h>

#include "dequant.cuh"

namespace bl::cuda {

namespace {

template <int T>
__global__ void dequant_rows_k(const uint8_t * __restrict__ W, size_t row_bytes, int M, int K, float * __restrict__ out) {
    const int r = blockIdx.x;   // one block per row: gridDim.y would cap the rows at 65535 (an expert tensor has 1.3M)
    const int n_groups = K / 8;
    const uint8_t * row = W + static_cast<size_t>(r) * row_bytes;
    for (int gg = threadIdx.x; gg < n_groups; gg += blockDim.x) {
        const int blk = gg / Grp<T>::groups, g = gg % Grp<T>::groups;
        float v[8];
        dequant8<T>(row + static_cast<size_t>(blk) * Grp<T>::bytes, g, v);
        float * o = out + static_cast<size_t>(r) * K + 8 * gg;
        for (int j = 0; j < 8; ++j) o[j] = v[j];
    }
}

template <int T>
__global__ void dequant_rows_bf16_k(const uint8_t * __restrict__ W, size_t row_bytes, int K, __nv_bfloat16 * __restrict__ out) {
    const int r = blockIdx.x;
    const int n_groups = K / 8;
    const uint8_t * row = W + static_cast<size_t>(r) * row_bytes;
    for (int gg = threadIdx.x; gg < n_groups; gg += blockDim.x) {
        const int blk = gg / Grp<T>::groups, g = gg % Grp<T>::groups;
        float v[8];
        dequant8<T>(row + static_cast<size_t>(blk) * Grp<T>::bytes, g, v);
        __nv_bfloat162 o2[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) o2[j] = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]);
        *reinterpret_cast<uint4 *>(out + static_cast<size_t>(r) * K + 8 * gg) = *reinterpret_cast<const uint4 *>(o2);
    }
}

constexpr int kRowsPerBlock = 4;   // warps per thread block

template <int T>
__global__ void matvec_k(const uint8_t * __restrict__ W, size_t row_bytes, int M, int K, const float * __restrict__ x,
                         float * __restrict__ y) {
    // block diagonal (matvec_blocks): group blockIdx.y has its own M rows, K inputs and M outputs
    W += static_cast<size_t>(blockIdx.y) * M * row_bytes;
    x += static_cast<size_t>(blockIdx.y) * K;
    y += static_cast<size_t>(blockIdx.y) * M;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * kRowsPerBlock + (threadIdx.x >> 5);
    if (r >= M) return;
    const uint8_t * row = W + static_cast<size_t>(r) * row_bytes;
    const int n_groups = K / 8;
    float acc = 0.f;
#pragma unroll 4
    for (int gg = lane; gg < n_groups; gg += 32) {
        const int blk = gg / Grp<T>::groups, g = gg % Grp<T>::groups;
        float v[8];
        dequant8<T>(row + static_cast<size_t>(blk) * Grp<T>::bytes, g, v);
        const float4 a = reinterpret_cast<const float4 *>(x)[2 * gg];
        const float4 b = reinterpret_cast<const float4 *>(x)[2 * gg + 1];
        acc += v[0] * a.x + v[1] * a.y + v[2] * a.z + v[3] * a.w + v[4] * b.x + v[5] * b.y + v[6] * b.z + v[7] * b.w;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[r] = acc;
}

template <template <int> class F, typename... A>
void dispatch(uint32_t t, A... a) {
    switch (t) {
        case F32:     return F<F32>::run(a...);
        case F16:     return F<F16>::run(a...);
        case BF16:    return F<BF16>::run(a...);
        case Q8_0:    return F<Q8_0>::run(a...);
        case Q4_K:    return F<Q4_K>::run(a...);
        case Q5_K:    return F<Q5_K>::run(a...);
        case Q6_K:    return F<Q6_K>::run(a...);
        case IQ2_XXS: return F<IQ2_XXS>::run(a...);
        case IQ2_XS:  return F<IQ2_XS>::run(a...);
        case IQ2_S:   return F<IQ2_S>::run(a...);
        case IQ3_XXS: return F<IQ3_XXS>::run(a...);
        case IQ3_S:   return F<IQ3_S>::run(a...);
        case IQ4_NL:  return F<IQ4_NL>::run(a...);
        case IQ4_XS:  return F<IQ4_XS>::run(a...);
        case Q2_0:    return F<Q2_0>::run(a...);
        default: throw std::runtime_error("bl::cuda: unsupported weight type " + std::to_string(t));
    }
}

void check_k(uint32_t t, int K) {
    int elems = 0;
    switch (t) {
        case F32: case F16: case BF16: elems = 8; break;
        case Q8_0: case IQ4_NL: elems = 32; break;
        case Q2_0: elems = 64; break;
        default: elems = 256;
    }
    if (K % elems) throw std::runtime_error("bl::cuda: row length " + std::to_string(K) + " is not whole blocks");
}

template <int T> struct DequantRows {
    static void run(const void * W, size_t rb, int M, int K, float * out, cudaStream_t s) {
        dequant_rows_k<T><<<M, 128, 0, s>>>(static_cast<const uint8_t *>(W), rb, M, K, out);
    }
};
template <int T> struct DequantRowsBf16 {
    static void run(const void * W, size_t rb, int M, int K, void * out, cudaStream_t s) {
        dequant_rows_bf16_k<T><<<M, 128, 0, s>>>(static_cast<const uint8_t *>(W), rb, K, static_cast<__nv_bfloat16 *>(out));
    }
};

template <int T> struct Matvec {
    static void run(const void * W, size_t rb, int M, int K, const float * x, float * y, cudaStream_t s, int G) {
        matvec_k<T><<<dim3((M + kRowsPerBlock - 1) / kRowsPerBlock, G), 32 * kRowsPerBlock, 0, s>>>(
            static_cast<const uint8_t *>(W), rb, M, K, x, y);
    }
};

}  // namespace

bool supported(uint32_t t) {
    switch (t) {
        case F32: case F16: case BF16: case Q8_0: case Q4_K: case Q5_K: case Q6_K: case IQ2_XXS: case IQ2_XS:
        case IQ2_S: case IQ3_XXS: case IQ3_S: case IQ4_NL: case IQ4_XS: case Q2_0: return true;
        default: return false;
    }
}

bool expert_supported(uint32_t t) {
    switch (t) {
        case IQ2_XXS: case IQ2_XS: case IQ2_S: case IQ3_XXS: case IQ3_S: case IQ4_NL: case IQ4_XS: case Q2_0: return true;
        default: return false;
    }
}

void dequant_rows(uint32_t t, const void * W, size_t rb, int M, int K, float * out, cudaStream_t s) {
    check_k(t, K);
    dispatch<DequantRows>(t, W, rb, M, K, out, s);
}

void dequant_rows_bf16(uint32_t t, const void * W, size_t rb, int M, int K, void * out, cudaStream_t s) {
    check_k(t, K);
    dispatch<DequantRowsBf16>(t, W, rb, M, K, out, s);
}

void matvec(uint32_t t, const void * W, size_t rb, int M, int K, const float * x, float * y, cudaStream_t s) {
    check_k(t, K);
    dispatch<Matvec>(t, W, rb, M, K, x, y, s, 1);
}

void matvec_blocks(uint32_t t, const void * W, size_t rb, int M, int K, int G, const float * x, float * y, cudaStream_t s) {
    check_k(t, K);
    dispatch<Matvec>(t, W, rb, M, K, x, y, s, G);
}

}  // namespace bl::cuda
