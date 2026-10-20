#include <cfloat>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "bl/cuda_prefill.h"
#include "bl/cuda_quant.h"
#include "dequant.cuh"
#include "decode32.cuh"
#include "kv.cuh"
#include "dp4a.cuh"

namespace bl::cuda {

namespace {

__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }
// lim > 0: DeepSeek-V4's clamps (gate at most lim, up within +-lim) before the SwiGLU
__device__ __forceinline__ float swiglu_lim(float g, float u, float lim) {
    if (lim > 0.f) { g = fminf(g, lim); u = fminf(fmaxf(u, -lim), lim); }
    return silu(g) * u;
}

cublasHandle_t handle() {
    static cublasHandle_t h = [] {
        cublasHandle_t x;
        if (cublasCreate(&x) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        cublasSetMathMode(x, CUBLAS_TF32_TENSOR_OP_MATH);
        return x;
    }();
    return h;
}

__global__ void silu_scale_k(const float * x, float * y, size_t n, float scale) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) y[i] = silu(x[i] * scale);
}

__global__ void hc_apply_k(const float * g, const float * xn, float * mixed, int D, int hc, int N) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<size_t>(N) * D) return;
    const size_t t = i / D;
    const int d = static_cast<int>(i % D);
    float m = 0.f;
    for (int c = 0; c < hc; ++c) {
        const size_t k = t * hc * D + static_cast<size_t>(c) * D + d;
        m += xn[k] * sigm(g[k]);
    }
    mixed[i] = m * (1.f / hc);
}

// one warp per query row (token t, head h); a block's 8 warps take consecutive rows of one kv head, so they read the
// same keys (L1). Lane l holds dims 8l .. 8l+7 (D = 256). Online softmax over the row's cells (sel: a list per token,
// or the last `window` cells up to pos0 + t).
template <bool Q8>
__global__ void attention_prefill_k(const float * __restrict__ q, KvCache kvc, const float * gate, int gsh, int gst, float * out,
                                    int heads, int kv_heads, int pos0, int N,
                                    float scale, AttnSel sel) {
    constexpr int D = 256;
    const int g = heads / kv_heads, kvh = blockIdx.y;
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= N * g) return;
    const int t = r / g, h = kvh * g + r % g, n_kv = pos0 + t + 1;
    const int * lst = nullptr;
    int lo = 0, n = n_kv;
    if (sel.list) { lst = sel.list + static_cast<size_t>(t) * sel.stride; n = sel.cnt[t]; }
    else if (sel.window > 0 && n_kv > sel.window) { lo = n_kv - sel.window; n = sel.window; }
    float qv[8];
    {
        const float4 * qp = reinterpret_cast<const float4 *>(q + (static_cast<size_t>(t) * heads + h) * D) + 2 * lane;
        const float4 a = qp[0], b = qp[1];
        qv[0] = a.x; qv[1] = a.y; qv[2] = a.z; qv[3] = a.w; qv[4] = b.x; qv[5] = b.y; qv[6] = b.z; qv[7] = b.w;
    }
    float m = -FLT_MAX, l = 0.f, acc[8] = {};
    for (int j0 = 0; j0 < n; j0 += 32) {
        const int my = j0 + lane < n ? (lst ? lst[j0 + lane] : lo + j0 + lane) : 0;
        const int np = min(32, n - j0);
        for (int j = 0; j < np; ++j) {
            const int p = __shfl_sync(0xffffffffu, my, j);
            const size_t row = kv_row(kvc, p, kv_heads, kvh);
            float kv[8], vv[8];
            kv_ld8<Q8>(kvc, 0, row, lane, kv);
            float d = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) d += qv[i] * kv[i];
            for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
            d *= scale;
            kv_ld8<Q8>(kvc, 1, row, lane, vv);
            if (d > m) {
                const float c = expf(m - d);
                l = l * c + 1.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[i] = acc[i] * c + vv[i];
                m = d;
            } else {
                const float e = expf(d - m);
                l += e;
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[i] += e * vv[i];
            }
        }
    }
    const float inv = 1.f / l;
    const float * gt = gate + static_cast<size_t>(t) * gst + h * gsh + 8 * lane;
    float * o = out + (static_cast<size_t>(t) * heads + h) * D + 8 * lane;
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = acc[j] * inv * sigm(gt[j]);
}

// ---- tensor-core attention: one block per (token, kv head). The token's cells are the same for all its heads (the
// indexer selects per token), so the g <= 16 q heads of a kv head are the 16 rows of mma.sync m16n8k16 (fp16 in,
// fp32 accumulate) and each tile of kPaC cells is gathered from the cache once for all of them. Per tile:
// S = Q K^T, online softmax (running max m, sum l per row), O = O * alpha + P V. Shared rows padded by 8 halves.
constexpr int kPaC = 32, kPaQS = 256 + 8, kPaPS = kPaC + 8;
__device__ __forceinline__ void mma_f16(float * c, const uint32_t * a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t ld_h2(const __half * p) { return *reinterpret_cast<const uint32_t *>(p); }

template <bool Q8>
__global__ void __launch_bounds__(128) attention_prefill_mma_k(const float * __restrict__ q, KvCache kvc, const float * gate, int gsh,
                                                               int gst, float * out, int heads, int kv_heads, int pos0, float scale,
                                                               AttnSel sel) {
    constexpr int D = 256, C = kPaC, QS = kPaQS, PS = kPaPS;
    __shared__ __align__(16) __half Qs[16 * QS];
    __shared__ __align__(16) __half Ks[C * QS];
    __shared__ __align__(16) __half Vs[C * QS];
    __shared__ __align__(16) __half Ps[16 * PS];
    __shared__ float Ss[16 * C];
    __shared__ float rs[16];   // per row: alpha of the current tile, then 1 / l
    __shared__ int cell_s[C];
    const int t = blockIdx.x, kvh = blockIdx.y, g = heads / kv_heads;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, tq = lane & 3;
    const int n_kv = pos0 + t + 1;
    const int * lst = nullptr;
    int lo = 0, n = n_kv;
    if (sel.list) { lst = sel.list + static_cast<size_t>(t) * sel.stride; n = sel.cnt[t]; }
    else if (sel.window > 0 && n_kv > sel.window) { lo = n_kv - sel.window; n = sel.window; }

    for (int i = tid; i < 16 * D; i += 128) {   // the q rows of this kv head, scaled; rows past g are zero
        const int r = i / D, d = i % D;
        const float v = r < g ? q[(static_cast<size_t>(t) * heads + kvh * g + r) * D + d] * scale : 0.f;
        Qs[r * QS + d] = __float2half_rn(v);
    }
    float m_r = -FLT_MAX, l_r = 0.f;   // softmax state of row tid / 8 (the 8 threads of a row hold the same values)
    float acc[8][4] = {};              // O: rows gq, gq + 8; dims 64 * warp + 8 * nt + 2 * tq (+1)
    for (int j0 = 0; j0 < n; j0 += C) {
        const int nc = min(C, n - j0);
        __syncthreads();   // the last tile's K, V, P are used up
        if (tid < C) cell_s[tid] = tid < nc ? (lst ? lst[j0 + tid] : lo + j0 + tid) : -1;
        __syncthreads();
        for (int i = tid; i < C * 32; i += 128) {   // a warp per cell row: 32 lanes x 8 values, K and V to fp16
            const int c = i >> 5, l8 = i & 31, p = cell_s[c];
            float kf[8] = {}, vf[8] = {};
            if (p >= 0) {
                const size_t row = kv_row(kvc, p, kv_heads, kvh);
                kv_ld8<Q8>(kvc, 0, row, l8, kf);
                kv_ld8<Q8>(kvc, 1, row, l8, vf);
            }
            uint4 ku, vu;
            __half2 * kh = reinterpret_cast<__half2 *>(&ku), * vh = reinterpret_cast<__half2 *>(&vu);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                kh[e] = __floats2half2_rn(kf[2 * e], kf[2 * e + 1]);
                vh[e] = __floats2half2_rn(vf[2 * e], vf[2 * e + 1]);
            }
            *reinterpret_cast<uint4 *>(Ks + c * QS + 8 * l8) = ku;
            *reinterpret_cast<uint4 *>(Vs + c * QS + 8 * l8) = vu;
        }
        __syncthreads();
        {   // S = Q K^T: warp w takes cells 8w .. 8w + 7
            float c4[4] = {};
            const __half * kr = Ks + (8 * warp + gq) * QS + 2 * tq;
#pragma unroll 4
            for (int k0 = 0; k0 < D; k0 += 16) {
                const uint32_t a[4] = {ld_h2(Qs + gq * QS + k0 + 2 * tq), ld_h2(Qs + (gq + 8) * QS + k0 + 2 * tq),
                                       ld_h2(Qs + gq * QS + k0 + 8 + 2 * tq), ld_h2(Qs + (gq + 8) * QS + k0 + 8 + 2 * tq)};
                mma_f16(c4, a, ld_h2(kr + k0), ld_h2(kr + k0 + 8));
            }
            const int col = 8 * warp + 2 * tq;
            Ss[gq * C + col] = c4[0];
            Ss[gq * C + col + 1] = c4[1];
            Ss[(gq + 8) * C + col] = c4[2];
            Ss[(gq + 8) * C + col + 1] = c4[3];
        }
        __syncthreads();
        {   // online softmax: thread -> row tid / 8, columns 4 * (tid % 8) .. + 3
            const int r = tid >> 3, c0 = 4 * (tid & 7);
            float sv[4], mx = -FLT_MAX;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                sv[e] = c0 + e < nc ? Ss[r * C + c0 + e] : -FLT_MAX;
                mx = fmaxf(mx, sv[e]);
            }
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mn = fmaxf(m_r, mx), al = expf(m_r - mn);
            float sum = 0.f;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float pe = c0 + e < nc ? expf(sv[e] - mn) : 0.f;
                Ps[r * PS + c0 + e] = __float2half_rn(pe);
                sum += pe;
            }
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            l_r = l_r * al + sum;
            m_r = mn;
            if ((tid & 7) == 0) rs[r] = al;
        }
        __syncthreads();
        {   // O = O * alpha + P V: warp w takes dims 64w .. 64w + 63; V's B fragments by ldmatrix.trans
            const float a0 = rs[gq], a1 = rs[gq + 8];
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) { acc[nt][0] *= a0; acc[nt][1] *= a0; acc[nt][2] *= a1; acc[nt][3] *= a1; }
#pragma unroll
            for (int k0 = 0; k0 < C; k0 += 16) {
                const uint32_t a[4] = {ld_h2(Ps + gq * PS + k0 + 2 * tq), ld_h2(Ps + (gq + 8) * PS + k0 + 2 * tq),
                                       ld_h2(Ps + gq * PS + k0 + 8 + 2 * tq), ld_h2(Ps + (gq + 8) * PS + k0 + 8 + 2 * tq)};
#pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    const uint32_t addr = static_cast<uint32_t>(
                        __cvta_generic_to_shared(Vs + (k0 + (lane & 15)) * QS + 64 * warp + 8 * nt));
                    uint32_t b0, b1;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(addr));
                    mma_f16(acc[nt], a, b0, b1);
                }
            }
        }
    }
    __syncthreads();
    if ((tid & 7) == 0) rs[tid >> 3] = 1.f / l_r;
    __syncthreads();
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int r = gq + 8 * half;
        if (r >= g) continue;
        const int h = kvh * g + r;
        const float inv = rs[r];
        const float * gt = gate + static_cast<size_t>(t) * gst + h * gsh;
        float * o = out + (static_cast<size_t>(t) * heads + h) * D;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int d = 64 * warp + 8 * nt + 2 * tq;
            o[d] = acc[nt][2 * half] * inv * sigm(gt[d]);
            o[d + 1] = acc[nt][2 * half + 1] * inv * sigm(gt[d + 1]);
        }
    }
}

__global__ void moe_gather_k(const BlockQ8 * xq, const int * tok, int A, int nb, BlockQ8 * xg) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;   // one int (4 bytes) of a block
    constexpr int W = sizeof(BlockQ8) / 4;
    if (i >= static_cast<size_t>(A) * nb * W) return;
    const size_t a = i / (static_cast<size_t>(nb) * W), rest = i % (static_cast<size_t>(nb) * W);
    reinterpret_cast<int *>(xg)[i] = reinterpret_cast<const int *>(xq + static_cast<size_t>(tok[a]) * nb)[rest];
}

constexpr int kMoeWarps = 8, kMoeNT = 16;   // the dp4a kernels' token tile

// grid (F / (8 * 2), items): 16 lanes per row of gate and up (D = 2560: 80 sub-blocks, 5 per lane), 2 rows per
// warp, the item's tokens at once
constexpr int kGuL = 16;
template <int F>
__global__ void moe_gate_up_k(const MoeItem * items, size_t gu_bytes, int Fd, int D, const BlockQ8 * __restrict__ xg, float * act,
                              float lim) {
    const MoeItem it = items[blockIdx.y];
    const int lane = threadIdx.x & 31, sub = lane % kGuL;
    const int r = (blockIdx.x * kMoeWarps + (threadIdx.x >> 5)) * (32 / kGuL) + lane / kGuL;
    if (r >= Fd) return;   // whole warps (Fd is even)
    const size_t rb = gu_bytes / Fd;
    const uint8_t * gate = it.base + static_cast<size_t>(r) * rb, * up = gate + gu_bytes;
    const int nb = D / 32;
    const BlockQ8 * x = xg + static_cast<size_t>(it.a0) * nb;
    float g[kMoeNT] = {}, u[kMoeNT] = {};
    for (int sb = sub; sb < nb; sb += kGuL) {
        dot32m<F, kMoeNT>(gate, sb, x, nb, it.n, g);
        dot32m<F, kMoeNT>(up, sb, x, nb, it.n, u);
    }
#pragma unroll
    for (int t = 0; t < kMoeNT; ++t) {
        if (t >= it.n) break;
        float gt = g[t], ut = u[t];
        for (int o = kGuL / 2; o > 0; o >>= 1) {
            gt += __shfl_xor_sync(0xffffffffu, gt, o);
            ut += __shfl_xor_sync(0xffffffffu, ut, o);
        }
        if (sub == 0) act[static_cast<size_t>(it.a0 + t) * Fd + r] = swiglu_lim(gt, ut, lim);
    }
}

// grid (D / 64, items): 4 lanes per row (F = 640: 20 sub-blocks, 5 per lane), 8 rows per warp
constexpr int kDLanes = 4, kDRows = kMoeWarps * (32 / kDLanes);
template <int F>
__global__ void moe_down_k(const MoeItem * items, size_t gu_bytes, size_t d_rb, int Fd, int D, const BlockQ8 * __restrict__ aq,
                           const int * tok, const float * w, float * h) {
    const MoeItem it = items[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * kDRows + warp * (32 / kDLanes) + lane / kDLanes;
    if (r >= D) return;
    const int nb = Fd / 32;
    const uint8_t * row = it.base + 2 * gu_bytes + static_cast<size_t>(r) * d_rb;
    const BlockQ8 * x = aq + static_cast<size_t>(it.a0) * nb;
    float v[kMoeNT] = {};
    for (int sb = lane % kDLanes; sb < nb; sb += kDLanes) dot32m<F, kMoeNT>(row, sb, x, nb, it.n, v);
#pragma unroll
    for (int t = 0; t < kMoeNT; ++t) {
        if (t >= it.n) break;
        float a = v[t];
        for (int o = kDLanes / 2; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane % kDLanes == 0) {
            const int ai = it.a0 + t;
            atomicAdd(h + static_cast<size_t>(tok[ai]) * D + r, w[ai] * a);
        }
    }
}

// ---- tensor cores: int8 mma.sync m16n8k16 over 64 rows x 64 tokens of one expert. Each 32-value sub-block is decoded
// once per block (int8 + the scale of each 16-value half) into shared memory; the activations are q8_1 already. The
// int32 product of a half is scaled (weight half-scale x activation scale) into float accumulators.
constexpr int kMmaRows = 64, kMmaTok = 64, kMmaPad = 9;   // ints per smem row: 8 + 1 (no bank conflicts)
static_assert(kMoeTile <= kMmaTok, "a work item must fit the tensor-core token tile");

__device__ __forceinline__ void mma16816(int (&c)[4], int a0, int a1, int b0) {
    c[0] = c[1] = c[2] = c[3] = 0;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}

// the item's activations for sub-block sb into shared memory: B [token][8 ints], xd [token]
__device__ __forceinline__ void load_acts(const BlockQ8 * x, int n, int nb, int sb, int (*B)[kMmaPad], float * xd) {
    for (int i = threadIdx.x; i < kMmaTok * 8; i += blockDim.x) {
        const int t = i >> 3, w = i & 7;
        B[t][w] = t < n ? ld_b4(x[static_cast<size_t>(t) * nb + sb].qs, w) : 0;
        if (w == 0) xd[t] = t < n ? __low2float(x[static_cast<size_t>(t) * nb + sb].ds) : 0.f;
    }
}

// grid (Fd / 64, items), 128 threads: gate and up of 64 rows for the item's tokens; act = silu(gate) * up
template <int F>
__global__ void __launch_bounds__(128) moe_gu_mma_k(const MoeItem * items, size_t gu_bytes, int Fd, int D, const BlockQ8 * __restrict__ xg,
                                                    float * act, float lim) {
    const MoeItem it = items[blockIdx.y];
    const int r0 = blockIdx.x * kMmaRows, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const size_t rb = gu_bytes / Fd;
    const int nb = D / 32, ntiles = (it.n + 7) / 8;
    const BlockQ8 * x = xg + static_cast<size_t>(it.a0) * nb;
    __shared__ int A[2][kMmaRows][kMmaPad];
    __shared__ float S[2][kMmaRows][2];
    __shared__ int B[kMmaTok][kMmaPad];
    __shared__ float xd[kMmaTok];
    float acc[2][kMmaTok / 8][4] = {};
    const int mat = threadIdx.x / kMmaRows, row = threadIdx.x % kMmaRows;   // this thread's decode: (gate | up, row)
    const uint8_t * wrow = it.base + mat * gu_bytes + static_cast<size_t>(r0 + row) * rb;
    const int ra = warp * 16 + g, rc = ra + 8;   // the rows of this thread's accumulator fragments
    for (int sb = 0; sb < nb; ++sb) {
        int gv[8];
        float s0, s1;
        decode32<F>(wrow, sb, gv, s0, s1);
#pragma unroll
        for (int j = 0; j < 8; ++j) A[mat][row][j] = gv[j];
        S[mat][row][0] = s0;
        S[mat][row][1] = s1;
        load_acts(x, it.n, nb, sb, B, xd);
        __syncthreads();
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int a0 = A[m][ra][4 * h + q], a1 = A[m][rc][4 * h + q];
                const float sa = S[m][ra][h], sc = S[m][rc][h];
#pragma unroll
                for (int nt = 0; nt < kMmaTok / 8; ++nt) {
                    if (nt >= ntiles) break;
                    int c[4];
                    mma16816(c, a0, a1, B[nt * 8 + g][4 * h + q]);
                    const float x0 = xd[nt * 8 + 2 * q], x1 = xd[nt * 8 + 2 * q + 1];
                    acc[m][nt][0] += static_cast<float>(c[0]) * sa * x0;
                    acc[m][nt][1] += static_cast<float>(c[1]) * sa * x1;
                    acc[m][nt][2] += static_cast<float>(c[2]) * sc * x0;
                    acc[m][nt][3] += static_cast<float>(c[3]) * sc * x1;
                }
            }
        __syncthreads();
    }
#pragma unroll
    for (int nt = 0; nt < kMmaTok / 8; ++nt) {
        if (nt >= ntiles) break;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = nt * 8 + 2 * q + (e & 1), r = r0 + (e < 2 ? ra : rc);
            if (t < it.n) act[static_cast<size_t>(it.a0 + t) * Fd + r] = swiglu_lim(acc[0][nt][e], acc[1][nt][e], lim);
        }
    }
}

// grid (D / 64, items), 128 threads: down of 64 rows for the item's tokens, h[tok] += w * value (atomics)
template <int F>
__global__ void __launch_bounds__(128) moe_down_mma_k(const MoeItem * items, size_t gu_bytes, size_t d_rb, int Fd, int D,
                                                      const BlockQ8 * __restrict__ aq, const int * tok, const float * w, float * h) {
    const MoeItem it = items[blockIdx.y];
    const int r0 = blockIdx.x * kMmaRows, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int nb = Fd / 32, ntiles = (it.n + 7) / 8;
    const BlockQ8 * x = aq + static_cast<size_t>(it.a0) * nb;
    __shared__ int A[kMmaRows][kMmaPad];
    __shared__ float S[kMmaRows][2];
    __shared__ int B[kMmaTok][kMmaPad];
    __shared__ float xd[kMmaTok];
    float acc[kMmaTok / 8][4] = {};
    const int ra = warp * 16 + g, rc = ra + 8;
    for (int sb = 0; sb < nb; ++sb) {
        if (threadIdx.x < kMmaRows) {
            int gv[8];
            float s0, s1;
            decode32<F>(it.base + 2 * gu_bytes + static_cast<size_t>(r0 + threadIdx.x) * d_rb, sb, gv, s0, s1);
#pragma unroll
            for (int j = 0; j < 8; ++j) A[threadIdx.x][j] = gv[j];
            S[threadIdx.x][0] = s0;
            S[threadIdx.x][1] = s1;
        }
        load_acts(x, it.n, nb, sb, B, xd);
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int a0 = A[ra][4 * hh + q], a1 = A[rc][4 * hh + q];
            const float sa = S[ra][hh], sc = S[rc][hh];
#pragma unroll
            for (int nt = 0; nt < kMmaTok / 8; ++nt) {
                if (nt >= ntiles) break;
                int c[4];
                mma16816(c, a0, a1, B[nt * 8 + g][4 * hh + q]);
                const float x0 = xd[nt * 8 + 2 * q], x1 = xd[nt * 8 + 2 * q + 1];
                acc[nt][0] += static_cast<float>(c[0]) * sa * x0;
                acc[nt][1] += static_cast<float>(c[1]) * sa * x1;
                acc[nt][2] += static_cast<float>(c[2]) * sc * x0;
                acc[nt][3] += static_cast<float>(c[3]) * sc * x1;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int nt = 0; nt < kMmaTok / 8; ++nt) {
        if (nt >= ntiles) break;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = nt * 8 + 2 * q + (e & 1), r = r0 + (e < 2 ? ra : rc);
            if (t < it.n) {
                const int ai = it.a0 + t;
                atomicAdd(h + static_cast<size_t>(tok[ai]) * D + r, w[ai] * acc[nt][e]);
            }
        }
    }
}

template <template <int> class Fn, typename... A>
void dispatch(uint32_t t, A... a) {
    switch (t) {
        case IQ2_XXS: return Fn<IQ2_XXS>::run(a...);
        case IQ2_XS:  return Fn<IQ2_XS>::run(a...);
        case IQ2_S:   return Fn<IQ2_S>::run(a...);
        case IQ3_XXS: return Fn<IQ3_XXS>::run(a...);
        case IQ3_S:   return Fn<IQ3_S>::run(a...);
        case IQ4_NL:  return Fn<IQ4_NL>::run(a...);
        case IQ4_XS:  return Fn<IQ4_XS>::run(a...);
        case Q2_0:    return Fn<Q2_0>::run(a...);
        default: throw std::runtime_error("prefill: expert format " + std::to_string(t) + " not supported");
    }
}
template <int F> struct GateUp {
    static void run(const MoeItem * items, int n, size_t gu, int Fd, int D, const void * xg, float * act, cudaStream_t s,
                    float lim) {
        static const bool dp4a = std::getenv("BL_MOE_DP4A") != nullptr;
        if (!dp4a && Fd % kMmaRows == 0) {
            moe_gu_mma_k<F><<<dim3(Fd / kMmaRows, n), 128, 0, s>>>(items, gu, Fd, D, static_cast<const BlockQ8 *>(xg), act, lim);
            return;
        }
        constexpr int rpb = kMoeWarps * (32 / kGuL);
        moe_gate_up_k<F><<<dim3((Fd + rpb - 1) / rpb, n), 32 * kMoeWarps, 0, s>>>(items, gu, Fd, D,
                                                                                            static_cast<const BlockQ8 *>(xg), act, lim);
    }
};
template <int F> struct Down {
    static void run(const MoeItem * items, int n, size_t gu, size_t d_rb, int Fd, int D, const void * aq, const int * tok,
                    const float * w, float * h, cudaStream_t s) {
        static const bool dp4a = std::getenv("BL_MOE_DP4A") != nullptr;
        if (!dp4a && D % kMmaRows == 0) {
            moe_down_mma_k<F><<<dim3(D / kMmaRows, n), 128, 0, s>>>(items, gu, d_rb, Fd, D, static_cast<const BlockQ8 *>(aq), tok, w, h);
            return;
        }
        moe_down_k<F><<<dim3((D + kDRows - 1) / kDRows, n), 32 * kMoeWarps, 0, s>>>(items, gu, d_rb, Fd, D,
                                                                                  static_cast<const BlockQ8 *>(aq), tok, w, h);
    }
};

__global__ void add_gated_k(float * h, const float * sh, const float * sg, int D, int N) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < static_cast<size_t>(N) * D) h[i] += sigm(sg[i / D]) * sh[i];
}
__global__ void swiglu_k(const float * g, const float * u, float * a, size_t n) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) a[i] = silu(g[i]) * u[i];
}
__global__ void copy_rows_k(const float * src, int lds, float * dst, int ldd, int D, int N) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < static_cast<size_t>(N) * D) dst[(i / D) * ldd + i % D] = src[(i / D) * lds + i % D];
}

__global__ void gather_rows_k(const uint32_t * src, size_t rw, const int * ids, uint32_t * dst) {
    const uint32_t * r = src + static_cast<size_t>(ids[blockIdx.x]) * rw;
    uint32_t * d = dst + static_cast<size_t>(blockIdx.x) * rw;
    for (size_t w = threadIdx.x; w < rw; w += blockDim.x) d[w] = r[w];
}

__global__ void copy_words_k(const int * src, int * dst, size_t n) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) dst[i] = reinterpret_cast<const volatile int *>(src)[i];
}

__global__ void to_bf16_k(const float * x, __nv_bfloat16 * y, size_t n) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}

unsigned blocks(size_t n) { return static_cast<unsigned>((n + 255) / 256); }

}  // namespace

void init_blas() { (void)handle(); }

void gemm_w(uint32_t type, const void * W, size_t rb, int M, int K, const float * x, int N, float * y, int ldy, float * scratch,
            size_t scratch_floats, cudaStream_t s) {
    cublasHandle_t hd = handle();
    cublasSetStream(hd, s);
    const float one = 1.f, zero = 0.f;
    if (type == 0) {   // F32: in place
        if (cublasSgemm(hd, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, static_cast<const float *>(W), K, x, K, &zero, y, ldy) !=
            CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasSgemm failed");
        return;
    }
    const int rows = static_cast<int>(std::min<size_t>(M, scratch_floats / K));
    if (rows < 1) throw std::runtime_error("gemm_w: scratch too small");
    for (int r0 = 0; r0 < M; r0 += rows) {
        const int m = std::min(rows, M - r0);
        dequant_rows(type, static_cast<const uint8_t *>(W) + static_cast<size_t>(r0) * rb, rb, m, K, scratch, s);
        // column-major: C (m x N, ld ldy) = Wd^T (m x K) . x (K x N)
        if (cublasSgemm(hd, CUBLAS_OP_T, CUBLAS_OP_N, m, N, K, &one, scratch, K, x, K, &zero, y + r0, ldy) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasSgemm failed");
    }
}

int moe_tile() { return std::getenv("BL_MOE_DP4A") ? 16 : kMoeTile; }

void gemm_bf16(const void * W16, int M, int K, const float * x, int N, float * y, int ldy, void * x16, cudaStream_t s) {
    cublasHandle_t hd = handle();
    cublasSetStream(hd, s);
    const size_t n = static_cast<size_t>(N) * K;
    to_bf16_k<<<blocks(n), 256, 0, s>>>(x, static_cast<__nv_bfloat16 *>(x16), n);
    const float one = 1.f, zero = 0.f;
    if (cublasGemmEx(hd, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, W16, CUDA_R_16BF, K, x16, CUDA_R_16BF, K, &zero, y, CUDA_R_32F, ldy,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasGemmEx (bf16) failed");
}

void gather_rows(const void * src, size_t row_bytes, const int * ids, int n, void * dst, cudaStream_t s) {
    if (row_bytes % 4) throw std::runtime_error("gather_rows: rows of " + std::to_string(row_bytes) + " bytes");
    if (n) gather_rows_k<<<n, 256, 0, s>>>(static_cast<const uint32_t *>(src), row_bytes / 4, ids, static_cast<uint32_t *>(dst));
}

void copy_words(const void * src, void * dst, size_t bytes, cudaStream_t s) {
    const size_t n = bytes / 4;
    if (n) copy_words_k<<<blocks(n), 256, 0, s>>>(static_cast<const int *>(src), static_cast<int *>(dst), n);
}

void silu_scale(const float * x, float * y, size_t n, float scale, cudaStream_t s) { silu_scale_k<<<blocks(n), 256, 0, s>>>(x, y, n, scale); }
void hc_apply(const float * g, const float * xn, float * mixed, int D, int hc, int N, cudaStream_t s) {
    hc_apply_k<<<blocks(static_cast<size_t>(N) * D), 256, 0, s>>>(g, xn, mixed, D, hc, N);
}
void attention_prefill(const float * q, const KvCache & kv, const float * gate, int gsh, int gst, float * out,
                       int heads, int kv_heads, int D, int pos0, int N, float scale, const AttnSel & sel, cudaStream_t s) {
    if (D != 256 || heads % kv_heads) throw std::runtime_error("attention_prefill: head dim " + std::to_string(D));
    static const bool old = [] { const char * e = std::getenv("BL_ATTN_OLD"); return e && std::atoi(e) != 0; }();   // testing
    if (!old && heads / kv_heads <= 16) {   // tensor cores: a block per (token, kv head)
        auto kern = kv.sk ? attention_prefill_mma_k<true> : attention_prefill_mma_k<false>;
        kern<<<dim3(N, kv_heads), 128, 0, s>>>(q, kv, gate, gsh, gst, out, heads, kv_heads, pos0, scale, sel);
        return;
    }
    const int rows = N * (heads / kv_heads);
    auto kern = kv.sk ? attention_prefill_k<true> : attention_prefill_k<false>;
    kern<<<dim3((rows + 7) / 8, kv_heads), 256, 0, s>>>(q, kv, gate, gsh, gst, out, heads, kv_heads, pos0, N, scale, sel);
}
void moe_gather(const void * xq, const int * tok, int A, int D, void * xg, cudaStream_t s) {
    const int nb = D / 32;
    moe_gather_k<<<blocks(static_cast<size_t>(A) * nb * (sizeof(BlockQ8) / 4)), 256, 0, s>>>(
        static_cast<const BlockQ8 *>(xq), tok, A, nb, static_cast<BlockQ8 *>(xg));
}
void moe_gate_up(uint32_t type, const MoeItem * items, int n_items, size_t gu_bytes, int F, int D, const void * xg, float * act,
                 cudaStream_t s, float limit) {
    if (n_items) dispatch<GateUp>(type, items, n_items, gu_bytes, F, D, xg, act, s, limit);
}
void moe_down(uint32_t type, const MoeItem * items, int n_items, size_t gu_bytes, size_t d_rb, int F, int D, const void * actq,
              const int * tok, const float * w, float * h, cudaStream_t s) {
    if (n_items) dispatch<Down>(type, items, n_items, gu_bytes, d_rb, F, D, actq, tok, w, h, s);
}
void add_gated(float * h, const float * sh, const float * sg, int D, int N, cudaStream_t s) {
    add_gated_k<<<blocks(static_cast<size_t>(N) * D), 256, 0, s>>>(h, sh, sg, D, N);
}
void swiglu(const float * g, const float * u, float * a, size_t n, cudaStream_t s) { swiglu_k<<<blocks(n), 256, 0, s>>>(g, u, a, n); }
void copy_rows(const float * src, int lds, float * dst, int ldd, int D, int N, cudaStream_t s) {
    copy_rows_k<<<blocks(static_cast<size_t>(N) * D), 256, 0, s>>>(src, lds, dst, ldd, D, N);
}

}  // namespace bl::cuda
