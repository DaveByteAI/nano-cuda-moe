#include <cmath>
#include <stdexcept>
#include <string>

#include "bl/cuda_pipeline.h"
#include "dequant.cuh"
#include "dp4a.cuh"

namespace bl::cuda {

namespace {

__global__ void window_begin_k(TokenState * ts, const int * toks, int T) {
    ts->seq += 1;
    for (int t = 0; t < T; ++t) ts->tok[t] = reinterpret_cast<const volatile int *>(toks)[t];
    ts->token = ts->tok[0];
}

__global__ void window_commit_k(TokenState * ts, const int * n) { ts->pos += *reinterpret_cast<const volatile int *>(n); }
__global__ void window_set_pos_k(TokenState * ts, const int * p) { ts->pos = *reinterpret_cast<const volatile int *>(p); }

// one block: thread 0 gathers the window's distinct experts and decides how each is served; all threads mirror the
// plan and the input to host memory; then the doorbell
__global__ void plan_k(int layer, int n_expert, int K, int T, const int * ids, const float * w, const uint8_t * const * table,
                       uint8_t * const * staging, float share, ExpertPlan * pd, ExpertPlan * ph, const float * x, float * xh,
                       int D, volatile uint32_t * bell, const TokenState * ts, const PfTable * pf, uint8_t * const * pf_slots,
                       const int * pred) {
    for (int i = threadIdx.x; i < T * D / 4; i += blockDim.x) reinterpret_cast<float4 *>(xh)[i] = reinterpret_cast<const float4 *>(x)[i];
    __shared__ ExpertPlan sp;
    __shared__ int s_id[kMaxT * kMaxK], s_first[kMaxT * kMaxK], s_pf[kMaxPf], s_npf;
    ExpertPlan & p = sp;
    const int TK = T * K, j = threadIdx.x;
    if (j == 0) {
        const volatile PfTable * vpf = pf;
        const bool pf_ok = vpf && vpf->seq == ts->seq;
        __threadfence_system();
        s_npf = pf_ok ? min(vpf->n, kMaxPf) : 0;
        for (int i = 0; i < s_npf; ++i) s_pf[i] = vpf->ids[i];
        p.T = T;
        p.K = K;
    }
    if (j < kMaxSlots) p.mask[j] = 0;
    if (j < TK) s_id[j] = ids[j];
    __syncthreads();
    // slots in order of first use: token-major, as the serial loop had them
    if (j < TK) {
        int f = j;
        for (int i = 0; i < j; ++i) if (s_id[i] == s_id[j]) { f = i; break; }
        s_first[j] = f;
    }
    __syncthreads();
    if (j < TK) {
        const int f = s_first[j];
        int slot = 0;
        for (int i = 0; i < f; ++i) slot += s_first[i] == i;
        if (f == j) p.ids[slot] = s_id[j];
        p.sel[j / K][j % K] = slot;
        p.w[j / K][j % K] = w[j];
        atomicOr(&p.mask[slot], 1 << (j / K));
        if (j == TK - 1) {
            int n = 0;
            for (int i = 0; i < TK; ++i) n += s_first[i] == i;
            p.n_slots = n;
        }
    }
    __syncthreads();
    if (j < p.n_slots) {   // the table lookups in parallel
        const uint8_t * b = table[static_cast<size_t>(layer) * n_expert + p.ids[j]];
        int ps = -1;
        if (!b) for (int i = 0; i < s_npf; ++i) if (s_pf[i] == p.ids[j]) ps = i;
        p.base[j] = b;
        p.idx[j] = ps;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int n_miss = 0;
        for (int s = 0; s < p.n_slots; ++s) n_miss += !p.base[s] && p.idx[s] < 0;
        const int n_cpu = static_cast<int>(lroundf(share * n_miss));
        int m = 0;
        p.n_dma = p.n_cpu = p.n_pf = 0;
        for (int s = 0; s < p.n_slots; ++s) {
            if (p.base[s]) { p.kind[s] = kHit; p.idx[s] = 0; }
            else if (p.idx[s] >= 0) { p.kind[s] = kPf; p.base[s] = pf_slots[p.idx[s]]; ++p.n_pf; }
            else if (m++ < n_cpu) { p.kind[s] = kCpu; p.idx[s] = p.n_cpu++; }
            else { p.kind[s] = kDma; p.idx[s] = p.n_dma; p.base[s] = staging[p.n_dma++]; }
        }
        for (int k = 0; k < kMaxK; ++k) p.pred[k] = pred && T == 1 && k < K ? pred[k] : -1;
    }
    __syncthreads();
    static_assert(sizeof(ExpertPlan) % 4 == 0, "plan copied as words");
    for (int i = threadIdx.x; i < static_cast<int>(sizeof(ExpertPlan) / 4); i += blockDim.x) {
        const uint32_t v = reinterpret_cast<const uint32_t *>(&sp)[i];
        reinterpret_cast<uint32_t *>(pd)[i] = v;
        reinterpret_cast<uint32_t *>(ph)[i] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();   // the plan and x reach host memory before the doorbell does
        *bell = ts->seq;
    }
}

__device__ unsigned long long g_wait_ns[4];   // per kind: the GPU's time spinning in wait_k (ns, %globaltimer)
__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void wait_k(const volatile uint32_t * flag, const TokenState * ts, const ExpertPlan * plan, int which) {
    if (which == kDma && plan->n_dma == 0) return;
    if (which == kCpu && plan->n_cpu == 0) return;
    if (which == kPf && plan->n_pf == 0) return;
    const uint32_t want = ts->seq;
    const unsigned long long t0 = gtime();
    while (*flag < want) __nanosleep(256);
    __threadfence_system();
    atomicAdd(&g_wait_ns[which], gtime() - t0);
}

__global__ void set_ptrs_k(const uint8_t ** table, PtrSet p) {
    if (threadIdx.x < p.n) table[p.idx[threadIdx.x]] = p.val[threadIdx.x];
}

__global__ void set_k(volatile uint32_t * flag, uint32_t v) {
    __threadfence_system();
    *flag = v;
}

constexpr int kExpWarps = 8;

__global__ void quantize_q8_k(const float * x, int nblocks, BlockQ8 * out) {
    const int blk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (blk < nblocks) quantize_block_q8(x + 32 * blk, out + blk, threadIdx.x & 31);
}

// grid (F / (kExpWarps * kGuRowsPerWarp), kMaxSlots): kGuLanes lanes per row of one slot (2 rows per warp: D = 2560
// has 80 sub-blocks, 5 per lane); gate and up for the tokens that chose the expert (SwiGLU fused)
constexpr int kGuLanes = 16, kGuRowsPerWarp = 32 / kGuLanes;
template <int F>
__global__ void plan_gate_up_k(const ExpertPlan * plan, int kinds, int T, size_t gu_bytes, int Fd, int D,
                               const BlockQ8 * __restrict__ xq, float * act, float limit) {
    const int e = blockIdx.y, lane = threadIdx.x & 31, sub = lane % kGuLanes;
    const int r = (blockIdx.x * kExpWarps + (threadIdx.x >> 5)) * kGuRowsPerWarp + lane / kGuLanes;
    if (e >= plan->n_slots || !((kinds >> plan->kind[e]) & 1) || r >= Fd) return;   // whole warps (Fd is even)
    const size_t rb = gu_bytes / Fd;
    const uint8_t * gate = plan->base[e] + static_cast<size_t>(r) * rb, * up = gate + gu_bytes;
    const int mask = plan->mask[e];
    float g[kMaxT] = {}, u[kMaxT] = {};
    const int xs = D / 32;
    for (int sb = sub; sb < xs; sb += kGuLanes) {
        dot32m<F, kMaxT>(gate, sb, xq, xs, T, g, mask);
        dot32m<F, kMaxT>(up, sb, xq, xs, T, u, mask);
    }
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) {
        if (t >= T) break;
        float gt = g[t], ut = u[t];
        for (int o = kGuLanes / 2; o > 0; o >>= 1) {
            gt += __shfl_xor_sync(0xffffffffu, gt, o);
            ut += __shfl_xor_sync(0xffffffffu, ut, o);
        }
        if (limit > 0.f) { gt = fminf(gt, limit); ut = fminf(fmaxf(ut, -limit), limit); }   // DeepSeek-V4's clamps
        if (sub == 0 && ((mask >> t) & 1)) act[(static_cast<size_t>(e) * kMaxT + t) * Fd + r] = gt / (1.f + expf(-gt)) * ut;
    }
}

// the block quantizes its slot's act (T tokens) into shared memory; 8 lanes per row, 4 rows per warp
constexpr int kDownLanes = 4, kDownRowsPerBlock = kExpWarps * (32 / kDownLanes);   // F = 640: 20 sub-blocks, 5 per lane
template <int F>
__global__ void plan_down_k(const ExpertPlan * plan, int kinds, int T, size_t gu_bytes, size_t d_rb, int Fd, int D,
                            const float * __restrict__ act, float * out) {
    const int e = blockIdx.y;
    if (e >= plan->n_slots || !((kinds >> plan->kind[e]) & 1)) return;
    extern __shared__ BlockQ8 aq[];   // [T][Fd/32]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nb = Fd / 32;
    const int mask = plan->mask[e];
    for (int blk = warp; blk < T * nb; blk += kExpWarps) {
        const int t = blk / nb;
        if ((mask >> t) & 1) quantize_block_q8(act + (static_cast<size_t>(e) * kMaxT + t) * Fd + 32 * (blk % nb), aq + blk, lane);
        else if (lane < 9) reinterpret_cast<int *>(aq + blk)[lane] = 0;   // a token that did not choose it: zeros
    }
    __syncthreads();
    const int r = blockIdx.x * kDownRowsPerBlock + warp * (32 / kDownLanes) + lane / kDownLanes;
    if (r >= D) return;
    const uint8_t * row = plan->base[e] + 2 * gu_bytes + static_cast<size_t>(r) * d_rb;
    float v[kMaxT] = {};
    for (int sb = lane % kDownLanes; sb < nb; sb += kDownLanes) dot32m<F, kMaxT>(row, sb, aq, nb, T, v, mask);
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) {
        if (t >= T) break;
        float a = v[t];
        for (int o = kDownLanes / 2; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane % kDownLanes == 0 && ((mask >> t) & 1)) out[(static_cast<size_t>(e) * kMaxT + t) * D + r] = a;
    }
}

// grid (D/256, T)
__global__ void plan_combine_k(const ExpertPlan * plan, const float * out, const volatile float * cpu_out, const float * sh,
                               const float * sg, int sgs, float * h, float * moe, int D) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= D) return;
    const int T = plan->T, K = plan->K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int s = plan->sel[t][k];
        const float v = plan->kind[s] == kCpu ? cpu_out[(static_cast<size_t>(plan->idx[s]) * T + t) * D + i]
                                              : out[(static_cast<size_t>(s) * kMaxT + t) * D + i];
        acc += plan->w[t][k] * v;
    }
    if (moe) moe[static_cast<size_t>(t) * D + i] = acc;
    h[static_cast<size_t>(t) * D + i] = acc + sh[static_cast<size_t>(t) * D + i] / (1.f + expf(-sg[t * sgs]));
}

size_t row_bytes_of(uint32_t t, int K) {
    switch (t) {
        case Q2_0:    return K / 64 * 18;
        case IQ4_NL:  return K / 32 * 18;
        case IQ4_XS:  return K / 256 * 136;
        case IQ2_XXS: return K / 256 * 66;
        case IQ2_XS:  return K / 256 * 74;
        case IQ2_S:   return K / 256 * 82;
        case IQ3_XXS: return K / 256 * 98;
        case IQ3_S:   return K / 256 * 110;
        default: throw std::runtime_error("expert format " + std::to_string(t) + " not supported");
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
        default: throw std::runtime_error("expert format " + std::to_string(t) + " not supported");
    }
}

template <int F> struct GateUp {
    static void run(const ExpertPlan * p, int kinds, int T, size_t gu, int Fd, int D, const void * xq, float * act, cudaStream_t s,
                    float limit) {
        constexpr int rpb = kExpWarps * kGuRowsPerWarp;
        plan_gate_up_k<F><<<dim3((Fd + rpb - 1) / rpb, kMaxSlots), 32 * kExpWarps, 0, s>>>(
            p, kinds, T, gu, Fd, D, static_cast<const BlockQ8 *>(xq), act, limit);   // (grid below: rows per block)
    }
};
template <int F> struct Down {
    static void run(const ExpertPlan * p, int kinds, int T, size_t gu, size_t d_rb, int Fd, int D, const float * act, float * out,
                    cudaStream_t s) {
        plan_down_k<F><<<dim3((D + kDownRowsPerBlock - 1) / kDownRowsPerBlock, kMaxSlots), 32 * kExpWarps,
                         static_cast<size_t>(T) * (Fd / 32) * sizeof(BlockQ8), s>>>(p, kinds, T, gu, d_rb, Fd, D, act, out);
    }
};

}  // namespace

void window_begin(TokenState * ts, const int * toks, int T, cudaStream_t s) { window_begin_k<<<1, 1, 0, s>>>(ts, toks, T); }
void window_commit(TokenState * ts, const int * n, cudaStream_t s) { window_commit_k<<<1, 1, 0, s>>>(ts, n); }
void window_set_pos(TokenState * ts, const int * p, cudaStream_t s) { window_set_pos_k<<<1, 1, 0, s>>>(ts, p); }

void plan_experts(int layer, int n_expert, int K, int T, const int * ids, const float * w, const uint8_t * const * table,
                  uint8_t * const * staging, float share, ExpertPlan * pd, ExpertPlan * ph, const float * x, float * xh, int D,
                  volatile uint32_t * bell, const TokenState * ts, const PfTable * pf, uint8_t * const * pf_slots,
                  const int * pred, cudaStream_t s) {
    plan_k<<<1, 256, 0, s>>>(layer, n_expert, K, T, ids, w, table, staging, share, pd, ph, x, xh, D, bell, ts, pf, pf_slots, pred);
}

void wait_flag(const volatile uint32_t * flag, const TokenState * ts, const ExpertPlan * plan, int which, cudaStream_t s) {
    wait_k<<<1, 1, 0, s>>>(flag, ts, plan, which);
}

void set_flag(volatile uint32_t * flag, uint32_t v, cudaStream_t s) { set_k<<<1, 1, 0, s>>>(flag, v); }

__device__ unsigned long long g_stamps[kMaxStamps];
__global__ void stamp_k(int i) { g_stamps[i] = gtime(); }
void stamp(int i, cudaStream_t s) { stamp_k<<<1, 1, 0, s>>>(i); }
void read_stamps(unsigned long long * out, int n) { cudaMemcpyFromSymbol(out, g_stamps, sizeof(unsigned long long) * n); }

void wait_times(double ms[4], bool reset) {
    unsigned long long ns[4];
    cudaMemcpyFromSymbol(ns, g_wait_ns, sizeof ns);
    for (int i = 0; i < 4; ++i) ms[i] = ns[i] * 1e-6;
    if (reset) {
        const unsigned long long z[4] = {};
        cudaMemcpyToSymbol(g_wait_ns, z, sizeof z);
    }
}

void set_ptrs(const uint8_t ** table, const PtrSet & p, cudaStream_t s) {
    if (p.n) set_ptrs_k<<<1, kMaxPtrSet, 0, s>>>(table, p);
}

size_t q8_bytes(int n) { return static_cast<size_t>(n / 32) * sizeof(BlockQ8); }

void quantize_q8(const float * x, int n, void * out, cudaStream_t s, int T) {
    const int blocks = T * n / 32;
    quantize_q8_k<<<(blocks + 7) / 8, 256, 0, s>>>(x, blocks, static_cast<BlockQ8 *>(out));
}

void plan_gate_up(uint32_t t, const ExpertPlan * p, int kinds, int T, size_t gu, int F, int D, const void * xq, float * act,
                  cudaStream_t s, float limit) {
    dispatch<GateUp>(t, p, kinds, T, gu, F, D, xq, act, s, limit);
}

void plan_down(uint32_t td, const ExpertPlan * p, int kinds, int T, size_t gu, int F, int D, const float * act, float * out,
               cudaStream_t s) {
    dispatch<Down>(td, p, kinds, T, gu, row_bytes_of(td, F), F, D, act, out, s);
}

void plan_combine(const ExpertPlan * p, int T, const float * out, const float * cpu_out, const float * sh, const float * sg, int sgs,
                  float * h, float * moe, int D, cudaStream_t s) {
    plan_combine_k<<<dim3((D + 255) / 256, T), 256, 0, s>>>(p, out, cpu_out, sh, sg, sgs, h, moe, D);
}

}  // namespace bl::cuda
