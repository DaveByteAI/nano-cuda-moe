// Device-driven expert pipeline: the GPU decides hits and misses itself, rings a doorbell in mapped host memory, and
// waits on flags the host sets once the missed experts are copied in (DMA) or computed (CPU). Nothing the host
// learns per token changes a kernel's arguments, so a whole token can be one CUDA graph.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace bl::cuda {

constexpr int kMaxK = 16;                  // routed experts per token (the model uses 10)
constexpr int kMaxT = 4;                   // tokens per verify window
constexpr int kMaxSlots = kMaxT * 10;      // distinct experts a layer can need in one window
constexpr int kMaxPf = 8;   // experts prefetched for the next layer
enum : int { kHit = 0, kDma = 1, kCpu = 2, kPf = 3 };

// the experts the host prefetched into a layer's prefetch slots (mapped host memory). Trusted only when `seq`
// matches the running token: the host writes ids, then seq; a plan that reads it too early ignores it.
struct PfTable {
    int      ids[kMaxPf];
    int      n;
    volatile uint32_t seq;
};

// one layer's routing decision for a window of T tokens, written by the GPU (device copy) and mirrored to mapped host
// memory: the distinct experts the tokens chose ("slots"), how each is served, and each token's choices
struct ExpertPlan {
    int             T, K, n_slots;
    int             ids[kMaxSlots];
    int             kind[kMaxSlots];  // kHit / kDma / kCpu / kPf
    int             idx[kMaxSlots];   // kDma: staging slot; kCpu: CPU result slot; kPf: prefetch slot
    int             mask[kMaxSlots];  // bit t: token t chose this expert
    const uint8_t * base[kMaxSlots];  // GPU address of [gate|up|down] for kHit / kDma / kPf
    int             sel[kMaxT][kMaxK];   // token t's k-th choice: a slot
    float           w[kMaxT][kMaxK];     // and its weight
    int             n_dma, n_cpu, n_pf;
    int             pred[kMaxK];      // the next layer's router on this layer's input (T == 1 only): its likely experts
};

// window counters in device memory (graph-safe): pos = position of the window's first token, seq = window sequence
// number (>= 1, the doorbells and flags compare against it), tok = the window's tokens
struct TokenState {
    int      token;          // tok[0]
    int      pos;
    uint32_t seq;
    int      tok[kMaxT];
};

// seq += 1, tok[0..T) = tokens (pinned host memory, read when the graph runs)
void window_begin(TokenState * ts, const int * tokens_host, int T, cudaStream_t s);
// pos += *n (pinned host memory): the accepted tokens of the last window
void window_commit(TokenState * ts, const int * n_host, cudaStream_t s);
// pos = *p (pinned host memory)
void window_set_pos(TokenState * ts, const int * p_host, cudaStream_t s);

// after the router: builds the plan from ids/w ([T][K] each), mirrors it and x ([T][D], the FFN input) to the host,
// rings doorbell = seq. pf/pf_slots: this layer's prefetch table and slot addresses; pred_ids: the next layer's
// predicted top-K (T == 1, or null)
void plan_experts(int layer, int n_expert, int K, int T, const int * ids, const float * w, const uint8_t * const * table,
                  uint8_t * const * staging, float cpu_share, ExpertPlan * plan_dev, ExpertPlan * plan_host_mapped,
                  const float * x, float * x_host_mapped, int D, volatile uint32_t * doorbell_mapped,
                  const TokenState * ts, const PfTable * pf, uint8_t * const * pf_slots, const int * pred_ids,
                  cudaStream_t s);

// testing: stamp i = %globaltimer (ns) when the stream reaches it; read them back once the stream is idle
constexpr int kMaxStamps = 4096;
void stamp(int i, cudaStream_t s);
void read_stamps(unsigned long long * out, int n);
// the GPU's total time spinning in wait_flag per kind (ms), since the last reset (call when the GPU is idle)
void wait_times(double ms[4], bool reset);
// spin until *flag >= ts->seq; skipped when the plan has no experts of kind `which` (kDma / kCpu / kPf)
void wait_flag(const volatile uint32_t * flag_mapped, const TokenState * ts, const ExpertPlan * plan, int which,
               cudaStream_t s);
// table[idx[i]] = val[i] for i < n (expert-table edits ordered on a stream with the copies they guard)
constexpr int kMaxPtrSet = 64;
struct PtrSet {
    int             n = 0;
    int             idx[kMaxPtrSet];
    const uint8_t * val[kMaxPtrSet];
};
void set_ptrs(const uint8_t ** table, const PtrSet & p, cudaStream_t s);

// *flag = value (for the copy stream, after the DMA copies)
void set_flag(volatile uint32_t * flag_mapped, uint32_t value, cudaStream_t s);

// the FFN input quantized to q8_1 (int8 per 32 values) for the experts' integer dot products; x [T][n] -> T rows
size_t q8_bytes(int n);
void   quantize_q8(const float * x, int n, void * out, cudaStream_t s, int T = 1);

// expert compute over the plan's slots whose kind is in `kinds` (bit mask), for all T tokens of the window:
//   act [slot][T][F], out [slot][T][D]; xq = quantize_q8 of the input ([T] rows of D/32 blocks)
// limit > 0: SwiGLU clamps first (gate . x at most limit, up . x within +-limit: DeepSeek-V4)
void plan_gate_up(uint32_t type_gu, const ExpertPlan * plan, int kinds, int T, size_t gu_bytes, int F, int D,
                  const void * xq, float * act, cudaStream_t s, float limit = 0.f);
void plan_down(uint32_t type_d, const ExpertPlan * plan, int kinds, int T, size_t gu_bytes, int F, int D,
               const float * act, float * out, cudaStream_t s);
// per token t: h[t] = sum_k w[t][k] * (expert sel[t][k]'s output for t) + sigmoid(sg[t]) * sh[t]
//   cpu_out_mapped [cpu slot][T][D]; moe alone into moe_out [T][D] if set; sh/h/moe [T][D]; sg [T] (stride sg_stride)
void plan_combine(const ExpertPlan * plan, int T, const float * out, const float * cpu_out_mapped, const float * sh,
                  const float * sg, int sg_stride, float * h, float * moe_out, int D, cudaStream_t s);

}  // namespace bl::cuda
