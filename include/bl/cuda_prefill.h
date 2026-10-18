// Prompt processing (prefill): every layer runs over the whole prompt before the next, so each routed expert crosses
// PCIe at most once per layer. Dense projections are cuBLAS GEMMs (TF32) on weights dequantized in row blocks; the
// experts are grouped by expert and computed with the dp4a dot products over tiles of 8 tokens.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bl/cuda_window.h"

namespace bl::cuda {

// y [N][M] = x [N][K] . W^T for a weight matrix W (M rows of K values, any supported format) - dequantized in blocks
// of rows into `scratch` (scratch_floats floats). ldy: the row stride of y (>= M; a column offset is y + offset).
void gemm_w(uint32_t type, const void * W, size_t row_bytes, int M, int K, const float * x, int N, float * y, int ldy,
            float * scratch, size_t scratch_floats, cudaStream_t s);

// y [N][M] = x . W^T with W in bf16 ([M][K]); x is rounded to bf16 into x16 (N*K values) first. fp32 accumulation.
void gemm_bf16(const void * W16, int M, int K, const float * x, int N, float * y, int ldy, void * x16, cudaStream_t s);

// dst row i = src row ids[i] (ids in device memory; row_bytes % 4 == 0)
void gather_rows(const void * src, size_t row_bytes, const int * ids, int n, void * dst, cudaStream_t s);
// a copy done by a kernel (bytes % 4 == 0): from mapped host memory it does not queue behind the copy engine's DMA
void copy_words(const void * src, void * dst, size_t bytes, cudaStream_t s);

// hyper-connection mix, N tokens: g [N][hc*D] = silu(lo / hc) . W_up^T is done by gemm_w on `lo_act`, then
//   mixed[t][i] = (1/hc) sum_c xn[t][c*D+i] * sigmoid(g[t][c*D+i])
void silu_scale(const float * x, float * y, size_t n, float scale, cudaStream_t s);
void hc_apply(const float * g, const float * xn, float * mixed, int D, int hc, int N, cudaStream_t s);

// causal attention of N query tokens at positions pos0.. over the cache cells sel names, gated: out [N][heads][D]
void attention_prefill(const float * q, const KvCache & kv, const float * gate, int gsh, int gst, float * out,
                       int heads, int kv_heads, int D, int pos0, int N, float scale, const AttnSel & sel, cudaStream_t s);

// the routed experts of N tokens, grouped by expert. An item is up to kMoeTile consecutive assignments of one expert.
constexpr int kMoeTile = 64;   // the most tokens per item (the tensor-core kernels' tile)
int moe_tile();                // tokens per item to use: kMoeTile, or 16 with the dp4a kernels (BL_MOE_DP4A=1, testing)
struct MoeItem {
    const uint8_t * base;   // the expert's [gate | up | down]
    int             a0, n;  // assignments a0 .. a0+n-1
};
// xg [A][D/32] q8_1 rows of the assignments' tokens, gathered from xq [N][D/32]
void moe_gather(const void * xq, const int * tok, int A, int D, void * xg, cudaStream_t s);
// act [A][F] = silu(gate . x) * (up . x); limit > 0 clamps first (gate at most limit, up within +-limit: DeepSeek-V4)
void moe_gate_up(uint32_t type, const MoeItem * items, int n_items, size_t gu_bytes, int F, int D, const void * xg, float * act,
                 cudaStream_t s, float limit = 0.f);
// h[tok[a]] += w[a] * (down . act[a])  (act quantized: actq [A][F/32]); atomics: the sum order varies
void moe_down(uint32_t type, const MoeItem * items, int n_items, size_t gu_bytes, size_t d_rb, int F, int D, const void * actq,
              const int * tok, const float * w, float * h, cudaStream_t s);
// h[t] += sigmoid(sg[t]) * sh[t]   (h [N][D])
void add_gated(float * h, const float * sh, const float * sg, int D, int N, cudaStream_t s);
// a = silu(g) * u, elementwise
void swiglu(const float * g, const float * u, float * a, size_t n, cudaStream_t s);
// y = x for rows: dst [N][D] row r = src row (r) of a [N][ld] matrix
void copy_rows(const float * src, int lds, float * dst, int ldd, int D, int N, cudaStream_t s);

}  // namespace bl::cuda
