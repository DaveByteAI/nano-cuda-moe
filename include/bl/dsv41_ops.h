// The deepseek41 forward's own small kernels (one token, fp32). Formulas: docs/DSV41.md; each names the model.py /
// kernel.py / engram.py code it reproduces. The quantized matvecs are bl::cuda::matvec, the norms rmsnorm_rows.
#pragma once

#include <cuda_runtime.h>

namespace bl::cuda::dsv41 {

// rotate the last n_rot values of each of `heads` rows of D, adjacent pairs as complex numbers (apply_rotary_emb):
// angle = pos * freq[i]; inverse = the conjugate rotation (the attention output gets the query's rotation removed)
void rope_tail(float * x, int heads, int D, int n_rot, int pos, const float * freq, bool inverse, cudaStream_t s);

// hc_split_sinkhorn: m = raw mixes (fn . flat R) [2+hc][hc]; R the stream [hc][D] (its rms scales m);
// pre [hc], post [hc], comb [hc][hc] (comb[j][k]: residual copy j into output copy k)
void hc_split(const float * m, const float * R, const float * scale, const float * base, int hc, int D, float norm_eps,
              float hc_eps, int iters, float * pre, float * post, float * comb, cudaStream_t s);
// a few KB between device memory and pinned host memory (UVA: the kernel reads or writes the host pointer) by a
// kernel, not a copy engine: a small copy then never queues behind a 9 MB expert copy on the same engine
void copy_small(void * dst, const void * src, size_t bytes, cudaStream_t s);
// fn . flat R (fn [M][hc*D] fp32) into mix, then hc_split from it: two launches, R read once (ss: 1 float of scratch)
void hc_fn_split(const float * fn, int M, const float * R, const float * scale, const float * base, int hc, int D,
                 float norm_eps, float hc_eps, int iters, float * mix, float * ss, float * pre, float * post, float * comb,
                 cudaStream_t s);
// out[d] = sum_j pre[j] * R[j][d]
void hc_pre(const float * R, const float * pre, float * out, int hc, int D, cudaStream_t s);
// out[k][d] = post[k] * h[d] + sum_j comb[j][k] * res[j][d]   (out must not alias res)
void hc_post(const float * h, const float * res, const float * post, const float * comb, float * out, int hc, int D,
             cudaStream_t s);

// Engram.forward: kv = [hc][D] keys then [D] value; qk = q_weight * k_weight [hc][D]; R [hc][D] gets the gated value
void engram_gate(float * R, const float * kv, const float * qk, int hc, int D, float eps, cudaStream_t s);

// sparse_attn for one query: H heads of D against keys gathered from two sources (rows idx_a[0..n_a) of A and rows
// idx_b[0..n_b) of B, each [rows][D]; K = V), plus a per-head sink logit in the softmax denominator
void attention_sink(const float * q, const float * A, const int * idx_a, int n_a, const float * B, const int * idx_b,
                    int n_b, const float * sink, float * out, int H, int D, float scale, cudaStream_t s);

// Compressor at ratio 2, one completed group: out[d] = sum_i softmax_i(score[i][d]) * kv[i][d]   (i over 2 slots)
void pool2(const float * kv, const float * score, float * out, int D, cudaStream_t s);

// Indexer: out[t] = sum_h relu(q[h] . keys[t]) * w[h] * wscale   (q [H][Di], keys [N][Di])
void indexer_score(const float * q, const float * w, float wscale, const float * keys, int N, int H, int Di, float * out,
                   cudaStream_t s);

// the k = min(k, n) largest of score[0..n) as indices in increasing order (ties: the lower index first), one block
// (radix select on the float bits, then an ordered scan): the indexer's top-k without a trip to the host
void topk_sorted(const float * score, int n, int k, int * out, cudaStream_t s);

// Gate (sqrtsoftplus, noaux_tc): s = sqrt(softplus(logit)); the top k by s + bias; w = s / sum(s) * scale
void router_topk(const float * logits, const float * bias, int n, int k, float scale, int * ids, float * w, cudaStream_t s);

// Expert: out = silu(min(g, limit)) * clamp(u, -limit, limit)
void swiglu_clamp(const float * g, const float * u, float * out, int n, float limit, cudaStream_t s);

// ---- the same over T tokens at once (prefill). Row t of every [T][...] array is token t.

// positions pos0 + t * step (step: 1 for tokens, the compress ratio for compressed latents)
void rope_tail_rows(float * x, int T, int heads, int D, int n_rot, int pos0, int step, const float * freq, bool inverse,
                    cudaStream_t s);
// m [T][(2+hc)*hc], R [T][hc][D] -> pre [T][hc], post [T][hc], comb [T][hc][hc]
void hc_split_rows(const float * m, const float * R, const float * scale, const float * base, int hc, int D, int T,
                   float norm_eps, float hc_eps, int iters, float * pre, float * post, float * comb, cudaStream_t s);
void hc_pre_rows(const float * R, const float * pre, float * out, int hc, int D, int T, cudaStream_t s);
void hc_post_rows(const float * h, const float * res, const float * post, const float * comb, float * out, int hc, int D,
                  int T, cudaStream_t s);
// kv [T][(hc+1)*D]
void engram_gate_rows(float * R, const float * kv, const float * qk, int hc, int D, int T, float eps, cudaStream_t s);
// logits [T][n] -> ids [T][k], w [T][k]
void router_topk_rows(const float * logits, const float * bias, int n, int k, float scale, int T, int * ids, float * w,
                      cudaStream_t s);
// out[t][j] for j < n_vis[t] (device array), row stride ld; q [T][H][Di], w [T][H]
void indexer_score_rows(const float * q, const float * w, float wscale, const float * keys, const int * n_vis, int n_max,
                        int H, int Di, int T, float * out, int ld, cudaStream_t s);
// query t: the contiguous rows [a0[t], a0[t] + na[t]) of A, and rows idx_b[t][0..nb[t]) of B (idx_b row stride ldb);
// n_max = the largest na + nb (shared memory); q, out [T][H][D]
void attention_sink_rows(const float * q, const float * A, const int * a0, const int * na, const float * B, const int * idx_b,
                         int ldb, const int * nb, int n_max, const float * sink, float * out, int H, int D, int T, float scale,
                         cudaStream_t s);
// n groups of 2 consecutive rows: out[g] = pool of rows 2g, 2g+1 of kv/score ([2n][D])
void pool2_rows(const float * kv, const float * score, float * out, int D, int n, cudaStream_t s);

}  // namespace bl::cuda::dsv41
