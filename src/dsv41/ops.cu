#include "bl/dsv41_ops.h"

#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

namespace bl::cuda::dsv41 {

namespace {

__device__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum for blockDim.x a multiple of 32 (<= 1024); every thread gets the result
__device__ float block_sum(float v) {
    __shared__ float part[32];
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    float t = l < (blockDim.x >> 5) ? part[l] : 0.f;
    return warp_sum(t);
}
__device__ float block_max(float v) {
    __shared__ float part[32];
    v = warp_max(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    float t = l < (blockDim.x >> 5) ? part[l] : -FLT_MAX;
    return warp_max(t);
}

__global__ void rope_tail_k(float * x, int D, int n_rot, int pos, const float * freq, float sign) {
    const int h = blockIdx.x, i = threadIdx.x;   // i: pair index in [0, n_rot / 2)
    if (i >= n_rot / 2) return;
    float * p = x + static_cast<size_t>(h) * D + (D - n_rot) + 2 * i;
    // torch.outer(arange, freqs) in fp32, then polar: the angle is one fp32 product
    const float ang = static_cast<float>(pos) * freq[i];
    float sn, cs;
    sincosf(ang, &sn, &cs);
    sn *= sign;
    const float a = p[0], b = p[1];
    p[0] = a * cs - b * sn;
    p[1] = a * sn + b * cs;
}

// one block: the rms of the flattened stream, then one thread does the split and the Sinkhorn iterations
__device__ void hc_split_warp(const float * m, float ss, const float * scale, const float * base, int hc, int n,
                              float norm_eps, float hc_eps, int iters, float * pre, float * post, float * comb);

__global__ void hc_split_k(const float * m, const float * R, const float * scale, const float * base, int hc, int n,
                           float norm_eps, float hc_eps, int iters, float * pre, float * post, float * comb) {
    {   // row blockIdx.x of a batch (one block per token)
        const int t = blockIdx.x, mix = (2 + hc) * hc;
        m += static_cast<size_t>(t) * mix; R += static_cast<size_t>(t) * n;
        pre += t * hc; post += t * hc; comb += t * hc * hc;
    }
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += R[i] * R[i];
    ss = block_sum(ss);
    if (threadIdx.x >= 32) return;
    hc_split_warp(m, ss, scale, base, hc, n, norm_eps, hc_eps, iters, pre, post, comb);
}

// m[r] = fn[r] . R (fn [M][K] fp32, K = hc * D), one block per row; block 0 also stores sum(R^2) (every block reads
// all of R anyway), and the last block to finish runs hc_split's Sinkhorn on them. Replaces a warp-per-row matvec
// (24 rows: 6 blocks on the whole GPU), hc_split's own R pass and its launch.
__global__ void hc_fn_k(const float * __restrict__ W, int K, const float * __restrict__ R, float * m, float * ss_out,
                        unsigned * done, const float * scale, const float * base, int hc, float norm_eps, float hc_eps,
                        int iters, float * pre, float * post, float * comb) {
    const float4 * w = reinterpret_cast<const float4 *>(W + static_cast<size_t>(blockIdx.x) * K);
    const float4 * r = reinterpret_cast<const float4 *>(R);
    float d = 0.f, ss = 0.f;
    for (int i = threadIdx.x; i < K / 4; i += blockDim.x) {
        const float4 a = w[i], b = r[i];
        d += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        ss += b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
    }
    d = block_sum(d);
    ss = block_sum(ss);
    // the last block to finish runs the Sinkhorn (one launch fewer): its row results are visible after the fence
    __shared__ bool last;
    if (threadIdx.x == 0) {
        m[blockIdx.x] = d;
        if (blockIdx.x == 0) *ss_out = ss;
        __threadfence();
        last = atomicAdd(done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || threadIdx.x >= 32) return;
    __threadfence();
    hc_split_warp(m, *reinterpret_cast<volatile float *>(ss_out), scale, base, hc, K, norm_eps, hc_eps, iters, pre, post, comb);
    if (threadIdx.x == 0) *done = 0;   // for the next launch (stream order)
}

// warp 0 of hc_split: from the raw mixes and sum(R^2) to pre, post, comb
__device__ void hc_split_warp(const float * m, float ss, const float * scale, const float * base, int hc, int n,
                              float norm_eps, float hc_eps, int iters, float * pre, float * post, float * comb) {
    // lane l = j * hc + k holds comb element (j, k) in a register (hc a power of two, hc * hc <= 32):
    // a row's elements are hc neighbouring lanes (xor 1 .. hc/2), a column's are hc apart (xor hc .. hc*hc/2)
    const int l = threadIdx.x, j = l / hc, k = l % hc, nn = hc * hc;
    const float r = rsqrtf(ss / n + norm_eps);
    if (l < hc) {
        pre[l]  = 1.f / (1.f + expf(-(m[l] * r * scale[0] + base[l]))) + hc_eps;
        post[l] = 2.f / (1.f + expf(-(m[hc + l] * r * scale[1] + base[hc + l])));
    }
    const bool on = l < nn;
    float c = on ? m[2 * hc + l] * r * scale[2] + base[2 * hc + l] : 0.f;
    auto row_reduce = [&](float v, bool is_max) {
        for (int o = 1; o < hc; o <<= 1) {
            const float w = __shfl_xor_sync(0xffffffffu, v, o);
            v = is_max ? fmaxf(v, w) : v + w;
        }
        return v;
    };
    auto col_sum = [&](float v) {
        for (int o = hc; o < nn; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        return v;
    };
    // softmax over each row, + eps
    const float mx = row_reduce(on ? c : -FLT_MAX, true);
    const float e = on ? expf(c - mx) : 0.f;
    c = on ? e / row_reduce(e, false) + hc_eps : 0.f;
    c = on ? c / (col_sum(c) + hc_eps) : 0.f;
    for (int it = 1; it < iters; ++it) {
        c = on ? c / (row_reduce(c, false) + hc_eps) : 0.f;
        c = on ? c / (col_sum(c) + hc_eps) : 0.f;
    }
    if (on) comb[j * hc + k] = c;
}

__global__ void hc_pre_k(const float * R, const float * pre, float * out, int hc, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float s = 0.f;
    for (int j = 0; j < hc; ++j) s += pre[j] * R[static_cast<size_t>(j) * D + d];
    out[d] = s;
}

__global__ void hc_post_k(const float * h, const float * res, const float * post, const float * comb, float * out, int hc,
                          int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, k = blockIdx.y;
    if (d >= D) return;
    float s = post[k] * h[d];
    for (int j = 0; j < hc; ++j) s += comb[j * hc + k] * res[static_cast<size_t>(j) * D + d];
    out[static_cast<size_t>(k) * D + d] = s;
}

// one block per hc copy
__global__ void engram_gate_k(float * R, const float * kv, const float * qk, int hc, int D, float eps) {
    const int c = blockIdx.x;
    R += static_cast<size_t>(blockIdx.y) * hc * D;           // token blockIdx.y of a batch
    kv += static_cast<size_t>(blockIdx.y) * (hc + 1) * D;
    float * x = R + static_cast<size_t>(c) * D;
    const float * key = kv + static_cast<size_t>(c) * D;
    const float * w = qk + static_cast<size_t>(c) * D;
    const float * value = kv + static_cast<size_t>(hc) * D;
    float sx = 0.f, sk = 0.f, sd = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sx += x[i] * x[i];
        sk += key[i] * key[i];
        sd += x[i] * w[i] * key[i];
    }
    sx = block_sum(sx);
    sk = block_sum(sk);
    sd = block_sum(sd);
    const float dot = sd * rsqrtf(sx / D + eps) * rsqrtf(sk / D + eps) * rsqrtf(static_cast<float>(D));
    // signed sqrt before the sigmoid; copysign keeps +0 positive like the reference
    const float g = 1.f / (1.f + expf(-copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot)));
    for (int i = threadIdx.x; i < D; i += blockDim.x) x[i] += g * value[i];
}

// one block per head; scores in dynamic shared memory
__global__ void attention_sink_k(const float * q, const float * A, const int * idx_a, int n_a, const float * B,
                                 const int * idx_b, int n_b, const float * sink, float * out, int D, float scale) {
    extern __shared__ float sm[];
    float * qs = sm;            // [D]
    float * sc = sm + D;        // [n_a + n_b]
    const int h = blockIdx.x, n = n_a + n_b;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int i = threadIdx.x; i < D; i += blockDim.x) qs[i] = q[static_cast<size_t>(h) * D + i];
    __syncthreads();
    auto key = [&](int j) -> const float * {
        return j < n_a ? A + static_cast<size_t>(idx_a[j]) * D : B + static_cast<size_t>(idx_b[j - n_a]) * D;
    };
    for (int j = warp; j < n; j += nw) {
        const float * k = key(j);
        float s = 0.f;
        for (int i = lane; i < D; i += 32) s += qs[i] * k[i];
        s = warp_sum(s);
        if (lane == 0) sc[j] = s * scale;
    }
    __syncthreads();
    float mx = sink[h];
    for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, sc[j]);
    mx = block_max(mx);
    float z = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) z += (sc[j] = expf(sc[j] - mx));
    z = block_sum(z) + expf(sink[h] - mx);
    __syncthreads();
    const float inv = 1.f / z;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float o = 0.f;
        for (int j = 0; j < n; ++j) o += sc[j] * key(j)[i];
        out[static_cast<size_t>(h) * D + i] = o * inv;
    }
}

// split-K attention for one query (flash-decoding): block (hb, c) takes heads hb*kHG .. +kHG and keys c*kKC .. +kKC,
// reads each of its keys once for all kHG heads (K = V is shared by the heads) and writes the chunk's max, sum of
// exponentials and unnormalised output; attn_combine_k merges the chunks with the sink. 8x less L2 traffic than a
// block per head reading every key.
constexpr int kHG = 8, kKC = 64;
__global__ void attn_split_k(const float * q, const float * A, const int * idx_a, int n_a, const float * B,
                             const int * idx_b, int n_b, int D, float scale, int S, float * om, float * ol, float * oo) {
    extern __shared__ float sm[];
    float * qs = sm;              // [kHG][D]
    float * sc = sm + kHG * D;    // [kHG][kKC]
    const int hb = blockIdx.x, c = blockIdx.y, n = n_a + n_b;
    const int j0 = c * kKC, nk = min(kKC, n - j0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int i = threadIdx.x; i < kHG * D; i += blockDim.x) qs[i] = q[static_cast<size_t>(hb) * kHG * D + i];
    __syncthreads();
    auto key = [&](int j) -> const float * {
        return j < n_a ? A + static_cast<size_t>(idx_a[j]) * D : B + static_cast<size_t>(idx_b[j - n_a]) * D;
    };
    for (int jj = warp; jj < nk; jj += nw) {
        const float * k = key(j0 + jj);
        float acc[kHG];
#pragma unroll
        for (int h = 0; h < kHG; ++h) acc[h] = 0.f;
        for (int i = lane; i < D; i += 32) {
            const float kv = k[i];
#pragma unroll
            for (int h = 0; h < kHG; ++h) acc[h] += qs[h * D + i] * kv;
        }
#pragma unroll
        for (int h = 0; h < kHG; ++h) acc[h] = warp_sum(acc[h]);
        if (lane == 0)
#pragma unroll
            for (int h = 0; h < kHG; ++h) sc[h * kKC + jj] = acc[h] * scale;
    }
    __syncthreads();
    if (warp < kHG) {   // warp h: the chunk's softmax statistics of head h
        const int h = warp;
        float m = -FLT_MAX;
        for (int jj = lane; jj < nk; jj += 32) m = fmaxf(m, sc[h * kKC + jj]);
        m = warp_max(m);
        float l = 0.f;
        for (int jj = lane; jj < nk; jj += 32) {
            const float e = expf(sc[h * kKC + jj] - m);
            sc[h * kKC + jj] = e;
            l += e;
        }
        l = warp_sum(l);
        if (lane == 0) {
            om[(hb * kHG + h) * S + c] = m;
            ol[(hb * kHG + h) * S + c] = l;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float o[kHG];
#pragma unroll
        for (int h = 0; h < kHG; ++h) o[h] = 0.f;
        for (int jj = 0; jj < nk; ++jj) {
            const float kv = key(j0 + jj)[i];
#pragma unroll
            for (int h = 0; h < kHG; ++h) o[h] += sc[h * kKC + jj] * kv;
        }
#pragma unroll
        for (int h = 0; h < kHG; ++h) oo[(static_cast<size_t>(hb * kHG + h) * S + c) * D + i] = o[h];
    }
}

__global__ void attn_combine_k(const float * om, const float * ol, const float * oo, const float * sink, float * out, int D,
                               int S) {
    const int h = blockIdx.x;
    float mx = sink[h];
    for (int c = 0; c < S; ++c) mx = fmaxf(mx, om[h * S + c]);
    float z = expf(sink[h] - mx);
    for (int c = 0; c < S; ++c) z += ol[h * S + c] * expf(om[h * S + c] - mx);
    const float inv = 1.f / z;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float o = 0.f;
        for (int c = 0; c < S; ++c) o += expf(om[h * S + c] - mx) * oo[(static_cast<size_t>(h) * S + c) * D + i];
        out[static_cast<size_t>(h) * D + i] = o * inv;
    }
}

__global__ void pool2_k(const float * kv, const float * score, float * out, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const float s0 = score[d], s1 = score[D + d], mx = fmaxf(s0, s1);
    const float e0 = expf(s0 - mx), e1 = expf(s1 - mx);
    out[d] = (kv[d] * e0 + kv[D + d] * e1) / (e0 + e1);
}

// one warp per key
__global__ void indexer_score_k(const float * q, const float * w, float wscale, const float * keys, int N, int H, int Di,
                                float * out) {
    const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= N) return;
    const float * k = keys + static_cast<size_t>(t) * Di;
    float acc = 0.f;
    for (int h = 0; h < H; ++h) {
        float s = 0.f;
        for (int i = lane; i < Di; i += 32) s += q[static_cast<size_t>(h) * Di + i] * k[i];
        s = warp_sum(s);
        acc += fmaxf(s, 0.f) * w[h];
    }
    if (lane == 0) out[t] = acc * wscale;
}

// one block of n threads (n <= 1024): the k largest of s + bias, one argmax round each
__global__ void router_topk_k(const float * logits, const float * bias, int n, int k, float scale, int * ids, float * w) {
    logits += static_cast<size_t>(blockIdx.x) * n;   // token blockIdx.x of a batch
    ids += blockIdx.x * k;
    w += blockIdx.x * k;
    __shared__ float sel[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ float wsum;
    const int i = threadIdx.x;
    const float x = i < n ? logits[i] : 0.f;
    // F.softplus (threshold 20: x itself above it), then sqrt
    const float sp = x > 20.f ? x : log1pf(expf(x));
    const float sc = sqrtf(sp);
    if (i < n) sel[i] = sc + bias[i];
    if (i == 0) wsum = 0.f;
    __syncthreads();
    for (int r = 0; r < k; ++r) {
        float v = i < n ? sel[i] : -FLT_MAX;
        int id = i;
        for (int o = 16; o > 0; o >>= 1) {   // argmax, the lower index on a tie (torch.topk's order on ties is not defined)
            const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
            const int id2 = __shfl_xor_sync(0xffffffffu, id, o);
            if (v2 > v || (v2 == v && id2 < id)) { v = v2; id = id2; }
        }
        if ((i & 31) == 0) { bv[i >> 5] = v; bi[i >> 5] = id; }
        __syncthreads();
        if (i == 0) {
            float best = -FLT_MAX;
            int b = 0;
            for (int j = 0; j < (static_cast<int>(blockDim.x) + 31) / 32; ++j)
                if (bv[j] > best || (bv[j] == best && bi[j] < b)) { best = bv[j]; b = bi[j]; }
            ids[r] = b;
            sel[b] = -FLT_MAX;
        }
        __syncthreads();
        if (i == ids[r]) { w[r] = sc; atomicAdd(&wsum, sc); }
        __syncthreads();
    }
    if (i < k) w[i] = w[i] / (wsum + 1e-20f) * scale;
}

// order-preserving float -> uint32 (larger float, larger key)
__device__ uint32_t fkey(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// one block of 1024 threads; thread i owns the contiguous chunk [i * per, (i+1) * per)
__global__ void topk_sorted_k(const float * score, int n, int k, int * out) {
    __shared__ uint32_t sT;
    __shared__ int cnt;
    __shared__ int scan[1024];
    const int tid = threadIdx.x, nt = blockDim.x, per = (n + nt - 1) / nt;
    const int b0 = min(n, tid * per), b1 = min(n, b0 + per);
    if (tid == 0) sT = 0;
    __syncthreads();
    // the k-th largest key: the largest T with #(key >= T) >= k, one bit at a time from the top
    for (int bit = 31; bit >= 0; --bit) {
        const uint32_t t = sT | (1u << bit);
        int c = 0;
        for (int i = b0; i < b1; ++i) c += fkey(score[i]) >= t;
        if (tid == 0) cnt = 0;
        __syncthreads();
        atomicAdd(&cnt, c);
        __syncthreads();
        if (tid == 0 && cnt >= k) sT = t;
        __syncthreads();
    }
    const uint32_t T = sT;
    int gt = 0;
    for (int i = b0; i < b1; ++i) gt += fkey(score[i]) > T;
    if (tid == 0) cnt = 0;
    __syncthreads();
    atomicAdd(&cnt, gt);
    __syncthreads();
    const int need_eq = k - cnt;   // equal keys to take, lowest indices first
    // ordered scan: equal keys before mine, then the takes before mine
    int eq = 0;
    for (int i = b0; i < b1; ++i) eq += fkey(score[i]) == T;
    scan[tid] = eq;
    __syncthreads();
    for (int o = 1; o < nt; o <<= 1) {   // inclusive Hillis-Steele scan
        const int v = tid >= o ? scan[tid - o] : 0;
        __syncthreads();
        scan[tid] += v;
        __syncthreads();
    }
    int eq_before = scan[tid] - eq;
    __syncthreads();
    int take = 0;
    {
        int e = eq_before;
        for (int i = b0; i < b1; ++i) {
            const uint32_t kk = fkey(score[i]);
            if (kk > T) ++take;
            else if (kk == T && e++ < need_eq) ++take;
        }
    }
    scan[tid] = take;
    __syncthreads();
    for (int o = 1; o < nt; o <<= 1) {
        const int v = tid >= o ? scan[tid - o] : 0;
        __syncthreads();
        scan[tid] += v;
        __syncthreads();
    }
    int w = scan[tid] - take, e = eq_before;
    for (int i = b0; i < b1; ++i) {
        const uint32_t kk = fkey(score[i]);
        if (kk > T || (kk == T && e++ < need_eq)) out[w++] = i;
    }
}

__global__ void swiglu_clamp_k(const float * g, const float * u, float * out, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gg = fminf(g[i], limit), uu = fminf(fmaxf(u[i], -limit), limit);
    out[i] = gg / (1.f + expf(-gg)) * uu;
}

// ---------------------------------------------------------------- batched (prefill) versions

__global__ void rope_tail_rows_k(float * x, int heads, int D, int n_rot, int pos0, int step, const float * freq, float sign) {
    const int t = blockIdx.y, h = blockIdx.x, i = threadIdx.x;
    if (i >= n_rot / 2) return;
    float * p = x + (static_cast<size_t>(t) * heads + h) * D + (D - n_rot) + 2 * i;
    const float ang = static_cast<float>(pos0 + t * step) * freq[i];
    float sn, cs;
    sincosf(ang, &sn, &cs);
    sn *= sign;
    const float a = p[0], b = p[1];
    p[0] = a * cs - b * sn;
    p[1] = a * sn + b * cs;
}

__global__ void hc_pre_rows_k(const float * R, const float * pre, float * out, int hc, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (d >= D) return;
    const float * r = R + static_cast<size_t>(t) * hc * D;
    float s = 0.f;
    for (int j = 0; j < hc; ++j) s += pre[t * hc + j] * r[static_cast<size_t>(j) * D + d];
    out[static_cast<size_t>(t) * D + d] = s;
}

__global__ void hc_post_rows_k(const float * h, const float * res, const float * post, const float * comb, float * out, int hc,
                               int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, k = blockIdx.y, t = blockIdx.z;
    if (d >= D) return;
    const float * r = res + static_cast<size_t>(t) * hc * D;
    const float * cm = comb + static_cast<size_t>(t) * hc * hc;
    float s = post[t * hc + k] * h[static_cast<size_t>(t) * D + d];
    for (int j = 0; j < hc; ++j) s += cm[j * hc + k] * r[static_cast<size_t>(j) * D + d];
    out[(static_cast<size_t>(t) * hc + k) * D + d] = s;
}

__global__ void indexer_score_rows_k(const float * q, const float * w, float wscale, const float * keys, const int * n_vis, int H,
                                     int Di, float * out, int ld) {
    const int t = blockIdx.y, j = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (j >= n_vis[t]) return;
    const float * k = keys + static_cast<size_t>(j) * Di;
    const float * qt = q + static_cast<size_t>(t) * H * Di;
    float acc = 0.f;
    for (int h = 0; h < H; ++h) {
        float sum = 0.f;
        for (int i = lane; i < Di; i += 32) sum += qt[static_cast<size_t>(h) * Di + i] * k[i];
        sum = warp_sum(sum);
        acc += fmaxf(sum, 0.f) * w[t * H + h];
    }
    if (lane == 0) out[static_cast<size_t>(t) * ld + j] = acc * wscale;
}

__global__ void attention_sink_rows_k(const float * q, const float * A, const int * a0, const int * na, const float * B,
                                      const int * idx_b, int ldb, const int * nb, const float * sink, float * out, int H, int D,
                                      float scale) {
    extern __shared__ float sm[];
    const int h = blockIdx.x, t = blockIdx.y;
    const int n_a = na[t], n_b = nb[t], base_a = a0[t], n = n_a + n_b;
    const int * ib = idx_b + static_cast<size_t>(t) * ldb;
    float * qs = sm;
    float * sc = sm + D;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const float * qh = q + (static_cast<size_t>(t) * H + h) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) qs[i] = qh[i];
    __syncthreads();
    auto key = [&](int j) -> const float * {
        return j < n_a ? A + static_cast<size_t>(base_a + j) * D : B + static_cast<size_t>(ib[j - n_a]) * D;
    };
    for (int j = warp; j < n; j += nw) {
        const float * k = key(j);
        float sum = 0.f;
        for (int i = lane; i < D; i += 32) sum += qs[i] * k[i];
        sum = warp_sum(sum);
        if (lane == 0) sc[j] = sum * scale;
    }
    __syncthreads();
    float mx = sink[h];
    for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, sc[j]);
    mx = block_max(mx);
    float z = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) z += (sc[j] = expf(sc[j] - mx));
    z = block_sum(z) + expf(sink[h] - mx);
    __syncthreads();
    const float inv = 1.f / z;
    float * o = out + (static_cast<size_t>(t) * H + h) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < n; ++j) acc += sc[j] * key(j)[i];
        o[i] = acc * inv;
    }
}

__global__ void pool2_rows_k(const float * kv, const float * score, float * out, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, g = blockIdx.y;
    if (d >= D) return;
    const float * k0 = kv + static_cast<size_t>(2 * g) * D, * s0 = score + static_cast<size_t>(2 * g) * D;
    const float a = s0[d], b = s0[D + d], mx = fmaxf(a, b);
    const float e0 = expf(a - mx), e1 = expf(b - mx);
    out[static_cast<size_t>(g) * D + d] = (k0[d] * e0 + k0[D + d] * e1) / (e0 + e1);
}

}  // namespace

void rope_tail(float * x, int heads, int D, int n_rot, int pos, const float * freq, bool inverse, cudaStream_t s) {
    rope_tail_k<<<heads, 32 * ((n_rot / 2 + 31) / 32), 0, s>>>(x, D, n_rot, pos, freq, inverse ? -1.f : 1.f);
}

void hc_split(const float * m, const float * R, const float * scale, const float * base, int hc, int D, float norm_eps,
              float hc_eps, int iters, float * pre, float * post, float * comb, cudaStream_t s) {
    hc_split_k<<<1, 256, 0, s>>>(m, R, scale, base, hc, hc * D, norm_eps, hc_eps, iters, pre, post, comb);
}

namespace {
// one row per block, 1024 threads, float4 (the shared rmsnorm_rows: 256 threads, 20 dependent loads each on a 5120 row)
__global__ void rmsnorm1_k(const float * x0, float * y0, int D0, const float * w0, const float * x1, float * y1, int D1,
                           const float * w1, float eps) {
    const float * x = blockIdx.x ? x1 : x0;
    float * y = blockIdx.x ? y1 : y0;
    const float * w = blockIdx.x ? w1 : w0;
    const int D = blockIdx.x ? D1 : D0;
    const float4 * x4 = reinterpret_cast<const float4 *>(x);
    float ss = 0.f;
    for (int i = threadIdx.x; i < D / 4; i += blockDim.x) {
        const float4 v = x4[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / D + eps);
    const float4 * w4 = reinterpret_cast<const float4 *>(w);
    float4 * y4 = reinterpret_cast<float4 *>(y);
    for (int i = threadIdx.x; i < D / 4; i += blockDim.x) {
        const float4 v = x4[i], g = w4[i];
        y4[i] = make_float4(v.x * sc * g.x, v.y * sc * g.y, v.z * sc * g.z, v.w * sc * g.w);
    }
}

struct CopySegs { uint32_t * dst[4]; const uint32_t * src[4]; int words[4]; int n; };
__global__ void copy_segs_k(CopySegs c) {   // blockIdx.y: the segment
    if (static_cast<int>(blockIdx.y) >= c.n) return;
    const int k = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < c.words[k]; i += gridDim.x * blockDim.x) c.dst[k][i] = c.src[k][i];
}

__global__ void copy_k(uint32_t * __restrict__ dst, const uint32_t * __restrict__ src, size_t words) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < words; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        dst[i] = src[i];
}
}  // namespace

void rmsnorm(const float * x, float * y, int D, const float * w, float eps, cudaStream_t s) {
    if (D % 4) throw std::runtime_error("rmsnorm: D not a multiple of 4");
    rmsnorm1_k<<<1, 1024, 0, s>>>(x, y, D, w, x, y, D, w, eps);
}

void rmsnorm2(const float * x0, float * y0, int D0, const float * w0, const float * x1, float * y1, int D1, const float * w1,
              float eps, cudaStream_t s) {
    if (D0 % 4 || D1 % 4) throw std::runtime_error("rmsnorm2: D not a multiple of 4");
    rmsnorm1_k<<<2, 1024, 0, s>>>(x0, y0, D0, w0, x1, y1, D1, w1, eps);
}

void copy_small_n(void * const * dst, const void * const * src, const size_t * bytes, int n, cudaStream_t s) {
    if (n < 1 || n > 4) throw std::runtime_error("copy_small_n: 1 to 4 segments");
    CopySegs c{};
    int most = 0;
    for (int k = 0; k < n; ++k) {
        if (bytes[k] % 4) throw std::runtime_error("copy_small_n: not whole words");
        c.dst[k] = static_cast<uint32_t *>(dst[k]);
        c.src[k] = static_cast<const uint32_t *>(src[k]);
        c.words[k] = static_cast<int>(bytes[k] / 4);
        most = std::max(most, c.words[k]);
    }
    c.n = n;
    copy_segs_k<<<dim3(std::max(1, std::min((most + 255) / 256, 16)), n), 256, 0, s>>>(c);
}

void copy_small(void * dst, const void * src, size_t bytes, cudaStream_t s) {
    if (bytes % 4) throw std::runtime_error("copy_small: not whole words");
    const size_t w = bytes / 4;
    if (w == 0) return;
    const int blocks = static_cast<int>(std::min<size_t>((w + 255) / 256, 64));
    copy_k<<<blocks, 256, 0, s>>>(static_cast<uint32_t *>(dst), static_cast<const uint32_t *>(src), w);
}

void hc_fn_split(const float * fn, int M, const float * R, const float * scale, const float * base, int hc, int D,
                 float norm_eps, float hc_eps, int iters, float * mix, float * ss, float * pre, float * post, float * comb,
                 cudaStream_t s) {
    static unsigned * done = nullptr;   // the blocks finished; zero between launches
    if (!done) {
        if (cudaMalloc(&done, sizeof(unsigned)) != cudaSuccess || cudaMemset(done, 0, sizeof(unsigned)) != cudaSuccess)
            throw std::runtime_error("hc_fn_split: counter");
    }
    hc_fn_k<<<M, 512, 0, s>>>(fn, hc * D, R, mix, ss, done, scale, base, hc, norm_eps, hc_eps, iters, pre, post, comb);
}

void hc_pre(const float * R, const float * pre, float * out, int hc, int D, cudaStream_t s) {
    hc_pre_k<<<(D + 255) / 256, 256, 0, s>>>(R, pre, out, hc, D);
}

void hc_post(const float * h, const float * res, const float * post, const float * comb, float * out, int hc, int D,
             cudaStream_t s) {
    hc_post_k<<<dim3((D + 255) / 256, hc), 256, 0, s>>>(h, res, post, comb, out, hc, D);
}

void engram_gate(float * R, const float * kv, const float * qk, int hc, int D, float eps, cudaStream_t s) {
    engram_gate_k<<<hc, 256, 0, s>>>(R, kv, qk, hc, D, eps);
}

void attention_sink(const float * q, const float * A, const int * idx_a, int n_a, const float * B, const int * idx_b,
                    int n_b, const float * sink, float * out, int H, int D, float scale, cudaStream_t s) {
    static const bool old = std::getenv("BL_DSV41_ATTN_OLD") != nullptr;
    const int n = n_a + n_b;
    if (old || H % kHG || D % 32 || n == 0) {
        const size_t smem = sizeof(float) * (D + n_a + n_b);
        attention_sink_k<<<H, 256, smem, s>>>(q, A, idx_a, n_a, B, idx_b, n_b, sink, out, D, scale);
        return;
    }
    const int S = (n + kKC - 1) / kKC;
    // the chunks' partial results: grown when needed, reused in stream order
    static float * scratch = nullptr;
    static size_t have = 0;
    const size_t need = static_cast<size_t>(H) * S * (D + 2);
    if (need > have) {
        if (scratch) { cudaStreamSynchronize(s); cudaFree(scratch); }
        if (cudaMalloc(&scratch, need * sizeof(float)) != cudaSuccess) throw std::runtime_error("attention scratch");
        have = need;
    }
    float * om = scratch, * ol = scratch + static_cast<size_t>(H) * S, * oo = scratch + 2 * static_cast<size_t>(H) * S;
    const size_t smem = sizeof(float) * (kHG * D + kHG * kKC);
    attn_split_k<<<dim3(H / kHG, S), 256, smem, s>>>(q, A, idx_a, n_a, B, idx_b, n_b, D, scale, S, om, ol, oo);
    attn_combine_k<<<H, 256, 0, s>>>(om, ol, oo, sink, out, D, S);
}

void pool2(const float * kv, const float * score, float * out, int D, cudaStream_t s) {
    pool2_k<<<(D + 255) / 256, 256, 0, s>>>(kv, score, out, D);
}

void indexer_score(const float * q, const float * w, float wscale, const float * keys, int N, int H, int Di, float * out,
                   cudaStream_t s) {
    if (N <= 0) return;
    indexer_score_k<<<(N + 7) / 8, 256, 0, s>>>(q, w, wscale, keys, N, H, Di, out);
}

void router_topk(const float * logits, const float * bias, int n, int k, float scale, int * ids, float * w, cudaStream_t s) {
    router_topk_k<<<1, 32 * ((n + 31) / 32), 0, s>>>(logits, bias, n, k, scale, ids, w);
}

void topk_sorted(const float * score, int n, int k, int * out, cudaStream_t s) {
    if (n <= 0 || k <= 0) return;
    topk_sorted_k<<<1, 1024, 0, s>>>(score, n, k < n ? k : n, out);
}

void swiglu_clamp(const float * g, const float * u, float * out, int n, float limit, cudaStream_t s) {
    swiglu_clamp_k<<<(n + 255) / 256, 256, 0, s>>>(g, u, out, n, limit);
}

void rope_tail_rows(float * x, int T, int heads, int D, int n_rot, int pos0, int step, const float * freq, bool inverse,
                    cudaStream_t s) {
    if (T > 0) rope_tail_rows_k<<<dim3(heads, T), 32 * ((n_rot / 2 + 31) / 32), 0, s>>>(x, heads, D, n_rot, pos0, step, freq, inverse ? -1.f : 1.f);
}
void hc_split_rows(const float * m, const float * R, const float * scale, const float * base, int hc, int D, int T,
                   float norm_eps, float hc_eps, int iters, float * pre, float * post, float * comb, cudaStream_t s) {
    if (T > 0) hc_split_k<<<T, 256, 0, s>>>(m, R, scale, base, hc, hc * D, norm_eps, hc_eps, iters, pre, post, comb);
}
void hc_pre_rows(const float * R, const float * pre, float * out, int hc, int D, int T, cudaStream_t s) {
    if (T > 0) hc_pre_rows_k<<<dim3((D + 255) / 256, T), 256, 0, s>>>(R, pre, out, hc, D);
}
void hc_post_rows(const float * h, const float * res, const float * post, const float * comb, float * out, int hc, int D,
                  int T, cudaStream_t s) {
    if (T > 0) hc_post_rows_k<<<dim3((D + 255) / 256, hc, T), 256, 0, s>>>(h, res, post, comb, out, hc, D);
}
void engram_gate_rows(float * R, const float * kv, const float * qk, int hc, int D, int T, float eps, cudaStream_t s) {
    if (T > 0) engram_gate_k<<<dim3(hc, T), 256, 0, s>>>(R, kv, qk, hc, D, eps);
}
void router_topk_rows(const float * logits, const float * bias, int n, int k, float scale, int T, int * ids, float * w,
                      cudaStream_t s) {
    if (T > 0) router_topk_k<<<T, 32 * ((n + 31) / 32), 0, s>>>(logits, bias, n, k, scale, ids, w);
}
void indexer_score_rows(const float * q, const float * w, float wscale, const float * keys, const int * n_vis, int n_max,
                        int H, int Di, int T, float * out, int ld, cudaStream_t s) {
    if (T > 0 && n_max > 0) indexer_score_rows_k<<<dim3((n_max + 7) / 8, T), 256, 0, s>>>(q, w, wscale, keys, n_vis, H, Di, out, ld);
}
void attention_sink_rows(const float * q, const float * A, const int * a0, const int * na, const float * B, const int * idx_b,
                         int ldb, const int * nb, int n_max, const float * sink, float * out, int H, int D, int T, float scale,
                         cudaStream_t s) {
    if (T > 0)
        attention_sink_rows_k<<<dim3(H, T), 256, sizeof(float) * (D + n_max), s>>>(q, A, a0, na, B, idx_b, ldb, nb, sink, out, H, D, scale);
}
void pool2_rows(const float * kv, const float * score, float * out, int D, int n, cudaStream_t s) {
    if (n > 0) pool2_rows_k<<<dim3((D + 255) / 256, n), 256, 0, s>>>(kv, score, out, D);
}

}  // namespace bl::cuda::dsv41
