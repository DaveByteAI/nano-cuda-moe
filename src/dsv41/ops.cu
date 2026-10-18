#include "bl/dsv41_ops.h"

#include <cfloat>

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
__global__ void hc_split_k(const float * m, const float * R, const float * scale, const float * base, int hc, int n,
                           float norm_eps, float hc_eps, int iters, float * pre, float * post, float * comb) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += R[i] * R[i];
    ss = block_sum(ss);
    if (threadIdx.x != 0) return;
    const float r = rsqrtf(ss / n + norm_eps);
    for (int j = 0; j < hc; ++j) {
        pre[j]  = 1.f / (1.f + expf(-(m[j] * r * scale[0] + base[j]))) + hc_eps;
        post[j] = 2.f / (1.f + expf(-(m[hc + j] * r * scale[1] + base[hc + j])));
    }
    float c[8][8];   // hc <= 8
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k) c[j][k] = m[2 * hc + j * hc + k] * r * scale[2] + base[2 * hc + j * hc + k];
    // softmax over each row, + eps
    for (int j = 0; j < hc; ++j) {
        float mx = -FLT_MAX;
        for (int k = 0; k < hc; ++k) mx = fmaxf(mx, c[j][k]);
        float s = 0.f;
        for (int k = 0; k < hc; ++k) s += (c[j][k] = expf(c[j][k] - mx));
        for (int k = 0; k < hc; ++k) c[j][k] = c[j][k] / s + hc_eps;
    }
    auto cols = [&]() {
        for (int k = 0; k < hc; ++k) {
            float s = 0.f;
            for (int j = 0; j < hc; ++j) s += c[j][k];
            for (int j = 0; j < hc; ++j) c[j][k] /= s + hc_eps;
        }
    };
    cols();
    for (int it = 1; it < iters; ++it) {
        for (int j = 0; j < hc; ++j) {
            float s = 0.f;
            for (int k = 0; k < hc; ++k) s += c[j][k];
            for (int k = 0; k < hc; ++k) c[j][k] /= s + hc_eps;
        }
        cols();
    }
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k) comb[j * hc + k] = c[j][k];
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

__global__ void swiglu_clamp_k(const float * g, const float * u, float * out, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gg = fminf(g[i], limit), uu = fminf(fmaxf(u[i], -limit), limit);
    out[i] = gg / (1.f + expf(-gg)) * uu;
}

}  // namespace

void rope_tail(float * x, int heads, int D, int n_rot, int pos, const float * freq, bool inverse, cudaStream_t s) {
    rope_tail_k<<<heads, 32 * ((n_rot / 2 + 31) / 32), 0, s>>>(x, D, n_rot, pos, freq, inverse ? -1.f : 1.f);
}

void hc_split(const float * m, const float * R, const float * scale, const float * base, int hc, int D, float norm_eps,
              float hc_eps, int iters, float * pre, float * post, float * comb, cudaStream_t s) {
    hc_split_k<<<1, 256, 0, s>>>(m, R, scale, base, hc, hc * D, norm_eps, hc_eps, iters, pre, post, comb);
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
    const size_t smem = sizeof(float) * (D + n_a + n_b);
    attention_sink_k<<<H, 256, smem, s>>>(q, A, idx_a, n_a, B, idx_b, n_b, sink, out, D, scale);
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

void swiglu_clamp(const float * g, const float * u, float * out, int n, float limit, cudaStream_t s) {
    swiglu_clamp_k<<<(n + 255) / 256, 256, 0, s>>>(g, u, out, n, limit);
}

}  // namespace bl::cuda::dsv41
