// test_dsv41_ops: each deepseek41 kernel against a plain CPU transcription of model.py / kernel.py on random data.
// No model needed; a few seconds. Prints the max abs error per kernel and fails above the tolerance.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "bl/dsv41_ops.h"

using namespace bl::cuda::dsv41;
using Vec = std::vector<float>;

namespace {

std::mt19937 rng(1234);
Vec rnd(size_t n, float s = 1.f) {
    std::normal_distribution<float> d(0.f, s);
    Vec v(n);
    for (auto & x : v) x = d(rng);
    return v;
}
template <typename T> T * dev(const std::vector<T> & h) {
    T * d = nullptr;
    cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <typename T> std::vector<T> host(const T * d, size_t n) {
    std::vector<T> h(n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
int fails = 0;
void check(const char * name, const Vec & got, const Vec & want, float tol) {
    float m = 0.f, ref = 0.f;
    for (size_t i = 0; i < got.size(); ++i) {
        m = std::max(m, std::fabs(got[i] - want[i]));
        ref = std::max(ref, std::fabs(want[i]));
    }
    const bool ok = m <= tol * std::max(ref, 1.f);
    std::printf("%-16s max |err| %.3g (max |ref| %.3g) %s\n", name, m, ref, ok ? "ok" : "FAIL");
    if (!ok) ++fails;
}
float sigm(float x) { return 1.f / (1.f + std::exp(-x)); }

}  // namespace

int main() {
    const int D = 512, H = 64, NR = 64;

    {   // rope_tail: forward against complex multiplication, then inverse undoes it
        Vec x = rnd(H * D), f(NR / 2);
        for (int i = 0; i < NR / 2; ++i) f[i] = 1.f / std::pow(10000.f, 2.f * i / NR);
        const int pos = 777;
        Vec want = x;
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < NR / 2; ++i) {
                float * p = want.data() + h * D + D - NR + 2 * i;
                const float a = pos * f[i], c = std::cos(a), s = std::sin(a), x0 = p[0], x1 = p[1];
                p[0] = x0 * c - x1 * s;
                p[1] = x0 * s + x1 * c;
            }
        float * dx = dev(x), * df = dev(f);
        rope_tail(dx, H, D, NR, pos, df, false, 0);
        check("rope_tail", host(dx, x.size()), want, 1e-4f);
        rope_tail(dx, H, D, NR, pos, df, true, 0);
        check("rope inverse", host(dx, x.size()), x, 1e-4f);
    }
    {   // hc_split: hc_split_sinkhorn_kernel transcribed
        const int hc = 4, E = 5120, n = hc * E, mix = (2 + hc) * hc, iters = 20;
        const float eps = 1e-6f, neps = 1e-20f;
        Vec R = rnd(n), m = rnd(mix, 30.f), scale = {0.5f, 0.7f, 0.9f}, base = rnd(mix);
        double ss = 0;
        for (float v : R) ss += v * v;
        const float r = 1.f / std::sqrt(static_cast<float>(ss / n) + neps);
        Vec pre(hc), post(hc), comb(hc * hc);
        for (int j = 0; j < hc; ++j) {
            pre[j] = sigm(m[j] * r * scale[0] + base[j]) + eps;
            post[j] = 2 * sigm(m[hc + j] * r * scale[1] + base[hc + j]);
        }
        float c[4][4];
        for (int j = 0; j < hc; ++j)
            for (int k = 0; k < hc; ++k) c[j][k] = m[2 * hc + j * hc + k] * r * scale[2] + base[2 * hc + j * hc + k];
        for (int j = 0; j < hc; ++j) {
            float mx = *std::max_element(c[j], c[j] + hc), s = 0;
            for (int k = 0; k < hc; ++k) s += (c[j][k] = std::exp(c[j][k] - mx));
            for (int k = 0; k < hc; ++k) c[j][k] = c[j][k] / s + eps;
        }
        auto colnorm = [&]() {
            for (int k = 0; k < hc; ++k) {
                float s = 0;
                for (int j = 0; j < hc; ++j) s += c[j][k];
                for (int j = 0; j < hc; ++j) c[j][k] /= s + eps;
            }
        };
        colnorm();
        for (int it = 1; it < iters; ++it) {
            for (int j = 0; j < hc; ++j) {
                float s = 0;
                for (int k = 0; k < hc; ++k) s += c[j][k];
                for (int k = 0; k < hc; ++k) c[j][k] /= s + eps;
            }
            colnorm();
        }
        for (int j = 0; j < hc; ++j)
            for (int k = 0; k < hc; ++k) comb[j * hc + k] = c[j][k];
        float * dR = dev(R), * dm = dev(m), * ds = dev(scale), * db = dev(base), * dp = dev(Vec(hc)), * dq = dev(Vec(hc)),
              * dc = dev(Vec(hc * hc));
        hc_split(dm, dR, ds, db, hc, E, neps, eps, iters, dp, dq, dc, 0);
        check("hc_split pre", host(dp, hc), pre, 1e-5f);
        check("hc_split post", host(dq, hc), post, 1e-5f);
        check("hc_split comb", host(dc, hc * hc), comb, 1e-5f);

        // hc_pre / hc_post with these coefficients
        Vec h = rnd(E), want_pre(E, 0.f), want_post(n);
        for (int d = 0; d < E; ++d)
            for (int j = 0; j < hc; ++j) want_pre[d] += pre[j] * R[j * E + d];
        for (int k = 0; k < hc; ++k)
            for (int d = 0; d < E; ++d) {
                float s = post[k] * h[d];
                for (int j = 0; j < hc; ++j) s += comb[j * hc + k] * R[j * E + d];
                want_post[k * E + d] = s;
            }
        float * dout = dev(Vec(E)), * dh = dev(h), * dpost = dev(Vec(n));
        hc_pre(dR, dp, dout, hc, E, 0);
        check("hc_pre", host(dout, E), want_pre, 1e-5f);
        hc_post(dh, dR, dq, dc, dpost, hc, E, 0);
        check("hc_post", host(dpost, n), want_post, 1e-5f);

        // engram_gate
        Vec kv = rnd((hc + 1) * E), qk = rnd(n, 0.3f), want = R;
        for (int cc = 0; cc < hc; ++cc) {
            double sx = 0, sk = 0, sd = 0;
            for (int d = 0; d < E; ++d) {
                const float xv = R[cc * E + d], kk = kv[cc * E + d];
                sx += xv * xv; sk += kk * kk; sd += xv * qk[cc * E + d] * kk;
            }
            const float dot = static_cast<float>(sd) / std::sqrt(static_cast<float>(sx / E) + neps) /
                              std::sqrt(static_cast<float>(sk / E) + neps) / std::sqrt(static_cast<float>(E));
            const float g = sigm(std::copysign(std::sqrt(std::max(std::fabs(dot), 1e-6f)), dot));
            for (int d = 0; d < E; ++d) want[cc * E + d] += g * kv[hc * E + d];
        }
        float * dkv = dev(kv), * dqk = dev(qk);
        engram_gate(dR, dkv, dqk, hc, E, neps, 0);
        check("engram_gate", host(dR, n), want, 1e-4f);
    }
    {   // attention_sink over a window of 100 and 300 picked compressed rows
        const int nA = 100, nrowsB = 1000, nB = 300;
        Vec q = rnd(H * D, 0.2f), A = rnd(128 * D), B = rnd(nrowsB * D), sink = rnd(H);
        std::vector<int> ia(nA), ib(nB);
        std::iota(ia.begin(), ia.end(), 0);
        for (int i = 0; i < nB; ++i) ib[i] = (i * 7) % nrowsB;
        const float scale = 1.f / std::sqrt(static_cast<float>(D));
        Vec want(H * D);
        for (int h = 0; h < H; ++h) {
            std::vector<double> s(nA + nB);
            for (int j = 0; j < nA + nB; ++j) {
                const float * k = j < nA ? &A[ia[j] * D] : &B[ib[j - nA] * D];
                double t = 0;
                for (int d = 0; d < D; ++d) t += q[h * D + d] * k[d];
                s[j] = t * scale;
            }
            double mx = *std::max_element(s.begin(), s.end()), z = std::exp(sink[h] - mx);
            for (auto & v : s) z += (v = std::exp(v - mx));
            for (int d = 0; d < D; ++d) {
                double o = 0;
                for (int j = 0; j < nA + nB; ++j) o += s[j] * (j < nA ? A[ia[j] * D + d] : B[ib[j - nA] * D + d]);
                want[h * D + d] = static_cast<float>(o / z);
            }
        }
        float * dq = dev(q), * dA = dev(A), * dB = dev(B), * dsk = dev(sink), * dout = dev(Vec(H * D));
        int * dia = dev(ia), * dib = dev(ib);
        attention_sink(dq, dA, dia, nA, dB, dib, nB, dsk, dout, H, D, scale, 0);
        check("attention_sink", host(dout, H * D), want, 1e-4f);
    }
    {   // pool2
        Vec kv = rnd(2 * D), sc = rnd(2 * D, 3.f), want(D);
        for (int d = 0; d < D; ++d) {
            const float mx = std::max(sc[d], sc[D + d]), e0 = std::exp(sc[d] - mx), e1 = std::exp(sc[D + d] - mx);
            want[d] = (kv[d] * e0 + kv[D + d] * e1) / (e0 + e1);
        }
        float * dkv = dev(kv), * dsc = dev(sc), * dout = dev(Vec(D));
        pool2(dkv, dsc, dout, D, 0);
        check("pool2", host(dout, D), want, 1e-5f);
    }
    {   // indexer_score
        const int Hi = 32, Di = 128, N = 777;
        const float ws = 0.0123f;
        Vec q = rnd(Hi * Di), w = rnd(Hi), k = rnd(N * Di), want(N);
        for (int t = 0; t < N; ++t) {
            double acc = 0;
            for (int h = 0; h < Hi; ++h) {
                double s = 0;
                for (int i = 0; i < Di; ++i) s += q[h * Di + i] * k[t * Di + i];
                acc += std::max(s, 0.0) * w[h];
            }
            want[t] = static_cast<float>(acc * ws);
        }
        float * dq = dev(q), * dw = dev(w), * dk = dev(k), * dout = dev(Vec(N));
        indexer_score(dq, dw, ws, dk, N, Hi, Di, dout, 0);
        check("indexer_score", host(dout, N), want, 1e-4f);
    }
    {   // router_topk: Gate.forward with sqrtsoftplus
        const int n = 384, k = 6;
        Vec l = rnd(n, 3.f), b = rnd(n, 0.1f), s(n);
        for (int i = 0; i < n; ++i) s[i] = std::sqrt(l[i] > 20.f ? l[i] : std::log1p(std::exp(l[i])));
        std::vector<int> id(n);
        std::iota(id.begin(), id.end(), 0);
        std::partial_sort(id.begin(), id.begin() + k, id.end(), [&](int a, int c) { return s[a] + b[a] > s[c] + b[c]; });
        Vec want_w(k), want_id(k);
        float sum = 0;
        for (int i = 0; i < k; ++i) sum += s[id[i]];
        for (int i = 0; i < k; ++i) { want_w[i] = s[id[i]] / (sum + 1e-20f) * 1.5f; want_id[i] = static_cast<float>(id[i]); }
        float * dl = dev(l), * db = dev(b), * dw = dev(Vec(k));
        int * did = dev(std::vector<int>(k));
        router_topk(dl, db, n, k, 1.5f, did, dw, 0);
        const auto gi = host(did, k);
        Vec gid(k);
        for (int i = 0; i < k; ++i) gid[i] = static_cast<float>(gi[i]);
        check("router ids", gid, want_id, 0.f);
        check("router weights", host(dw, k), want_w, 1e-5f);
    }
    {   // topk_sorted against nth_element + sort (ties: lower index first), with many ties and n below / above k
        for (int n : {300, 512, 4000, 16384}) {
            Vec sc(n);
            std::uniform_int_distribution<int> q(0, 200);
            for (auto & v : sc) v = static_cast<float>(q(rng)) * 0.25f - 20.f;   // coarse: lots of equal scores
            const int k = std::min(512, n);
            std::vector<int> id(n);
            std::iota(id.begin(), id.end(), 0);
            std::nth_element(id.begin(), id.begin() + k - 1, id.end(), [&](int a, int b) { return sc[a] > sc[b] || (sc[a] == sc[b] && a < b); });
            id.resize(k);
            std::sort(id.begin(), id.end());
            float * ds = dev(sc);
            int * dout = dev(std::vector<int>(k));
            topk_sorted(ds, n, 512, dout, 0);
            const auto got = host(dout, k);
            Vec g(k), w(k);
            for (int i = 0; i < k; ++i) { g[i] = static_cast<float>(got[i]); w[i] = static_cast<float>(id[i]); }
            char name[32];
            std::snprintf(name, sizeof name, "topk n=%d", n);
            check(name, g, w, 0.f);
        }
    }
    {   // swiglu_clamp
        const int n = 2304;
        Vec g = rnd(n, 8.f), u = rnd(n, 8.f), want(n);
        for (int i = 0; i < n; ++i) {
            const float gg = std::min(g[i], 10.f), uu = std::clamp(u[i], -10.f, 10.f);
            want[i] = gg * sigm(gg) * uu;
        }
        float * dg = dev(g), * du = dev(u), * dout = dev(Vec(n));
        swiglu_clamp(dg, du, dout, n, 10.f, 0);
        check("swiglu_clamp", host(dout, n), want, 1e-5f);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    std::printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails ? 1 : 0;
}
