// bench_dsv41_experts: DeepSeek-V4.1's routed experts (IQ2_XXS, D 5120, F 2304) on real tensors of whatever shards are
// on disk: correctness of the GPU matvec and of CpuExperts against a double-precision reference, then the time of one
// expert on the GPU (resident, cold L2: a different expert each time) and of one layer's 6 on the CPU (cold L3).
//   bench_dsv41_experts SHARD1.gguf [n_gpu_experts=256] [cpu_threads=8]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bl/cpu_experts.h"
#include "bl/cuda_ops.h"
#include "bl/cuda_quant.h"
#include "bl/dsv41_ops.h"
#include "bl/gguf.h"

namespace {

void ck(cudaError_t e, const char * w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s SHARD1.gguf [n_gpu_experts] [cpu_threads]\n", argv[0]); return 2; }
    const int n_gpu = argc > 2 ? std::atoi(argv[2]) : 256, n_thr = argc > 3 ? std::atoi(argv[3]) : 8;
    bl::GgufModel g(argv[1]);
    const int D = 5120, F = 2304, E = 384;
    const float lim = 10.f;

    // the layers whose three expert tensors are all on disk
    std::vector<int> layers;
    for (int il = 0; il < 40; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const bl::Tensor * a = g.find(p + "ffn_gate_exps.weight"), * b = g.find(p + "ffn_up_exps.weight"), * c = g.find(p + "ffn_down_exps.weight");
        if (a && b && c && a->data && b->data && c->data) layers.push_back(il);
    }
    if (layers.empty()) { std::fprintf(stderr, "no complete expert layer on disk\n"); return 1; }
    std::printf("expert layers on disk: %zu (%d..%d)\n", layers.size(), layers.front(), layers.back());
    const uint32_t T = g.at("blk." + std::to_string(layers[0]) + ".ffn_gate_exps.weight").type_id;
    const size_t gu_rb = bl::row_bytes(T, D), d_rb = bl::row_bytes(T, F);
    const size_t gu_b = gu_rb * F, d_b = d_rb * D, per = 2 * gu_b + d_b;
    std::printf("format %s: %.2f MB per expert\n", bl::type_geometry(T).name, per / 1e6);

    auto expert_ptr = [&](int il, int e, int part) -> const uint8_t * {
        const std::string p = "blk." + std::to_string(il) + ".";
        const char * n[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        return g.at(p + n[part]).data + static_cast<size_t>(e) * (part < 2 ? gu_b : d_b);
    };

    // an activation like the model's: rms-normalized, so dots land in the range the clamps were made for
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(D);
    for (auto & v : x) v = nd(rng);

    // ---- correctness on one expert: double reference from the dequantized rows
    const int il0 = layers[0], e0 = 17;
    std::vector<uint8_t> blob(per);
    std::memcpy(blob.data(), expert_ptr(il0, e0, 0), gu_b);
    std::memcpy(blob.data() + gu_b, expert_ptr(il0, e0, 1), gu_b);
    std::memcpy(blob.data() + 2 * gu_b, expert_ptr(il0, e0, 2), d_b);
    uint8_t * dblob;
    ck(cudaMalloc(&dblob, per), "malloc");
    ck(cudaMemcpy(dblob, blob.data(), per, cudaMemcpyHostToDevice), "copy");
    float * dw;
    ck(cudaMalloc(&dw, sizeof(float) * D * F), "malloc");
    std::vector<float> wg(static_cast<size_t>(F) * D), wu(wg.size()), wd(static_cast<size_t>(D) * F);
    bl::cuda::dequant_rows(T, dblob, gu_rb, F, D, dw, 0);
    ck(cudaMemcpy(wg.data(), dw, wg.size() * 4, cudaMemcpyDeviceToHost), "d2h");
    bl::cuda::dequant_rows(T, dblob + gu_b, gu_rb, F, D, dw, 0);
    ck(cudaMemcpy(wu.data(), dw, wu.size() * 4, cudaMemcpyDeviceToHost), "d2h");
    bl::cuda::dequant_rows(T, dblob + 2 * gu_b, d_rb, D, F, dw, 0);
    ck(cudaMemcpy(wd.data(), dw, wd.size() * 4, cudaMemcpyDeviceToHost), "d2h");
    std::vector<double> act(F), ref(D, 0.0);
    double gmax = 0, umax = 0;
    for (int r = 0; r < F; ++r) {
        double gs = 0, us = 0;
        for (int k = 0; k < D; ++k) { gs += static_cast<double>(wg[static_cast<size_t>(r) * D + k]) * x[k]; us += static_cast<double>(wu[static_cast<size_t>(r) * D + k]) * x[k]; }
        gmax = std::max(gmax, std::fabs(gs)); umax = std::max(umax, std::fabs(us));
        gs = std::min(gs, 10.0); us = std::clamp(us, -10.0, 10.0);
        act[r] = gs / (1.0 + std::exp(-gs)) * us;
    }
    for (int r = 0; r < D; ++r) {
        double s = 0;
        for (int k = 0; k < F; ++k) s += static_cast<double>(wd[static_cast<size_t>(r) * F + k]) * act[k];
        ref[r] = s;
    }
    auto rel = [&](const std::vector<float> & o) {
        double num = 0, den = 0;
        for (int i = 0; i < D; ++i) { num += (o[i] - ref[i]) * (o[i] - ref[i]); den += ref[i] * ref[i]; }
        return std::sqrt(num / den);
    };
    std::printf("layer %d expert %d: max |gate.x| %.2f, max |up.x| %.2f (clamp at 10)\n", il0, e0, gmax, umax);

    float * dx, * dg, * du, * da, * dout;
    ck(cudaMalloc(&dx, D * 4), "m"); ck(cudaMalloc(&dg, F * 4), "m"); ck(cudaMalloc(&du, F * 4), "m");
    ck(cudaMalloc(&da, F * 4), "m"); ck(cudaMalloc(&dout, D * 4), "m");
    ck(cudaMemcpy(dx, x.data(), D * 4, cudaMemcpyHostToDevice), "x");
    auto gpu_expert = [&](const uint8_t * b, cudaStream_t s) {
        bl::cuda::matvec(T, b, gu_rb, F, D, dx, dg, s);
        bl::cuda::matvec(T, b + gu_b, gu_rb, F, D, dx, du, s);
        bl::cuda::dsv41::swiglu_clamp(dg, du, da, F, lim, s);
        bl::cuda::matvec(T, b + 2 * gu_b, d_rb, D, F, da, dout, s);
    };
    gpu_expert(dblob, 0);
    std::vector<float> og(D);
    ck(cudaMemcpy(og.data(), dout, D * 4, cudaMemcpyDeviceToHost), "out");
    std::printf("GPU matvec (fp32 activations)   rel err %.2e\n", rel(og));

    bl::CpuExperts cpu(n_thr);
    std::vector<float> oc(D, 0.f);
    bl::CpuExperts::Job job{blob.data(), 1, oc.data()};
    cpu.run(T, T, gu_b, F, D, x.data(), 1, &job, 1, lim);
    std::printf("CPU ggml dots (q8_K activations) rel err %.2e\n", rel(oc));

    // ---- GPU speed: n_gpu experts resident, each launch a different one (cold L2)
    const int ng = std::min<int>(n_gpu, static_cast<int>(layers.size()) * E);
    uint8_t * arena;
    ck(cudaMalloc(&arena, per * ng), "arena");
    std::vector<uint8_t> tmp(per);
    for (int i = 0; i < ng; ++i) {
        const int il = layers[i % layers.size()], e = (i * 37) % E;
        ck(cudaMemcpy(arena + per * i, expert_ptr(il, e, 0), gu_b, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(arena + per * i + gu_b, expert_ptr(il, e, 1), gu_b, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(arena + per * i + 2 * gu_b, expert_ptr(il, e, 2), d_b, cudaMemcpyHostToDevice), "up");
    }
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < ng; ++i) gpu_expert(arena + per * i, 0);   // warm-up
    cudaEventRecord(a);
    const int reps = 3;
    for (int r = 0; r < reps; ++r)
        for (int i = 0; i < ng; ++i) gpu_expert(arena + per * i, 0);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    const double us_e = 1000.0 * ms / (reps * ng);
    std::printf("GPU: %.1f us per expert (%.0f GB/s), %d resident, cold L2 -> %.2f ms per layer of 6\n", us_e,
                per / (us_e * 1e3), ng, 6 * us_e / 1000);

    // ---- PCIe: one expert from pinned host memory
    uint8_t * pin;
    ck(cudaMallocHost(&pin, per * 6), "pinned");
    for (int k = 0; k < 6; ++k) std::memcpy(pin + per * k, arena ? blob.data() : blob.data(), per);
    cudaEventRecord(a);
    for (int r = 0; r < 10; ++r) ck(cudaMemcpyAsync(arena, pin, per * 6, cudaMemcpyHostToDevice, 0), "h2d");
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms, a, b);
    std::printf("PCIe: 6 experts pinned -> VRAM %.2f ms (%.1f GB/s)\n", ms / 10, 6.0 * per * 10 / (ms * 1e6));

    // ---- CPU speed: one layer's 6 experts at T = 1 (and 4), cycling over 64 different experts (cold L3)
    const int nc = std::min<int>(64, static_cast<int>(layers.size()) * E);
    std::vector<std::vector<uint8_t>> host(nc, std::vector<uint8_t>(per));
    for (int i = 0; i < nc; ++i) {
        const int il = layers[i % layers.size()], e = (i * 53 + 5) % E;
        std::memcpy(host[i].data(), expert_ptr(il, e, 0), gu_b);
        std::memcpy(host[i].data() + gu_b, expert_ptr(il, e, 1), gu_b);
        std::memcpy(host[i].data() + 2 * gu_b, expert_ptr(il, e, 2), d_b);
    }
    for (int Tt : {1, 4}) {
        std::vector<float> xs(static_cast<size_t>(Tt) * D), outs(static_cast<size_t>(6) * Tt * D);
        for (int t = 0; t < Tt; ++t) std::copy(x.begin(), x.end(), xs.begin() + t * D);
        std::vector<bl::CpuExperts::Job> jobs(6);
        double best = 1e9;
        for (int r = 0; r < 4; ++r) {
            const double t0 = now();
            for (int s = 0; s < nc / 6; ++s) {
                for (int k = 0; k < 6; ++k) jobs[k] = {host[s * 6 + k].data(), (1 << Tt) - 1, outs.data() + static_cast<size_t>(k) * Tt * D};
                cpu.run(T, T, gu_b, F, D, xs.data(), Tt, jobs.data(), 6, lim);
            }
            best = std::min(best, (now() - t0) / (nc / 6));
        }
        std::printf("CPU (%d threads): %.2f ms per layer of 6 at T=%d (%.1f GB/s)\n", n_thr, best * 1e3, Tt, 6.0 * per / (best * 1e9));
    }
    return 0;
}
