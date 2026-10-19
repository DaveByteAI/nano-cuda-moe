// bench_dsv41_dense: DeepSeek-V4.1's dense matrices on the GPU, the phase-1 matvec against matvec_group (the decode
// path's fused kernel, dp4a for Q8_0), one token, cold L2 (each launch a different layer's copy of the matrix).
//   bench_dsv41_dense SHARD1.gguf
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bl/cuda_fused.h"
#include "bl/cuda_quant.h"
#include "bl/gguf.h"

namespace {
void ck(cudaError_t e, const char * w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}
}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s SHARD1.gguf\n", argv[0]); return 2; }
    bl::GgufModel g(argv[1]);
    const int L = 40;
    for (const char * name : {"attn_q_b.weight", "attn_output_b.weight", "attn_q_a.weight", "ffn_gate_shexp.weight",
                              "ffn_down_shexp.weight", "attn_output_a.weight", "hc_attn_fn.weight", "ffn_gate_inp.weight"}) {
        std::vector<void *> dW;
        uint32_t type = 0;
        int M = 0, K = 0;
        size_t rb = 0, bytes = 0;
        for (int il = 0; il < L; ++il) {
            const bl::Tensor & t = g.at("blk." + std::to_string(il) + "." + name);
            type = t.type_id;
            K = static_cast<int>(t.shape[0]);
            M = static_cast<int>(t.elements() / K);
            rb = bl::row_bytes(type, K);
            bytes = t.nbytes;
            void * d;
            ck(cudaMalloc(&d, t.nbytes), "malloc");
            ck(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice), "upload");
            dW.push_back(d);
        }
        std::vector<float> x(K);
        std::mt19937 rng(3);
        std::normal_distribution<float> nd;
        for (auto & v : x) v = nd(rng);
        float * dx, * y1, * y2;
        ck(cudaMalloc(&dx, K * 4), "m"); ck(cudaMalloc(&y1, M * 4), "m"); ck(cudaMalloc(&y2, M * 4), "m");
        ck(cudaMemcpy(dx, x.data(), K * 4, cudaMemcpyHostToDevice), "x");
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        auto time = [&](auto fn) {
            for (int il = 0; il < L; ++il) fn(il);
            cudaEventRecord(a);
            for (int r = 0; r < 5; ++r)
                for (int il = 0; il < L; ++il) fn(il);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms;
            cudaEventElapsedTime(&ms, a, b);
            return 1000.0 * ms / (5 * L);
        };
        const double t1 = time([&](int il) { bl::cuda::matvec(type, dW[il], rb, M, K, dx, y1, 0); });
        double t2 = -1;
        try {
            t2 = time([&](int il) {
                bl::cuda::MvGroup gr;
                gr.K = K;
                gr.add(type, dW[il], rb, M, y2);
                bl::cuda::matvec_group(gr, dx, 0);
            });
        } catch (const std::exception & e) {
            std::printf("%-22s group: %s\n", name, e.what());
            cudaMemcpy(y2, y1, M * 4, cudaMemcpyDeviceToDevice);
        }
        std::vector<float> h1(M), h2(M);
        ck(cudaMemcpy(h1.data(), y1, M * 4, cudaMemcpyDeviceToHost), "y");
        ck(cudaMemcpy(h2.data(), y2, M * 4, cudaMemcpyDeviceToHost), "y");
        double num = 0, den = 0;
        for (int i = 0; i < M; ++i) { num += (h1[i] - h2[i]) * (h1[i] - h2[i]); den += h1[i] * h1[i]; }
        std::printf("%-22s %-5s %6d x %-6d %6.2f MB: matvec %7.1f us (%4.0f GB/s), group %7.1f us (%4.0f GB/s), rel diff %.1e\n",
                    name, bl::type_geometry(type).name, M, K, bytes / 1e6, t1, bytes / (t1 * 1e3), t2, bytes / (t2 * 1e3),
                    std::sqrt(num / den));
        std::fflush(stdout);
        for (void * d : dW) cudaFree(d);
        cudaFree(dx); cudaFree(y1); cudaFree(y2);
    }
    {   // the output head (BF16, 1.3 GB): one copy
        const bl::Tensor & t = g.at("output.weight");
        const int K = static_cast<int>(t.shape[0]), M = static_cast<int>(t.elements() / K);
        const size_t rb = bl::row_bytes(t.type_id, K);
        void * d;
        float * dx, * y;
        ck(cudaMalloc(&d, t.nbytes), "m");
        ck(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice), "up");
        ck(cudaMalloc(&dx, K * 4), "m"); ck(cudaMalloc(&y, M * 4), "m");
        ck(cudaMemset(dx, 0, K * 4), "x");
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        for (int pass = 0; pass < 2; ++pass) {
            cudaEventRecord(a);
            for (int r = 0; r < 5; ++r) {
                if (pass == 0) bl::cuda::matvec(t.type_id, d, rb, M, K, dx, y, 0);
                else {
                    bl::cuda::MvGroup gr;
                    gr.K = K;
                    gr.add(t.type_id, d, rb, M, y);
                    bl::cuda::matvec_group(gr, dx, 0);
                }
            }
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms;
            cudaEventElapsedTime(&ms, a, b);
            std::printf("output.weight BF16 %d x %d: %s %.2f ms (%.0f GB/s)\n", M, K, pass ? "group " : "matvec", ms / 5,
                        t.nbytes / (ms / 5 * 1e6));
        }
    }
    return 0;
}
