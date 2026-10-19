// bench_dsv41_cpu: the CPU experts on cold memory, as in hybrid decode: real experts copied into a pool much larger
// than L3, `n` distinct experts per call, one token. Per expert time and the DRAM rate it implies.
//   bench_dsv41_cpu SHARD1.gguf [threads]
#include <sys/mman.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "bl/cpu_experts.h"
#include "bl/gguf.h"
#include "ggml-cpu.h"
#include "ggml.h"

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s SHARD1.gguf [threads]\n", argv[0]); return 2; }
    const int threads = argc > 2 ? std::atoi(argv[2]) : 8;
    bl::GgufModel g(argv[1]);
    ggml_cpu_init();
    const bl::Tensor & tg = g.at("blk.0.ffn_gate_exps.weight"), & tu = g.at("blk.0.ffn_up_exps.weight"),
                     & td = g.at("blk.0.ffn_down_exps.weight");
    const int D = static_cast<int>(tg.shape[0]), F = static_cast<int>(tg.shape[1]), E = static_cast<int>(tg.shape[2]);
    const size_t gu = tg.nbytes / E, dn = td.nbytes / E, per = 2 * gu + dn;
    // 256 experts, 2.3 GB: far beyond the 16 MB L3 (BL_POOL=1: one expert, hot in L3)
    const int pool_n = std::getenv("BL_POOL") ? std::atoi(std::getenv("BL_POOL")) : 256;
    void * mem = mmap(nullptr, per * pool_n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(mem, per * pool_n, MADV_HUGEPAGE);
    uint8_t * pool = static_cast<uint8_t *>(mem);
    for (int i = 0; i < pool_n; ++i) {   // experts i % E of layer 0 (read through the page cache once)
        const int e = i % E;
        std::memcpy(pool + i * per, tg.data + e * gu, gu);
        std::memcpy(pool + i * per + gu, tu.data + e * gu, gu);
        std::memcpy(pool + i * per + 2 * gu, td.data + e * dn, dn);
    }
    std::vector<float> x(D), out(static_cast<size_t>(8) * D);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd;
    for (auto & v : x) v = nd(rng);
    {   // the bare kernel, one thread: ggml's vec_dot over every gate and up row of the pool's experts in turn
        const auto * tr = ggml_get_type_traits_cpu(static_cast<ggml_type>(tg.type_id));
        std::vector<uint8_t> xq(ggml_row_size(tr->vec_dot_type, D));
        ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(x.data(), xq.data(), D);
        const size_t rb = gu / F;
        float acc = 0, v;
        const auto t0 = std::chrono::steady_clock::now();
        const int ne = std::min(pool_n, 64);
        for (int i = 0; i < ne; ++i)
            for (int r = 0; r < 2 * F; ++r) { tr->vec_dot(D, &v, 0, pool + static_cast<size_t>(i) * per + r * rb, 0, xq.data(), 0, 1); acc += v; }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("bare vec_dot, 1 thread, gate+up of %d experts: %.2f GB/s%s\n", ne, ne * 2.0 * gu / s / 1e9, acc == 1.f ? "!" : "");
    }
    bl::CpuExperts cpu(threads);
    for (int n : {1, 2, 4}) {
        int next = 0;
        auto call = [&] {
            std::vector<bl::CpuExperts::Job> jobs(n);
            for (int j = 0; j < n; ++j) { jobs[j] = {pool + static_cast<size_t>(next) * per, 1, out.data() + j * D}; next = (next + 1) % pool_n; }
            cpu.run(tg.type_id, td.type_id, gu, F, D, x.data(), 1, jobs.data(), n, 10.f);
        };
        for (int i = 0; i < 64; ++i) call();
        const int calls = 600 / n;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < calls; ++i) call();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%d threads, %d experts a call: %.3f ms per expert (%.1f GB/s)\n", threads, n, 1e3 * s / (calls * n),
                    calls * n * per / s / 1e9);
    }
    return 0;
}
