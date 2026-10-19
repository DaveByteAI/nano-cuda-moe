// CPU expert throughput: single-thread ggml vec_dot per format, and the CpuExperts pool on whole experts.
//   bench_cpu SHARD1.gguf [threads]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "bl/cpu_experts.h"
#include "bl/cpu_iq.h"
#include "bl/cpu_q2.h"
#include "bl/gguf.h"
#include "ggml-cpu.h"
#include "ggml.h"

using clk = std::chrono::steady_clock;

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s SHARD1.gguf [threads]\n", argv[0]); return 2; }
    const int threads = argc > 2 ? std::atoi(argv[2]) : 8;
    bl::GgufModel g(argv[1]);
    ggml_cpu_init();
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-1.f, 1.f);

    // one layer per (gate/up format, down format) pair
    std::map<std::pair<uint32_t, uint32_t>, int> pick;
    for (int il = 0; g.find("blk." + std::to_string(il) + ".ffn_gate_exps.weight"); ++il) {   // every layer with routed experts
        const auto & gt = g.at("blk." + std::to_string(il) + ".ffn_gate_exps.weight");
        const auto & dt = g.at("blk." + std::to_string(il) + ".ffn_down_exps.weight");
        pick.emplace(std::make_pair(gt.type_id, dt.type_id), il);
    }
    bl::CpuExperts pool(threads);
    const int D = 2560, F = 640, NE = 512, n_exp = 2, reps = 200;
    std::vector<float> x(D);
    for (auto & v : x) v = U(rng);

    {   // our AVX2 Q2_0 dot against ggml's (scalar) one: values and speed
        const auto & dt = g.at("blk.1.ffn_down_exps.weight");
        const int K = 640, rows = 2560 * 8;
        const size_t rb = ggml_row_size(GGML_TYPE_Q2_0, K);
        std::vector<float> a(K);
        for (auto & v : a) v = U(rng);
        std::vector<uint8_t> xp(bl::cpu::q8planes_bytes(K)), xq(ggml_row_size(GGML_TYPE_Q8_0, K));
        bl::cpu::q8planes_quantize(a.data(), K, xp.data());
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(a.data(), xq.data(), K);
        const auto * tr = ggml_get_type_traits_cpu(GGML_TYPE_Q2_0);
        double max_rel = 0, ref_sum = 0;
        for (int r = 0; r < 256; ++r) {
            float ref, ours = bl::cpu::q2_0_dot(dt.data + r * rb, xp.data(), K);
            tr->vec_dot(K, &ref, 0, dt.data + r * rb, 0, xq.data(), 0, 1);
            ref_sum += std::fabs(ref);
            max_rel = std::max(max_rel, static_cast<double>(std::fabs(ours - ref)));
        }
        float sink = 0;
        auto t0 = clk::now();
        for (int rep = 0; rep < 5; ++rep)
            for (int r = 0; r < rows; ++r) sink += bl::cpu::q2_0_dot(dt.data + r * rb, xp.data(), K);
        const double ours_s = std::chrono::duration<double>(clk::now() - t0).count();
        std::printf("Q2_0 AVX2 dot: max |ours - ggml| %.2e (mean |ggml| %.2e); %.2f GB/s single thread%s\n", max_rel,
                    ref_sum / 256, 5.0 * rows * rb / ours_s / 1e9, sink == 1.2345f ? "!" : "");
    }
    std::printf("single thread vec_dot (one row against q8 x), and the %d-thread pool on %d experts at a time\n", threads, n_exp);
    for (auto [types, il] : pick) {
        const auto & gt = g.at("blk." + std::to_string(il) + ".ffn_gate_exps.weight");
        const auto & ut = g.at("blk." + std::to_string(il) + ".ffn_up_exps.weight");
        const auto & dt = g.at("blk." + std::to_string(il) + ".ffn_down_exps.weight");
        const size_t gu = gt.nbytes / NE, db = dt.nbytes / NE, eb = 2 * gu + db;
        // pack 16 experts [gate|up|down] like the engine's pinned store (warm in RAM, cold in cache)
        const int pool_n = 16;
        std::vector<uint8_t> store(eb * pool_n);
        for (int e = 0; e < pool_n; ++e) {
            std::memcpy(store.data() + e * eb, gt.data + e * gu, gu);
            std::memcpy(store.data() + e * eb + gu, ut.data + e * gu, gu);
            std::memcpy(store.data() + e * eb + 2 * gu, dt.data + e * db, db);
        }
        // single-thread rows
        auto st_rate = [&](ggml_type t, int K, const uint8_t * rows, size_t rb, int n_rows) {
            const auto * tr = ggml_get_type_traits_cpu(t);
            std::vector<uint8_t> xq(ggml_row_size(tr->vec_dot_type, K));
            std::vector<float> xx(K);
            for (auto & v : xx) v = U(rng);
            ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(xx.data(), xq.data(), K);
            float s = 0, acc = 0;
            const auto t0 = clk::now();
            for (int rep = 0; rep < 20; ++rep)
                for (int r = 0; r < n_rows; ++r) { tr->vec_dot(K, &s, 0, rows + r * rb, 0, xq.data(), 0, 1); acc += s; }
            const double sec = std::chrono::duration<double>(clk::now() - t0).count();
            if (acc == 12345.f) std::printf("!");
            return 20.0 * n_rows * rb / sec / 1e9;
        };
        const double r_gu = st_rate(static_cast<ggml_type>(gt.type_id), D, store.data(), gu / F, F * 2);
        if (bl::cpu::iq_supported(gt.type_id)) {   // the multi-token kernel: against ggml per token, and its speed
            const auto * tr = ggml_get_type_traits_cpu(static_cast<ggml_type>(gt.type_id));
            const size_t xrb = ggml_row_size(tr->vec_dot_type, D), rb = gu / F;
            std::vector<uint8_t> xq(4 * xrb);
            std::vector<float> xx(D);
            const void * xs[4];
            for (int t = 0; t < 4; ++t) {
                for (auto & v : xx) v = U(rng);
                ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(xx.data(), xq.data() + t * xrb, D);
                xs[t] = xq.data() + t * xrb;
            }
            double max_err = 0, mean_ref = 0;
            for (int r = 0; r < 64; ++r) {
                const uint8_t * gr = store.data() + r * rb, * ur = store.data() + gu + r * rb;
                float gv[4], uv[4];
                bl::cpu::iq_gate_up(gt.type_id, gr, ur, D, xs, 4, gv, uv);
                for (int t = 0; t < 4; ++t) {
                    float rg, ru;
                    tr->vec_dot(D, &rg, 0, gr, 0, xs[t], 0, 1);
                    tr->vec_dot(D, &ru, 0, ur, 0, xs[t], 0, 1);
                    max_err = std::max({max_err, static_cast<double>(std::fabs(gv[t] - rg)), static_cast<double>(std::fabs(uv[t] - ru))});
                    mean_ref += std::fabs(rg) / 256;
                }
            }
            auto iq_rate = [&](int nt) {
                float gv[4], uv[4], acc = 0;
                const auto t0 = clk::now();
                for (int rep = 0; rep < 20; ++rep)
                    for (int e = 0; e < pool_n; ++e)
                        for (int r = 0; r < F; ++r) {
                            const uint8_t * base = store.data() + e * eb;
                            bl::cpu::iq_gate_up(gt.type_id, base + r * rb, base + gu + r * rb, D, xs, nt, gv, uv);
                            acc += gv[0];
                        }
                const double sec = std::chrono::duration<double>(clk::now() - t0).count();
                if (acc == 12345.f) std::printf("!");
                return 20.0 * pool_n * 2 * gu / sec / 1e9;
            };
            std::printf("layer %2d  %-8s multi-token gate/up: max |ours - ggml| %.2e (mean |ggml| %.2e); 1 thread %.2f GB/s at 1 token, "
                        "%.2f GB/s at 4 tokens (ggml: %.2f per token)\n", il, ggml_type_name(static_cast<ggml_type>(gt.type_id)), max_err,
                        mean_ref, iq_rate(1), iq_rate(4), r_gu);
        }
        const double r_d = st_rate(static_cast<ggml_type>(dt.type_id), F, store.data() + 2 * gu, db / D, D);

        std::vector<float> out(static_cast<size_t>(n_exp) * D);
        bl::CpuExperts::Job jobs[n_exp];
        const auto t0 = clk::now();
        for (int rep = 0; rep < reps; ++rep) {
            for (int e = 0; e < n_exp; ++e) jobs[e] = {store.data() + ((rep * n_exp + e) % pool_n) * eb, 1, out.data() + e * D};
            pool.run(gt.type_id, dt.type_id, gu, F, D, x.data(), 1, jobs, n_exp);
        }
        const double sec = std::chrono::duration<double>(clk::now() - t0).count();
        std::printf("layer %2d  %-8s/%-7s  1-thread gate/up %5.2f GB/s, down %5.2f GB/s | pool %5.1f GB/s (%.0f us per call)\n",
                    il, ggml_type_name(static_cast<ggml_type>(gt.type_id)), ggml_type_name(static_cast<ggml_type>(dt.type_id)),
                    r_gu, r_d, reps * n_exp * eb / sec / 1e9, sec / reps * 1e6);
    }
    return 0;
}
