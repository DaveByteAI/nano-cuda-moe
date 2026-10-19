// bench_dsv41_ram: the RAM tier's disk reads alone (src/dsv41/ram_tier.h): random experts of the real files, fetched
// `batch` at a time like a decode layer's misses, the throughput per IO thread count.
//   bench_dsv41_ram SHARD1.gguf [seconds]
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../src/dsv41/ram_tier.h"
#include "bl/gguf.h"

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s SHARD1.gguf [seconds]\n", argv[0]); return 2; }
    const double secs = argc > 2 ? std::atof(argv[2]) : 4;
    bl::GgufModel g(argv[1]);
    std::vector<int> fds;
    for (size_t i = 0; i < g.n_shards(); ++i) fds.push_back(::open(g.shard_path(static_cast<uint32_t>(i)).c_str(), O_RDONLY | O_DIRECT));
    struct L { int fd[3]; uint64_t off[3]; };
    std::vector<L> layers;
    size_t len = 0;
    for (int il = 0; il < 40; ++il) {
        L l;
        int k = 0;
        for (const char * n : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
            const bl::Tensor & t = g.at("blk." + std::to_string(il) + "." + n);
            l.fd[k] = fds.at(t.shard);
            l.off[k] = g.file_offset(t);
            len = t.nbytes / 384;
            ++k;
        }
        layers.push_back(l);
    }
    const size_t per = 3 * len;
    const bool scatter = std::getenv("BL_SCATTER") != nullptr;
    for (int io : {2, 3, 4})
        for (int batch : {1, 4}) {
            bl::ExpertRam ram(size_t{4} << 30, per, io, true);
            std::mt19937 r(io * 10 + batch);
            int key_base = 0;
            uint64_t n = 0;
            const auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
                std::vector<int> keys(batch);
                std::vector<bl::ExpertRam::Src> src(batch);
                std::vector<const uint8_t *> out(batch);
                for (int b = 0; b < batch; ++b) {
                    const int il = r() % 40, e = r() % 384;
                    keys[b] = key_base++;   // always a miss
                    for (int k = 0; k < 3; ++k) {
                        // BL_SCATTER=1 (an experiment): each part from another random layer and expert
                        const int jl = scatter ? static_cast<int>(r() % 40) : il, je = scatter ? static_cast<int>(r() % 384) : e;
                        src[b].fd[k] = layers[jl].fd[k]; src[b].off[k] = layers[jl].off[k] + uint64_t(je) * len; src[b].len[k] = len;
                    }
                }
                ram.fetch(keys.data(), src.data(), batch, out.data());
                n += batch;
            }
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("io threads %d, batch %d: %.2f GB/s (%.2f ms per expert)\n", io, batch, n * per / el / 1e9, 1e3 * el / n);
            std::fflush(stdout);
        }
    return 0;
}
