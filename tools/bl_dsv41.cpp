// bl-dsv41: the deepseek41 phase-A engine on a token list (docs/DSV41.md).
//
//   bl-dsv41 --model SHARD1.gguf (--tokens 1,2,3 | --tokens-file F) [--gen N] [--ctx N]
//            [--save-logits FILE] [--dump DIR [--filter REGEX]]
//
// Feeds the tokens one step each, prints the top-5 after each and then greedy-generates N more. --save-logits
// writes every fed position's logits (T rows of n_vocab float32: tools/logits_compare.py against bl-ref-dump).
// --dump writes the probes in bl-ref-dump's format (DIR/index.tsv + DIR/data.bin), one entry per tensor per step,
// named like llama.cpp's deepseek41.cpp graph ("attn_norm-12", "l_last-39", "result_output").
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bl/dsv41.h"

namespace {

std::vector<int> parse_ids(const std::string & s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        out.push_back(std::stoi(tok));
    }
    return out;
}

void usage() {
    std::fprintf(stderr, "bl-dsv41 --model SHARD1.gguf (--tokens 1,2,3 | --tokens-file F) [--gen N] [--ctx N]\n"
                         "         [--save-logits FILE] [--dump DIR [--filter REGEX]]\n");
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model, tokens_s, tokens_file, save_logits, dump_dir, filter = ".*";
    int gen = 0, ctx = 4096;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--tokens") tokens_s = next();
        else if (a == "--tokens-file") tokens_file = next();
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--ctx") ctx = std::stoi(next());
        else if (a == "--save-logits") save_logits = next();
        else if (a == "--dump") dump_dir = next();
        else if (a == "--filter") filter = next();
        else { usage(); return 2; }
    }
    if (model.empty() || (tokens_s.empty() && tokens_file.empty())) { usage(); return 2; }
    if (!tokens_file.empty()) {
        std::ifstream f(tokens_file);
        std::stringstream ss;
        ss << f.rdbuf();
        tokens_s = ss.str();
    }
    const std::vector<int> prompt = parse_ids(tokens_s);
    if (prompt.empty()) { std::fprintf(stderr, "no tokens\n"); return 2; }

    try {
        const auto t0 = std::chrono::steady_clock::now();
        bl::Dsv41Engine eng(model, ctx);
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("loaded in %.1f s\n", std::chrono::duration<double>(t1 - t0).count());
        const int V = eng.config().n_vocab;

        std::ofstream data, index;
        uint64_t off = 0;
        std::regex re(filter);
        std::vector<float> buf;
        if (!dump_dir.empty()) {
            data.open(dump_dir + "/data.bin", std::ios::binary);
            index.open(dump_dir + "/index.tsv");
            if (!data || !index) { std::fprintf(stderr, "cannot write into %s\n", dump_dir.c_str()); return 1; }
            index << "name\ttype\tne0\tne1\tne2\tne3\toffset\tbytes\n";
            eng.set_probe([&](const std::string & name, const float * dev, int n) {
                if (!std::regex_match(name, re)) return;
                buf.resize(n);
                cudaMemcpy(buf.data(), dev, n * 4, cudaMemcpyDeviceToHost);
                data.write(reinterpret_cast<const char *>(buf.data()), n * 4);
                index << name << "\tf32\t" << n << "\t1\t1\t1\t" << off << "\t" << n * 4 << "\n";
                off += static_cast<uint64_t>(n) * 4;
            });
        }
        std::ofstream lg;
        if (!save_logits.empty()) lg.open(save_logits, std::ios::binary);

        auto top = [&](const std::vector<float> & l, int k) {
            std::vector<int> id(V);
            std::iota(id.begin(), id.end(), 0);
            std::partial_sort(id.begin(), id.begin() + k, id.end(), [&](int a, int b) { return l[a] > l[b]; });
            id.resize(k);
            return id;
        };
        int last = -1;
        for (size_t i = 0; i < prompt.size(); ++i) {
            const auto & l = eng.step(prompt[i]);
            if (lg) lg.write(reinterpret_cast<const char *>(l.data()), V * 4);
            const auto t = top(l, 5);
            std::printf("pos %4zu tok %6d -> top5", i, prompt[i]);
            for (int id : t) std::printf(" %d(%.2f)", id, l[id]);
            std::printf("\n");
            std::fflush(stdout);
            last = t[0];
        }
        std::vector<int> out;
        for (int i = 0; i < gen; ++i) {
            out.push_back(last);
            const auto & l = eng.step(last);
            last = top(l, 1)[0];
        }
        if (gen > 0) {
            std::printf("generated:");
            for (int id : out) std::printf(" %d", id);
            std::printf("\n");
        }
        std::printf("%s\n", eng.report().c_str());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
