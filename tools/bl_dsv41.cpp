// bl-dsv41: the deepseek41 phase-A engine on a token list (docs/DSV41.md).
//
//   bl-dsv41 --model SHARD1.gguf (--tokens 1,2,3 | --tokens-file F) [--gen N] [--ctx N]
//            [--save-logits FILE] [--dump DIR [--filter REGEX]]
//   bl-dsv41 --model SHARD1.gguf --tokens-file F --engram-rows     (the Engram row ids per position, no weights)
//   bl-dsv41 ... --prefill          the prompt through the chunked prefill (then --gen steps)
//   bl-dsv41 ... --check-prefill    prefill's logits at every position against one step per token: KL, top-1
//   bl-dsv41 ... --force N          the first N tokens through prefill, the rest one step each (teacher-forced
//                                   decode: the same work every run, for timing); --stats-after counts those steps
//
// Feeds the tokens one step each, prints the top-5 after each and then greedy-generates N more. --save-logits
// writes every fed position's logits (T rows of n_vocab float32: tools/logits_compare.py against bl-ref-dump).
// --dump writes the probes in bl-ref-dump's format (DIR/index.tsv + DIR/data.bin), one entry per tensor per step,
// named like llama.cpp's deepseek41.cpp graph ("attn_norm-12", "l_last-39", "result_output").
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
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
    int gen = 0, ctx = 4096, stats_after = -1, force = -1, fchunk = 0;
    bool engram_rows = false, use_prefill = false, check_prefill = false;
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
        else if (a == "--engram-rows") engram_rows = true;
        else if (a == "--stats-after") stats_after = std::stoi(next());   // the report covers only the steps after N generated
        else if (a == "--force") force = std::stoi(next());
        else if (a == "--chunk") fchunk = std::stoi(next());   // --force: the stepped tokens K at a time through prefill
        else if (a == "--prefill") use_prefill = true;          // the prompt through prefill(), then --gen steps
        else if (a == "--check-prefill") check_prefill = true;  // prefill's logits against stepping each token
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

    if (engram_rows) {   // the Engram row ids per position, from the metadata alone (tools/dsv41_engram_check.py)
        try {
            bl::GgufModel g(model);
            const bl::Dsv41Config c = bl::Dsv41Config::from_gguf(g);
            std::vector<uint64_t> rows(c.engram_layers.size() * c.engram_cols());
            for (size_t p = 0; p < prompt.size(); ++p) {
                bl::dsv41_engram_rows(c, prompt.data(), static_cast<int>(p), rows.data());
                for (size_t i = 0; i < rows.size(); ++i) std::printf("%s%llu", i ? "," : "", static_cast<unsigned long long>(rows[i]));
                std::printf("\n");
            }
        } catch (const std::exception & e) {
            std::fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
        return 0;
    }

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
        auto log_softmax = [&](const float * l, std::vector<double> & out) {
            out.resize(V);
            double mx = -1e300, z = 0;
            for (int i = 0; i < V; ++i) mx = std::max(mx, static_cast<double>(l[i]));
            for (int i = 0; i < V; ++i) z += std::exp(l[i] - mx);
            for (int i = 0; i < V; ++i) out[i] = l[i] - mx - std::log(z);
        };
        if (check_prefill) {
            std::vector<float> all;
            const auto a = std::chrono::steady_clock::now();
            eng.prefill(prompt, &all);
            const auto b = std::chrono::steady_clock::now();
            std::printf("prefill %zu tokens: %.2f s\n", prompt.size(), std::chrono::duration<double>(b - a).count());
            eng.reset();
            double kl_sum = 0, kl_max = 0;
            int agree = 0;
            std::vector<double> lp, ls;
            for (size_t i = 0; i < prompt.size(); ++i) {
                const auto & l = eng.step(prompt[i]);
                log_softmax(all.data() + i * V, lp);
                log_softmax(l.data(), ls);
                double kl = 0;
                for (int j = 0; j < V; ++j) kl += std::exp(ls[j]) * (ls[j] - lp[j]);
                kl_sum += kl;
                kl_max = std::max(kl_max, kl);
                agree += top(l, 1)[0] == top(std::vector<float>(all.begin() + i * V, all.begin() + (i + 1) * V), 1)[0];
            }
            std::printf("prefill vs steps over %zu positions: mean KL %.5f, max %.5f, top-1 agreement %d/%zu\n", prompt.size(),
                        kl_sum / prompt.size(), kl_max, agree, prompt.size());
            std::printf("%s\n", eng.report().c_str());
            return 0;
        }
        if (force >= 0) {   // teacher-forced: prefill the head, then step every remaining token whatever was predicted
            const std::vector<int> head(prompt.begin(), prompt.begin() + std::min<size_t>(force, prompt.size()));
            if (!head.empty()) eng.prefill(head);
            int agree = 0, n = 0;
            if (fchunk > 1) {   // a verify step's cost: K tokens at once
                double t = 0;
                int nc = 0;
                for (size_t i = head.size(); i + fchunk <= prompt.size(); i += fchunk, ++nc) {
                    if (stats_after >= 0 && i == head.size() + stats_after / fchunk * fchunk) { eng.reset_stats(); t = 0; nc = 0; }
                    const auto a = std::chrono::steady_clock::now();
                    eng.prefill(std::vector<int>(prompt.begin() + i, prompt.begin() + i + fchunk));
                    t += std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
                }
                std::printf("forced %d chunks of %d: %.2f ms per chunk, %.2f ms per token\n", nc, fchunk, 1e3 * t / nc,
                            1e3 * t / nc / fchunk);
                std::printf("%s\n", eng.report().c_str());
                return 0;
            }
            for (size_t i = head.size(); i < prompt.size(); ++i, ++n) {
                if (n == stats_after) eng.reset_stats();
                const auto & l = eng.step(prompt[i]);
                if (i + 1 < prompt.size()) agree += top(l, 1)[0] == prompt[i + 1];
            }
            std::printf("forced %d steps, top-1 = next token %d/%d\n", n, agree, std::max(n - 1, 0));
            std::printf("%s\n", eng.report().c_str());
            return 0;
        }
        if (use_prefill) {
            std::vector<float> all;
            const auto & l = eng.prefill(prompt, lg ? &all : nullptr);
            if (lg) lg.write(reinterpret_cast<const char *>(all.data()), all.size() * 4);
            const auto t = top(l, 5);
            std::printf("prefill %zu tokens -> top5", prompt.size());
            for (int id : t) std::printf(" %d(%.2f)", id, l[id]);
            std::printf("\n");
            last = t[0];
        }
        for (size_t i = 0; i < (use_prefill ? 0 : prompt.size()); ++i) {
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
            if (i == stats_after) eng.reset_stats();
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
