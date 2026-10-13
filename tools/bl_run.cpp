// bl-run: feed a token list through the phase-1 engine, optionally compare every checkpoint with a bl-ref-dump.
//   bl-run --model SHARD1.gguf --tokens-file F [--ref DIR] [--gen N] [--ctx N]
// With --ref, each probe of token t is compared with token t's slice of the reference tensor of the same name and
// occurrence (a layer has two "hc_mixed"); a reference that holds only the last token is compared at the last token.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bl/engine.h"

namespace {

struct RefTensor {
    int64_t ne[4];
    uint64_t offset, bytes;
};

struct Ref {
    std::map<std::string, std::vector<RefTensor>> by_name;
    std::vector<float> data;
    bool load(const std::string & dir) {
        std::ifstream idx(dir + "/index.tsv");
        if (!idx) return false;
        std::string line;
        std::getline(idx, line);   // header
        while (std::getline(idx, line)) {
            std::istringstream is(line);
            std::string name, type;
            RefTensor t;
            int graph;
            is >> name >> type >> t.ne[0] >> t.ne[1] >> t.ne[2] >> t.ne[3] >> t.offset >> t.bytes >> graph;
            if (type == "f32") by_name[name].push_back(t);
        }
        std::ifstream f(dir + "/data.bin", std::ios::binary | std::ios::ate);
        const auto n = f.tellg();
        f.seekg(0);
        data.resize(static_cast<size_t>(n) / 4);
        f.read(reinterpret_cast<char *>(data.data()), n);
        return true;
    }
};

struct Stat {
    double worst_rel = 0;
    int    worst_tok = -1, compared = 0;
};

std::vector<int> read_tokens(const std::string & path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    for (char & ch : s)
        if (ch == ',') ch = ' ';
    std::istringstream is(s);
    std::vector<int> v;
    for (int x; is >> x;) v.push_back(x);
    return v;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model, tokens_file, ref_dir, save_logits, sweep;
    int gen = 0, ctx = 4096, repeat = 1, warm = 0, window = 1, save_last = 0;
    std::string specs, pf_logits, cache_file, save_cache;
    float min_p = 0.5f;
    bool prepass = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model") model = next();
        else if (a == "--tokens-file") tokens_file = tokens_file.empty() ? next() : tokens_file + ";" + next();
        else if (a == "--ref") ref_dir = next();
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--ctx") ctx = bl::parse_ctx(next());
        else if (a == "--save-logits") save_logits = next();
        else if (a == "--save-last") save_last = std::stoi(next());   // with --save-logits: only the last N positions
        else if (a == "--set") sweep = next();          // key=v1,v2,...: one run per value, same loaded model
        else if (a == "--repeat") repeat = std::stoi(next());   // runs per setting (the expert cache stays warm)
        else if (a == "--warm") warm = std::stoi(next());       // passes to warm the expert cache, then it is frozen
        else if (a == "--window") window = std::stoi(next());   // feed the prompt in verify windows of this many tokens
        else if (a == "--spec") specs = next();   // generate() with up to this many MTP drafts (0: none); a list: one run each
        else if (a == "--prefill-logits") pf_logits = next();   // testing: prefill path, every position's logits
        else if (a == "--min-p") min_p = std::stof(next());     // chain another draft while the last one's p >= this
        else if (a == "--cache-file") cache_file = next();      // start with this saved expert cache
        else if (a == "--save-cache") save_cache = next();      // save the expert cache at the end
        else if (a == "--no-prepass") prepass = false;          // with several files: skip feeding the extra ones first
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || tokens_file.empty()) {
        std::fprintf(stderr, "usage: %s --model SHARD1.gguf --tokens-file F [--ref DIR] [--gen N] [--ctx N]\n", argv[0]);
        return 2;
    }
    try {
        // several --tokens-file: each runs from a reset state, in one process (one model load)
        std::vector<std::vector<int>> files;
        {
            std::stringstream fs(tokens_file);
            for (std::string f; std::getline(fs, f, ';');) files.push_back(read_tokens(f));
        }
        std::vector<int> toks = files[0];
        int T = static_cast<int>(toks.size());
        auto t_load = std::chrono::steady_clock::now();
        bl::Engine eng(model, ctx, cache_file);
        struct SaveAtEnd {   // every return path below
            bl::Engine & e;
            const std::string & path;
            ~SaveAtEnd() {
                if (path.empty()) return;
                try { e.save_cache(path); std::printf("expert cache saved to %s\n", path.c_str()); }
                catch (const std::exception & x) { std::fprintf(stderr, "%s\n", x.what()); }
            }
        } save_at_end{eng, save_cache};
        std::fprintf(stderr, "loaded in %.1f s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count());

        Ref ref;
        const bool compare = !ref_dir.empty() && ref.load(ref_dir);
        if (!ref_dir.empty() && !compare) { std::fprintf(stderr, "cannot read %s\n", ref_dir.c_str()); return 1; }
        std::vector<std::string> order;
        std::map<std::string, Stat> stats;
        std::map<std::string, int> occ;
        int cur_tok = 0;
        std::vector<float> host;
        if (compare) {
            eng.set_probe([&](const std::string & name, const float * dev, int n) {
                const int k = occ[name]++;
                const std::string key = k ? name + "#" + std::to_string(k) : name;
                auto it = ref.by_name.find(name);
                if (it == ref.by_name.end() || k >= static_cast<int>(it->second.size())) return;
                const RefTensor & rt = it->second[k];
                const int64_t total = rt.ne[0] * rt.ne[1] * rt.ne[2] * rt.ne[3];
                const float * r = nullptr;
                if (total == static_cast<int64_t>(n) * T) r = ref.data.data() + rt.offset / 4 + static_cast<size_t>(cur_tok) * n;
                else if (total == n && cur_tok == T - 1) r = ref.data.data() + rt.offset / 4;
                if (!r) return;
                host.resize(n);
                cudaMemcpy(host.data(), dev, sizeof(float) * n, cudaMemcpyDeviceToHost);
                double num = 0, den = 0;
                for (int i = 0; i < n; ++i) {
                    const double d = static_cast<double>(host[i]) - r[i];
                    num += d * d;
                    den += static_cast<double>(r[i]) * r[i];
                }
                const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
                if (!stats.count(key)) order.push_back(key);
                Stat & s = stats[key];
                ++s.compared;
                if (rel > s.worst_rel || s.worst_tok < 0) { s.worst_rel = rel; s.worst_tok = cur_tok; }
            });
        }

        std::vector<float> last;
        std::ofstream lf;
        if (!save_logits.empty()) lf.open(save_logits, std::ios::binary);   // one row of logits per prompt position
        auto run_prompt = [&]() {
            auto t0 = std::chrono::steady_clock::now();
            if (window > 1) {   // verify windows, all accepted
                for (int t = 0; t < T; t += window) {
                    const int w = std::min(window, T - t);
                    const bool keep = lf.is_open() && (save_last <= 0 || t + w > T - save_last);
                    eng.verify(&toks[t], w, t + w == T || keep);
                    if (keep) {
                        const int V = eng.config().n_vocab;
                        const int from = save_last > 0 ? std::max(0, (T - save_last) - t) : 0;
                        lf.write(reinterpret_cast<const char *>(eng.logits().data() + static_cast<size_t>(from) * V),
                                 static_cast<std::streamsize>((static_cast<size_t>(w) - from) * V * 4));
                    }
                    if (t + w == T) last.assign(eng.logits().end() - eng.config().n_vocab, eng.logits().end());
                    eng.commit(w);
                }
            } else
            for (int t = 0; t < T; ++t) {
                cur_tok = t;
                occ.clear();
                last = eng.step(toks[t], t == T - 1 || gen > 0 || lf.is_open());
                if (lf.is_open()) lf.write(reinterpret_cast<const char *>(last.data()), static_cast<std::streamsize>(last.size() * 4));
            }
            const double prompt_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "prompt: %d tokens in %.2f s (%.2f tok/s)\n", T, prompt_s, T / prompt_s);
        };
        if (warm > 0) {   // comparable runs: adapt the cache to the prompt first, then freeze it
            for (int r = 0; r < warm; ++r) { eng.reset(); run_prompt(); }
            eng.set("adapt_every", 0);
            std::printf("warmed (%d passes), cache frozen: %s\n", warm, eng.report().c_str());
        }
        if (!sweep.empty() && specs.empty()) {   // key=v1,v2,...: run the prompt `repeat` times per value
            const auto eq = sweep.find('=');
            const std::string key = sweep.substr(0, eq);
            std::stringstream vs(sweep.substr(eq + 1));
            for (std::string v; std::getline(vs, v, ',');) {
                eng.set(key, std::stod(v));
                for (int r = 0; r < repeat; ++r) {
                    eng.reset();
                    eng.reset_stats();
                    std::printf("== %s=%s run %d\n", key.c_str(), v.c_str(), r + 1);
                    std::fflush(stdout);
                    run_prompt();
                    std::printf("%s\n", eng.report().c_str());
                    std::fflush(stdout);
                }
            }
            return 0;
        }
        for (size_t fi = 1; prepass && fi < files.size(); ++fi) {   // the extra files first, the first one last (it is reported)
            toks = files[fi];
            T = static_cast<int>(toks.size());
            eng.reset();
            run_prompt();
            eng.reset();
        }
        toks = files[0];
        T = static_cast<int>(toks.size());
        if (!pf_logits.empty()) {   // testing: the prefill path's logits at every prompt position
            eng.set_probe(nullptr);
            const std::vector<float> l = eng.prefill_logits(toks);
            const size_t V = eng.config().n_vocab, rows = l.size() / V, keep = save_last > 0 ? std::min<size_t>(save_last, rows) : rows;
            std::ofstream f(pf_logits, std::ios::binary);
            f.write(reinterpret_cast<const char *>(l.data() + (rows - keep) * V), static_cast<std::streamsize>(keep * V * 4));
            std::printf("prefill logits: %zu positions -> %s\n", keep, pf_logits.c_str());
            if (save_logits.empty()) return 0;
            eng.reset();   // then the window path (--save-logits) on the same prompt, same loaded model
        }
        if (!specs.empty()) {   // engine-side greedy generation (speculative with the MTP when spec > 0), every file
            eng.set_probe(nullptr);
            std::vector<std::string> svals{""};   // --set key=v1,v2 sweeps here too
            std::string skey;
            if (!sweep.empty()) {
                const auto eq = sweep.find('=');
                skey = sweep.substr(0, eq);
                svals.clear();
                std::stringstream vs(sweep.substr(eq + 1));
                for (std::string v; std::getline(vs, v, ',');) svals.push_back(v);
            }
            for (const std::string & setv : svals) {
            if (!skey.empty()) { eng.set(skey, std::stod(setv)); std::printf("==== %s=%s\n", skey.c_str(), setv.c_str()); }
            std::stringstream ss(specs);
            for (std::string sv; std::getline(ss, sv, ',');) {
            double sum_tok = 0, sum_s = 0;
            for (size_t fi = 0; fi < files.size(); ++fi)
            for (int r = 0; r < repeat; ++r) {
                toks = files[fi];
                T = static_cast<int>(toks.size());
                const int spec = std::stoi(sv);
                std::printf("== spec %d file %zu run %d\n", spec, fi, r + 1);
                eng.reset();
                eng.reset_stats();
                auto g0 = std::chrono::steady_clock::now();
                std::chrono::steady_clock::time_point g1;
                int n = 0;
                const std::vector<int> out = eng.generate(toks, std::max(gen, 1), spec, min_p, [&](int) {
                    if (n++ == 0) g1 = std::chrono::steady_clock::now();
                    return true;
                });
                const auto g2 = std::chrono::steady_clock::now();
                const double ps = std::chrono::duration<double>(g1 - g0).count(), ds = std::chrono::duration<double>(g2 - g1).count();
                std::printf("generated:");
                for (int id : out) std::printf(" %d", id);
                std::printf("\nprompt: %d tokens in %.2f s; decode: %zu tokens in %.2f s (%.2f tok/s)\n%s\n", T, ps,
                            out.size() - 1, ds, (out.size() - 1) / ds, eng.report().c_str());
                std::fflush(stdout);
                sum_tok += static_cast<double>(out.size() - 1);
                sum_s += ds;
            }
            std::printf("== spec %s overall: %.0f tokens in %.2f s: %.2f tok/s\n", sv.c_str(), sum_tok, sum_s, sum_tok / sum_s);
            }
            }
            return 0;
        }
        for (int r = 0; r < repeat; ++r) {
            if (r) { eng.reset(); eng.reset_stats(); }
            run_prompt();
            if (repeat > 1) std::printf("run %d: %s\n", r + 1, eng.report().c_str());
        }

        auto top = [](const std::vector<float> & l, int k) {
            std::vector<std::pair<float, int>> v;
            for (int i = 0; i < static_cast<int>(l.size()); ++i) v.push_back({l[i], i});
            std::partial_sort(v.begin(), v.begin() + k, v.end(), [](auto a, auto b) { return a.first > b.first; });
            v.resize(k);
            return v;
        };
        std::printf("top-5 after the prompt:");
        for (auto [lv, id] : top(last, 5)) std::printf(" %d(%.3f)", id, lv);
        std::printf("\n");

        if (compare) {
            std::printf("\n%-28s %10s %6s\n", "checkpoint", "worst rel", "token");
            for (const auto & k : order) {
                const Stat & s = stats[k];
                std::printf("%-28s %10.2e %6d%s\n", k.c_str(), s.worst_rel, s.worst_tok, s.worst_rel > 5e-2 ? "  <--" : "");
            }
        }

        if (gen > 0) {
            eng.set_probe(nullptr);
            std::vector<int> out;
            int next = top(last, 1)[0].second;
            auto g0 = std::chrono::steady_clock::now();
            for (int i = 0; i < gen; ++i) {
                out.push_back(next);
                const auto & l = eng.step(next, true);
                next = top(l, 1)[0].second;
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - g0).count();
            std::printf("generated:");
            for (int id : out) std::printf(" %d", id);
            std::printf("\ndecode: %d tokens in %.2f s (%.2f tok/s)\n", gen, s, gen / s);
        }
        std::printf("%s\n", eng.report().c_str());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
