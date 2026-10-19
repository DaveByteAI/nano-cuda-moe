// The engine: the qwen4exp forward on one GPU with the routed experts spread over VRAM, pinned RAM and the CPU.
//   - dense weights resident in VRAM in their file formats
//   - routed experts: an LRU cache per layer in VRAM; misses computed by the CPU or copied in over PCIe
//   - the GPU decides hits and misses itself and rings a doorbell in host memory; a whole window is one CUDA graph
//   - a "window" of T tokens (T <= 4) runs through the model at once: verify() computes it without changing the
//     state, commit(n) then keeps its first n tokens. step() is verify + commit of one token.
//   - QSA: the indexer's top-k blocks of 4 cells (the model's sparse attention); fp16 KV
// Every checkpoint of a one-token window calls the probe with the name llama.cpp's qwen4exp graph gives the tensor.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bl/config.h"
#include "bl/gguf.h"

namespace bl {

// sampling settings for generate(): temperature 0 = greedy
struct Sampling {
    float    temperature = 0.f;
    float    top_p = 1.f;
    int      top_k = 40;      // at most 256 candidates either way
    uint64_t seed = 0;        // 0: keep the engine's random stream going
};

// a context length as the tools take it: tokens, or with a k suffix (x 1024): "32768", "64k", "128k", "256k"
inline int parse_ctx(const std::string & s) {
    size_t n = 0;
    const long v = std::stol(s, &n);
    const std::string rest = s.substr(n);
    if (rest.empty()) return static_cast<int>(v);
    if (rest == "k" || rest == "K") return static_cast<int>(v * 1024);
    throw std::invalid_argument("--ctx " + s + ": a number of tokens, or with k (64k, 128k, 256k)");
}

class Engine {
public:
    // cache_file: an expert cache saved by save_cache(); when it exists, the VRAM cache starts with those experts
    Engine(const std::string & shard1, int max_ctx, const std::string & cache_file = "");
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    const ModelConfig & config() const;
    void reset();
    int  position() const;   // tokens committed so far

    // the window tokens[0..T) at positions position().. : the argmax after each of them (ids, T values), and their
    // logits when asked ([T][n_vocab]). The committed state is not changed.
    const std::vector<int> & verify(const int * tokens, int T, bool want_logits);
    const std::vector<float> & logits() const;   // of the last verify that asked for them
    // keep the first n tokens of the last verify window (1 <= n <= T)
    void commit(int n);

    // feeds one token at the next position; returns the logits when asked (else an empty vector)
    const std::vector<float> & step(int token, bool want_logits);

    // Greedy decoding from the current state: feeds the pending token (if any) and the prompt, returns up to max_new
    // tokens; on_token returning false stops after that token; a full context stops too. With the MTP draft layer,
    // each round verifies the last token plus up to `spec` (<= 3) drafts; drafts after the first are made while the
    // previous one's probability is >= min_p. spec = 0: one token per window.
    // Session state: after a call, position() covers everything fed and emitted except the last emitted token, which
    // is pending() and is fed first by the next generate() - so turns chain: generate(turn 1), generate(turn 2), ...
    // The low-level verify/commit/step know nothing of the pending token; reset() drops it.
    bool has_mtp() const;
    // With sampling (temperature > 0) the drafts are judged by speculative rejection sampling: the output follows the
    // sampled distribution exactly, drafts or not.
    std::vector<int> generate(const std::vector<int> & prompt, int max_new, int spec = 3, float min_p = 0.5f,
                              const std::function<bool(int)> & on_token = nullptr, const Sampling * sampling = nullptr);
    int  pending() const;   // the last emitted token, not yet fed (-1: none)
    // testing: the prompt through the prefill path (in chunks), the last chunk's logits ([rows][n_vocab])
    std::vector<float> prefill_logits(const std::vector<int> & tokens);

    // the experts now in the VRAM cache, most recently used first (call between generations)
    void save_cache(const std::string & path) const;

    // expert cache hit rate, where the time goes, cache activity
    std::string report() const;
    void reset_stats();

    // runtime settings: cpu_share (one-token windows), cpu_share_win (longer ones), lru, lru_max, prefetch, adapt_every (0 = off), adapt_max, adapt_decay, adapt_margin
    void set(const std::string & key, double value);

    // name ("hc_mixed-12", ...), device pointer to n floats; called synchronously in forward order (T = 1 only)
    using Probe = std::function<void(const std::string & name, const float * dev, int n)>;
    void set_probe(Probe p);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace bl
