// DeepSeek-V4.1-Flash (`deepseek41`): hyperparameters read from GGUF metadata and checked against the shapes the
// kernels are written for, and the engine. See docs/DSV41.md for every formula.
//
// Phase A (this file's engine): correct first. One token per step, prefill is a loop of steps; dense weights in
// VRAM in their file formats; the 6 routed experts of a layer are copied from the mapped file into a VRAM staging
// buffer and run by the generic matvec; Engram rows are read from the mapped tables by the host. Every checkpoint
// calls the probe with the name llama.cpp's deepseek41.cpp gives the tensor, for comparison with bl-ref-dump.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "bl/gguf.h"

namespace bl {

struct Dsv41Config {
    int n_layer = 0, n_embd = 0, n_vocab = 0;
    int n_head = 0, head_dim = 0, rope_dims = 0;       // one KV head of head_dim (K = V)
    int q_lora = 0, o_groups = 0, o_lora = 0;
    int window = 0;
    double rope_base = 0, compress_rope_base = 0;      // sliding-window layers / layers with a compressed stream
    double yarn_factor = 0, yarn_beta_fast = 0, yarn_beta_slow = 0;
    int    yarn_orig_ctx = 0;
    double rms_eps = 0;

    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0;
    double expert_scale = 0, swiglu_limit = 0;

    int hc = 0, hc_sinkhorn = 0;
    double hc_eps = 0;

    int idx_heads = 0, idx_dim = 0, idx_top_k = 0;
    std::vector<int> compress_ratio;    // per layer: 0 sliding window only, 1 or 2 a compressed stream
    std::vector<int> kv_source;         // per layer: the layer whose compressed KV it reads (-1: none)
    std::vector<int> idx_key_source;    // per layer: the layer whose indexer keys it reads
    std::vector<int> topk_source;       // per layer: the layer whose top-k choice it uses

    int engram_heads = 0, engram_dim = 0, engram_ngram = 0, engram_pad = 0;
    std::vector<int> engram_layers;                 // layer ids
    std::vector<uint64_t> engram_mult;              // [engram layer][ngram]
    std::vector<uint64_t> engram_prime, engram_offset;   // [engram layer][(ngram-1) * heads]
    std::vector<int64_t> engram_token_map;          // token id -> compressed id

    int64_t context_length = 0;
    int bos = 0, eos = 0;

    int hc_dim() const { return hc * n_embd; }
    int hc_mix() const { return (2 + hc) * hc; }
    int engram_cols() const { return (engram_ngram - 1) * engram_heads; }   // rows read per token per engram layer
    int engram_index(int il) const;                                          // -1: no engram on this layer
    bool is_kv_source(int il) const { return compress_ratio[il] > 0 && kv_source[il] == il; }
    bool owns_idx_key(int il) const { return compress_ratio[il] > 0 && idx_key_source[il] == il; }
    bool is_topk_source(int il) const { return compress_ratio[il] > 0 && topk_source[il] == il; }

    static Dsv41Config from_gguf(const GgufModel & g);   // throws on anything this engine does not implement
};

// the Engram row ids of position p (n_cols per engram layer, layer-major), from the token ids up to p.
// tok[0..p] are the ids at positions 0..p of the sequence. Exactly engram.py NgramHashState (text only).
void dsv41_engram_rows(const Dsv41Config & c, const int * tok, int p, uint64_t * rows);

class Dsv41Engine {
public:
    Dsv41Engine(const std::string & shard1, int max_ctx);
    ~Dsv41Engine();
    Dsv41Engine(const Dsv41Engine &) = delete;
    Dsv41Engine & operator=(const Dsv41Engine &) = delete;

    const Dsv41Config & config() const;
    void reset();
    int  position() const;

    // feeds one token at position(); returns the logits ([n_vocab])
    const std::vector<float> & step(int token);

    // name ("attn_norm-12", ...), device pointer to n floats; called synchronously in forward order
    using Probe = std::function<void(const std::string & name, const float * dev, int n)>;
    void set_probe(Probe p);

    std::string report() const;   // where the time went

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace bl
