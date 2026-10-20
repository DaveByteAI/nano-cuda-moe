#include "bl/dsv41.h"

#include <stdexcept>
#include <string>
#include <variant>

namespace bl {

namespace {

void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error("unsupported model: " + what);
}

std::vector<double> float_array(const GgufModel & g, const std::string & key) {
    auto it = g.metadata().find(key);
    if (it == g.metadata().end()) throw std::runtime_error("GGUF: no key " + key);
    auto arr = std::get_if<MetaArray>(&it->second);
    if (!arr) throw std::runtime_error("GGUF: " + key + " is not an array");
    std::vector<double> out;
    for (const auto & v : *arr) {
        if (auto d = std::get_if<double>(&v)) out.push_back(*d);
        else if (auto i = std::get_if<int64_t>(&v)) out.push_back(static_cast<double>(*i));
        else if (auto u = std::get_if<uint64_t>(&v)) out.push_back(static_cast<double>(*u));
        else throw std::runtime_error("GGUF: " + key + " holds a non-number");
    }
    return out;
}

std::vector<uint64_t> u64_array(const GgufModel & g, const std::string & key) {
    std::vector<uint64_t> out;
    for (int64_t v : g.get_int_array(key)) out.push_back(static_cast<uint64_t>(v));
    return out;
}

}  // namespace

int Dsv41Config::engram_index(int il) const {
    for (size_t e = 0; e < engram_layers.size(); ++e)
        if (engram_layers[e] == il) return static_cast<int>(e);
    return -1;
}

Dsv41Config Dsv41Config::from_gguf(const GgufModel & g) {
    require(g.get_str("general.architecture") == "deepseek41", "architecture is not deepseek41");
    const std::string a = "deepseek41.";
    auto I = [&](const std::string & k) { return static_cast<int>(g.get_int(a + k)); };
    auto F = [&](const std::string & k) { return g.get_float(a + k); };

    Dsv41Config c;
    c.n_layer   = I("block_count");
    c.n_embd    = I("embedding_length");
    c.n_head    = I("attention.head_count");
    require(I("attention.head_count_kv") == 1, "more than one KV head");
    c.head_dim  = I("attention.key_length");
    require(I("attention.value_length") == c.head_dim, "key and value head sizes differ");
    c.rope_dims = I("rope.dimension_count");
    c.q_lora    = I("attention.q_lora_rank");
    c.o_groups  = I("attention.output_group_count");
    c.o_lora    = I("attention.output_lora_rank");
    c.window    = I("attention.sliding_window");
    c.rope_base          = F("rope.freq_base");
    c.compress_rope_base = F("attention.compress_rope_freq_base");
    require(g.get_str(a + "rope.scaling.type") == "yarn", "rope scaling is not yarn");
    c.yarn_factor    = F("rope.scaling.factor");
    c.yarn_orig_ctx  = I("rope.scaling.original_context_length");
    c.yarn_beta_fast = F("rope.scaling.yarn_beta_fast");
    c.yarn_beta_slow = F("rope.scaling.yarn_beta_slow");
    c.rms_eps        = F("attention.layer_norm_rms_epsilon");

    c.n_expert      = I("expert_count");
    c.n_expert_used = I("expert_used_count");
    c.n_ff_exp      = I("expert_feed_forward_length");
    require(I("expert_shared_count") == 1, "not exactly one shared expert");
    require(I("expert_gating_func") == 4, "expert gating is not sqrt(softplus)");
    require(g.get_int(a + "expert_weights_norm") != 0, "routing weights not normalized");
    c.expert_scale = F("expert_weights_scale");
    const auto clamp_e = float_array(g, a + "swiglu_clamp_exp"), clamp_s = float_array(g, a + "swiglu_clamp_shexp");
    require(static_cast<int>(clamp_e.size()) == c.n_layer && clamp_s.size() == clamp_e.size(), "swiglu clamp lengths");
    c.swiglu_limit = clamp_e[0];
    for (size_t i = 0; i < clamp_e.size(); ++i)
        require(clamp_e[i] == c.swiglu_limit && clamp_s[i] == c.swiglu_limit, "swiglu clamp differs between layers");

    c.hc          = I("hyper_connection.count");
    c.hc_sinkhorn = I("hyper_connection.sinkhorn_iterations");
    c.hc_eps      = F("hyper_connection.epsilon");
    require(c.hc == 4, "hyper_connection.count != 4");

    c.idx_heads = I("attention.indexer.head_count");
    c.idx_dim   = I("attention.indexer.key_length");
    c.idx_top_k = I("attention.indexer.top_k");
    for (int64_t r : g.get_int_array(a + "attention.compress_ratios")) c.compress_ratio.push_back(static_cast<int>(r));
    require(static_cast<int>(c.compress_ratio.size()) >= c.n_layer, "compress_ratios shorter than block_count");
    c.compress_ratio.resize(c.n_layer);   // the tail belongs to the DSpark layers, not in this file

    c.engram_heads = I("engram.head_count");
    c.engram_dim   = I("engram.key_length");
    c.engram_ngram = I("engram.max_ngram_size");
    c.engram_pad   = I("engram.pad_id");
    for (int64_t l : g.get_int_array(a + "engram.layer_ids")) c.engram_layers.push_back(static_cast<int>(l));
    c.engram_mult   = u64_array(g, a + "engram.multipliers");
    c.engram_prime  = u64_array(g, a + "engram.primes");
    c.engram_offset = u64_array(g, a + "engram.offsets");
    c.engram_token_map = g.get_int_array(a + "engram.token_map");
    const size_t ne = c.engram_layers.size(), ncol = static_cast<size_t>(c.engram_cols());
    require(ne > 0 && c.engram_ngram >= 2, "engram layout");
    require(c.engram_mult.size() == ne * c.engram_ngram, "engram multipliers != layers * ngram");
    require(c.engram_prime.size() == ne * ncol && c.engram_offset.size() == ne * ncol, "engram primes/offsets");
    for (uint64_t p : c.engram_prime) require(p != 0, "engram prime of zero");

    c.context_length = g.get_int(a + "context_length");
    c.bos = static_cast<int>(g.get_int("tokenizer.ggml.bos_token_id"));
    c.eos = static_cast<int>(g.get_int("tokenizer.ggml.eos_token_id"));

    const Tensor & emb = g.at("token_embd.weight");
    require(emb.shape.size() == 2 && emb.shape[0] == c.n_embd, "token_embd shape");
    c.n_vocab = static_cast<int>(emb.shape[1]);
    require(static_cast<int>(c.engram_token_map.size()) >= c.n_vocab, "engram token map shorter than the vocabulary");

    // which layer publishes the stream each layer reads: the file says it, by which tensors a layer carries
    // (llama.cpp deepseek41.cpp load_arch_tensors)
    c.kv_source.assign(c.n_layer, -1);
    c.idx_key_source.assign(c.n_layer, -1);
    c.topk_source.assign(c.n_layer, -1);
    int last_kv = -1, last_key = -1, last_idx = -1;
    for (int il = 0; il < c.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        if (g.find(p + "attn_compressor_kv.weight")) last_kv = il;
        if (g.find(p + "indexer.attn_k.weight")) last_key = il;
        if (g.find(p + "indexer.attn_q_b.weight")) last_idx = il;
        if (c.compress_ratio[il] == 0) continue;
        require(last_kv >= 0 && last_key >= 0 && last_idx >= 0, "layer " + std::to_string(il) + " reads a stream nobody publishes");
        require(c.compress_ratio[il] == c.compress_ratio[last_kv], "layer " + std::to_string(il) + " ratio differs from its source");
        c.kv_source[il] = last_kv;
        c.idx_key_source[il] = last_key;
        c.topk_source[il] = last_idx;
    }
    // not in the GGUF: the release's config.json (candidate_source_layer_id 20, candidate_topk_blocks 2048,
    // candidate_block_size 8), on the first ratio-1 layer, which is an indexer source in this checkpoint
    for (int il = 0; il < c.n_layer && c.cand_source < 0; ++il)
        if (c.compress_ratio[il] == 1 && c.is_topk_source(il)) c.cand_source = il;
    if (c.cand_source >= 0) {
        require(c.cand_source == 20, "the candidate source is layer 20 in the release (got " + std::to_string(c.cand_source) + ")");
        c.cand_blocks = 2048;
        c.cand_bs = 8;
    }

    // the shapes the kernels assume (docs/DSV41.md); a different checkpoint fails here, not mid-forward
    auto shape = [&](const std::string & n, std::vector<int64_t> s) {
        const Tensor & t = g.at(n);
        require(t.shape == s, n + " has an unexpected shape");
    };
    const int64_t E = c.n_embd, H = c.n_head, D = c.head_dim, HD = c.hc_dim(), MIX = c.hc_mix();
    for (int il = 0; il < c.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        shape(p + "attn_norm.weight", {E});
        shape(p + "attn_sinks.weight", {H});
        shape(p + "attn_q_a.weight", {E, c.q_lora});
        shape(p + "attn_q_a_norm.weight", {c.q_lora});
        shape(p + "attn_q_b.weight", {c.q_lora, H * D});
        shape(p + "attn_kv.weight", {E, D});
        shape(p + "attn_kv_a_norm.weight", {D});
        shape(p + "attn_output_a.weight", {H * D / c.o_groups, static_cast<int64_t>(c.o_groups) * c.o_lora});
        shape(p + "attn_output_b.weight", {static_cast<int64_t>(c.o_groups) * c.o_lora, E});
        for (const char * s : {"hc_attn_", "hc_ffn_"}) {
            shape(p + s + "fn.weight", {HD, MIX});
            shape(p + s + "base.weight", {MIX});
            shape(p + s + "scale.weight", {3});
        }
        shape(p + "ffn_norm.weight", {E});
        shape(p + "ffn_gate_inp.weight", {E, c.n_expert});
        shape(p + "exp_probs_b.bias", {c.n_expert});
        shape(p + "ffn_gate_exps.weight", {E, c.n_ff_exp, c.n_expert});
        shape(p + "ffn_up_exps.weight", {E, c.n_ff_exp, c.n_expert});
        shape(p + "ffn_down_exps.weight", {c.n_ff_exp, E, c.n_expert});
        shape(p + "ffn_gate_shexp.weight", {E, c.n_ff_exp});
        shape(p + "ffn_up_shexp.weight", {E, c.n_ff_exp});
        shape(p + "ffn_down_shexp.weight", {c.n_ff_exp, E});
        if (c.is_kv_source(il)) {
            shape(p + "attn_compressor_kv.weight", {E, D});
            shape(p + "attn_compressor_norm.weight", {D});
            if (c.compress_ratio[il] > 1) shape(p + "attn_compressor_gate.weight", {E, D});
            require(c.compress_ratio[il] <= 2, "compress ratio above 2");
        }
        if (c.owns_idx_key(il)) {
            shape(p + "indexer.attn_k.weight", {D, c.idx_dim});
            shape(p + "indexer.k_norm.weight", {c.idx_dim});
        }
        if (c.is_topk_source(il)) {
            shape(p + "indexer.attn_q_b.weight", {c.q_lora, static_cast<int64_t>(c.idx_heads) * c.idx_dim});
            shape(p + "indexer.proj.weight", {E, c.idx_heads});
        }
        const int eg = c.engram_index(il);
        if (eg >= 0) {
            shape(p + "engram_wkv.weight", {static_cast<int64_t>(c.engram_cols()) * c.engram_dim, E * (c.hc + 1)});
            shape(p + "engram_q.weight", {E, c.hc});
            shape(p + "engram_k.weight", {E, c.hc});
            // the tables live in their own files (8 and 9 of 10), which may not be downloaded yet: the engine
            // then runs without Engram and says so
            if (const Tensor * t = g.find(p + "engram_embd.weight")) {
                require(t->shape.size() == 2 && t->shape[0] == c.engram_dim, p + "engram_embd shape");
                for (size_t b = 0; b < ncol; ++b)   // the last bucket has to end inside the table
                    require(c.engram_offset[eg * ncol + b] + c.engram_prime[eg * ncol + b] <= static_cast<uint64_t>(t->shape[1]),
                            p + "engram_embd has too few rows for its buckets");
            }
        }
    }
    shape("output.weight", {E, c.n_vocab});
    shape("output_norm.weight", {E});
    return c;
}

void dsv41_engram_rows(const Dsv41Config & c, const int * tok, int p, uint64_t * rows) {
    const int n = c.engram_ngram, heads = c.engram_heads, ncol = c.engram_cols();
    auto map = [&](int t) { return static_cast<uint64_t>(c.engram_token_map.at(t)); };
    const uint64_t pad = map(c.engram_pad);
    std::vector<uint64_t> ctx(n);
    for (int s = 0; s < n; ++s) ctx[s] = p - s >= 0 ? map(tok[p - s]) : pad;   // look-back stops at the sequence start
    for (size_t e = 0; e < c.engram_layers.size(); ++e) {
        const uint64_t * mult = c.engram_mult.data() + e * n;
        const uint64_t * prime = c.engram_prime.data() + e * ncol;
        const uint64_t * off = c.engram_offset.data() + e * ncol;
        // compressed ids stay under 2^17 and the multipliers under 2^47, so no product overflows 64 bits
        uint64_t rolling = ctx[0] * mult[0];
        for (int s = 1; s < n; ++s) {
            rolling ^= ctx[s] * mult[s];
            for (int h = 0; h < heads; ++h) {
                const int b = (s - 1) * heads + h;
                rows[e * ncol + b] = rolling % prime[b] + off[b];
            }
        }
    }
}

}  // namespace bl
