// Dsv41Engine, phase A: the deepseek41 forward one token at a time, written to be checked against llama.cpp tensor
// by tensor (docs/DSV41.md). Speed is not the point yet: each layer's 6 routed experts are copied from the mapped
// file into VRAM (page cache or NVMe), the indexer's top-k is chosen on the host, and the stream syncs often.
#include "bl/dsv41.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <unordered_map>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <sys/mman.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "bl/cpu_experts.h"
#include "bl/cuda_ops.h"
#include "bl/cuda_pipeline.h"
#include "bl/cuda_quant.h"
#include "bl/dsv41_ops.h"

namespace bl {

namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

float bf16_to_float(uint16_t v) {
    const uint32_t b = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

// a weight matrix in VRAM, in its file format: M rows of K values
struct Mat {
    uint32_t type = 0;
    int      M = 0, K = 0;
    size_t   row_bytes = 0;
    void *   d = nullptr;
};

// the rotary frequencies of one kind of layer (precompute_freqs_cis): YaRN when orig_ctx > 0
std::vector<float> rope_freqs(int dim, double base, int orig_ctx, double factor, double beta_fast, double beta_slow) {
    std::vector<float> f(dim / 2);
    for (int i = 0; i < dim / 2; ++i)   // torch: 1 / base ** (arange(0, dim, 2) / dim), in fp32
        f[i] = 1.f / std::pow(static_cast<float>(base), static_cast<float>(2 * i) / static_cast<float>(dim));
    if (orig_ctx > 0) {
        auto corrected = [&](double rot) { return dim * std::log(orig_ctx / (rot * 2 * M_PI)) / (2 * std::log(base)); };
        const double low = std::max(std::floor(corrected(beta_fast)), 0.0);
        const double high = std::min(std::ceil(corrected(beta_slow)), static_cast<double>(dim - 1));
        for (int i = 0; i < dim / 2; ++i) {
            const float ramp = std::clamp(static_cast<float>((i - low) / std::max(high - low, 1e-3)), 0.f, 1.f);
            const float smooth = 1.f - ramp;
            f[i] = f[i] / static_cast<float>(factor) * (1.f - smooth) + f[i] * smooth;
        }
    }
    return f;
}

}  // namespace

struct Dsv41Engine::Impl {
    GgufModel   g;
    Dsv41Config c;
    int         max_ctx;
    cudaStream_t st = nullptr;
    std::vector<void *> allocs;
    std::map<std::string, Mat> mats;
    std::map<std::string, float *> vecs;
    Probe probe;

    // per layer: the routed experts in the mapped file
    struct LayerExperts { const uint8_t * gate = nullptr, * up = nullptr, * down = nullptr; };
    std::vector<LayerExperts> experts;
    size_t gu_expert_bytes = 0, d_expert_bytes = 0, gu_row_bytes = 0, d_row_bytes = 0;
    uint32_t expert_type = 0;

    // Engram tables in the mapped file (nullptr: not downloaded, Engram skipped)
    std::vector<const uint8_t *> engram_table;
    size_t engram_row_bytes = 0;
    bool engram_on = true;

    const uint8_t * embd = nullptr;   // token_embd, BF16 rows in the mapped file

    // device buffers
    float * R = nullptr, * R2 = nullptr;                   // the stream [hc][D] and its next value
    float * mix = nullptr, * pre = nullptr, * a_pre = nullptr, * post = nullptr, * comb = nullptr;
    float * x = nullptr, * qr = nullptr, * q = nullptr, * kvt = nullptr, * o = nullptr, * oa = nullptr, * ao = nullptr;
    float * lat = nullptr, * ik = nullptr, * iq = nullptr, * iw = nullptr, * isc = nullptr;
    float * logit_e = nullptr, * wsel = nullptr, * gbuf = nullptr, * ubuf = nullptr, * abuf = nullptr, * ebuf = nullptr;
    float * moe = nullptr, * shx = nullptr, * ffo = nullptr;
    int * ids = nullptr, * win_idx = nullptr, * topk_idx = nullptr;
    float * emb = nullptr, * ekv = nullptr;
    uint8_t * erows = nullptr, * erows_h = nullptr;
    // routed experts in VRAM: an LRU cache of slots, each one expert's [gate | up | down]; misses come from the mapped
    // file (page cache or disk). BL_DSV41_CACHE_GB sets its size (default: what is free after everything else, - 1.5 GB)
    uint8_t * xslots = nullptr;
    int n_slots = 0;
    std::vector<int> slot_key;                              // (layer * n_expert + expert) held by each slot, -1 free
    std::unordered_map<int, std::list<int>::iterator> where;
    std::list<int> lru;                                     // slots, most recently used first
    uint64_t hits = 0, misses = 0;
    // BL_DSV41_DP4A=1: the experts through the decode pipeline's dp4a kernels (q8_1 activations, 2.6x faster);
    // default the fp32 matvec, exact against a double reference (for the comparisons with llama.cpp)
    bool dp4a = std::getenv("BL_DSV41_DP4A") && std::atoi(std::getenv("BL_DSV41_DP4A")) != 0;
    // BL_DSV41_HYBRID=1: VRAM hits on the GPU (dp4a) while the CPU computes the misses in place in the mapped file
    // (the page cache is the RAM tier, the disk behind it the third); BL_DSV41_ADMIT misses per layer and token are
    // then copied into the VRAM cache (default 1)
    bool hybrid = std::getenv("BL_DSV41_HYBRID") && std::atoi(std::getenv("BL_DSV41_HYBRID")) != 0;
    int admit = std::getenv("BL_DSV41_ADMIT") ? std::atoi(std::getenv("BL_DSV41_ADMIT")) : 1;
    std::unique_ptr<CpuExperts> cpu;
    float * x_h = nullptr, * cpu_out_h = nullptr, * cpu_out_d = nullptr;
    double t_cpu = 0, t_gpu_wait = 0;
    uint64_t cpu_experts = 0;
    cuda::ExpertPlan * plan_h = nullptr, * plan_d = nullptr;
    void * xq = nullptr;
    float * pact = nullptr, * pout = nullptr;
    float * logits_d = nullptr;
    std::vector<float *> win;                               // per layer [window][head_dim] ring
    std::vector<float *> comp_kv, idx_keys;                 // per source layer: [max_ctx / ratio][...]
    std::vector<float *> cst_kv, cst_sc;                    // per ratio-2 source: the pending group [2][head_dim]
    std::vector<float *> engram_qk;                         // per engram layer [hc][D]
    float * freq_swa = nullptr, * freq_comp = nullptr;

    std::vector<int> toks;                                  // every token fed so far
    std::vector<float> logits;
    int n_topk = 0;

    // timing
    double t_expert_copy = 0, t_total = 0, t_engram = 0;
    int n_steps = 0;

    template <typename T> T * dalloc(size_t n) {
        void * p = nullptr;
        ck(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc");
        ck(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)), "cudaMemset");
        allocs.push_back(p);
        return static_cast<T *>(p);
    }

    const Tensor & need(const std::string & n) {
        const Tensor & t = g.at(n);
        if (!t.data) throw std::runtime_error(n + ": bytes not on disk (incomplete download?)");
        return t;
    }

    void upload_mat(const std::string & n) {
        const Tensor & t = need(n);
        Mat m;
        m.type = t.type_id;
        m.K = static_cast<int>(t.shape[0]);
        m.M = static_cast<int>(t.elements() / m.K);
        m.row_bytes = row_bytes(t.type_id, m.K);
        if (!cuda::supported(m.type)) throw std::runtime_error(n + ": unsupported type");
        ck(cudaMalloc(&m.d, t.nbytes), "cudaMalloc");
        allocs.push_back(m.d);
        ck(cudaMemcpy(m.d, t.data, t.nbytes, cudaMemcpyHostToDevice), "upload");
        mats[n] = m;
    }

    std::vector<float> host_floats(const std::string & n) {
        const Tensor & t = need(n);
        std::vector<float> f(static_cast<size_t>(t.elements()));
        if (t.type_id == 0) std::memcpy(f.data(), t.data, f.size() * 4);
        else if (t.type_id == 30) {
            for (size_t i = 0; i < f.size(); ++i) {
                uint16_t v;
                std::memcpy(&v, t.data + 2 * i, 2);
                f[i] = bf16_to_float(v);
            }
        } else throw std::runtime_error(n + ": vector of unexpected type");
        return f;
    }
    float * upload_floats(const std::vector<float> & f) {
        float * d = dalloc<float>(f.size());
        ck(cudaMemcpy(d, f.data(), f.size() * 4, cudaMemcpyHostToDevice), "upload");
        return d;
    }
    void upload_vec(const std::string & n) { vecs[n] = upload_floats(host_floats(n)); }

    const Mat & M(const std::string & n) const { return mats.at(n); }
    const float * V(const std::string & n) const { return vecs.at(n); }
    void mv(const std::string & n, const float * in, float * out) {
        const Mat & m = M(n);
        cuda::matvec(m.type, m.d, m.row_bytes, m.M, m.K, in, out, st);
    }
    void norm(const float * in, float * out, int D, const std::string & w) {
        cuda::rmsnorm_rows(in, out, 1, D, V(w), 1, static_cast<float>(c.rms_eps), 1.f, st);
    }
    void P(const std::string & name, int il, const float * d, int n) {
        if (!probe) return;
        ck(cudaStreamSynchronize(st), "probe sync");
        probe(il >= 0 ? name + "-" + std::to_string(il) : name, d, n);
    }

    Impl(const std::string & shard1, int ctx) : g(shard1), c(Dsv41Config::from_gguf(g)), max_ctx(ctx) {
        if (max_ctx > 16384) throw std::runtime_error("phase A: --ctx at most 16384 (the two-level candidate mask is not implemented)");
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        const int D = c.n_embd, HD = c.head_dim, H = c.n_head;

        // dense weights
        for (int il = 0; il < c.n_layer; ++il) {
            const std::string p = "blk." + std::to_string(il) + ".";
            for (const char * n : {"attn_q_a", "attn_q_b", "attn_kv", "attn_output_a", "attn_output_b", "hc_attn_fn",
                                   "hc_ffn_fn", "ffn_gate_inp", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"})
                upload_mat(p + n + ".weight");
            for (const char * n : {"attn_norm.weight", "ffn_norm.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
                                   "attn_sinks.weight", "hc_attn_base.weight", "hc_attn_scale.weight", "hc_ffn_base.weight",
                                   "hc_ffn_scale.weight", "exp_probs_b.bias"})
                upload_vec(p + n);
            if (c.is_kv_source(il)) {
                upload_mat(p + "attn_compressor_kv.weight");
                upload_vec(p + "attn_compressor_norm.weight");
                if (c.compress_ratio[il] > 1) upload_mat(p + "attn_compressor_gate.weight");
            }
            if (c.owns_idx_key(il)) {
                upload_mat(p + "indexer.attn_k.weight");
                upload_vec(p + "indexer.k_norm.weight");
            }
            if (c.is_topk_source(il)) {
                upload_mat(p + "indexer.attn_q_b.weight");
                upload_mat(p + "indexer.proj.weight");
            }
            if (c.engram_index(il) >= 0) {
                upload_mat(p + "engram_wkv.weight");
                // the reference only ever uses q_weight * k_weight
                auto qw = host_floats(p + "engram_q.weight"), kw = host_floats(p + "engram_k.weight");
                for (size_t i = 0; i < qw.size(); ++i) qw[i] *= kw[i];
                engram_qk.push_back(upload_floats(qw));
                const Tensor * t = g.find(p + "engram_embd.weight");
                engram_table.push_back(t && t->data ? t->data : nullptr);
                if (t) engram_row_bytes = row_bytes(t->type_id, t->shape[0]);
                if (!t || !t->data || t->type_id != 8) engram_on = false;
            }
            const Tensor & tg = need(p + "ffn_gate_exps.weight"), & tu = need(p + "ffn_up_exps.weight"),
                         & td = need(p + "ffn_down_exps.weight");
            if (il == 0) {
                expert_type = tg.type_id;
                if (tu.type_id != expert_type || td.type_id != expert_type) throw std::runtime_error("mixed expert formats");
                gu_row_bytes = row_bytes(expert_type, c.n_embd);
                d_row_bytes = row_bytes(expert_type, c.n_ff_exp);
                gu_expert_bytes = gu_row_bytes * c.n_ff_exp;
                d_expert_bytes = d_row_bytes * c.n_embd;
            }
            if (tg.type_id != expert_type || tu.type_id != expert_type || td.type_id != expert_type)
                throw std::runtime_error(p + ": expert formats differ from layer 0");
            experts.push_back({tg.data, tu.data, td.data});
        }
        if (!engram_on) std::fprintf(stderr, "dsv41: WARNING: Engram tables missing (files 8, 9) - running WITHOUT Engram; logits will be wrong\n");
        upload_mat("output.weight");
        upload_vec("output_norm.weight");
        embd = need("token_embd.weight").data;
        if (need("token_embd.weight").type_id != 30) throw std::runtime_error("token_embd is not BF16");

        // rotary tables
        freq_swa = upload_floats(rope_freqs(c.rope_dims, c.rope_base, 0, 1, 0, 0));
        freq_comp = upload_floats(rope_freqs(c.rope_dims, c.compress_rope_base, c.yarn_orig_ctx, c.yarn_factor,
                                             c.yarn_beta_fast, c.yarn_beta_slow));

        // state and scratch
        R = dalloc<float>(c.hc_dim()); R2 = dalloc<float>(c.hc_dim());
        mix = dalloc<float>(c.hc_mix()); pre = dalloc<float>(c.hc); a_pre = dalloc<float>(c.hc);
        post = dalloc<float>(c.hc); comb = dalloc<float>(c.hc * c.hc);
        x = dalloc<float>(D); qr = dalloc<float>(c.q_lora); q = dalloc<float>(static_cast<size_t>(H) * HD);
        kvt = dalloc<float>(HD); o = dalloc<float>(static_cast<size_t>(H) * HD);
        oa = dalloc<float>(static_cast<size_t>(c.o_groups) * c.o_lora); ao = dalloc<float>(D);
        lat = dalloc<float>(HD); ik = dalloc<float>(c.idx_dim);
        iq = dalloc<float>(static_cast<size_t>(c.idx_heads) * c.idx_dim); iw = dalloc<float>(c.idx_heads);
        isc = dalloc<float>(max_ctx);
        logit_e = dalloc<float>(c.n_expert); wsel = dalloc<float>(c.n_expert_used); ids = dalloc<int>(c.n_expert_used);
        gbuf = dalloc<float>(c.n_ff_exp); ubuf = dalloc<float>(c.n_ff_exp); abuf = dalloc<float>(c.n_ff_exp);
        ebuf = dalloc<float>(D); moe = dalloc<float>(D); shx = dalloc<float>(D); ffo = dalloc<float>(D);
        win_idx = dalloc<int>(c.window); topk_idx = dalloc<int>(c.idx_top_k);
        {
            std::vector<int> w(c.window);
            std::iota(w.begin(), w.end(), 0);
            ck(cudaMemcpy(win_idx, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "upload");
        }
        const int ncol = c.engram_cols();
        emb = dalloc<float>(static_cast<size_t>(ncol) * c.engram_dim);
        ekv = dalloc<float>(static_cast<size_t>(c.hc + 1) * D);
        erows = dalloc<uint8_t>(static_cast<size_t>(ncol) * std::max<size_t>(engram_row_bytes, 1));
        ck(cudaMallocHost(&erows_h, static_cast<size_t>(ncol) * std::max<size_t>(engram_row_bytes, 1)), "pinned");
        logits_d = dalloc<float>(c.n_vocab);
        logits.resize(c.n_vocab);

        win.resize(c.n_layer, nullptr);
        comp_kv.resize(c.n_layer, nullptr); idx_keys.resize(c.n_layer, nullptr);
        cst_kv.resize(c.n_layer, nullptr); cst_sc.resize(c.n_layer, nullptr);
        for (int il = 0; il < c.n_layer; ++il) {
            win[il] = dalloc<float>(static_cast<size_t>(c.window) * HD);
            if (c.is_kv_source(il)) {
                const int r = c.compress_ratio[il];
                comp_kv[il] = dalloc<float>(static_cast<size_t>(max_ctx / r + 1) * HD);
                if (r > 1) { cst_kv[il] = dalloc<float>(2 * HD); cst_sc[il] = dalloc<float>(2 * HD); }
            }
            if (c.owns_idx_key(il)) idx_keys[il] = dalloc<float>(static_cast<size_t>(max_ctx / c.compress_ratio[il] + 1) * c.idx_dim);
        }
        ck(cudaDeviceSynchronize(), "load");

        if (hybrid) {
            dp4a = true;
            const int nt = std::getenv("BL_CPU_THREADS") ? std::atoi(std::getenv("BL_CPU_THREADS")) : 8;
            cpu = std::make_unique<CpuExperts>(nt);
            ck(cudaMallocHost(&x_h, sizeof(float) * D), "pinned");
            ck(cudaMallocHost(&cpu_out_h, sizeof(float) * c.n_expert_used * D), "pinned");
            cpu_out_d = dalloc<float>(static_cast<size_t>(c.n_expert_used) * D);
        }
        if (dp4a) {
            ck(cudaMallocHost(&plan_h, sizeof(cuda::ExpertPlan)), "pinned");
            plan_d = dalloc<cuda::ExpertPlan>(1);
            xq = dalloc<uint8_t>(cuda::q8_bytes(D));
            pact = dalloc<float>(static_cast<size_t>(cuda::kMaxSlots) * cuda::kMaxT * c.n_ff_exp);
            pout = dalloc<float>(static_cast<size_t>(cuda::kMaxSlots) * cuda::kMaxT * D);
        }
        // the expert cache takes the rest of VRAM
        const size_t per = 2 * gu_expert_bytes + d_expert_bytes;
        size_t fr = 0, to = 0;
        ck(cudaMemGetInfo(&fr, &to), "cudaMemGetInfo");
        size_t want = fr > (3ull << 29) ? fr - (3ull << 29) : 0;
        if (const char * e = std::getenv("BL_DSV41_CACHE_GB")) want = std::min(want, static_cast<size_t>(std::atof(e) * (1ull << 30)));
        n_slots = std::max(static_cast<int>(want / per), c.n_expert_used);
        xslots = dalloc<uint8_t>(static_cast<size_t>(n_slots) * per);
        slot_key.assign(n_slots, -1);
        for (int i = 0; i < n_slots; ++i) lru.push_back(i);
        std::fprintf(stderr, "dsv41: expert cache %d slots (%.1f GB, %.1f%% of all experts)\n", n_slots,
                     static_cast<double>(n_slots) * per / 1e9, 100.0 * n_slots / (static_cast<double>(c.n_layer) * c.n_expert));
    }

    // the VRAM slot holding expert e of layer il; copied in on a miss. Slots used in this layer stay put: the LRU
    // tail is never one of them because they were just moved to the front.
    uint8_t * expert_slot(int il, int e) {
        const size_t per = 2 * gu_expert_bytes + d_expert_bytes;
        const int key = il * c.n_expert + e;
        auto it = where.find(key);
        if (it != where.end()) {
            ++hits;
            lru.splice(lru.begin(), lru, it->second);
            return xslots + static_cast<size_t>(*it->second) * per;
        }
        ++misses;
        const int slot = lru.back();
        lru.pop_back();
        if (slot_key[slot] >= 0) where.erase(slot_key[slot]);
        slot_key[slot] = key;
        lru.push_front(slot);
        where[key] = lru.begin();
        uint8_t * dst = xslots + static_cast<size_t>(slot) * per;
        const size_t ee = static_cast<size_t>(e);
        ck(cudaMemcpyAsync(dst, experts[il].gate + ee * gu_expert_bytes, gu_expert_bytes, cudaMemcpyHostToDevice, st), "expert");
        ck(cudaMemcpyAsync(dst + gu_expert_bytes, experts[il].up + ee * gu_expert_bytes, gu_expert_bytes, cudaMemcpyHostToDevice, st), "expert");
        ck(cudaMemcpyAsync(dst + 2 * gu_expert_bytes, experts[il].down + ee * d_expert_bytes, d_expert_bytes, cudaMemcpyHostToDevice, st), "expert");
        return dst;
    }

    ~Impl() {
        if (st) cudaStreamSynchronize(st);
        for (void * p : allocs) cudaFree(p);
        if (erows_h) cudaFreeHost(erows_h);
        if (plan_h) cudaFreeHost(plan_h);
        if (x_h) cudaFreeHost(x_h);
        if (cpu_out_h) cudaFreeHost(cpu_out_h);
        if (st) cudaStreamDestroy(st);
    }

    void reset() {
        toks.clear();
        n_topk = 0;
        for (int il = 0; il < c.n_layer; ++il) {   // only the windows matter (they are read before they are filled)
            ck(cudaMemsetAsync(win[il], 0, sizeof(float) * c.window * c.head_dim, st), "reset");
        }
        ck(cudaStreamSynchronize(st), "reset");
    }

    void engram(int il, int eg, const uint64_t * rows) {
        const int ncol = c.engram_cols(), D = c.n_embd;
        const auto t0 = std::chrono::steady_clock::now();
        for (int b = 0; b < ncol; ++b)
            std::memcpy(erows_h + b * engram_row_bytes, engram_table[eg] + rows[eg * ncol + b] * engram_row_bytes, engram_row_bytes);
        t_engram += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ck(cudaMemcpyAsync(erows, erows_h, ncol * engram_row_bytes, cudaMemcpyHostToDevice, st), "engram rows");
        cuda::dequant_rows(8 /* Q8_0 */, erows, engram_row_bytes, ncol, c.engram_dim, emb, st);
        mv("blk." + std::to_string(il) + ".engram_wkv.weight", emb, ekv);
        cuda::dsv41::engram_gate(R, ekv, engram_qk[eg], c.hc, D, static_cast<float>(c.rms_eps), st);
        P("engram_out", il, R, c.hc_dim());
    }

    void attention(int il, int pos) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int HD = c.head_dim, H = c.n_head, rd = c.rope_dims, ratio = c.compress_ratio[il];
        const float * freq = ratio ? freq_comp : freq_swa;

        mv(p + "attn_q_a.weight", x, qr);
        norm(qr, qr, c.q_lora, p + "attn_q_a_norm.weight");
        P("qr", il, qr, c.q_lora);
        mv(p + "attn_q_b.weight", qr, q);
        cuda::dsv41::rope_tail(q, H, HD, rd, pos, freq, false, st);
        P("q", il, q, H * HD);

        mv(p + "attn_kv.weight", x, kvt);
        norm(kvt, kvt, HD, p + "attn_kv_a_norm.weight");
        cuda::dsv41::rope_tail(kvt, 1, HD, rd, pos, freq, false, st);
        P("kv", il, kvt, HD);
        ck(cudaMemcpyAsync(win[il] + static_cast<size_t>(pos % c.window) * HD, kvt, HD * 4, cudaMemcpyDeviceToDevice, st), "window");
        const int n_win = std::min(pos + 1, c.window);

        int n_comp_sel = 0;
        const float * comp = nullptr;
        if (ratio > 0) {
            const int src = c.kv_source[il];
            if (c.is_kv_source(il)) compress(il, pos);
            comp = comp_kv[src];
            const int n_vis = (pos + 1) / ratio;
            if (c.is_topk_source(il)) select_topk(il, pos, n_vis);
            n_comp_sel = n_topk;
        }
        cuda::dsv41::attention_sink(q, win[il], win_idx, n_win, comp, topk_idx, n_comp_sel, V(p + "attn_sinks.weight"), o, H,
                                    HD, 1.f / std::sqrt(static_cast<float>(HD)), st);
        P("attn_out_raw", il, o, H * HD);
        cuda::dsv41::rope_tail(o, H, HD, rd, pos, freq, true, st);
        P("attn_derope", il, o, H * HD);
        // wo_a is block diagonal over the groups: group g's rows only see its own heads
        const Mat & wa = M(p + "attn_output_a.weight");
        const int gin = H * HD / c.o_groups;
        for (int gi = 0; gi < c.o_groups; ++gi)
            cuda::matvec(wa.type, static_cast<const uint8_t *>(wa.d) + wa.row_bytes * c.o_lora * gi, wa.row_bytes, c.o_lora,
                         wa.K, o + static_cast<size_t>(gi) * gin, oa + static_cast<size_t>(gi) * c.o_lora, st);
        P("attn_wo_a", il, oa, c.o_groups * c.o_lora);
        mv(p + "attn_output_b.weight", oa, ao);
        P("attn_out", il, ao, c.n_embd);
    }

    // a KV source layer: its compressor (and, when it owns them, the indexer keys) for the group this token completes
    void compress(int il, int pos) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int HD = c.head_dim, ratio = c.compress_ratio[il], rd = c.rope_dims;
        if (ratio == 1) {
            mv(p + "attn_compressor_kv.weight", x, lat);
        } else {
            const int slot = pos % ratio;
            mv(p + "attn_compressor_kv.weight", x, cst_kv[il] + slot * HD);
            mv(p + "attn_compressor_gate.weight", x, cst_sc[il] + slot * HD);
            if ((pos + 1) % ratio != 0) return;   // the group is still filling
            cuda::dsv41::pool2(cst_kv[il], cst_sc[il], lat, HD, st);
        }
        norm(lat, lat, HD, p + "attn_compressor_norm.weight");
        const int j = pos / ratio, jpos = j * ratio;   // a latent stands for the first token of its group
        P("comp_kv_pre", il, lat, HD);
        if (c.owns_idx_key(il)) {   // index keys from the latent before its RoPE
            mv(p + "indexer.attn_k.weight", lat, ik);
            norm(ik, ik, c.idx_dim, p + "indexer.k_norm.weight");
            cuda::dsv41::rope_tail(ik, 1, c.idx_dim, rd, jpos, freq_comp, false, st);
            P("idx_k_new", il, ik, c.idx_dim);
            ck(cudaMemcpyAsync(idx_keys[il] + static_cast<size_t>(j) * c.idx_dim, ik, c.idx_dim * 4, cudaMemcpyDeviceToDevice, st), "idx k");
        }
        cuda::dsv41::rope_tail(lat, 1, HD, rd, jpos, freq_comp, false, st);
        P("comp_kv", il, lat, HD);
        ck(cudaMemcpyAsync(comp_kv[il] + static_cast<size_t>(j) * HD, lat, HD * 4, cudaMemcpyDeviceToDevice, st), "comp kv");
    }

    // an indexer source layer: the top-k compressed positions this query attends to (shared with the layers after it)
    void select_topk(int il, int pos, int n_vis) {
        n_topk = 0;
        if (n_vis == 0) return;
        const std::string p = "blk." + std::to_string(il) + ".";
        mv(p + "indexer.attn_q_b.weight", qr, iq);
        cuda::dsv41::rope_tail(iq, c.idx_heads, c.idx_dim, c.rope_dims, pos, freq_comp, false, st);
        mv(p + "indexer.proj.weight", x, iw);
        const float wscale = 1.f / std::sqrt(static_cast<float>(c.idx_dim) * c.idx_heads);
        cuda::dsv41::indexer_score(iq, iw, wscale, idx_keys[c.idx_key_source[il]], n_vis, c.idx_heads, c.idx_dim, isc, st);
        P("idx_score", il, isc, n_vis);
        // on the GPU, in position order like the reference's re-sort; how many is known here, so no sync
        const int k = std::min(c.idx_top_k, n_vis);
        cuda::dsv41::topk_sorted(isc, n_vis, k, topk_idx, st);
        n_topk = k;
    }

    void moe_ffn(int il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int D = c.n_embd, F = c.n_ff_exp, K = c.n_expert_used;
        const float lim = static_cast<float>(c.swiglu_limit);
        mv(p + "ffn_gate_inp.weight", x, logit_e);
        cuda::dsv41::router_topk(logit_e, V(p + "exp_probs_b.bias"), c.n_expert, K, static_cast<float>(c.expert_scale), ids, wsel, st);
        std::vector<int> hid(K);
        std::vector<float> hw(K);
        ck(cudaMemcpyAsync(hid.data(), ids, K * 4, cudaMemcpyDeviceToHost, st), "ids");
        ck(cudaMemcpyAsync(hw.data(), wsel, K * 4, cudaMemcpyDeviceToHost, st), "w");
        ck(cudaStreamSynchronize(st), "router");

        if (hybrid) {
            moe_hybrid(il, hid, hw);
            shared_and_out(il);
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<const uint8_t *> base_k(K);
        for (int k = 0; k < K; ++k) base_k[k] = expert_slot(il, hid[k]);
        ck(cudaStreamSynchronize(st), "experts");
        t_expert_copy += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        ck(cudaMemsetAsync(moe, 0, D * 4, st), "moe");
        if (dp4a) {   // one plan of K resident slots (the previous layer's plan copy finished at the router's sync)
            std::memset(plan_h, 0, sizeof *plan_h);
            plan_h->T = 1; plan_h->K = K; plan_h->n_slots = K;
            for (int k = 0; k < K; ++k) {
                plan_h->ids[k] = hid[k]; plan_h->kind[k] = cuda::kHit; plan_h->mask[k] = 1; plan_h->base[k] = base_k[k];
                plan_h->sel[0][k] = k; plan_h->w[0][k] = hw[k];
            }
            ck(cudaMemcpyAsync(plan_d, plan_h, sizeof *plan_h, cudaMemcpyHostToDevice, st), "plan");
            cuda::quantize_q8(x, D, xq, st);
            cuda::plan_gate_up(expert_type, plan_d, 1 << cuda::kHit, 1, gu_expert_bytes, F, D, xq, pact, st, lim);
            cuda::plan_down(expert_type, plan_d, 1 << cuda::kHit, 1, gu_expert_bytes, F, D, pact, pout, st);
            for (int k = 0; k < K; ++k) cuda::axpy(moe, pout + static_cast<size_t>(k) * cuda::kMaxT * D, hw[k], D, st);
        }
        for (int k = 0; k < (dp4a ? 0 : K); ++k) {
            const uint8_t * base = base_k[k];
            cuda::matvec(expert_type, base, gu_row_bytes, F, D, x, gbuf, st);
            cuda::matvec(expert_type, base + gu_expert_bytes, gu_row_bytes, F, D, x, ubuf, st);
            cuda::dsv41::swiglu_clamp(gbuf, ubuf, abuf, F, lim, st);
            cuda::matvec(expert_type, base + 2 * gu_expert_bytes, d_row_bytes, D, F, abuf, ebuf, st);
            cuda::axpy(moe, ebuf, hw[k], D, st);
        }
        shared_and_out(il);
    }

    void shared_and_out(int il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int D = c.n_embd, F = c.n_ff_exp;
        const float lim = static_cast<float>(c.swiglu_limit);
        P("ffn_moe_out", il, moe, D);
        mv(p + "ffn_gate_shexp.weight", x, gbuf);
        mv(p + "ffn_up_shexp.weight", x, ubuf);
        cuda::dsv41::swiglu_clamp(gbuf, ubuf, abuf, F, lim, st);
        mv(p + "ffn_down_shexp.weight", abuf, shx);
        P("ffn_shexp", il, shx, D);
        ck(cudaMemcpyAsync(ffo, moe, D * 4, cudaMemcpyDeviceToDevice, st), "ffn");
        cuda::add(ffo, shx, D, st);
        P("ffn_out", il, ffo, D);
    }

    // hits on the GPU, misses on the CPU at the same time; then a few misses admitted into the VRAM cache
    void moe_hybrid(int il, const std::vector<int> & hid, const std::vector<float> & hw) {
        const int D = c.n_embd, F = c.n_ff_exp, K = c.n_expert_used;
        const float lim = static_cast<float>(c.swiglu_limit);
        std::vector<int> gk, ck_;   // indices k of the hits / the misses
        std::vector<const uint8_t *> gbase;
        for (int k = 0; k < K; ++k) {
            auto it = where.find(il * c.n_expert + hid[k]);
            if (it != where.end()) {
                ++hits;
                lru.splice(lru.begin(), lru, it->second);
                gk.push_back(k);
                gbase.push_back(xslots + static_cast<size_t>(*it->second) * (2 * gu_expert_bytes + d_expert_bytes));
            } else {
                ++misses;
                ck_.push_back(k);
            }
        }
        ck(cudaMemsetAsync(moe, 0, D * 4, st), "moe");
        if (!gk.empty()) {   // the GPU part is queued first and runs while the CPU works
            std::memset(plan_h, 0, sizeof *plan_h);
            const int n = static_cast<int>(gk.size());
            plan_h->T = 1; plan_h->K = n; plan_h->n_slots = n;
            for (int i = 0; i < n; ++i) {
                plan_h->ids[i] = hid[gk[i]]; plan_h->kind[i] = cuda::kHit; plan_h->mask[i] = 1; plan_h->base[i] = gbase[i];
                plan_h->sel[0][i] = i; plan_h->w[0][i] = hw[gk[i]];
            }
            ck(cudaMemcpyAsync(plan_d, plan_h, sizeof *plan_h, cudaMemcpyHostToDevice, st), "plan");
            cuda::quantize_q8(x, D, xq, st);
            cuda::plan_gate_up(expert_type, plan_d, 1 << cuda::kHit, 1, gu_expert_bytes, F, D, xq, pact, st, lim);
            cuda::plan_down(expert_type, plan_d, 1 << cuda::kHit, 1, gu_expert_bytes, F, D, pact, pout, st);
            for (int i = 0; i < n; ++i) cuda::axpy(moe, pout + static_cast<size_t>(i) * cuda::kMaxT * D, hw[gk[i]], D, st);
        }
        if (!ck_.empty()) {
            const auto t0 = std::chrono::steady_clock::now();
            // x was final before the router's sync; a pinned copy is a few microseconds
            ck(cudaMemcpy(x_h, x, D * 4, cudaMemcpyDeviceToHost), "x");
            const int n = static_cast<int>(ck_.size());
            std::vector<CpuExperts::Job> jobs(n);
            const long pg = sysconf(_SC_PAGESIZE);
            auto willneed = [&](const uint8_t * a, size_t len) {   // start the disk reads of all misses at once
                const uintptr_t b = reinterpret_cast<uintptr_t>(a) & ~static_cast<uintptr_t>(pg - 1);
                madvise(reinterpret_cast<void *>(b), len + (reinterpret_cast<uintptr_t>(a) - b), MADV_WILLNEED);
            };
            for (int j = 0; j < n; ++j) {
                const size_t e = static_cast<size_t>(hid[ck_[j]]);
                const uint8_t * gp = experts[il].gate + e * gu_expert_bytes, * up = experts[il].up + e * gu_expert_bytes,
                              * dp = experts[il].down + e * d_expert_bytes;
                willneed(gp, gu_expert_bytes); willneed(up, gu_expert_bytes); willneed(dp, d_expert_bytes);
                jobs[j] = {gp, 1, cpu_out_h + static_cast<size_t>(j) * D, up, dp};
            }
            cpu->run(expert_type, expert_type, gu_expert_bytes, F, D, x_h, 1, jobs.data(), n, lim);
            cpu_experts += n;
            t_cpu += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            ck(cudaMemcpyAsync(cpu_out_d, cpu_out_h, sizeof(float) * n * D, cudaMemcpyHostToDevice, st), "cpu out");
            for (int j = 0; j < n; ++j) cuda::axpy(moe, cpu_out_d + static_cast<size_t>(j) * D, hw[ck_[j]], D, st);
            // admit a few: their pages are hot now, so the copies come from RAM
            for (int j = 0; j < std::min(n, admit); ++j) {
                expert_slot(il, hid[ck_[j]]);
                --misses;   // expert_slot counted a miss for what was already counted above
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        ck(cudaStreamSynchronize(st), "moe");   // cpu_out_h is rewritten by the next layer
        t_gpu_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    }

    const std::vector<float> & step(int token) {
        if (static_cast<int>(toks.size()) >= max_ctx) throw std::runtime_error("context full");
        if (token < 0 || token >= c.n_vocab) throw std::runtime_error("token id out of range");
        const auto t0 = std::chrono::steady_clock::now();
        const int pos = static_cast<int>(toks.size()), D = c.n_embd, hc = c.hc;
        toks.push_back(token);

        std::vector<uint64_t> erow(c.engram_layers.size() * c.engram_cols());
        if (engram_on) {
            dsv41_engram_rows(c, toks.data(), pos, erow.data());
            // every row of both Engram layers is known now: start all their disk reads at once, so layer 1 waits
            // about one read instead of 24 in a row, and layer 14's arrive while layers 0..13 compute
            const long pg = sysconf(_SC_PAGESIZE);
            const int ncol = c.engram_cols();
            for (size_t e = 0; e < c.engram_layers.size(); ++e)
                for (int b = 0; b < ncol; ++b) {
                    const uint8_t * a = engram_table[e] + erow[e * ncol + b] * engram_row_bytes;
                    const uintptr_t pa = reinterpret_cast<uintptr_t>(a) & ~static_cast<uintptr_t>(pg - 1);
                    madvise(reinterpret_cast<void *>(pa), engram_row_bytes + (reinterpret_cast<uintptr_t>(a) - pa), MADV_WILLNEED);
                }
        }

        {   // the embedding row, repeated into the hc copies
            std::vector<float> e(D);
            const uint8_t * row = embd + static_cast<size_t>(token) * D * 2;
            for (int i = 0; i < D; ++i) {
                uint16_t v;
                std::memcpy(&v, row + 2 * i, 2);
                e[i] = bf16_to_float(v);
            }
            ck(cudaMemcpyAsync(x, e.data(), D * 4, cudaMemcpyHostToDevice, st), "embd");
            cuda::repeat(x, R, D, hc, st);
            ck(cudaStreamSynchronize(st), "embd");
        }
        {   // layer 0 has no previous sublayer: the identity mix picks copy 0
            std::vector<float> one(hc, 0.f);
            one[0] = 1.f;
            ck(cudaMemcpyAsync(pre, one.data(), hc * 4, cudaMemcpyHostToDevice, st), "pre");
            ck(cudaStreamSynchronize(st), "pre");
        }
        const float eps = static_cast<float>(c.rms_eps), hc_eps = static_cast<float>(c.hc_eps);

        for (int il = 0; il < c.n_layer; ++il) {
            const std::string p = "blk." + std::to_string(il) + ".";
            const int eg = c.engram_index(il);
            if (eg >= 0 && engram_on) engram(il, eg, erow.data());

            // attention sublayer: collapses with the mix the previous sublayer made
            mv(p + "hc_attn_fn.weight", R, mix);
            cuda::dsv41::hc_split(mix, R, V(p + "hc_attn_scale.weight"), V(p + "hc_attn_base.weight"), hc, D, eps, hc_eps,
                                  c.hc_sinkhorn, a_pre, post, comb, st);
            cuda::dsv41::hc_pre(R, pre, x, hc, D, st);
            P("hc_attn_pre", il, x, D);
            norm(x, x, D, p + "attn_norm.weight");
            P("attn_norm", il, x, D);
            attention(il, pos);
            cuda::dsv41::hc_post(ao, R, post, comb, R2, hc, D, st);
            std::swap(R, R2);
            P("hc_attn_post", il, R, c.hc_dim());

            // FFN sublayer: collapses with the attention's mix; its own pre goes to the next layer's attention
            mv(p + "hc_ffn_fn.weight", R, mix);
            cuda::dsv41::hc_split(mix, R, V(p + "hc_ffn_scale.weight"), V(p + "hc_ffn_base.weight"), hc, D, eps, hc_eps,
                                  c.hc_sinkhorn, pre, post, comb, st);
            cuda::dsv41::hc_pre(R, a_pre, x, hc, D, st);
            P("hc_ffn_pre", il, x, D);
            norm(x, x, D, p + "ffn_norm.weight");
            P("ffn_norm", il, x, D);
            moe_ffn(il);
            cuda::dsv41::hc_post(ffo, R, post, comb, R2, hc, D, st);
            std::swap(R, R2);
            P("l_last", il, R, c.hc_dim());
        }
        // the last FFN's mix collapses the copies (there is no separate hc head)
        cuda::dsv41::hc_pre(R, pre, x, hc, D, st);
        P("hc_out", -1, x, D);
        norm(x, x, D, "output_norm.weight");
        P("result_norm", -1, x, D);
        mv("output.weight", x, logits_d);
        P("result_output", -1, logits_d, c.n_vocab);
        ck(cudaMemcpyAsync(logits.data(), logits_d, c.n_vocab * 4, cudaMemcpyDeviceToHost, st), "logits");
        ck(cudaStreamSynchronize(st), "step");
        t_total += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++n_steps;
        return logits;
    }
};

Dsv41Engine::Dsv41Engine(const std::string & shard1, int max_ctx) : p_(std::make_unique<Impl>(shard1, max_ctx)) {}
Dsv41Engine::~Dsv41Engine() = default;
const Dsv41Config & Dsv41Engine::config() const { return p_->c; }
void Dsv41Engine::reset() { p_->reset(); }
int Dsv41Engine::position() const { return static_cast<int>(p_->toks.size()); }
const std::vector<float> & Dsv41Engine::step(int token) { return p_->step(token); }
void Dsv41Engine::set_probe(Probe p) { p_->probe = std::move(p); }

std::string Dsv41Engine::report() const {
    std::ostringstream os;
    const auto & q = *p_;
    if (q.n_steps == 0) return "no steps";
    os << q.n_steps << " steps, " << 1000.0 * q.t_total / q.n_steps << " ms/token (expert copies "
       << 1000.0 * q.t_expert_copy / q.n_steps << " ms, Engram rows " << 1000.0 * q.t_engram / q.n_steps << " ms); "
       << "expert cache hit rate " << (q.hits + q.misses ? 100.0 * q.hits / (q.hits + q.misses) : 0.0) << "% of "
       << q.hits + q.misses;
    if (q.hybrid)
        os << "; CPU experts " << static_cast<double>(q.cpu_experts) / q.n_steps << "/token in " << 1000.0 * q.t_cpu / q.n_steps
           << " ms, then GPU wait " << 1000.0 * q.t_gpu_wait / q.n_steps << " ms";
    return os.str();
}

}  // namespace bl
