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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "bl/cpu_experts.h"
#include "bl/cuda_ops.h"
#include "bl/cuda_pipeline.h"
#include "bl/cuda_prefill.h"
#include "bl/cuda_quant.h"
#include "bl/dsv41_ops.h"
#include "ram_tier.h"

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
    struct LayerExperts {
        const uint8_t * gate = nullptr, * up = nullptr, * down = nullptr;   // in the mapped files
        int      fd[3] = {-1, -1, -1};                                     // O_DIRECT descriptors of their shards
        uint64_t off[3] = {};                                               // file offsets of expert 0's part
    };
    // RAM tier (src/dsv41/ram_tier.h): BL_DSV41_RAM_GB (default: MemAvailable - 12 GB), BL_DSV41_IO_THREADS (2, with the reads cut into 1.5 MB pieces: this NVMe reads fastest with 3-6 MB in flight)
    std::unique_ptr<ExpertRam> ram;
    std::vector<int> direct_fd;   // per opened shard
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
    float * hc_ss = nullptr;   // hc_mix's sum of squares
    uint32_t * route_h = nullptr;
    float * logits_h = nullptr, * step_h = nullptr;   // pinned: the logits; a step's embedding row and first mix   // pinned: [0, 64) the picked ids, [64, 128) their weights, [128, 192) the prediction
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
    // admissions (hybrid decode) copy on their own stream so the compute stream does not queue behind them; a slot
    // whose copy may still be running carries its event, and the compute stream waits on it before the first use
    cudaStream_t cst = nullptr;
    std::vector<cudaEvent_t> slot_ready;
    std::vector<char> slot_pending;
    cudaEvent_t ev_compute = nullptr, ev_admit = nullptr;   // compute stream's progress / the last admission
    bool admit_pending = false;
    void use_slot(int slot) {
        if (!slot_pending[slot]) return;
        ck(cudaStreamWaitEvent(st, slot_ready[slot], 0), "wait admission");
        slot_pending[slot] = 0;
    }
    // BL_DSV41_DP4A=1: the experts through the decode pipeline's dp4a kernels (q8_1 activations, 2.6x faster);
    // default the fp32 matvec, exact against a double reference (for the comparisons with llama.cpp)
    bool dp4a = std::getenv("BL_DSV41_DP4A") && std::atoi(std::getenv("BL_DSV41_DP4A")) != 0;
    // BL_DSV41_HYBRID=1: VRAM hits on the GPU (dp4a) while the CPU computes the misses in place in the mapped file
    // (the page cache is the RAM tier, the disk behind it the third); BL_DSV41_ADMIT misses per layer and token are
    // then copied into the VRAM cache (default 1)
    bool hybrid = std::getenv("BL_DSV41_HYBRID") && std::atoi(std::getenv("BL_DSV41_HYBRID")) != 0;
    int admit = std::getenv("BL_DSV41_ADMIT") ? std::atoi(std::getenv("BL_DSV41_ADMIT")) : 1;
    // BL_DSV41_EXCLUSIVE (default 1): an admitted expert leaves the RAM tier and one evicted from VRAM is copied back
    // into it, so the two tiers hold different experts (28% fewer disk reads in tools/dsv41_cache_sim.py)
    bool exclusive = !std::getenv("BL_DSV41_EXCLUSIVE") || std::atoi(std::getenv("BL_DSV41_EXCLUSIVE")) != 0;
    uint64_t demoted = 0;
    std::unique_ptr<CpuExperts> cpu;
    float * x_h = nullptr, * cpu_out_h = nullptr, * cpu_out_d = nullptr;
    double t_cpu = 0, t_gpu_wait = 0, t_ram = 0, t_router = 0, t_admit = 0;
    // host time per stretch of a decode step (launches included): engram, attention sublayer, FFN pre, moe_ffn
    double t_host[4] = {};
    // BL_DSV41_GPUTIME=1 (measurement): GPU time from the end of one layer's MoE to the next layer's router
    bool gputime = std::getenv("BL_DSV41_GPUTIME") != nullptr;
    cudaEvent_t ev_a = nullptr, ev_b = nullptr;
    bool ev_a_set = false;
    double t_gpu_pre = 0, t_span = 0;
    double t_gpu_wait_by[3] = {};   // layers with no miss / misses all on the CPU / an early admission
    uint64_t n_gpu_wait_by[3] = {};
    std::chrono::steady_clock::time_point span0;   // t_router: waiting for the GPU at the router
    // BL_DSV41_PRED=1 (measurement): layer il+1's router on layer il's FFN input; are the misses that go to disk among
    // its picks? (prefetch candidates)
    bool pred_stats = std::getenv("BL_DSV41_PRED") && std::atoi(std::getenv("BL_DSV41_PRED")) != 0;
    int pred_k = std::getenv("BL_DSV41_PRED_K") ? std::atoi(std::getenv("BL_DSV41_PRED_K")) : 6;
    // BL_DSV41_PREFETCH=K (hybrid decode, default 2; 0 = off): that prediction's top K are read into the RAM tier in the
    // background while this layer computes
    int prefetch_k = std::getenv("BL_DSV41_PREFETCH") ? std::atoi(std::getenv("BL_DSV41_PREFETCH")) : 2;
    // BL_DSV41_PREFETCH_DIST=d (default 2): the prediction is for layer il + d (a read takes ~6 ms, a layer ~3.5);
    // the measurements (BL_DSV41_PRED, BL_DSV41_TRACE) assume 1
    int prefetch_dist = std::getenv("BL_DSV41_PREFETCH_DIST") ? std::atoi(std::getenv("BL_DSV41_PREFETCH_DIST")) : 2;
    // BL_DSV41_TRACE=FILE (measurement): per decode step and layer "pos layer: 6 picks | next layer's top 24" (cache
    // policy simulations: tools/dsv41_cache_sim.py)
    FILE * trace = std::getenv("BL_DSV41_TRACE") ? std::fopen(std::getenv("BL_DSV41_TRACE"), "w") : nullptr;
    int cur_pos = 0;
    std::vector<int> pred_ids;   // the prediction for the next layer
    int * pred_ids_d = nullptr;
    float * pred_w_d = nullptr, * pred_logit = nullptr;
    uint64_t pred_disk = 0, pred_disk_hit = 0, pred_all = 0, pred_all_hit = 0;
    uint64_t cpu_experts = 0;
    cuda::ExpertPlan * plan_h = nullptr, * plan_d = nullptr;
    cuda::ExpertPlan * plan2_h = nullptr, * plan2_d = nullptr;   // hybrid: the misses admitted early, a second plan
    // BL_DSV41_EARLY=n (default 1): with two misses or more in a layer, up to n of them are copied into VRAM first and
    // computed on the GPU (the admission copy would happen anyway), the CPU takes the rest
    int early = std::getenv("BL_DSV41_EARLY") ? std::atoi(std::getenv("BL_DSV41_EARLY")) : 1;
    // BL_DSV41_EARLY_ADAPT=1: up to BL_DSV41_EARLY, ~0.36 of the layer's misses
    // BL_DSV41_GHOST=G (default 1000, 0 = off): an expert enters VRAM only on its second VRAM miss within the last G
    // misses (each admission costs 9 MB up and 9 MB back over PCIe and DRAM; tools/dsv41_cache_sim.py: 49% fewer
    // admissions, 1.5% fewer misses)
    int ghost_n = std::getenv("BL_DSV41_GHOST") ? std::atoi(std::getenv("BL_DSV41_GHOST")) : 1000;
    std::list<int> ghost_lru;
    std::unordered_map<int, std::list<int>::iterator> ghost_at;
    bool ghost_check(int key) {   // was it missed recently? (and remember this miss)
        if (ghost_n <= 0) return true;
        auto it = ghost_at.find(key);
        const bool seen = it != ghost_at.end();
        if (seen) ghost_lru.erase(it->second);
        ghost_lru.push_front(key);
        ghost_at[key] = ghost_lru.begin();
        if (static_cast<int>(ghost_lru.size()) > ghost_n) {
            ghost_at.erase(ghost_lru.back());
            ghost_lru.pop_back();
        }
        return seen;
    }
    bool adaptive_early = std::getenv("BL_DSV41_EARLY_ADAPT") && std::atoi(std::getenv("BL_DSV41_EARLY_ADAPT")) != 0;
    uint64_t gpu_misses = 0;
    // VRAM slots kept free (their expert already copied back to RAM, off the critical path) so an early admission is
    // one host-to-device copy on its own stream, waiting for nothing else; BL_DSV41_FREE_SLOTS (default 4)
    int free_target = std::getenv("BL_DSV41_FREE_SLOTS") ? std::atoi(std::getenv("BL_DSV41_FREE_SLOTS")) : 4;
    std::vector<int> free_vslots;
    cudaStream_t ust = nullptr;
    void * xq = nullptr;
    float * pact = nullptr, * pout = nullptr;
    float * logits_d = nullptr;
    std::vector<float *> win;                               // per layer [window][head_dim] ring
    std::vector<float *> comp_kv, idx_keys;                 // per source layer: [max_ctx / ratio][...]
    std::vector<float *> cst_kv, cst_sc;                    // per ratio-2 source: the pending group [2][head_dim]
    std::vector<float *> engram_qk;                         // per engram layer [hc][D]
    float * freq_swa = nullptr, * freq_comp = nullptr;

    // prefill: chunks of up to kChunk tokens through the whole model, layer by layer (weights read once per chunk)
    static constexpr int kChunk = 64;
    struct Pf {
        float * R, * R2, * mix, * pre, * a_pre, * post, * comb, * x, * qr, * q, * kv, * o, * og, * oa, * ao;
        float * win, * ckv, * csc, * lat, * ik, * iq, * iw, * isc, * logit_e, * w, * moe, * g, * u, * a, * sh, * ffo;
        float * emb, * ekv, * act, * gscratch, * logits;
        int * ids, * a0, * na, * nb, * nvis, * topk, * tok, * win_ids;
        float * wa;
        void * xq, * xg, * actq;
        cuda::MoeItem * items;
        uint8_t * erows;
        uint8_t * erows_h;
        size_t gscratch_floats;
    } pf{};
    std::vector<int> pf_nb;                                 // the current top-k counts per chunk token (host)

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

    // a then b in one allocation (same format and row length): mats[a], mats[b] and mats[fused] = both stacked
    void upload_pair(const std::string & a, const std::string & b, const std::string & fused) {
        const Tensor & ta = need(a), & tb = need(b);
        Mat ma, mb;
        for (auto [t, m] : {std::pair<const Tensor *, Mat *>{&ta, &ma}, {&tb, &mb}}) {
            m->type = t->type_id;
            m->K = static_cast<int>(t->shape[0]);
            m->M = static_cast<int>(t->elements() / m->K);
            m->row_bytes = row_bytes(t->type_id, m->K);
            if (!cuda::supported(m->type)) throw std::runtime_error(a + ": unsupported type");
        }
        if (ma.type != mb.type || ma.K != mb.K) throw std::runtime_error(a + " / " + b + ": cannot share a launch");
        void * d = nullptr;
        ck(cudaMalloc(&d, ta.nbytes + tb.nbytes), "cudaMalloc");
        allocs.push_back(d);
        ck(cudaMemcpy(d, ta.data, ta.nbytes, cudaMemcpyHostToDevice), "upload");
        ck(cudaMemcpy(static_cast<uint8_t *>(d) + ta.nbytes, tb.data, tb.nbytes, cudaMemcpyHostToDevice), "upload");
        ma.d = d;
        mb.d = static_cast<uint8_t *>(d) + ta.nbytes;
        Mat mf = ma;
        mf.M = ma.M + mb.M;
        mats[a] = ma;
        mats[b] = mb;
        mats[fused] = mf;
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
    // the hyper-connection mixes of one sublayer from the stream R: fused when fn is fp32 (it is in these files)
    void hc_mix(const std::string & fn, const std::string & scale, const std::string & base, float * pre_o, float * post_o,
                float * comb_o) {
        const Mat & m = M(fn);
        const float eps = static_cast<float>(c.rms_eps), hc_eps = static_cast<float>(c.hc_eps);
        if (m.type == 0 && m.K == c.hc * c.n_embd && m.K % 4 == 0) {
            cuda::dsv41::hc_fn_split(static_cast<const float *>(m.d), m.M, R, V(scale), V(base), c.hc, c.n_embd, eps, hc_eps,
                                     c.hc_sinkhorn, mix, hc_ss, pre_o, post_o, comb_o, st);
            return;
        }
        mv(fn, R, mix);
        cuda::dsv41::hc_split(mix, R, V(scale), V(base), c.hc, c.n_embd, eps, hc_eps, c.hc_sinkhorn, pre_o, post_o, comb_o, st);
    }
    void norm(const float * in, float * out, int D, const std::string & w) {
        cuda::dsv41::rmsnorm(in, out, D, V(w), static_cast<float>(c.rms_eps), st);
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
            // the pairs that read the same input share one allocation: each is a slice of a fused matrix (one launch)
            upload_pair(p + "attn_q_a.weight", p + "attn_kv.weight", p + "attn_qkv_a.fused");
            upload_pair(p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight", p + "ffn_gate_up_shexp.fused");
            for (const char * n : {"attn_q_b", "attn_output_a", "attn_output_b", "hc_attn_fn", "hc_ffn_fn", "ffn_gate_inp",
                                   "ffn_down_shexp"})
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
            LayerExperts le;
            le.gate = tg.data; le.up = tu.data; le.down = td.data;
            const Tensor * parts[3] = {&tg, &tu, &td};
            for (int k = 0; k < 3; ++k) {
                if (direct_fd.empty())
                    for (size_t i = 0; i < g.n_shards(); ++i) {
                        int fd = ::open(g.shard_path(static_cast<uint32_t>(i)).c_str(), O_RDONLY | O_DIRECT);
                        if (fd < 0) fd = ::open(g.shard_path(static_cast<uint32_t>(i)).c_str(), O_RDONLY);   // no O_DIRECT here
                        direct_fd.push_back(fd);
                    }
                le.fd[k] = direct_fd.at(parts[k]->shard);
                le.off[k] = g.file_offset(*parts[k]);
            }
            experts.push_back(le);
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
        mix = dalloc<float>(c.hc_mix()); pre = dalloc<float>(c.hc); a_pre = dalloc<float>(c.hc); hc_ss = dalloc<float>(1);
        post = dalloc<float>(c.hc); comb = dalloc<float>(c.hc * c.hc);
        x = dalloc<float>(D); q = dalloc<float>(static_cast<size_t>(H) * HD);
        qr = dalloc<float>(c.q_lora + HD); kvt = qr + c.q_lora;   // [qr | kv]: attn_qkv_a's output
        o = dalloc<float>(static_cast<size_t>(H) * HD);
        oa = dalloc<float>(static_cast<size_t>(c.o_groups) * c.o_lora); ao = dalloc<float>(D);
        lat = dalloc<float>(HD); ik = dalloc<float>(c.idx_dim);
        iq = dalloc<float>(static_cast<size_t>(c.idx_heads) * c.idx_dim); iw = dalloc<float>(c.idx_heads);
        isc = dalloc<float>(max_ctx);
        logit_e = dalloc<float>(c.n_expert); wsel = dalloc<float>(c.n_expert_used); ids = dalloc<int>(c.n_expert_used);
        gbuf = dalloc<float>(2 * c.n_ff_exp); ubuf = gbuf + c.n_ff_exp; abuf = dalloc<float>(c.n_ff_exp);   // [g | u]
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
        ck(cudaMallocHost(&route_h, 192 * 4), "pinned");
        ck(cudaMallocHost(&logits_h, static_cast<size_t>(c.n_vocab) * 4), "pinned");
        ck(cudaMallocHost(&step_h, static_cast<size_t>(D + c.hc) * 4), "pinned");
        pred_ids_d = dalloc<int>(64); pred_w_d = dalloc<float>(64); pred_logit = dalloc<float>(c.n_expert);
        if (dp4a) {
            ck(cudaMallocHost(&plan_h, sizeof(cuda::ExpertPlan)), "pinned");
            plan_d = dalloc<cuda::ExpertPlan>(1);
            ck(cudaMallocHost(&plan2_h, sizeof(cuda::ExpertPlan)), "pinned");
            plan2_d = dalloc<cuda::ExpertPlan>(1);
            xq = dalloc<uint8_t>(cuda::q8_bytes(D));
            pact = dalloc<float>(static_cast<size_t>(cuda::kMaxSlots) * cuda::kMaxT * c.n_ff_exp);
            pout = dalloc<float>(static_cast<size_t>(cuda::kMaxSlots) * cuda::kMaxT * D);
        }
        {   // prefill buffers (before the expert cache, which takes the rest)
            const size_t T = kChunk, HQ = static_cast<size_t>(H) * HD, A = T * c.n_expert_used;
            pf.R = dalloc<float>(T * c.hc_dim()); pf.R2 = dalloc<float>(T * c.hc_dim());
            pf.mix = dalloc<float>(T * c.hc_mix()); pf.pre = dalloc<float>(T * c.hc); pf.a_pre = dalloc<float>(T * c.hc);
            pf.post = dalloc<float>(T * c.hc); pf.comb = dalloc<float>(T * c.hc * c.hc);
            pf.x = dalloc<float>(T * D); pf.qr = dalloc<float>(T * c.q_lora); pf.q = dalloc<float>(T * HQ);
            pf.kv = dalloc<float>(T * HD); pf.o = dalloc<float>(T * HQ); pf.og = dalloc<float>(T * HQ / c.o_groups);
            pf.oa = dalloc<float>(T * c.o_groups * c.o_lora); pf.ao = dalloc<float>(T * D);
            pf.win = dalloc<float>((T + c.window) * HD);
            pf.ckv = dalloc<float>((T + 1) * HD); pf.csc = dalloc<float>((T + 1) * HD);
            pf.lat = dalloc<float>(T * HD); pf.ik = dalloc<float>(T * c.idx_dim);
            pf.iq = dalloc<float>(T * c.idx_heads * c.idx_dim); pf.iw = dalloc<float>(T * c.idx_heads);
            pf.isc = dalloc<float>(T * static_cast<size_t>(max_ctx));
            pf.logit_e = dalloc<float>(T * c.n_expert); pf.w = dalloc<float>(A); pf.ids = dalloc<int>(A);
            pf.moe = dalloc<float>(T * D); pf.g = dalloc<float>(T * c.n_ff_exp); pf.u = dalloc<float>(T * c.n_ff_exp);
            pf.a = dalloc<float>(T * c.n_ff_exp); pf.sh = dalloc<float>(T * D); pf.ffo = dalloc<float>(T * D);
            pf.emb = dalloc<float>(T * c.engram_cols() * c.engram_dim); pf.ekv = dalloc<float>(T * (c.hc + 1) * D);
            pf.act = dalloc<float>(A * c.n_ff_exp);
            pf.a0 = dalloc<int>(T); pf.na = dalloc<int>(T); pf.nb = dalloc<int>(T); pf.nvis = dalloc<int>(T);
            pf.topk = dalloc<int>(T * c.idx_top_k); pf.tok = dalloc<int>(A); pf.wa = dalloc<float>(A);
            pf.win_ids = dalloc<int>(c.window);
            pf.xq = dalloc<uint8_t>(cuda::q8_bytes(D) * T); pf.xg = dalloc<uint8_t>(cuda::q8_bytes(D) * A);
            pf.actq = dalloc<uint8_t>(cuda::q8_bytes(c.n_ff_exp) * A);
            pf.items = dalloc<cuda::MoeItem>(A);
            pf.erows = dalloc<uint8_t>(T * c.engram_cols() * std::max<size_t>(engram_row_bytes, 1));
            ck(cudaMallocHost(&pf.erows_h, T * c.engram_cols() * std::max<size_t>(engram_row_bytes, 1)), "pinned");
            pf.gscratch_floats = size_t{16} << 20;   // 64 MB: gemm_w dequantizes a weight matrix into it, block by block
            pf.gscratch = dalloc<float>(pf.gscratch_floats);
            pf.logits = nullptr;                      // all-position logits only when asked (allocated then)
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
        ck(cudaStreamCreateWithFlags(&cst, cudaStreamNonBlocking), "copy stream");
        ck(cudaStreamCreateWithFlags(&ust, cudaStreamNonBlocking), "copy stream");
        slot_ready.assign(n_slots, nullptr);
        slot_pending.assign(n_slots, 0);
        for (auto & e : slot_ready) ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
        ck(cudaEventCreateWithFlags(&ev_compute, cudaEventDisableTiming), "event");
        ck(cudaEventCreateWithFlags(&ev_admit, cudaEventDisableTiming), "event");
        for (int i = 0; i < n_slots; ++i) lru.push_back(i);
        std::fprintf(stderr, "dsv41: expert cache %d slots (%.1f GB, %.1f%% of all experts)\n", n_slots,
                     static_cast<double>(n_slots) * per / 1e9, 100.0 * n_slots / (static_cast<double>(c.n_layer) * c.n_expert));
        {   // the RAM tier: what is available now, less room for the system and the Engram rows' page cache
            size_t avail = 0;
            if (FILE * f = std::fopen("/proc/meminfo", "r")) {
                char line[256];
                while (std::fgets(line, sizeof line, f))
                    if (std::sscanf(line, "MemAvailable: %zu kB", &avail) == 1) { avail *= 1024; break; }
                std::fclose(f);
            }
            // 12 GB left: the system, the Engram rows' pages, and any file I/O (a download) in the page cache
            size_t want = avail > (size_t{12} << 30) ? avail - (size_t{12} << 30) : 0;
            if (const char * e = std::getenv("BL_DSV41_RAM_GB")) want = static_cast<size_t>(std::atof(e) * (1ull << 30));
            // the pool is pinned (never swapped): asking for more than MemAvailable - 8 GB hung this box once
            const size_t cap = avail > (size_t{8} << 30) ? avail - (size_t{8} << 30) : 0;
            if (want > cap) {
                std::fprintf(stderr, "dsv41: RAM tier capped at %.1f GB (MemAvailable - 8 GB)\n", cap / 1e9);
                want = cap;
            }
            const int iot = std::getenv("BL_DSV41_IO_THREADS") ? std::atoi(std::getenv("BL_DSV41_IO_THREADS")) : 2;
            want = std::max(want, per * static_cast<size_t>(c.n_expert));   // at least one layer's worth
            ram = std::make_unique<ExpertRam>(want, per, iot, true);
            std::fprintf(stderr, "dsv41: RAM tier %d slots (%.1f GB, %.1f%% of all experts)%s\n", ram->slots(), ram->bytes() / 1e9,
                         100.0 * ram->slots() / (static_cast<double>(c.n_layer) * c.n_expert), ram->pinned() ? ", pinned" : "");
        }
    }

    // expert e of layer il into a VRAM slot on the copy stream (hybrid decode's admissions)
    // the LRU tail out of VRAM into the free list (copied back to RAM first when exclusive)
    void free_vslot() {
        const size_t per = 2 * gu_expert_bytes + d_expert_bytes;
        const int slot = lru.back();
        lru.pop_back();
        const int old = slot_key[slot];
        if (old >= 0) where.erase(old);
        slot_key[slot] = -1;
        ck(cudaEventRecord(ev_compute, st), "event");   // queued compute may still read it
        ck(cudaStreamWaitEvent(cst, ev_compute, 0), "wait compute");
        if (slot_pending[slot]) ck(cudaStreamWaitEvent(cst, slot_ready[slot], 0), "wait admission");
        cudaEvent_t ev = slot_ready[slot];
        if (exclusive && old >= 0)
            if (uint8_t * back = ram->insert(old, ev)) {
                ck(cudaMemcpyAsync(back, xslots + static_cast<size_t>(slot) * per, per, cudaMemcpyDeviceToHost, cst), "demote");
                ++demoted;
            }
        ck(cudaEventRecord(ev, cst), "event");
        slot_pending[slot] = 1;
        free_vslots.push_back(slot);
    }
    // expert e of layer il into a free VRAM slot, one copy on the urgent stream (exclusive tiers only)
    void admit_urgent(int il, int e) {
        const size_t per = 2 * gu_expert_bytes + d_expert_bytes;
        const int key = il * c.n_expert + e;
        const int slot = free_vslots.back();
        free_vslots.pop_back();
        slot_key[slot] = key;
        lru.push_front(slot);
        where[key] = lru.begin();
        const uint8_t * src = nullptr;
        ram_get(il, &e, 1, &src);
        if (slot_pending[slot]) ck(cudaStreamWaitEvent(ust, slot_ready[slot], 0), "wait demotion");
        ck(cudaMemcpyAsync(xslots + static_cast<size_t>(slot) * per, src, per, cudaMemcpyHostToDevice, ust), "admit");
        ck(cudaEventRecord(slot_ready[slot], ust), "event");
        ram->drop(key, slot_ready[slot]);
        slot_pending[slot] = 1;
    }

    void admit_expert(int il, int e) {
        const size_t per = 2 * gu_expert_bytes + d_expert_bytes;
        const int key = il * c.n_expert + e;
        if (where.count(key)) return;
        const int slot = lru.back();
        lru.pop_back();
        const int old = slot_key[slot];
        if (old >= 0) where.erase(old);
        slot_key[slot] = key;
        lru.push_front(slot);
        where[key] = lru.begin();
        const uint8_t * src = nullptr;
        ram_get(il, &e, 1, &src);
        uint8_t * dslot = xslots + static_cast<size_t>(slot) * per;
        // the evicted expert may still be read by queued compute: the copy waits for it
        ck(cudaEventRecord(ev_compute, st), "event");
        ck(cudaStreamWaitEvent(cst, ev_compute, 0), "wait compute");
        if (slot_pending[slot]) ck(cudaStreamWaitEvent(cst, slot_ready[slot], 0), "wait admission");
        // exclusive tiers: the evicted expert back into RAM first (same stream, so before the slot is overwritten),
        // and the admitted one out of RAM once its copy is done
        cudaEvent_t ev = slot_ready[slot];
        if (exclusive && old >= 0)
            if (uint8_t * back = ram->insert(old, ev)) {
                ck(cudaMemcpyAsync(back, dslot, per, cudaMemcpyDeviceToHost, cst), "demote");
                ++demoted;
            }
        ck(cudaMemcpyAsync(dslot, src, per, cudaMemcpyHostToDevice, cst), "admit");
        ck(cudaEventRecord(ev, cst), "event");
        ck(cudaEventRecord(ev_admit, cst), "event");
        if (exclusive) ram->drop(key, ev);
        slot_pending[slot] = 1;
        admit_pending = true;
    }

    ExpertRam::Src src_of(int il, int e) const {
        ExpertRam::Src s{};
        const size_t len[3] = {gu_expert_bytes, gu_expert_bytes, d_expert_bytes};
        for (int k = 0; k < 3; ++k) {
            s.fd[k] = experts[il].fd[k];
            s.off[k] = experts[il].off[k] + static_cast<uint64_t>(e) * len[k];
            s.len[k] = len[k];
        }
        return s;
    }
    // the experts e[0..n) of layer il resident in RAM (read in parallel where missing); out[i] = expert e[i]'s slot
    void ram_get(int il, const int * e, int n, const uint8_t ** out) {
        std::vector<int> keys(n);
        std::vector<ExpertRam::Src> src(n);
        for (int i = 0; i < n; ++i) { keys[i] = il * c.n_expert + e[i]; src[i] = src_of(il, e[i]); }
        if (admit_pending && !exclusive) {   // an admission copy may still read a RAM slot this fetch could evict
            ck(cudaEventSynchronize(ev_admit), "admission");
            admit_pending = false;
        }
        ram->fetch(keys.data(), src.data(), n, out);
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
            use_slot(*it->second);
            return xslots + static_cast<size_t>(*it->second) * per;
        }
        ++misses;
        const int slot = lru.back();
        use_slot(slot);   // a pending admission into the slot being reused finishes first
        lru.pop_back();
        if (slot_key[slot] >= 0) where.erase(slot_key[slot]);
        slot_key[slot] = key;
        lru.push_front(slot);
        where[key] = lru.begin();
        uint8_t * dst = xslots + static_cast<size_t>(slot) * per;
        const uint8_t * src = nullptr;   // from the RAM tier (read from disk first if it is not there)
        ram_get(il, &e, 1, &src);
        ck(cudaMemcpyAsync(dst, src, per, cudaMemcpyHostToDevice, st), "expert");
        return dst;
    }

    ~Impl() {
        if (st) cudaStreamSynchronize(st);
        if (trace) std::fclose(trace);
        if (cst) { cudaStreamSynchronize(cst); cudaStreamDestroy(cst); }
        if (ust) { cudaStreamSynchronize(ust); cudaStreamDestroy(ust); }
        for (auto e : slot_ready) if (e) cudaEventDestroy(e);
        if (ev_compute) cudaEventDestroy(ev_compute);
        if (ev_admit) cudaEventDestroy(ev_admit);
        for (void * p : allocs) cudaFree(p);
        if (erows_h) cudaFreeHost(erows_h);
        if (plan_h) cudaFreeHost(plan_h);
        if (plan2_h) cudaFreeHost(plan2_h);
        if (route_h) cudaFreeHost(route_h);
        if (logits_h) cudaFreeHost(logits_h);
        if (step_h) cudaFreeHost(step_h);
        if (pf.erows_h) cudaFreeHost(pf.erows_h);
        ram.reset();
        for (int fd : direct_fd) if (fd >= 0) ::close(fd);
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

        mv(p + "attn_qkv_a.fused", x, qr);   // qr and kvt
        cuda::dsv41::rmsnorm2(qr, qr, c.q_lora, V(p + "attn_q_a_norm.weight"), kvt, kvt, HD, V(p + "attn_kv_a_norm.weight"),
                              static_cast<float>(c.rms_eps), st);
        P("qr", il, qr, c.q_lora);
        mv(p + "attn_q_b.weight", qr, q);
        cuda::dsv41::rope_tail(q, H, HD, rd, pos, freq, false, st);
        P("q", il, q, H * HD);


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
        if (wa.K != gin) throw std::runtime_error("attn_output_a: unexpected row length");
        cuda::matvec_blocks(wa.type, wa.d, wa.row_bytes, c.o_lora, wa.K, c.o_groups, o, oa, st);
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
        if (gputime && ev_a_set) ck(cudaEventRecord(ev_b, st), "event");
        const auto tr0 = std::chrono::steady_clock::now();
        // into pinned buffers: a copy into pageable memory is a synchronous staged copy (~10 us each, 80 a token)
        // through kernels into pinned memory: the copy engines may be busy with expert copies for ~1 ms
        cuda::dsv41::copy_small(route_h, ids, K * 4, st);
        cuda::dsv41::copy_small(route_h + 64, wsel, K * 4, st);
        if (hybrid) cuda::dsv41::copy_small(x_h, x, D * 4, st);   // the CPU experts' input
        // the next layer's router on this layer's FFN input: the experts it will likely pick (one sync with this one's)
        std::vector<int> next_pred;
        const int nk = std::min(std::max({pred_stats ? pred_k : 0, hybrid ? prefetch_k : 0, trace ? 24 : 0}), 64);
        const int pl = il + (hybrid && prefetch_k > 0 ? prefetch_dist : 1);
        if (nk > 0 && pl < c.n_layer) {
            const std::string pn = "blk." + std::to_string(pl) + ".";
            mv(pn + "ffn_gate_inp.weight", x, pred_logit);
            cuda::dsv41::router_topk(pred_logit, V(pn + "exp_probs_b.bias"), c.n_expert, nk, 1.f, pred_ids_d, pred_w_d, st);
            next_pred.resize(nk);
            cuda::dsv41::copy_small(route_h + 128, pred_ids_d, nk * 4, st);
        }
        ck(cudaStreamSynchronize(st), "router");
        std::vector<int> hid(reinterpret_cast<const int *>(route_h), reinterpret_cast<const int *>(route_h) + K);
        std::vector<float> hw(reinterpret_cast<const float *>(route_h + 64), reinterpret_cast<const float *>(route_h + 64) + K);
        if (!next_pred.empty()) std::memcpy(next_pred.data(), route_h + 128, nk * 4);
        if (gputime && ev_a_set) {
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, ev_a, ev_b), "elapsed");
            t_gpu_pre += ms / 1e3;
            t_span += std::chrono::duration<double>(std::chrono::steady_clock::now() - span0).count();
            ev_a_set = false;
        }
        t_router += std::chrono::duration<double>(std::chrono::steady_clock::now() - tr0).count();
        if (trace) {
            std::fprintf(trace, "%d %d:", cur_pos, il);
            for (int k = 0; k < K; ++k) std::fprintf(trace, " %d", hid[k]);
            std::fprintf(trace, " |");
            for (int v : next_pred) std::fprintf(trace, " %d", v);
            std::fprintf(trace, "\n");
        }
        if (pred_stats && il > 0 && !pred_ids.empty()) {
            for (int k = 0; k < K; ++k) {
                const bool predicted = std::find(pred_ids.begin(), pred_ids.end(), hid[k]) != pred_ids.end();
                ++pred_all; pred_all_hit += predicted;
                const int key = il * c.n_expert + hid[k];
                if (!where.count(key) && !ram->contains(key)) { ++pred_disk; pred_disk_hit += predicted; }
            }
        }
        if (pred_stats) pred_ids.assign(next_pred.begin(), next_pred.begin() + std::min<size_t>(pred_k, next_pred.size()));
        if (hybrid) {
            moe_hybrid(il, hid, hw, next_pred);
            if (gputime) {
                if (!ev_a) { ck(cudaEventCreate(&ev_a), "event"); ck(cudaEventCreate(&ev_b), "event"); }
                ck(cudaEventRecord(ev_a, st), "event");
                ev_a_set = true;
                span0 = std::chrono::steady_clock::now();
            }
            shared_and_out(il, true);
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

    // the shared expert into shx (needs only x: hybrid decode queues it before the CPU experts, so it overlaps them)
    void shared_expert(int il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        mv(p + "ffn_gate_up_shexp.fused", x, gbuf);   // gbuf and ubuf
        cuda::dsv41::swiglu_clamp(gbuf, ubuf, abuf, c.n_ff_exp, static_cast<float>(c.swiglu_limit), st);
        mv(p + "ffn_down_shexp.weight", abuf, shx);
    }
    void shared_and_out(int il, bool shexp_done = false) {
        const int D = c.n_embd;
        P("ffn_moe_out", il, moe, D);
        if (!shexp_done) shared_expert(il);
        P("ffn_shexp", il, shx, D);
        ck(cudaMemcpyAsync(ffo, moe, D * 4, cudaMemcpyDeviceToDevice, st), "ffn");
        cuda::add(ffo, shx, D, st);
        P("ffn_out", il, ffo, D);
    }

    // hits on the GPU, misses on the CPU at the same time; then a few misses admitted into the VRAM cache
    void moe_hybrid(int il, const std::vector<int> & hid, const std::vector<float> & hw, const std::vector<int> & next_pred) {
        const int D = c.n_embd, F = c.n_ff_exp, K = c.n_expert_used;
        const float lim = static_cast<float>(c.swiglu_limit);
        std::vector<int> gk, ck_;   // indices k of the hits / the misses
        std::vector<const uint8_t *> gbase;
        for (int k = 0; k < K; ++k) {
            auto it = where.find(il * c.n_expert + hid[k]);
            if (it != where.end()) {
                ++hits;
                lru.splice(lru.begin(), lru, it->second);
                use_slot(*it->second);
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
            cuda::dsv41::copy_small(plan_d, plan_h, sizeof *plan_h, st);
            cuda::quantize_q8(x, D, xq, st);
            cuda::plan_gate_up(expert_type, plan_d, 1 << cuda::kHit, 1, gu_expert_bytes, F, D, xq, pact, st, lim);
            cuda::plan_down(expert_type, plan_d, 1 << cuda::kHit, 1, gu_expert_bytes, F, D, pact, pout, st);
            for (int i = 0; i < n; ++i) cuda::axpy(moe, pout + static_cast<size_t>(i) * cuda::kMaxT * D, hw[gk[i]], D, st);
        }
        shared_expert(il);   // on the GPU while the CPU works
        const int n = static_cast<int>(ck_.size());
        std::vector<const uint8_t *> slot(n);
        std::vector<int> rslot(n);
        std::vector<char> admissible(n, 1);
        int nready = n, npass_ready = 0;
        const auto t0 = std::chrono::steady_clock::now();
        if (n) {   // the misses from the RAM tier; the ones not there start reading from disk, all at once
            std::vector<int> keys(n);
            std::vector<ExpertRam::Src> src(n);
            for (int j = 0; j < n; ++j) { keys[j] = il * c.n_expert + hid[ck_[j]]; src[j] = src_of(il, hid[ck_[j]]); }
            if (admit_pending && !exclusive) {   // an admission copy may still read a RAM slot this fetch could evict
                ck(cudaEventSynchronize(ev_admit), "admission");
                admit_pending = false;
            }
            std::vector<char> ready(n), pass(n);
            ram->start(keys.data(), src.data(), n, slot.data(), rslot.data(), ready.data());
            for (int j = 0; j < n; ++j) pass[j] = ghost_check(keys[j]);
            // the usable ones first (computed while the disk reads run), among them the admissible ones first
            std::vector<int> ord(n);
            std::iota(ord.begin(), ord.end(), 0);
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
                return std::make_pair(!ready[a], !pass[a]) < std::make_pair(!ready[b], !pass[b]);
            });
            nready = static_cast<int>(std::count(ready.begin(), ready.end(), 1));
            for (int j = 0; j < n; ++j) npass_ready += ready[j] && pass[j];
            std::vector<int> ck2(n), rs2(n);
            std::vector<char> pa2(n);
            std::vector<const uint8_t *> sl2(n);
            for (int j = 0; j < n; ++j) { ck2[j] = ck_[ord[j]]; rs2[j] = rslot[ord[j]]; sl2[j] = slot[ord[j]]; pa2[j] = pass[ord[j]]; }
            ck_.swap(ck2); rslot.swap(rs2); slot.swap(sl2); admissible.swap(pa2);
        }
        // the next layer's likely experts that are in neither tier: read in the background while this layer computes
        // (after this layer's fetch, so its slots are at the LRU front and no prefetch evicts them)
        const int pl = il + prefetch_dist;
        if (prefetch_k > 0 && pl < c.n_layer && !next_pred.empty()) {
            std::vector<int> keys;
            std::vector<ExpertRam::Src> src;
            for (int k = 0; k < std::min<int>(prefetch_k, next_pred.size()); ++k) {
                const int key = pl * c.n_expert + next_pred[k];
                if (where.count(key) || ram->contains(key)) continue;
                keys.push_back(key);
                src.push_back(src_of(pl, next_pred[k]));
            }
            if (!keys.empty()) {
                if (admit_pending && !exclusive) {   // an admission copy may still read a RAM slot a prefetch could evict
                    ck(cudaEventSynchronize(ev_admit), "admission");
                    admit_pending = false;
                }
                ram->prefetch(keys.data(), src.data(), static_cast<int>(keys.size()));
            }
        }
        // the first ng misses: admitted now and computed on the GPU once their copy lands, while the CPU does the rest
        // how many: balance a GPU miss (~0.8 ms: the copy, then compute) against a CPU one (~0.45 ms): k ~ 0.36 n
        const int want = adaptive_early ? std::max(1, static_cast<int>((n + 1) * 0.36f)) : early;
        const int ng = n >= 2 && exclusive ? std::min({want, early, n - 1, npass_ready, static_cast<int>(free_vslots.size())}) : 0;
        if (ng > 0) {
            const auto ta = std::chrono::steady_clock::now();
            std::memset(plan2_h, 0, sizeof *plan2_h);
            plan2_h->T = 1; plan2_h->K = ng; plan2_h->n_slots = ng;
            for (int j = 0; j < ng; ++j) {
                const int e = hid[ck_[j]];
                admit_urgent(il, e);
                const int vs = *where.at(il * c.n_expert + e);
                use_slot(vs);   // the compute stream waits for the copy
                plan2_h->ids[j] = e; plan2_h->kind[j] = cuda::kHit; plan2_h->mask[j] = 1;
                plan2_h->base[j] = xslots + static_cast<size_t>(vs) * (2 * gu_expert_bytes + d_expert_bytes);
                plan2_h->sel[0][j] = j; plan2_h->w[0][j] = hw[ck_[j]];
            }
            t_admit += std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count();
            cuda::dsv41::copy_small(plan2_d, plan2_h, sizeof *plan2_h, st);
            if (gk.empty()) cuda::quantize_q8(x, D, xq, st);
            cuda::plan_gate_up(expert_type, plan2_d, 1 << cuda::kHit, 1, gu_expert_bytes, F, D, xq, pact, st, lim);
            cuda::plan_down(expert_type, plan2_d, 1 << cuda::kHit, 1, gu_expert_bytes, F, D, pact, pout, st);
            for (int j = 0; j < ng; ++j) cuda::axpy(moe, pout + static_cast<size_t>(j) * cuda::kMaxT * D, hw[ck_[j]], D, st);
            gpu_misses += ng;
        }
        if (n > ng) {
            const int nc = n - ng;
            // x_h: copied with the router's results
            std::vector<CpuExperts::Job> jobs(nc);
            for (int j = 0; j < nc; ++j) jobs[j] = {slot[ng + j], 1, cpu_out_h + static_cast<size_t>(j) * D};
            // the resident ones while the disk reads run, then the rest once they land
            const int n1 = std::max(nready - ng, 0);
            cpu->run(expert_type, expert_type, gu_expert_bytes, F, D, x_h, 1, jobs.data(), n1, lim);
            if (n1 < nc) {   // from disk: gate/up as soon as those parts are in, down after its part lands
                const int * rs = rslot.data() + ng + n1;
                auto timed_wait = [&](bool gu_only) {
                    const auto tw = std::chrono::steady_clock::now();
                    ram->wait(rs, nc - n1, gu_only);
                    t_ram += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count();
                };
                timed_wait(true);
                cpu->run(expert_type, expert_type, gu_expert_bytes, F, D, x_h, 1, jobs.data() + n1, nc - n1, lim,
                         [&] { timed_wait(false); });
            }
            cpu_experts += nc;
            t_cpu += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            cuda::dsv41::copy_small(cpu_out_d, cpu_out_h, sizeof(float) * nc * D, st);
            for (int j = 0; j < nc; ++j) cuda::axpy(moe, cpu_out_d + static_cast<size_t>(j) * D, hw[ck_[ng + j]], D, st);
            if (ng == 0) {   // admit a few: their pages are hot now, so the copies come from RAM
                const auto ta = std::chrono::steady_clock::now();
                for (int j = 0, a = 0; j < n && a < admit; ++j)
                    if (admissible[j]) { admit_expert(il, hid[ck_[j]]); ++a; }
                t_admit += std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count();
            }
        }
        if (exclusive && early > 0)   // free slots for the next early admissions, the copies back behind this layer's work
            while (static_cast<int>(free_vslots.size()) < free_target && static_cast<int>(lru.size()) > 2 * c.n_expert_used) free_vslot();
        const auto t1 = std::chrono::steady_clock::now();
        ck(cudaStreamSynchronize(st), "moe");   // cpu_out_h is rewritten by the next layer
        const double gw = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        t_gpu_wait += gw;
        t_gpu_wait_by[ng > 0 ? 2 : (n > 0 ? 1 : 0)] += gw;
        ++n_gpu_wait_by[ng > 0 ? 2 : (n > 0 ? 1 : 0)];
    }

    // y [N][M] = x [N][K] . W^T for the named matrix
    void gemm(const std::string & n, const float * in, int N, float * out, int ldy = -1) {
        const Mat & m = M(n);
        cuda::gemm_w(m.type, m.d, m.row_bytes, m.M, m.K, in, N, out, ldy < 0 ? m.M : ldy, pf.gscratch, pf.gscratch_floats, st);
    }
    void norm_rows(float * xin, int rows, int D, const std::string & w) {
        cuda::rmsnorm_rows(xin, xin, rows, D, V(w), 1, static_cast<float>(c.rms_eps), 1.f, st);
    }
    template <typename T> void up(T * dst, const std::vector<T> & v) {
        ck(cudaMemcpyAsync(dst, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, st), "upload");
    }

    // T tokens at positions pos0.. (all already in toks), through every layer; logits of every position into
    // pf.logits when all_logits, else only the last one into logits_d
    void prefill_chunk(int pos0, int T, bool all_logits) {
        const int D = c.n_embd, hc = c.hc, HD = c.head_dim, H = c.n_head, rd = c.rope_dims;
        const float eps = static_cast<float>(c.rms_eps), hc_eps = static_cast<float>(c.hc_eps);
        // embeddings, repeated into the hc copies; the one-hot pre mix for layer 0
        {
            std::vector<float> e(static_cast<size_t>(T) * D);
            for (int t = 0; t < T; ++t) {
                const uint8_t * row = embd + static_cast<size_t>(toks[pos0 + t]) * D * 2;
                for (int i = 0; i < D; ++i) {
                    uint16_t v;
                    std::memcpy(&v, row + 2 * i, 2);
                    e[static_cast<size_t>(t) * D + i] = bf16_to_float(v);
                }
            }
            up(pf.x, e);
            for (int t = 0; t < T; ++t) cuda::repeat(pf.x + static_cast<size_t>(t) * D, pf.R + static_cast<size_t>(t) * c.hc_dim(), D, hc, st);
            std::vector<float> one(static_cast<size_t>(T) * hc, 0.f);
            for (int t = 0; t < T; ++t) one[static_cast<size_t>(t) * hc] = 1.f;
            up(pf.pre, one);
            ck(cudaStreamSynchronize(st), "embd");
        }
        // Engram rows of every chunk token, all reads started together
        const int ncol = c.engram_cols();
        std::vector<uint64_t> erow(static_cast<size_t>(T) * c.engram_layers.size() * ncol);
        if (engram_on) {
            const long pg = sysconf(_SC_PAGESIZE);
            for (int t = 0; t < T; ++t) {
                uint64_t * r = erow.data() + static_cast<size_t>(t) * c.engram_layers.size() * ncol;
                dsv41_engram_rows(c, toks.data(), pos0 + t, r);
                for (size_t e = 0; e < c.engram_layers.size(); ++e)
                    for (int b = 0; b < ncol; ++b) {
                        const uint8_t * a = engram_table[e] + r[e * ncol + b] * engram_row_bytes;
                        const uintptr_t pa = reinterpret_cast<uintptr_t>(a) & ~static_cast<uintptr_t>(pg - 1);
                        madvise(reinterpret_cast<void *>(pa), engram_row_bytes + (reinterpret_cast<uintptr_t>(a) - pa), MADV_WILLNEED);
                    }
            }
        }
        // per-position counts the attention needs (host-known): window range in pf.win, visible compressed entries
        const int W0 = std::min(pos0, c.window - 1);   // earlier raw KV the first query sees
        std::vector<int> a0(T), na(T);
        for (int t = 0; t < T; ++t) {
            const int p = pos0 + t, first = std::max(0, p - c.window + 1);
            a0[t] = first - (pos0 - W0);
            na[t] = p - first + 1;
        }
        up(pf.a0, a0);
        up(pf.na, na);
        const int na_max = *std::max_element(na.begin(), na.end());
        pf_nb.assign(T, 0);

        for (int il = 0; il < c.n_layer; ++il) {
            const std::string p = "blk." + std::to_string(il) + ".";
            const int eg = c.engram_index(il);
            if (eg >= 0 && engram_on) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int t = 0; t < T; ++t)
                    for (int b = 0; b < ncol; ++b) {
                        const uint64_t r = erow[(static_cast<size_t>(t) * c.engram_layers.size() + eg) * ncol + b];
                        std::memcpy(pf.erows_h + (static_cast<size_t>(t) * ncol + b) * engram_row_bytes,
                                    engram_table[eg] + r * engram_row_bytes, engram_row_bytes);
                    }
                t_engram += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                ck(cudaMemcpyAsync(pf.erows, pf.erows_h, static_cast<size_t>(T) * ncol * engram_row_bytes, cudaMemcpyHostToDevice, st), "rows");
                cuda::dequant_rows(8 /* Q8_0 */, pf.erows, engram_row_bytes, T * ncol, c.engram_dim, pf.emb, st);
                gemm(p + "engram_wkv.weight", pf.emb, T, pf.ekv);
                cuda::dsv41::engram_gate_rows(pf.R, pf.ekv, engram_qk[eg], hc, D, T, eps, st);
            }

            // ---- attention sublayer
            gemm(p + "hc_attn_fn.weight", pf.R, T, pf.mix);
            cuda::dsv41::hc_split_rows(pf.mix, pf.R, V(p + "hc_attn_scale.weight"), V(p + "hc_attn_base.weight"), hc, D, T, eps, hc_eps,
                                       c.hc_sinkhorn, pf.a_pre, pf.post, pf.comb, st);
            cuda::dsv41::hc_pre_rows(pf.R, pf.pre, pf.x, hc, D, T, st);
            norm_rows(pf.x, T, D, p + "attn_norm.weight");
            attention_chunk(il, pos0, T, W0, na_max);
            cuda::dsv41::hc_post_rows(pf.ao, pf.R, pf.post, pf.comb, pf.R2, hc, D, T, st);
            std::swap(pf.R, pf.R2);

            // ---- FFN sublayer
            gemm(p + "hc_ffn_fn.weight", pf.R, T, pf.mix);
            cuda::dsv41::hc_split_rows(pf.mix, pf.R, V(p + "hc_ffn_scale.weight"), V(p + "hc_ffn_base.weight"), hc, D, T, eps, hc_eps,
                                       c.hc_sinkhorn, pf.pre, pf.post, pf.comb, st);
            cuda::dsv41::hc_pre_rows(pf.R, pf.a_pre, pf.x, hc, D, T, st);
            norm_rows(pf.x, T, D, p + "ffn_norm.weight");
            moe_chunk(il, T);
            cuda::dsv41::hc_post_rows(pf.ffo, pf.R, pf.post, pf.comb, pf.R2, hc, D, T, st);
            std::swap(pf.R, pf.R2);
        }
        cuda::dsv41::hc_pre_rows(pf.R, pf.pre, pf.x, hc, D, T, st);
        norm_rows(pf.x, T, D, "output_norm.weight");
        if (all_logits) gemm("output.weight", pf.x, T, pf.logits + static_cast<size_t>(pos0 - pf_logits_from) * c.n_vocab);
        else mv("output.weight", pf.x + static_cast<size_t>(T - 1) * D, logits_d);
        ck(cudaStreamSynchronize(st), "prefill chunk");
        (void)rd; (void)HD; (void)H;
    }
    int pf_logits_from = 0;

    void attention_chunk(int il, int pos0, int T, int W0, int na_max) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int HD = c.head_dim, H = c.n_head, rd = c.rope_dims, ratio = c.compress_ratio[il], D = c.n_embd;
        const float * freq = ratio ? freq_comp : freq_swa;
        gemm(p + "attn_q_a.weight", pf.x, T, pf.qr);
        norm_rows(pf.qr, T, c.q_lora, p + "attn_q_a_norm.weight");
        gemm(p + "attn_q_b.weight", pf.qr, T, pf.q);
        cuda::dsv41::rope_tail_rows(pf.q, T, H, HD, rd, pos0, 1, freq, false, st);
        // window keys: the W0 earlier positions from the ring, then this chunk's
        if (W0 > 0) {
            std::vector<int> ids(W0);
            for (int i = 0; i < W0; ++i) ids[i] = (pos0 - W0 + i) % c.window;
            up(pf.win_ids, ids);
            cuda::gather_rows(win[il], static_cast<size_t>(HD) * 4, pf.win_ids, W0, pf.win, st);
        }
        float * kvc = pf.win + static_cast<size_t>(W0) * HD;
        gemm(p + "attn_kv.weight", pf.x, T, kvc);
        norm_rows(kvc, T, HD, p + "attn_kv_a_norm.weight");
        cuda::dsv41::rope_tail_rows(kvc, T, 1, HD, rd, pos0, 1, freq, false, st);
        // the ring keeps the last window of positions
        for (int t = std::max(0, T - c.window); t < T; ++t)
            ck(cudaMemcpyAsync(win[il] + static_cast<size_t>((pos0 + t) % c.window) * HD, kvc + static_cast<size_t>(t) * HD, HD * 4,
                               cudaMemcpyDeviceToDevice, st), "ring");

        int nb_max = 0;
        const float * comp = pf.win;   // unused when no compressed keys
        if (ratio > 0) {
            if (c.is_kv_source(il)) compress_chunk(il, pos0, T);
            comp = comp_kv[c.kv_source[il]];
            if (c.is_topk_source(il)) topk_chunk(il, pos0, T);
            for (int v : pf_nb) nb_max = std::max(nb_max, v);
        } else {
            std::vector<int> z(T, 0);
            up(pf.nb, z);
        }
        cuda::dsv41::attention_sink_rows(pf.q, pf.win, pf.a0, pf.na, comp, pf.topk, c.idx_top_k, pf.nb, na_max + nb_max,
                                         V(p + "attn_sinks.weight"), pf.o, H, HD, T, 1.f / std::sqrt(static_cast<float>(HD)), st);
        if (ratio == 0) {   // the counts of the last top-k source stay for the layers after it; only zero them here
            std::vector<int> nb(pf_nb);
            up(pf.nb, nb);
        }
        cuda::dsv41::rope_tail_rows(pf.o, T, H, HD, rd, pos0, 1, freq, true, st);
        const Mat & wa = M(p + "attn_output_a.weight");
        const int gin = H * HD / c.o_groups;
        for (int gi = 0; gi < c.o_groups; ++gi) {
            cuda::copy_rows(pf.o + static_cast<size_t>(gi) * gin, H * HD, pf.og, gin, gin, T, st);
            cuda::gemm_w(wa.type, static_cast<const uint8_t *>(wa.d) + wa.row_bytes * c.o_lora * gi, wa.row_bytes, c.o_lora, wa.K, pf.og,
                         T, pf.oa + static_cast<size_t>(gi) * c.o_lora, c.o_groups * c.o_lora, pf.gscratch, pf.gscratch_floats, st);
        }
        gemm(p + "attn_output_b.weight", pf.oa, T, pf.ao);
        (void)D;
    }

    void compress_chunk(int il, int pos0, int T) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int HD = c.head_dim, ratio = c.compress_ratio[il], rd = c.rope_dims;
        int n = 0, j0 = 0;   // latents produced, the first one's index
        if (ratio == 1) {
            gemm(p + "attn_compressor_kv.weight", pf.x, T, pf.lat);
            n = T;
            j0 = pos0;
        } else {   // ratio 2: a pending token from before the chunk (odd pos0) pairs with the chunk's first
            const int s0 = pos0 % 2;
            if (s0) {
                ck(cudaMemcpyAsync(pf.ckv, cst_kv[il], HD * 4, cudaMemcpyDeviceToDevice, st), "pending");
                ck(cudaMemcpyAsync(pf.csc, cst_sc[il], HD * 4, cudaMemcpyDeviceToDevice, st), "pending");
            }
            gemm(p + "attn_compressor_kv.weight", pf.x, T, pf.ckv + static_cast<size_t>(s0) * HD);
            gemm(p + "attn_compressor_gate.weight", pf.x, T, pf.csc + static_cast<size_t>(s0) * HD);
            n = (s0 + T) / 2;
            j0 = (pos0 - s0) / 2;
            cuda::dsv41::pool2_rows(pf.ckv, pf.csc, pf.lat, HD, n, st);
            if ((s0 + T) % 2) {   // the chunk's last token waits for its partner (slot 0: its position is even)
                ck(cudaMemcpyAsync(cst_kv[il], pf.ckv + static_cast<size_t>(s0 + T - 1) * HD, HD * 4, cudaMemcpyDeviceToDevice, st), "pending");
                ck(cudaMemcpyAsync(cst_sc[il], pf.csc + static_cast<size_t>(s0 + T - 1) * HD, HD * 4, cudaMemcpyDeviceToDevice, st), "pending");
            }
        }
        if (n == 0) return;
        norm_rows(pf.lat, n, HD, p + "attn_compressor_norm.weight");
        if (c.owns_idx_key(il)) {
            gemm(p + "indexer.attn_k.weight", pf.lat, n, pf.ik);
            norm_rows(pf.ik, n, c.idx_dim, p + "indexer.k_norm.weight");
            cuda::dsv41::rope_tail_rows(pf.ik, n, 1, c.idx_dim, rd, j0 * ratio, ratio, freq_comp, false, st);
            ck(cudaMemcpyAsync(idx_keys[il] + static_cast<size_t>(j0) * c.idx_dim, pf.ik, static_cast<size_t>(n) * c.idx_dim * 4,
                               cudaMemcpyDeviceToDevice, st), "idx k");
        }
        cuda::dsv41::rope_tail_rows(pf.lat, n, 1, HD, rd, j0 * ratio, ratio, freq_comp, false, st);
        ck(cudaMemcpyAsync(comp_kv[il] + static_cast<size_t>(j0) * HD, pf.lat, static_cast<size_t>(n) * HD * 4, cudaMemcpyDeviceToDevice, st),
           "comp kv");
    }

    void topk_chunk(int il, int pos0, int T) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int ratio = c.compress_ratio[il];
        std::vector<int> nvis(T);
        int nmax = 0;
        for (int t = 0; t < T; ++t) { nvis[t] = (pos0 + t + 1) / ratio; nmax = std::max(nmax, nvis[t]); }
        for (int t = 0; t < T; ++t) pf_nb[t] = std::min(c.idx_top_k, nvis[t]);
        up(pf.nb, pf_nb);
        if (nmax == 0) return;
        up(pf.nvis, nvis);
        gemm(p + "indexer.attn_q_b.weight", pf.qr, T, pf.iq);
        cuda::dsv41::rope_tail_rows(pf.iq, T, c.idx_heads, c.idx_dim, c.rope_dims, pos0, 1, freq_comp, false, st);
        gemm(p + "indexer.proj.weight", pf.x, T, pf.iw);
        const float wscale = 1.f / std::sqrt(static_cast<float>(c.idx_dim) * c.idx_heads);
        cuda::dsv41::indexer_score_rows(pf.iq, pf.iw, wscale, idx_keys[c.idx_key_source[il]], pf.nvis, nmax, c.idx_heads, c.idx_dim, T,
                                        pf.isc, max_ctx, st);
        for (int t = 0; t < T; ++t)
            if (nvis[t] > 0) cuda::dsv41::topk_sorted(pf.isc + static_cast<size_t>(t) * max_ctx, nvis[t], pf_nb[t], pf.topk + t * c.idx_top_k, st);
    }

    void moe_chunk(int il, int T) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int D = c.n_embd, F = c.n_ff_exp, K = c.n_expert_used, A = T * K;
        const float lim = static_cast<float>(c.swiglu_limit);
        gemm(p + "ffn_gate_inp.weight", pf.x, T, pf.logit_e);
        cuda::dsv41::router_topk_rows(pf.logit_e, V(p + "exp_probs_b.bias"), c.n_expert, K, static_cast<float>(c.expert_scale), T,
                                      pf.ids, pf.w, st);
        std::vector<int> hid(A);
        std::vector<float> hw(A);
        ck(cudaMemcpyAsync(hid.data(), pf.ids, A * 4, cudaMemcpyDeviceToHost, st), "ids");
        ck(cudaMemcpyAsync(hw.data(), pf.w, A * 4, cudaMemcpyDeviceToHost, st), "w");
        ck(cudaStreamSynchronize(st), "router");
        // the assignments grouped by expert; each expert made resident once for the chunk
        std::vector<int> order(A);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return hid[a] < hid[b]; });
        std::vector<int> tok(A);
        std::vector<float> wv(A);
        std::vector<cuda::MoeItem> items;
        const auto t0 = std::chrono::steady_clock::now();
        {   // the experts not in VRAM: into RAM in one parallel read first, so the copies below come from memory
            std::vector<int> need;
            for (int i = 0; i < A; ++i) {
                const int e = hid[order[i]];
                if ((i == 0 || e != hid[order[i - 1]]) && !where.count(il * c.n_expert + e)) need.push_back(e);
            }
            std::vector<const uint8_t *> dummy(need.size());
            if (!need.empty()) ram_get(il, need.data(), static_cast<int>(need.size()), dummy.data());
        }
        const int tile = cuda::moe_tile();
        for (int i = 0; i < A;) {
            int j = i;
            while (j < A && hid[order[j]] == hid[order[i]]) ++j;
            const uint8_t * base = expert_slot(il, hid[order[i]]);
            for (int a0 = i; a0 < j; a0 += tile) items.push_back({base, a0, std::min(tile, j - a0)});
            i = j;
        }
        for (int a = 0; a < A; ++a) { tok[a] = order[a] / K; wv[a] = hw[order[a]]; }
        t_expert_copy += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        up(pf.tok, tok);
        up(pf.wa, wv);
        up(pf.items, items);
        ck(cudaMemsetAsync(pf.moe, 0, static_cast<size_t>(T) * D * 4, st), "moe");
        cuda::quantize_q8(pf.x, D, pf.xq, st, T);
        cuda::moe_gather(pf.xq, pf.tok, A, D, pf.xg, st);
        cuda::moe_gate_up(expert_type, pf.items, static_cast<int>(items.size()), gu_expert_bytes, F, D, pf.xg, pf.act, st, lim);
        cuda::quantize_q8(pf.act, F, pf.actq, st, A);
        cuda::moe_down(expert_type, pf.items, static_cast<int>(items.size()), gu_expert_bytes, d_row_bytes, F, D, pf.actq, pf.tok,
                       pf.wa, pf.moe, st);
        // the shared expert
        gemm(p + "ffn_gate_shexp.weight", pf.x, T, pf.g);
        gemm(p + "ffn_up_shexp.weight", pf.x, T, pf.u);
        cuda::dsv41::swiglu_clamp(pf.g, pf.u, pf.a, T * F, lim, st);
        gemm(p + "ffn_down_shexp.weight", pf.a, T, pf.sh);
        ck(cudaMemcpyAsync(pf.ffo, pf.moe, static_cast<size_t>(T) * D * 4, cudaMemcpyDeviceToDevice, st), "ffn");
        cuda::add(pf.ffo, pf.sh, T * D, st);
        ck(cudaStreamSynchronize(st), "moe");   // the host vectors above go out of scope
    }

    // the prompt in chunks; returns the last position's logits (all positions' in all_out when given)
    const std::vector<float> & prefill(const std::vector<int> & prompt, std::vector<float> * all_out) {
        if (prompt.empty()) return logits;
        if (static_cast<int>(toks.size() + prompt.size()) > max_ctx) throw std::runtime_error("context full");
        const auto t0 = std::chrono::steady_clock::now();
        const int pos_start = static_cast<int>(toks.size());
        for (int t : prompt) {
            if (t < 0 || t >= c.n_vocab) throw std::runtime_error("token id out of range");
            toks.push_back(t);
        }
        const int n = static_cast<int>(prompt.size());
        if (all_out) {
            ck(cudaMalloc(reinterpret_cast<void **>(&pf.logits), static_cast<size_t>(n) * c.n_vocab * 4), "logits");
            pf_logits_from = pos_start;
        }
        for (int i = 0; i < n; i += kChunk) prefill_chunk(pos_start + i, std::min(kChunk, n - i), all_out != nullptr);
        if (all_out) {
            all_out->resize(static_cast<size_t>(n) * c.n_vocab);
            ck(cudaMemcpy(all_out->data(), pf.logits, all_out->size() * 4, cudaMemcpyDeviceToHost), "logits");
            cudaFree(pf.logits);
            pf.logits = nullptr;
            std::copy(all_out->end() - c.n_vocab, all_out->end(), logits.begin());
        } else {
            ck(cudaMemcpy(logits.data(), logits_d, c.n_vocab * 4, cudaMemcpyDeviceToHost), "logits");
        }
        t_prefill += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        n_prefill += n;
        return logits;
    }
    double t_prefill = 0;
    long n_prefill = 0;

    const std::vector<float> & step(int token) {
        if (static_cast<int>(toks.size()) >= max_ctx) throw std::runtime_error("context full");
        if (token < 0 || token >= c.n_vocab) throw std::runtime_error("token id out of range");
        const auto t0 = std::chrono::steady_clock::now();
        const int pos = static_cast<int>(toks.size()), D = c.n_embd, hc = c.hc;
        toks.push_back(token);
        cur_pos = pos;

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
            // layer 0 has no previous sublayer: the identity mix picks copy 0 (pre, after the row in step_h)
            ck(cudaStreamSynchronize(st), "step_h");   // the previous step's reads of step_h are done
            std::memcpy(step_h, e.data(), D * 4);
            for (int j = 0; j < hc; ++j) step_h[D + j] = j == 0 ? 1.f : 0.f;
            cuda::dsv41::copy_small(x, step_h, D * 4, st);
            cuda::dsv41::copy_small(pre, step_h + D, hc * 4, st);
            cuda::repeat(x, R, D, hc, st);
        }

        for (int il = 0; il < c.n_layer; ++il) {
            const std::string p = "blk." + std::to_string(il) + ".";
            const int eg = c.engram_index(il);
            auto h0 = std::chrono::steady_clock::now();
            auto lap = [&](int i) {
                const auto h1 = std::chrono::steady_clock::now();
                t_host[i] += std::chrono::duration<double>(h1 - h0).count();
                h0 = h1;
            };
            if (eg >= 0 && engram_on) engram(il, eg, erow.data());
            lap(0);

            // attention sublayer: collapses with the mix the previous sublayer made
            hc_mix(p + "hc_attn_fn.weight", p + "hc_attn_scale.weight", p + "hc_attn_base.weight", a_pre, post, comb);
            cuda::dsv41::hc_pre(R, pre, x, hc, D, st);
            P("hc_attn_pre", il, x, D);
            norm(x, x, D, p + "attn_norm.weight");
            P("attn_norm", il, x, D);
            attention(il, pos);
            cuda::dsv41::hc_post(ao, R, post, comb, R2, hc, D, st);
            std::swap(R, R2);
            P("hc_attn_post", il, R, c.hc_dim());
            lap(1);

            // FFN sublayer: collapses with the attention's mix; its own pre goes to the next layer's attention
            hc_mix(p + "hc_ffn_fn.weight", p + "hc_ffn_scale.weight", p + "hc_ffn_base.weight", pre, post, comb);
            cuda::dsv41::hc_pre(R, a_pre, x, hc, D, st);
            P("hc_ffn_pre", il, x, D);
            norm(x, x, D, p + "ffn_norm.weight");
            P("ffn_norm", il, x, D);
            lap(2);
            moe_ffn(il);
            cuda::dsv41::hc_post(ffo, R, post, comb, R2, hc, D, st);
            std::swap(R, R2);
            P("l_last", il, R, c.hc_dim());
            lap(3);
        }
        // the last FFN's mix collapses the copies (there is no separate hc head)
        cuda::dsv41::hc_pre(R, pre, x, hc, D, st);
        P("hc_out", -1, x, D);
        norm(x, x, D, "output_norm.weight");
        P("result_norm", -1, x, D);
        mv("output.weight", x, logits_d);
        P("result_output", -1, logits_d, c.n_vocab);
        // by kernel into pinned memory (a pageable copy is staged and synchronous, and the D2H engine may be busy
        // with an expert going back to RAM)
        cuda::dsv41::copy_small(logits_h, logits_d, c.n_vocab * 4, st);
        ck(cudaStreamSynchronize(st), "step");
        std::memcpy(logits.data(), logits_h, c.n_vocab * 4);
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
const std::vector<float> & Dsv41Engine::prefill(const std::vector<int> & t, std::vector<float> * all) { return p_->prefill(t, all); }
void Dsv41Engine::set_probe(Probe p) { p_->probe = std::move(p); }
void Dsv41Engine::reset_stats() {
    auto & q = *p_;
    q.t_total = q.t_expert_copy = q.t_engram = q.t_cpu = q.t_gpu_wait = q.t_ram = q.t_router = q.t_admit = q.t_prefill = 0;
    for (double & t : q.t_host) t = 0;
    q.t_gpu_pre = q.t_span = 0;
    for (int i = 0; i < 3; ++i) { q.t_gpu_wait_by[i] = 0; q.n_gpu_wait_by[i] = 0; }
    q.n_steps = 0;
    q.n_prefill = 0;
    q.hits = q.misses = q.cpu_experts = q.demoted = q.gpu_misses = 0;
    q.pred_disk = q.pred_disk_hit = q.pred_all = q.pred_all_hit = 0;
    if (q.ram) { q.ram->hits = q.ram->reads = q.ram->prefetched = q.ram->prefetch_used = q.ram->prefetch_waits = 0; q.ram->read_s = 0; }
}

std::string Dsv41Engine::report() const {
    std::ostringstream os;
    const auto & q = *p_;
    if (q.n_steps == 0 && q.n_prefill == 0) return "no steps";
    if (q.n_prefill) os << "prefill " << q.n_prefill << " tokens in " << q.t_prefill << " s (" << q.n_prefill / q.t_prefill << " tok/s); ";
    const int ns = std::max(q.n_steps, 1);
    os << q.n_steps << " steps, " << 1000.0 * q.t_total / ns << " ms/token (expert copies, prefill included: "
       << 1000.0 * q.t_expert_copy / ns << " ms, Engram rows " << 1000.0 * q.t_engram / ns << " ms); "
       << "expert cache hit rate " << (q.hits + q.misses ? 100.0 * q.hits / (q.hits + q.misses) : 0.0) << "% of "
       << q.hits + q.misses;
    if (q.hybrid)
        os << "; GPU-computed misses " << static_cast<double>(q.gpu_misses) / ns << "/token; CPU experts " << static_cast<double>(q.cpu_experts) / ns << "/token in " << 1000.0 * q.t_cpu / ns
           << " ms (of it RAM tier / disk " << 1000.0 * q.t_ram / ns << " ms), then GPU wait " << 1000.0 * q.t_gpu_wait / ns << " ms; router wait " << 1000.0 * q.t_router / ns
           << " ms (GPU time MoE end -> router " << 1000.0 * q.t_gpu_pre / ns << " ms, wall " << 1000.0 * q.t_span / ns << " ms), admissions " << 1000.0 * q.t_admit / ns << " ms; GPU wait by layer kind (no miss / CPU only / early): "
           << 1000.0 * q.t_gpu_wait_by[0] / ns << " / " << 1000.0 * q.t_gpu_wait_by[1] / ns << " / " << 1000.0 * q.t_gpu_wait_by[2] / ns
           << " ms over " << 1.0 * q.n_gpu_wait_by[0] / ns << " / " << 1.0 * q.n_gpu_wait_by[1] / ns << " / " << 1.0 * q.n_gpu_wait_by[2] / ns
           << " layers; host per step: engram " << 1000.0 * q.t_host[0] / ns
           << " ms, attention sublayer " << 1000.0 * q.t_host[1] / ns << ", FFN pre " << 1000.0 * q.t_host[2] / ns << ", MoE + post "
           << 1000.0 * q.t_host[3] / ns << " ms";
    if (q.pred_stats)
        os << "; next-layer prediction (top " << q.pred_k << "): " << 100.0 * q.pred_all_hit / std::max<uint64_t>(q.pred_all, 1)
           << "% of all picks, " << 100.0 * q.pred_disk_hit / std::max<uint64_t>(q.pred_disk, 1) << "% of the " << q.pred_disk << " disk reads";
    if (q.ram)
        os << "; RAM tier: " << q.ram->hits << " hits, " << q.ram->reads << " disk reads in " << q.ram->read_s << " s ("
           << (q.ram->read_s > 0 ? q.ram->reads * 9.12e-3 / q.ram->read_s : 0.0) << " GB/s)";
    if (q.exclusive) os << "; exclusive tiers: " << q.demoted << " demoted to RAM";
    if (q.ram && q.prefetch_k > 0)
        os << "; prefetch (top " << q.prefetch_k << "): " << q.ram->prefetched << " read, " << q.ram->prefetch_used
           << " used (" << q.ram->prefetch_waits << " still in flight when needed)";
    return os.str();
}

}  // namespace bl
