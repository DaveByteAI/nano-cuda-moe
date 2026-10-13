#include "bl/engine.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <type_traits>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <random>
#include <unordered_map>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "bl/cpu_experts.h"
#include "bl/cuda_fused.h"
#include "bl/cuda_ops.h"
#include "bl/cuda_pipeline.h"
#include "bl/cuda_quant.h"
#include "bl/cuda_prefill.h"
#include "bl/cuda_sample.h"
#include "bl/cuda_window.h"

namespace bl {

namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// a weight matrix in VRAM, still in its file format: M rows of K values
struct Mat {
    uint32_t type = 0;
    int      M = 0, K = 0;
    size_t   row_bytes = 0;
    void *   d = nullptr;
};

float half_to_float(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {   // subnormal
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; ++e; } while (!(m & 0x400));
            bits = sign | static_cast<uint32_t>(127 - 15 - e) << 23 | (m & 0x3ff) << 13;
        }
    } else if (exp == 31) bits = sign | 0x7f800000u | man << 13;
    else bits = sign | (exp + 127 - 15) << 23 | man << 13;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint16_t float_to_half(float f) {   // round to nearest even; the scales here are normal numbers
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int e = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (e <= 0) {   // subnormal half (or zero)
        if (e < -10) return static_cast<uint16_t>(sign);
        man |= 0x800000u;
        const int shift = 14 - e;
        uint32_t h = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return static_cast<uint16_t>(sign | h);
    }
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    uint32_t h = static_cast<uint32_t>(e) << 10 | man >> 13;
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return static_cast<uint16_t>(sign | h);
}

template <typename T> T * host_mapped(size_t n, std::vector<void *> & owned) {
    void * p = nullptr;
    ck(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc mapped");
    std::memset(p, 0, n * sizeof(T));
    owned.push_back(p);
    return static_cast<T *>(p);
}
template <typename T> T * dev_of(T * host) {
    void * d = nullptr;
    ck(cudaHostGetDevicePointer(&d, host, 0), "cudaHostGetDevicePointer");
    return static_cast<T *>(d);
}

constexpr int kT = cuda::kMaxT;

// What each GPU of a layer split holds for itself; with one GPU, the only set. Engine::Impl derives from it, so the
// code reads these as its own members; with several GPUs, Impl::bind(s) swaps stage s's set in.
struct Dev {
    int dev = 0;          // the CUDA device
    int l0 = 0, l1 = 0;   // its layers [l0, l1); the last stage also runs the head and the MTP layer
    cudaStream_t st = nullptr, cp = nullptr, adm = nullptr;

    // workspace, [kT] rows each
    float *R = nullptr, *xn = nullptr, *lo = nullptr, *mixed = nullptr, *inj = nullptr, *h = nullptr, *t0 = nullptr,
          *t1 = nullptr, *t2 = nullptr, *t3 = nullptr, *t4 = nullptr, *t5 = nullptr, *moe = nullptr, *sh = nullptr,
          *emb_t = nullptr, *logits_d = nullptr;
    int *   ids_d = nullptr;      // router choices [kT][K]
    float * w_d = nullptr;
    int *   argmax_d = nullptr;   // [kT]
    float * gdn_scratch = nullptr;
    int *   sel_list = nullptr, * sel_cnt = nullptr;   // the QSA indexer's selection (position lists)
    float * sel_score = nullptr;
    void *  ple_rows_d = nullptr;
    cuda::TokenState * ts = nullptr;
    cuda::ExpertPlan * plan_d = nullptr;   // [layer]

    // the routed experts it caches (its layers'), its table of every expert's VRAM address, its copy slots
    uint8_t ** table_d = nullptr;
    uint8_t *  arena = nullptr;
    size_t     arena_bytes = 0;
    uint8_t *  staging[cuda::kMaxSlots] = {};
    uint8_t ** staging_d = nullptr;
    float *    exp_act = nullptr, *exp_out = nullptr;   // [slot][kT][F], [slot][kT][D]
    void *     xq_d = nullptr;                          // the FFN input as q8_1, [kT] rows
    uint8_t *  pf_slot[2][cuda::kMaxPf] = {};
    uint8_t ** pf_slots_d = nullptr;
    int *      pred_ids_d = nullptr;
    float *    pred_w_d = nullptr;

    // graphs: a window per T, and the commit
    cudaGraphExec_t graph[kT + 1] = {};
    cudaGraphExec_t commit_graph = nullptr;

    // a layer split: the residual it takes from the previous stage (mapped host memory, [kT][HD] rows: a window), and
    // the event its window records for the next stage
    float *     hand_h = nullptr, * hand_hd = nullptr;
    cudaEvent_t ev_hand = nullptr;

    // prefill buffers (see Impl::layout_prefill)
    float *P_R = nullptr, *P_mixed = nullptr, *P_h = nullptr, *P_sh = nullptr, *P_inj = nullptr, *P_rl = nullptr,
          *P_w = nullptr, *P_aw = nullptr, *P_act = nullptr, *P_g = nullptr, *P_u = nullptr, *P_sg = nullptr;
    float *P_xn = nullptr, *P_lo = nullptr, *P_a = nullptr, *P_b = nullptr, *P_c = nullptr, *P_d = nullptr, *P_e = nullptr,
          *P_pn = nullptr, *P_scr = nullptr, *P_ix = nullptr;
    size_t P_scr_n = 0, P_w16_n = 0, P_x16_n = 0;
    void  *P_w16 = nullptr, *P_x16 = nullptr;
    int   *P_ids = nullptr, *P_tok = nullptr, *P_nsc = nullptr, *P_tokd = nullptr;
    void  *P_xq = nullptr, *P_xg = nullptr, *P_aq = nullptr;
    cuda::TokenState * P_ts = nullptr;
    int   *P_ids_h = nullptr, *P_tok_h = nullptr, *P_tokd_h = nullptr;
    float *P_w_h = nullptr, *P_aw_h = nullptr;
    cuda::MoeItem * P_items_h = nullptr, * P_items_d = nullptr;
    int    P_items_cap = 0;
    uint8_t * P_ple_h = nullptr;
    void * P_ple_d = nullptr;
    cudaEvent_t ev_ready[2] = {}, ev_free[2] = {}, ev_pre = nullptr, ev_moe_done = nullptr;
    std::vector<uint8_t *> pf_ring;   // speculative copies of a layer's non-cached experts during its dense part
    std::vector<int> pre_ids;         // the experts the ring holds for the current layer, by slot
    std::vector<int> next_pre;        // the next layer's non-cached experts by predicted use
    float *P_rl2 = nullptr, *P_w2 = nullptr;
    int   *P_ids2 = nullptr, *P_ids2_h = nullptr;
    float * P_mixed_sc = nullptr;     // the current sub-chunk's attention input
    // the region of the expert arena lent to the prefill buffers (Impl::loan_out / loan_back)
    uint8_t * loan_base = nullptr;
    bool      loaned = false;
    size_t    loan_bytes = 0;
    std::vector<std::pair<int, uint8_t *>> loan_slots;   // (layer, slot) of the experts placed in the region
    // a layer's matrices as bf16 for the prefill GEMMs
    std::unordered_map<std::string, const void *> w16;
    size_t w16_used = 0;
};

}  // namespace

struct Engine::Impl : Dev {
    GgufModel   g;
    ModelConfig c;
    int         max_ctx;
    Probe       probe;
    int         pos = 0;
    uint32_t    seq = 0;             // windows run so far (never reset: the flags compare against it)
    std::vector<int> history;        // every committed token (the PLE n-gram hash reads the last ones)
    std::vector<int> win_tokens;     // the last verify window
    int         win_T = 0;

    std::vector<void *> allocs, host_allocs;
    std::unordered_map<std::string, Mat>     mats;
    std::unordered_map<std::string, float *> vecs;   // small float tensors (norms, conv kernels, gates)

    // per-layer state; the GDN state is read by a window and rewritten only by commit (gdn_scratch takes the window's)
    std::vector<float *> conv_state, gdn_state;
    std::vector<cuda::KvCache> kv;   // per layer (QSA layers and the MTP layer): int8 + scales, or fp16 (BL_KV_F16=1)
    // the QSA indexer: raw keys per cell and pooled block keys, per main QSA layer; the selection (position lists)
    std::vector<float *> ikraw, ikblk;
    static constexpr int kSelCells = 2048 + 3, kSelStride = 2056, kSelRows = 256;
    int     max_blocks = 0;
    float * ple_hist = nullptr;
    int     max_kv = 0;
    // a window's per-layer inputs that commit replays for the accepted tokens
    std::vector<float *> w_qkv, w_conv, w_beta, w_g;
    float * w_ple_norm = nullptr;

    int *   argmax_h = nullptr;   // mapped
    float * logits_h = nullptr;   // pinned [kT][V]
    std::vector<float> logits;
    std::vector<int>   ids_out;

    // window inputs, read by the GPU from pinned host memory when the graph runs
    int *     tokens_h = nullptr, * tokens_hd = nullptr;   // [kT]
    int *     ncommit_h = nullptr, * ncommit_hd = nullptr;
    uint8_t * ple_rows_h = nullptr;                        // [kT][heads] rows
    size_t    ple_row_bytes = 0;

    // ---- routed experts: a VRAM arena of the most-used ones, the rest (or all) in pinned RAM
    struct LayerExperts {
        uint32_t type_gu = 0, type_d = 0;
        size_t   gu_bytes = 0, d_bytes = 0;   // one expert's gate (= up) and down
        size_t   bytes() const { return 2 * gu_bytes + d_bytes; }
    };
    std::vector<LayerExperts> lx;
    std::vector<uint8_t *> exp_dev;            // [layer * n_expert + e]: VRAM address or nullptr
    std::vector<uint8_t *> exp_host;           // pinned RAM address (every expert when ram_all)
    uint8_t *  host_store = nullptr;
    size_t     cached = 0, cached_bytes = 0, host_bytes = 0;
    long long  n_hit = 0, n_dma = 0, n_cpu = 0;
    std::unique_ptr<CpuExperts> cpu;
    float      cpu_share = 0.7f;       // of the misses, the share the CPU computes (the rest: PCIe), one-token windows
    float      cpu_share_win = 1.0f;   // the same for longer windows (their misses compete with LRU admissions for PCIe)

    // the doorbell protocol, all in mapped pinned memory, one slot per layer
    cuda::ExpertPlan * plan_h = nullptr, * plan_hd = nullptr;
    float *    x_h = nullptr, * x_hd = nullptr;            // [layer][kT][D] the FFN input, for the CPU
    float *    cpu_out_h = nullptr, * cpu_out_hd = nullptr;   // [cpu slot][kT][D]
    uint32_t * bell_h = nullptr, * dma_flag_h = nullptr, * cpu_flag_h = nullptr;
    uint32_t * bell_hd = nullptr, * dma_flag_hd = nullptr, * cpu_flag_hd = nullptr;

    // next-layer prefetch (one-token windows; off by default with the LRU cache)
    cuda::PfTable * pf_h = nullptr, * pf_hd = nullptr;
    uint32_t * pf_flag_h = nullptr, * pf_flag_hd = nullptr;
    int        pf_max = 2;
    long long  n_pf_issued = 0, n_pf_used = 0;
    std::FILE * route_trace = nullptr;   // BL_ROUTE_TRACE: per token and layer, the selected expert ids

    // LRU cache per layer (needs every expert in pinned RAM: ram_all)
    bool       ram_all = false;
    bool       lru = true;
    int        lru_max = 10;
    uint64_t   use_clock = 0;
    std::vector<uint64_t> last_use;
    // testing (BL_STAGE_PROF=1): the window graph stamps the GPU clock at stage boundaries; verify() adds up the time
    // between consecutive stamps per stage (each stamp's own cost, measured by two back-to-back stamps, taken off)
    enum Stage { kStStart, kStCal, kStEmbed, kStHc, kStGdn, kStQsa, kStRoute, kStHit, kStShared, kStWait, kStCombine, kStHead, kStN };
    bool stage_prof = std::getenv("BL_STAGE_PROF") != nullptr;
    std::vector<int> * stamp_cats = nullptr;   // set while the window graph is being recorded
    std::vector<int> stamp_cat[kT + 1];
    double stage_ms[kStN] = {};
    long long stage_stamps[kStN] = {}, stage_windows = 0;
    void stamp(int cat) {
        if (!stamp_cats || static_cast<int>(stamp_cats->size()) >= cuda::kMaxStamps) return;
        cuda::stamp(static_cast<int>(stamp_cats->size()), st);
        stamp_cats->push_back(cat);
    }
    void add_stamps(int T) {
        const auto & cats = stamp_cat[T];
        if (cats.size() < 3) return;
        std::vector<unsigned long long> t(cats.size());
        cuda::read_stamps(t.data(), static_cast<int>(t.size()));
        for (size_t i = 1; i < t.size(); ++i) {
            stage_ms[cats[i]] += (t[i] - t[i - 1]) * 1e-6;
            ++stage_stamps[cats[i]];
        }
        ++stage_windows;
    }

    // the expert cache saved by save_cache() (most recently used first), restored at load: ranked before the profile
    std::string cache_file;
    std::vector<std::pair<int, int>> restored;
    long long  n_admit = 0;

    // periodic swaps (the cache when not LRU)
    std::vector<float> use;
    int        adapt_every = 32, adapt_max = 64;
    float      adapt_decay = 0.8f, adapt_margin = 1.0f;
    uint8_t *  swap_bounce = nullptr;
    size_t     max_expert_bytes = 0;
    long long  n_swaps = 0;

    // ---- MTP draft layer (mtp-q2_0.gguf, made by bl-mtp-pack: the checkpoint's mtp.* tensors, experts in Q2_0), an extra
    // "layer" il = n_layer with its own KV cache and position; its 512 experts all live in VRAM
    std::unique_ptr<GgufModel> mtp_g;
    bool       mtp = false;
    cuda::TokenState * mts = nullptr;
    int *      mtp_tok_h = nullptr, * mtp_tok_hd = nullptr, * mtp_pos_h = nullptr, * mtp_pos_hd = nullptr;
    float *    Rwin = nullptr, * mtp_Rin = nullptr, * mtp_Rout = nullptr, * mtp_logits = nullptr;
    int *      mtp_id_d = nullptr, * mtp_id_h = nullptr;
    float *    mtp_prob_d = nullptr, * mtp_prob_h = nullptr;
    cudaGraphExec_t mtp_graph[kT + 1][kT] = {};   // [catch-up rows][chained drafts]
    int *   dr_tok_d = nullptr, * dr_tok_h = nullptr;     // the drafts as token ids, [kT]
    float * dr_prob_d = nullptr, * dr_prob_h = nullptr;   // their probabilities
    double     t_draft_ms = 0;

    double t_token_ms = 0, t_bell_ms = 0, t_cpu_ms = 0, t_adapt_ms = 0, t_ple_ms = 0, t_launch_ms = 0;
    long long steps = 0, windows = 0;

    Impl(const std::string & path, int ctx, const GpuSplit & sp)
        : g(path), c(ModelConfig::from_gguf(g)), max_ctx(ctx), split_opt(sp), model_path(path) {}

    // ---- the stages of a layer split (one GPU: one stage, and nothing below ever switches)
    GpuSplit split_opt;
    std::vector<Dev> stages;          // stage s's set while it is not the current one (the current one's is *this)
    int cur = 0;                      // the current stage: its Dev is this object's
    std::vector<int> layer_stage;     // [layer]: the stage that runs it (the MTP layer, n_layer: the last)
    std::unordered_map<std::string, int> owner_of;   // split: the stage holding each matrix / vector (-1: every one)
    bool multi() const { return stages.size() > 1; }
    std::vector<int> gpu_list() const {   // the distinct GPUs, stage order
        std::vector<int> v;
        for (int s = 0; s < static_cast<int>(stages.size()); ++s) {
            const int d = s == cur ? dev : stages[s].dev;
            if (std::find(v.begin(), v.end(), d) == v.end()) v.push_back(d);
        }
        return v;
    }
    int  last() const { return static_cast<int>(stages.size()) - 1; }
    // make stage s current: its streams, buffers and graphs become this object's members, its GPU the thread's device
    void bind(int s) {
        if (s == cur) return;
        std::swap(static_cast<Dev &>(*this), stages[cur]);   // park the current set (stages[cur] held a placeholder)
        std::swap(static_cast<Dev &>(*this), stages[s]);
        cur = s;
        ck(cudaSetDevice(dev), "cudaSetDevice");
    }
    Dev & stage(int s) { return s == cur ? static_cast<Dev &>(*this) : stages[s]; }
    // an entry from another thread (the server's): CUDA's current device is per thread
    void enter() {
        if (multi() || dev != 0) ck(cudaSetDevice(dev), "cudaSetDevice");
    }
    // the stage whose GPU holds a tensor: a layer's go with the layer; the token embeddings are in mapped host memory
    // (every stage reads them); the rest - the head, the final mixer, the MTP's own - go with the last stage
    int tensor_stage(const std::string & n) const {
        if (n.rfind("blk.", 0) == 0) return layer_stage[std::min(std::atoi(n.c_str() + 4), c.n_layer)];
        if (n == "token_embd.weight" || n == "per_layer_token_embd.weight") return -1;
        return last();
    }
    void check_owner(const std::string & n) const {   // split: a stage reads only its own GPU's weights
        if (!multi()) return;
        auto it = owner_of.find(n);
        if (it != owner_of.end() && it->second >= 0 && it->second != cur)
            throw std::runtime_error(n + " is on stage " + std::to_string(it->second) + "'s GPU, used by stage " + std::to_string(cur));
    }
    std::string model_path;   // shard 1
    // next to shard 1 (the MTP layer: models/mtp-q2_0.gguf)
    std::string beside_model(const std::string & file) const {
        const auto s = model_path.rfind('/');
        return (s == std::string::npos ? std::string(".") : model_path.substr(0, s)) + "/" + file;
    }
    // the repository's data/ (the executable is build/<program>): files that ship with the code
    static std::string data_path(const std::string & file) {
        char exe[4096];
        const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        std::string d = n > 0 ? std::string(exe, static_cast<size_t>(n)) : std::string("./x");
        d = d.substr(0, d.rfind('/'));   // build/
        d = d.substr(0, d.rfind('/'));   // the repository
        return d + "/data/" + file;
    }

    template <typename T> T * dalloc(size_t n) {
        void * p = nullptr;
        if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) {
            size_t fr = 0, to = 0;
            cudaGetLastError();
            cudaMemGetInfo(&fr, &to);
            throw std::runtime_error("cudaMalloc of " + std::to_string(n * sizeof(T) >> 20) + " MiB failed, " +
                                     std::to_string(fr >> 20) + " MiB free");
        }
        ck(cudaMemset(p, 0, n * sizeof(T)), "cudaMemset");
        allocs.push_back(p);
        dalloc_bytes += n * sizeof(T);
        return static_cast<T *>(p);
    }
    size_t dalloc_bytes = 0;   // buffers (state, KV, scratch) allocated through dalloc

    const Tensor & need(const std::string & n) {
        const Tensor & t = g.at(n);
        if (!t.data) throw std::runtime_error(n + ": bytes not on disk (incomplete download?)");
        return t;
    }

    void upload_mat(const std::string & n) { upload_mat(n, need(n)); }
    // fast load: the main model's matrices are only allocated here; stream_weights() fills them in one pass
    bool fast_load = false;
    struct Pending { void * d; const Tensor * t; int stage; };
    std::vector<Pending> pending_dense;
    void upload_mat(const std::string & n, const Tensor & t, bool defer = false) {
        Mat m;
        m.type = t.type_id;
        m.K = static_cast<int>(t.shape[0]);
        m.M = static_cast<int>(t.elements() / m.K);
        m.row_bytes = row_bytes(t.type_id, m.K);
        if (!cuda::supported(m.type)) throw std::runtime_error(n + ": unsupported type");
        ck(cudaMalloc(&m.d, t.nbytes), "cudaMalloc");
        allocs.push_back(m.d);
        if (defer) pending_dense.push_back({m.d, &t, cur});
        else ck(cudaMemcpy(m.d, t.data, t.nbytes, cudaMemcpyHostToDevice), "upload");
        mats[n] = m;
    }

    // a matrix the GPU reads only a few rows of per window (the token embeddings: T rows) stays in mapped pinned host
    // memory, read over PCIe by the kernels: its VRAM goes to the expert cache instead (BL_EMBD_VRAM=1: in VRAM)
    void map_mat(const std::string & n, const Tensor & t) {
        Mat m;
        m.type = t.type_id;
        m.K = static_cast<int>(t.shape[0]);
        m.M = static_cast<int>(t.elements() / m.K);
        m.row_bytes = row_bytes(t.type_id, m.K);
        if (!cuda::supported(m.type)) throw std::runtime_error(n + ": unsupported type");
        uint8_t * h = host_mapped<uint8_t>(t.nbytes, host_allocs);
        std::memcpy(h, t.data, t.nbytes);
        m.d = dev_of(h);
        mats[n] = m;
        mapped_bytes += t.nbytes;
    }
    size_t mapped_bytes = 0;

    void upload_vec(const std::string & n) { upload_vec(n, need(n), 0.f); }
    void upload_vec(const std::string & n, const Tensor & t, float plus) {
        std::vector<float> f(static_cast<size_t>(t.elements()));
        if (t.type_id == 0) std::memcpy(f.data(), t.data, f.size() * 4);
        else if (t.type_id == 1 || t.type_id == 30) {
            for (size_t i = 0; i < f.size(); ++i) {
                uint16_t v;
                std::memcpy(&v, t.data + 2 * i, 2);
                if (t.type_id == 1) f[i] = half_to_float(v);
                else {
                    const uint32_t b = static_cast<uint32_t>(v) << 16;
                    std::memcpy(&f[i], &b, 4);
                }
            }
        } else throw std::runtime_error(n + ": vector of unexpected type");
        for (float & x : f) x += plus;
        float * d = dalloc<float>(f.size());
        ck(cudaMemcpy(d, f.data(), f.size() * 4, cudaMemcpyHostToDevice), "upload");
        vecs[n] = d;
    }

    const Mat & M(const std::string & n) const {
        auto it = mats.find(n);
        if (it == mats.end()) throw std::runtime_error("no matrix " + n);
        check_owner(n);
        return it->second;
    }
    float * V(const std::string & n) const {
        auto it = vecs.find(n);
        if (it == vecs.end()) throw std::runtime_error("no vector " + n);
        check_owner(n);
        return it->second;
    }

    void add(cuda::MvGroup & gr, const std::string & n, float * y) {
        const Mat & m = M(n);
        if (gr.n && gr.K != m.K) throw std::runtime_error(n + ": K differs within a matvec group");
        gr.K = m.K;
        gr.add(m.type, m.d, m.row_bytes, m.M, y);
    }
    void add_vec(cuda::MvGroup & gr, const std::string & n, int K, float * y) {   // an f32 vector as a 1-row matrix
        gr.K = K;
        gr.add(0 /*F32*/, V(n), static_cast<size_t>(K) * 4, 1, y);
    }
    void mm(const std::string & n, const float * x, float * y, int T) {
        cuda::MvGroup gr;
        add(gr, n, y);
        cuda::matmul_group(gr, x, T, st);
    }

    void emit(const std::string & name, const float * d, int n) {
        if (probe) {
            ck(cudaStreamSynchronize(st), "sync");
            probe(name, d, n);
        }
    }
    static std::string L(const char * base, int il) { return std::string(base) + "-" + std::to_string(il); }

    // ------------------------------------------------------------------------------------------------- loading
    // load phases, seconds (printed at the end of load)
    std::vector<std::pair<const char *, double>> load_times;
    std::chrono::steady_clock::time_point lt0 = std::chrono::steady_clock::now();
    void lap(const char * what) {
        const auto now = std::chrono::steady_clock::now();
        load_times.push_back({what, std::chrono::duration<double>(now - lt0).count()});
        lt0 = now;
    }

    // layer il's KV cache: max_ctx cells, or for the MTP layer (which attends only the last mtp_window positions) a
    // ring of mtp_ring cells when the context is longer
    // the MTP layer's dense attention: the last this many positions. A window of 32768 on a 32K prompt drafts no
    // better (acceptance 66.7% vs 65.6%) and costs 2.14 ms per MTP run instead of 1.12 (2026-10-05)
    int mtp_window = kSelCells;
    int mtp_ring = 0;         // a power of 2 >= mtp_window + 512 (a prefill sub-chunk (kSc) + a window)
    int mtp_window_max() const { return kv.size() > static_cast<size_t>(c.n_layer) && kv[c.n_layer].mask >= 0 ? mtp_ring - 512 : max_kv; }
    cuda::KvCache kv_alloc(int il) {
        static const bool f16 = [] { const char * e = std::getenv("BL_KV_F16"); return e && std::atoi(e) != 0; }();
        cuda::KvCache r;
        const int ring_n = mtp_ring;
        const bool ring = il == c.n_layer && max_ctx > ring_n;
        const size_t cells = ring ? ring_n : static_cast<size_t>(max_ctx), n = cells * c.n_head_kv * c.head_dim;
        if (ring) r.mask = ring_n - 1;
        if (f16) {
            r.k = dalloc<__half>(n);
            r.v = dalloc<__half>(n);
        } else {
            r.k = dalloc<int8_t>(n);
            r.v = dalloc<int8_t>(n);
            r.sk = dalloc<__half>(n / cuda::kKvGroup);
            r.sv = dalloc<__half>(n / cuda::kKvGroup);
        }
        return r;
    }

    // the stages: the GPUs and their layer ranges (one GPU: one stage over every layer); each gets its streams
    void setup_stages() {
        std::vector<int> gpus = split_opt.gpus;
        if (gpus.empty()) gpus = {0};
        const int n = static_cast<int>(gpus.size()), L = c.n_layer;
        std::vector<int> first = split_opt.first_layer;
        if (n > 1 && first.empty()) {   // auto: the order may change too
            const SplitPlan p = auto_split(gpus);
            gpus = p.gpus;
            first = p.first;
        }
        if (static_cast<int>(first.size()) != n - 1) throw std::runtime_error("--layer-split: give one first layer per GPU after the first");
        for (int s = 0; s + 1 < n; ++s)
            if (first[s] < 1 || first[s] >= L || (s && first[s] <= first[s - 1]))
                throw std::runtime_error("--layer-split: first layers must rise within 1.." + std::to_string(L - 1));
        stages.assign(n, Dev{});
        layer_stage.assign(L + 1, n - 1);
        for (int s = 0; s < n; ++s) {
            stages[s].dev = gpus[s];
            stages[s].l0 = s ? first[s - 1] : 0;
            stages[s].l1 = s + 1 < n ? first[s] : L;
            for (int il = stages[s].l0; il < stages[s].l1; ++il) layer_stage[il] = s;
        }
        static_cast<Dev &>(*this) = stages[0];   // stage 0 current; stages[0] is now the parking place
        cur = 0;
        ck(cudaSetDevice(dev), "cudaSetDevice");
        for (int s = 0; s < n; ++s) {
            bind(s);
            ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
            ck(cudaStreamCreateWithFlags(&cp, cudaStreamNonBlocking), "stream");
            ck(cudaStreamCreateWithFlags(&adm, cudaStreamNonBlocking), "stream");
        }
        bind(0);
        if (multi()) {
            std::string m = "layer split:";
            for (int s = 0; s < n; ++s)
                m += " layers " + std::to_string(stage(s).l0) + "-" + std::to_string(stage(s).l1 - 1) + " on GPU " + std::to_string(gpus[s]) + (s + 1 < n ? "," : "");
            std::fprintf(stderr, "%s; the head and the MTP layer on GPU %d\n", m.c_str(), gpus.back());
        }
    }

    // the experts in VRAM-cache order (the saved cache, then the shipped ranking): read once
    std::vector<std::pair<int, int>> ranked_;
    bool ranked_read = false;
    const std::vector<std::pair<int, int>> & ranking() {
        if (!ranked_read) { ranked_ = expert_ranking(); ranked_read = true; }
        return ranked_;
    }

    // --layer-split auto: the order of the GPUs and the first layer of each after the first. A candidate is scored by
    // placing the experts by rank the way load_experts() will - into each GPU's free VRAM less its layers' dense weights
    // and KV cache, an allowance for its buffers, the head and the MTP layer on the last, and the reserve - and summing
    // the placed experts' rank weights (the best-ranked count most). Ties: the fullest GPU keeps the most room (when
    // every expert fits, as on four 24 GB cards), then the more even split. Up to three GPUs every order and every split
    // is tried; beyond, the GPU with the most free VRAM goes last and the split starts in proportion to free VRAM, then
    // moves one boundary by one layer at a time while that scores better.
    struct SplitPlan { std::vector<int> gpus, first; long long score = -1, room = 0; int uneven = 0; };
    SplitPlan auto_split(std::vector<int> gpus) {
        const int n = static_cast<int>(gpus.size()), L = c.n_layer, NE = c.n_expert;
        std::unordered_map<int, long long> free_of;
        for (int d : gpus) {
            if (free_of.count(d)) continue;
            size_t fr = 0, to = 0;
            ck(cudaSetDevice(d), "cudaSetDevice");
            ck(cudaMemGetInfo(&fr, &to), "meminfo");
            free_of[d] = static_cast<long long>(fr);
        }
        std::vector<long long> dense(L + 1, 0), exp_b(L, 0);   // per layer; dense[L]: the head, the final mixer
        for (const auto & t : g.tensors()) {
            const std::string & nm = t.name;
            if (nm == "token_embd.weight" || nm == "per_layer_token_embd.weight") continue;
            const int il = nm.rfind("blk.", 0) == 0 ? std::atoi(nm.c_str() + 4) : L;
            if (nm.find("_exps.") != std::string::npos) exp_b[il] += static_cast<long long>(t.nbytes / NE);
            else dense[il] += static_cast<long long>(t.nbytes);
        }
        const long long kv_layer = static_cast<long long>(2.0 * max_ctx * c.n_head_kv * c.head_dim * (1.0 + 2.0 / cuda::kKvGroup))
                                 + static_cast<long long>(max_ctx / 4 + 1 + cuda::kIdxRing) * c.idx_dim * 4;
        for (int il = 0; il < L; ++il) if (c.is_qsa(il)) dense[il] += kv_layer;
        long long mtp_bytes = 0;
        if (std::ifstream f(beside_model("mtp-q2_0.gguf"), std::ios::binary | std::ios::ate); f) mtp_bytes = f.tellg();
        const char * rs = std::getenv("BL_VRAM_RESERVE_MB");
        const long long reserve = (rs ? std::atoll(rs) : 1024) << 20, fixed = 1ll << 30;
        const char * cap = std::getenv("BL_EXPERT_CACHE_MB");
        const auto & rk = ranking();

        auto eval = [&](const std::vector<int> & gp, const std::vector<int> & first) {
            SplitPlan p{gp, first};
            std::vector<int> slot(n);   // stage -> its GPU's budget (stages on one GPU share it)
            std::vector<long long> bud;
            for (int s = 0; s < n; ++s) {
                const auto it = std::find(gp.begin(), gp.begin() + s, gp[s]);
                slot[s] = it != gp.begin() + s ? slot[it - gp.begin()] : static_cast<int>(bud.size());
                if (slot[s] == static_cast<int>(bud.size())) bud.push_back(free_of[gp[s]] - reserve);
            }
            std::vector<int> slot_of(L);   // layer -> budget
            for (int il = 0, s = 0; il < L; ++il) {
                while (s + 1 < n && il >= first[s]) ++s;
                slot_of[il] = slot[s];
                bud[slot[s]] -= dense[il];
            }
            for (int s = 0; s < n; ++s) bud[slot[s]] -= fixed;
            bud[slot[n - 1]] -= dense[L] + mtp_bytes;
            if (cap) for (auto & b : bud) b = std::min(b, std::atoll(cap) << 20);
            p.score = 0;
            for (size_t i = 0; i < rk.size(); ++i) {
                const int il = rk[i].first;
                long long & b = bud[slot_of[il]];
                if (b >= exp_b[il]) { b -= exp_b[il]; p.score += static_cast<long long>(rk.size() - i); }
            }
            p.room = LLONG_MAX;
            for (long long b : bud) p.room = std::min(p.room, b);
            for (int s = 0; s < n; ++s) p.uneven += std::abs(((s + 1 < n ? first[s] : L) - (s ? first[s - 1] : 0)) * n - L);
            return p;
        };
        auto better = [](const SplitPlan & a, const SplitPlan & b) {
            if (a.score != b.score) return a.score > b.score;
            if (a.room != b.room) return a.room > b.room;
            return a.uneven < b.uneven;
        };
        SplitPlan best;
        if (n <= 3) {
            std::vector<int> order = gpus;
            std::sort(order.begin(), order.end());
            do {
                if (n == 2) {
                    for (int k = 1; k < L; ++k) if (auto p = eval(order, {k}); best.score < 0 || better(p, best)) best = p;
                } else {
                    for (int k1 = 1; k1 + 1 < L; ++k1)
                        for (int k2 = k1 + 1; k2 < L; ++k2)
                            if (auto p = eval(order, {k1, k2}); best.score < 0 || better(p, best)) best = p;
                }
            } while (std::next_permutation(order.begin(), order.end()));
        } else {
            const auto big = std::max_element(gpus.begin(), gpus.end(), [&](int x, int y) { return free_of[x] < free_of[y]; });
            std::rotate(big, big + 1, gpus.end());   // the most free VRAM last, the others in the given order
            long long tot = 0, acc = 0;
            for (int d : gpus) tot += free_of[d];
            std::vector<int> first;
            for (int s = 0; s + 1 < n; ++s) {
                acc += free_of[gpus[s]];
                first.push_back(std::clamp(static_cast<int>(static_cast<double>(acc) / tot * L + 0.5), s + 1, L - (n - 1 - s)));
                if (s && first[s] <= first[s - 1]) first[s] = first[s - 1] + 1;
            }
            best = eval(gpus, first);
            for (bool moved = true; moved;) {
                moved = false;
                for (int s = 0; s + 1 < n; ++s)
                    for (int dk : {-1, 1}) {
                        std::vector<int> f = best.first;
                        f[s] += dk;
                        if (f[s] < (s ? f[s - 1] + 1 : 1) || f[s] > (s + 2 < n ? f[s + 1] - 1 : L - 1)) continue;
                        if (auto p = eval(gpus, f); better(p, best)) { best = p; moved = true; }
                    }
            }
        }
        std::string m = "layer split auto: GPUs in the order";
        for (int d : best.gpus) m += " " + std::to_string(d);
        m += ", later ones from layer";
        for (int k : best.first) m += " " + std::to_string(k);
        std::fprintf(stderr, "%s\n", m.c_str());
        return best;
    }

    void load() {
        lap("context");
        setup_stages();
        fast_load = decide_fast_load();
        for (const auto & t : g.tensors()) {
            const std::string & n = t.name;
            if (n.find("_exps.") != std::string::npos || n == "per_layer_token_embd.weight") continue;
            const int s = tensor_stage(n);
            bind(s < 0 ? 0 : s);
            if (multi()) owner_of[n] = s;
            // vectors and the conv kernels are read element-wise as f32; everything else stays in its file format
            if (t.shape.size() == 1 || n.find("ssm_conv1d") != std::string::npos || n.find("ple_conv1d") != std::string::npos)
                upload_vec(n);
            else if (n == "token_embd.weight" && (multi() || !std::getenv("BL_EMBD_VRAM")))
                map_mat(n, need(n));   // (in VRAM it would be one GPU's; the first and the last stage both read it)
            else
                upload_mat(n, need(n), fast_load && t.shard == 0);
        }
        const Tensor & ple = g.at("per_layer_token_embd.weight");   // the PLE table must exist (shard 2)
        if (const Tensor & te = need("token_embd.weight"); !cuda::embed_supported(te.type_id))
            throw std::runtime_error("token_embd.weight: format " + std::string(CpuExperts::type_name(te.type_id)) +
                                     " is not handled by the embedding kernel - this model file is not supported");
        lap("dense weights");

        const int D = c.n_embd, HD = c.hc_dim();
        const int conv_dim = 2 * c.ssm_groups * c.ssm_state + c.ssm_inner;
        const size_t gdn_state_n = static_cast<size_t>(c.ssm_v_heads) * c.ssm_state * c.ssm_state;
        max_kv = max_ctx;   // QSA beyond the indexer's top-k: sparse over the selected cells
        if (c.idx_top_k + 3 != kSelCells) throw std::runtime_error("indexer top-k " + std::to_string(c.idx_top_k) + " unexpected");
        if (const char * e = std::getenv("BL_MTP_WINDOW")) mtp_window = std::atoi(e);
        mtp_ring = 1;
        while (mtp_ring < (mtp_window > 0 ? mtp_window : max_ctx) + 512) mtp_ring *= 2;
        if (const char * e = std::getenv("BL_MTP_RING")) mtp_ring = std::atoi(e);   // testing
        if (mtp_ring & (mtp_ring - 1)) throw std::runtime_error("BL_MTP_RING must be a power of 2");
        bind(last());
        load_mtp();
        lap("MTP layer");
        for (int il = 0; il <= c.n_layer; ++il) {   // il == n_layer: the MTP layer (attention only)
            bind(layer_stage[il]);
            const bool q = qsa_layer(il);
            ikraw.push_back(q && il < c.n_layer ? dalloc<float>(static_cast<size_t>(cuda::kIdxRing) * c.idx_dim) : nullptr);
            ikblk.push_back(q && il < c.n_layer ? dalloc<float>(static_cast<size_t>(max_ctx / 4 + 1) * c.idx_dim) : nullptr);
            if (il == c.n_layer && !mtp) { conv_state.push_back(nullptr); gdn_state.push_back(nullptr); kv.emplace_back();
                                            w_qkv.push_back(nullptr); w_conv.push_back(nullptr);
                                            w_beta.push_back(nullptr); w_g.push_back(nullptr); continue; }
            conv_state.push_back(q ? nullptr : dalloc<float>(static_cast<size_t>(conv_dim) * (c.ssm_conv - 1)));
            gdn_state.push_back(q ? nullptr : dalloc<float>(gdn_state_n));
            kv.push_back(q ? kv_alloc(il) : cuda::KvCache{});
            w_qkv.push_back(q ? nullptr : dalloc<float>(static_cast<size_t>(kT) * conv_dim));
            w_conv.push_back(q ? nullptr : dalloc<float>(static_cast<size_t>(kT) * conv_dim));
            w_beta.push_back(q ? nullptr : dalloc<float>(static_cast<size_t>(kT) * c.ssm_v_heads));
            w_g.push_back(q ? nullptr : dalloc<float>(static_cast<size_t>(kT) * c.ssm_v_heads));
        }
        max_blocks = max_ctx / 4 + 1;
        bind(layer_stage[c.ple_layer]);   // the PLE's state: with its layer
        ple_hist = dalloc<float>(static_cast<size_t>(c.ple_conv - 1) * c.ple_ngram * HD);
        w_ple_norm = dalloc<float>(static_cast<size_t>(kT) * HD);
        ple_row_bytes = row_bytes(ple.type_id, c.ple_dim);

        const size_t big = std::max<size_t>({static_cast<size_t>(HD), static_cast<size_t>(conv_dim),
                                             2ull * c.n_head * c.head_dim, static_cast<size_t>(c.n_expert)});
        for (int s = 0; s <= last(); ++s) {   // every stage's own workspace
            bind(s);
            gdn_scratch = dalloc<float>(gdn_state_n);
            sel_list = dalloc<int>(static_cast<size_t>(kSelRows) * kSelStride);
            sel_cnt = dalloc<int>(kSelRows);
            sel_score = dalloc<float>(static_cast<size_t>(kSelRows) * max_blocks);
            R = dalloc<float>(kT * HD); xn = dalloc<float>(kT * HD); lo = dalloc<float>(kT * c.hc_low_rank);
            mixed = dalloc<float>(kT * D); inj = dalloc<float>(kT * c.hc); h = dalloc<float>(kT * D);
            t0 = dalloc<float>(kT * big); t1 = dalloc<float>(kT * big); t2 = dalloc<float>(kT * big); t3 = dalloc<float>(kT * big);
            t4 = dalloc<float>(kT * big); t5 = dalloc<float>(kT * big);
            moe = dalloc<float>(kT * D); sh = dalloc<float>(kT * D); emb_t = dalloc<float>(kT * D);
            logits_d = dalloc<float>(static_cast<size_t>(kT) * c.n_vocab);
            ids_d = dalloc<int>(kT * cuda::kMaxK);
            w_d = dalloc<float>(kT * cuda::kMaxK);
            argmax_d = dalloc<int>(kT);
            ck(cudaMalloc(&ple_rows_d, static_cast<size_t>(kT) * c.ple_heads() * ple_row_bytes), "ple rows");
            allocs.push_back(ple_rows_d);
            ts = dalloc<cuda::TokenState>(1);
            plan_d = dalloc<cuda::ExpertPlan>(c.n_layer + 1);
            if (s > 0) {   // the residual from the previous stage, and that stage's event
                hand_h = host_mapped<float>(static_cast<size_t>(kT) * HD, host_allocs);
                hand_hd = dev_of(hand_h);
            }
            if (s < last()) ck(cudaEventCreateWithFlags(&ev_hand, cudaEventDisableTiming), "event");
        }
        bind(last());   // the head's and the MTP's buffers
        argmax_h = host_mapped<int>(kT, host_allocs);
        ck(cudaHostAlloc(reinterpret_cast<void **>(&logits_h), sizeof(float) * kT * c.n_vocab, cudaHostAllocPortable), "pinned");
        host_allocs.push_back(logits_h);

        smp_draft_h = host_mapped<int>(kT, host_allocs); smp_tok_h = host_mapped<int>(kT, host_allocs);
        smp_acc_h = host_mapped<int>(kT, host_allocs); smp_uni_h = host_mapped<float>(2 * kT, host_allocs);
        tokens_h = host_mapped<int>(kT, host_allocs);
        tokens_hd = dev_of(tokens_h);
        ncommit_h = host_mapped<int>(1, host_allocs);
        ncommit_hd = dev_of(ncommit_h);
        ple_rows_h = host_mapped<uint8_t>(static_cast<size_t>(kT) * c.ple_heads() * ple_row_bytes, host_allocs);
        reset_token_state();

        plan_h = host_mapped<cuda::ExpertPlan>(c.n_layer + 1, host_allocs);
        x_h = host_mapped<float>(static_cast<size_t>(c.n_layer + 1) * kT * D, host_allocs);
        Rwin = dalloc<float>(static_cast<size_t>(kT) * HD);
        if (mtp) {
            mts = dalloc<cuda::TokenState>(1);
            mtp_tok_h = host_mapped<int>(kT, host_allocs); mtp_tok_hd = dev_of(mtp_tok_h);
            mtp_pos_h = host_mapped<int>(kT, host_allocs); mtp_pos_hd = dev_of(mtp_pos_h);
            dr_tok_d = dalloc<int>(kT); dr_prob_d = dalloc<float>(kT);
            dr_tok_h = host_mapped<int>(kT, host_allocs); dr_prob_h = host_mapped<float>(kT, host_allocs);
            mtp_id_h = host_mapped<int>(1, host_allocs);
            mtp_prob_h = host_mapped<float>(1, host_allocs);
            mtp_Rin = dalloc<float>(static_cast<size_t>(kT) * HD);
            mtp_Rout = dalloc<float>(static_cast<size_t>(kT) * HD);
            mtp_logits = dalloc<float>(c.n_vocab);
            mtp_id_d = dalloc<int>(1);
            mtp_prob_d = dalloc<float>(1);
        }
        cpu_out_h = host_mapped<float>(static_cast<size_t>(cuda::kMaxSlots) * kT * D, host_allocs);
        bell_h = host_mapped<uint32_t>(c.n_layer + 1, host_allocs);
        dma_flag_h = host_mapped<uint32_t>(c.n_layer + 1, host_allocs);
        cpu_flag_h = host_mapped<uint32_t>(c.n_layer + 1, host_allocs);
        pf_h = host_mapped<cuda::PfTable>(c.n_layer + 1, host_allocs);
        pf_flag_h = host_mapped<uint32_t>(c.n_layer + 1, host_allocs);
        plan_hd = dev_of(plan_h); x_hd = dev_of(x_h); cpu_out_hd = dev_of(cpu_out_h);
        bell_hd = dev_of(bell_h); dma_flag_hd = dev_of(dma_flag_h); cpu_flag_hd = dev_of(cpu_flag_h);
        pf_hd = dev_of(pf_h); pf_flag_hd = dev_of(pf_flag_h);
        alloc_prefill();
        lap("state buffers");
        load_experts();
        if (pf_on) {   // cuBLAS's first calls take ~1 s: pay it at load, on decode buffers (the region holds experts)
            for (int s = 0; s <= last(); ++s) {
                bind(s);
                const Mat & up = M("blk." + std::to_string(l0) + ".hc_attn_up.weight");
                for (int n : {1, 4, 64}) {
                    cuda::gemm_bf16(up.d, up.M, up.K, t2, std::min(n, kT), t0, up.M, t1, st);
                    cuda::gemm_w(0, t2, static_cast<size_t>(up.K) * 4, 1, up.K, t3, std::min(n, kT), t4, 1, nullptr, 0, st);
                }
                ck(cudaStreamSynchronize(st), "warm-up");
            }
            bind(last());
        }
        lap("cuBLAS warm-up");
        if (std::getenv("BL_HC_Q8_SIM") && !multi()) sim_q8_hc();
        if (multi()) stage_prof = false;   // the stage stamps time one GPU's graph
        std::string msg = "load:";
        double tot = 0;
        for (auto [w, t] : load_times) { msg += " " + std::string(w) + " " + std::to_string(t).substr(0, std::to_string(t).find('.') + 2) + " s,"; tot += t; }
        std::fprintf(stderr, "%s total %.1f s\n", msg.c_str(), tot);
    }

    void reset_token_state() {   // every stage's copy
        cuda::TokenState t0s{};
        t0s.pos = 0;
        t0s.seq = seq;
        for (int s = 0; s <= last(); ++s) {
            bind(s);
            ck(cudaMemcpy(ts, &t0s, sizeof t0s, cudaMemcpyHostToDevice), "token state");
        }
        bind(last());
    }

    // cache file: "BLEC", then uint32 version 1, n_layer, n_expert, count; then count (uint16 layer, uint16 expert)
    // a ranking file ("BLEC": a saved cache, or data/expert-rank.bin): (layer, expert) pairs, the most wanted first
    std::vector<std::pair<int, int>> read_ranking(const std::string & path) {
        std::vector<std::pair<int, int>> r;
        if (path.empty()) return r;
        std::ifstream f(path, std::ios::binary);
        if (!f) return r;   // not there (a cache file before its first save)
        char magic[4];
        uint32_t hd[4];
        f.read(magic, 4);
        f.read(reinterpret_cast<char *>(hd), sizeof hd);
        if (!f || std::memcmp(magic, "BLEC", 4) != 0 || hd[0] != 1 || hd[1] != static_cast<uint32_t>(c.n_layer) ||
            hd[2] != static_cast<uint32_t>(c.n_expert)) {
            std::fprintf(stderr, "ranking %s: not for this model, ignored\n", path.c_str());
            return r;
        }
        std::vector<uint16_t> le(2 * static_cast<size_t>(hd[3]));
        f.read(reinterpret_cast<char *>(le.data()), static_cast<std::streamsize>(le.size() * 2));
        if (!f) { std::fprintf(stderr, "ranking %s: truncated, ignored\n", path.c_str()); return {}; }
        std::vector<char> seen(static_cast<size_t>(c.n_layer) * c.n_expert, 0);
        for (size_t i = 0; i < hd[3]; ++i) {
            const int il = le[2 * i], e = le[2 * i + 1];
            if (il >= c.n_layer || e >= c.n_expert || seen[static_cast<size_t>(il) * c.n_expert + e]) continue;
            seen[static_cast<size_t>(il) * c.n_expert + e] = 1;
            r.push_back({il, e});
        }
        return r;
    }

    // the experts now in the VRAM cache (model layers), most recently used first; written to a temporary file and
    // renamed, so a crash never leaves half a file
    void save_cache(const std::string & path) const {
        std::vector<std::pair<uint64_t, uint32_t>> held;   // (last use, layer << 16 | expert)
        for (int il = 0; il < c.n_layer; ++il)
            for (int e = 0; e < c.n_expert; ++e) {
                const size_t k = static_cast<size_t>(il) * c.n_expert + e;
                if (exp_dev[k]) held.push_back({last_use.empty() ? 0 : last_use[k], static_cast<uint32_t>(il) << 16 | e});
            }
        std::stable_sort(held.begin(), held.end(), [](auto a, auto b) { return a.first > b.first; });
        const std::string tmp = path + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary);
            if (!f) throw std::runtime_error("cannot write " + tmp);
            const uint32_t hd[4] = {1, static_cast<uint32_t>(c.n_layer), static_cast<uint32_t>(c.n_expert),
                                    static_cast<uint32_t>(held.size())};
            f.write("BLEC", 4);
            f.write(reinterpret_cast<const char *>(hd), sizeof hd);
            for (auto [u, le] : held) {
                const uint16_t v[2] = {static_cast<uint16_t>(le >> 16), static_cast<uint16_t>(le & 0xFFFF)};
                f.write(reinterpret_cast<const char *>(v), sizeof v);
            }
            if (!f) throw std::runtime_error("cannot write " + tmp);
        }
        if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot rename " + tmp + " to " + path);
    }

    std::vector<std::pair<int, int>> expert_ranking() {
        // the profile ranks (layer, expert) pairs by use; pairs it lacks follow, interleaved over the layers
        std::vector<std::pair<int, int>> ranked;
        std::vector<char> seen(static_cast<size_t>(c.n_layer) * c.n_expert, 0);
        restored = read_ranking(cache_file);
        if (!restored.empty()) std::fprintf(stderr, "expert cache: %zu experts restored from %s\n", restored.size(), cache_file.c_str());
        // then data/expert-rank.bin: the experts by how often they were routed to (tools/make_rank.py)
        const char * rk = std::getenv("BL_EXPERT_RANK");
        const std::string rank_path = rk ? rk : data_path("expert-rank.bin");
        const auto shipped = read_ranking(rank_path);
        if (shipped.empty()) std::fprintf(stderr, "expert ranking %s not found: the VRAM cache starts unranked\n", rank_path.c_str());
        for (const auto & list : {restored, shipped})
            for (auto [il, e] : list) {
                const size_t k = static_cast<size_t>(il) * c.n_expert + e;
                if (!seen[k]) { seen[k] = 1; ranked.push_back({il, e}); }
            }
        for (int e = 0; e < c.n_expert; ++e)
            for (int il = 0; il < c.n_layer; ++il)
                if (!seen[static_cast<size_t>(il) * c.n_expert + e]) ranked.push_back({il, e});
        return ranked;
    }

    // testing (BL_HC_Q8_SIM=1): the hyper-connection matrices as Q8_0 would give them - each BF16 matrix rounded to
    // Q8_0 (blocks of 32, scale amax / 127) and back, in place - to measure the precision before writing int8 kernels
    void sim_q8_hc() {
        size_t n_mats = 0, n_vals = 0;
        for (auto & [name, m] : mats) {
            if (name.find("hc_") == std::string::npos || m.type != 30 || m.K % 32) continue;
            const size_t n = static_cast<size_t>(m.M) * m.K;
            std::vector<uint16_t> w(n);
            ck(cudaMemcpy(w.data(), m.d, n * 2, cudaMemcpyDeviceToHost), "hc sim");
            for (size_t b = 0; b < n / 32; ++b) {
                float v[32], amax = 0.f;
                for (int j = 0; j < 32; ++j) {
                    const uint32_t u = static_cast<uint32_t>(w[32 * b + j]) << 16;
                    std::memcpy(&v[j], &u, 4);
                    amax = std::max(amax, std::fabs(v[j]));
                }
                const float d = half_to_float(float_to_half(amax / 127.f)), id = d ? 1.f / d : 0.f;
                for (int j = 0; j < 32; ++j) {
                    const float r = static_cast<float>(std::lround(v[j] * id)) * d;   // the Q8_0 value
                    uint32_t u;
                    std::memcpy(&u, &r, 4);
                    w[32 * b + j] = static_cast<uint16_t>((u + 0x7fff + ((u >> 16) & 1)) >> 16);   // to BF16, nearest even
                }
            }
            ck(cudaMemcpy(m.d, w.data(), n * 2, cudaMemcpyHostToDevice), "hc sim");
            ++n_mats;
            n_vals += n;
        }
        std::fprintf(stderr, "BL_HC_Q8_SIM: %zu hyper-connection matrices (%.1f M values) rounded through Q8_0\n", n_mats, n_vals / 1e6);
    }

    // a BF16 matrix as Q8_0 (ggml's reference rounding): half the bytes for the MTP's big projections
    void upload_mat_q8(const std::string & n, const Tensor & t) {
        const int K = static_cast<int>(t.shape[0]);
        const int64_t rows = t.elements() / K;
        if (t.type_id != 30 || K % 32) return upload_mat(n, t);
        struct Blk { uint16_t d; int8_t q[32]; };
        static_assert(sizeof(Blk) == 34, "block_q8_0");
        std::vector<Blk> out(static_cast<size_t>(rows) * K / 32);
        const uint16_t * src = reinterpret_cast<const uint16_t *>(t.data);
        for (size_t b = 0; b < out.size(); ++b) {
            float v[32], amax = 0.f;
            for (int j = 0; j < 32; ++j) {
                const uint32_t u = static_cast<uint32_t>(src[32 * b + j]) << 16;
                std::memcpy(&v[j], &u, 4);
                amax = std::max(amax, std::fabs(v[j]));
            }
            const float d = amax / 127.f, id = d ? 1.f / d : 0.f;
            for (int j = 0; j < 32; ++j) out[b].q[j] = static_cast<int8_t>(std::lround(v[j] * id));
            out[b].d = float_to_half(d);
        }
        Mat m;
        m.type = 8;   // Q8_0
        m.K = K;
        m.M = static_cast<int>(rows);
        m.row_bytes = static_cast<size_t>(K / 32) * 34;
        ck(cudaMalloc(&m.d, out.size() * 34), "cudaMalloc");
        allocs.push_back(m.d);
        ck(cudaMemcpy(m.d, out.data(), out.size() * 34, cudaMemcpyHostToDevice), "upload");
        mats[n] = m;
    }

    // the MTP's draft head over a token subset (data/draft-vocab.bin, int32 ids: tools/make_draft_vocab.py): rows of
    // output.weight gathered
    std::vector<int> dvocab;
    void load_draft_head() {
        const char * env = std::getenv("BL_DRAFT_VOCAB");
        const std::string path = env ? env : data_path("draft-vocab.bin");
        std::ifstream f(path, std::ios::binary);
        if (path == "0" || !f) { std::fprintf(stderr, "draft head: the full vocabulary\n"); return; }
        f.seekg(0, std::ios::end);
        const size_t n = static_cast<size_t>(f.tellg()) / 4;
        f.seekg(0);
        dvocab.resize(n);
        f.read(reinterpret_cast<char *>(dvocab.data()), static_cast<std::streamsize>(n * 4));
        const Mat & head = M("output.weight");
        for (size_t i = 0; i < n; ++i)
            if (dvocab[i] < 0 || dvocab[i] >= head.M) throw std::runtime_error(path + ": token id out of range");
        Mat m = head;
        m.M = static_cast<int>(n);
        ck(cudaMalloc(&m.d, n * head.row_bytes), "cudaMalloc");
        allocs.push_back(m.d);
        mats["mtp.head"] = m;
        dvocab_d = dalloc<int>(n);
        ck(cudaMemcpy(dvocab_d, dvocab.data(), n * 4, cudaMemcpyHostToDevice), "draft vocab");
        if (!fast_load) fill_draft_head();   // else once output.weight is loaded
        std::fprintf(stderr, "draft head: %zu tokens (%.0f MiB)\n", n, n * head.row_bytes / 1048576.0);
    }
    int * dvocab_d = nullptr;
    void fill_draft_head() {   // its rows gathered from output.weight on the GPU
        if (dvocab.empty()) return;
        const Mat & head = M("output.weight");
        cuda::gather_rows(head.d, head.row_bytes, dvocab_d, static_cast<int>(dvocab.size()), M("mtp.head").d, st);
        ck(cudaStreamSynchronize(st), "draft head");
    }

    // the MTP draft layer's weights under the names a main layer's would have ("blk.<n_layer>.*"), its mixer as
    // "mtp.output_hc_*". The file keeps the checkpoint's raw RMSNorm weights: Gemma-style, they scale by 1 + w.
    void load_mtp() {
        const char * env = std::getenv("BL_MTP");
        const std::string path = env ? env : beside_model("mtp-q2_0.gguf");   // tools/fetch_mtp.py + bl-mtp-pack
        if (path.empty() || path == "0" || !std::ifstream(path)) {
            std::fprintf(stderr, "no MTP draft layer (%s): speculative decoding off\n", path.c_str());
            return;
        }
        mtp_g = std::make_unique<GgufModel>(path);
        const std::string b = "blk." + std::to_string(c.n_layer) + ".";
        const std::pair<const char *, std::string> names[] = {
            {"mtp.layers.0.attn_hyper_connection.hc_norm.weight", b + "hc_attn_norm.weight"},
            {"mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight", b + "hc_attn_down.weight"},
            {"mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight", b + "hc_attn_up.weight"},
            {"mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", b + "hc_attn_inject.weight"},
            {"mtp.layers.0.mlp_hyper_connection.hc_norm.weight", b + "hc_ffn_norm.weight"},
            {"mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight", b + "hc_ffn_down.weight"},
            {"mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight", b + "hc_ffn_up.weight"},
            {"mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight", b + "hc_ffn_inject.weight"},
            {"mtp.layers.0.self_attn.q_proj.weight", b + "attn_q.weight"},
            {"mtp.layers.0.self_attn.k_proj.weight", b + "attn_k.weight"},
            {"mtp.layers.0.self_attn.v_proj.weight", b + "attn_v.weight"},
            {"mtp.layers.0.self_attn.o_proj.weight", b + "attn_output.weight"},
            {"mtp.layers.0.self_attn.q_norm.weight", b + "attn_q_norm.weight"},
            {"mtp.layers.0.self_attn.k_norm.weight", b + "attn_k_norm.weight"},
            {"mtp.layers.0.mlp.gate.weight", b + "ffn_gate_inp.weight"},
            {"mtp.layers.0.mlp.shared_expert.gate_proj.weight", b + "ffn_gate_shexp.weight"},
            {"mtp.layers.0.mlp.shared_expert.up_proj.weight", b + "ffn_up_shexp.weight"},
            {"mtp.layers.0.mlp.shared_expert.down_proj.weight", b + "ffn_down_shexp.weight"},
            {"mtp.layers.0.mlp.shared_expert_gate.weight", b + "ffn_gate_inp_shexp.weight"},
            {"mtp.hyper_connection_mixer.hc_norm.weight", "mtp.output_hc_norm.weight"},
            {"mtp.hyper_connection_mixer.input_mix_weight_down.weight", "mtp.output_hc_down.weight"},
            {"mtp.hyper_connection_mixer.input_mix_weight_up.weight", "mtp.output_hc_up.weight"},
            {"mtp.fc_embedding.weight", "mtp.fc_embedding.weight"},
            {"mtp.fc_hidden.weight", "mtp.fc_hidden.weight"},
            {"mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_embedding.weight"},
            {"mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_hidden.weight"},
        };
        for (const auto & [src, dst] : names) {
            const Tensor & t = mtp_g->at(src);
            if (!t.data) throw std::runtime_error(std::string(src) + ": bytes not on disk");
            const bool is_vec = t.type_id == 0 || dst.find("ffn_gate_inp_shexp") != std::string::npos;
            if (is_vec) upload_vec(dst, t, t.type_id == 0 ? 1.f : 0.f);   // the F32 tensors are the RMSNorm weights
            else if (dst.find("hc_") != std::string::npos || dst.find("ffn_gate_inp") != std::string::npos) upload_mat(dst, t);
            else upload_mat_q8(dst, t);   // the projections
        }
        mtp = true;
        load_draft_head();
    }

    // the MTP layer's 512 experts (Q2_0, gate_up fused gate-first) all in VRAM, as [gate | up | down]
    uint8_t * mtp_arena = nullptr;
    void load_mtp_experts() {
        const int NE = c.n_expert;
        const Tensor & gu = mtp_g->at("mtp.layers.0.mlp.experts.gate_up_proj");
        const Tensor & dn = mtp_g->at("mtp.layers.0.mlp.experts.down_proj");
        if (gu.shape.size() != 3 || gu.shape[2] != NE || gu.shape[1] != 2 * c.n_ff_exp)
            throw std::runtime_error("MTP experts: unexpected gate_up shape");
        LayerExperts & L_ = lx[c.n_layer];
        L_ = {gu.type_id, dn.type_id, gu.nbytes / NE / 2, dn.nbytes / NE};
        mtp_arena = dalloc<uint8_t>(L_.bytes() * NE);
        for (int e = 0; e < NE; ++e) {
            uint8_t * dst = mtp_arena + e * L_.bytes();
            ck(cudaMemcpy(dst, gu.data + e * 2 * L_.gu_bytes, 2 * L_.gu_bytes, cudaMemcpyHostToDevice), "MTP experts");
            ck(cudaMemcpy(dst + 2 * L_.gu_bytes, dn.data + e * L_.d_bytes, L_.d_bytes, cudaMemcpyHostToDevice), "MTP experts");
        }
    }

    // ---- fast load: shard 1 in one sequential pass
    size_t host_map_bytes = 0;                        // host_store is an mmap (fast load), pinned per layer
    std::vector<std::pair<void *, size_t>> host_regs; // its pinned ranges

    // all the experts in RAM (the LRU's mode) and the main shard readable: then the dense matrices are only allocated
    // first and everything comes in one pass (stream_weights)
    bool decide_fast_load() {
        if (const char * e = std::getenv("BL_FAST_LOAD"); e && std::atoi(e) == 0) return false;
        if (const char * ra = std::getenv("BL_RAM_ALL"); ra && std::atoi(ra) == 0) return false;
        size_t total = 0;
        for (const auto & t : g.tensors())
            if (t.name.find("_exps.") != std::string::npos) {
                if (t.shard != 0) return false;
                total += t.nbytes;
            }
        const double ram = static_cast<double>(sysconf(_SC_PHYS_PAGES)) * sysconf(_SC_PAGE_SIZE);
        return ram >= total + 12e9;
    }

    // One sequential read of shard 1 (O_DIRECT when the filesystem allows: the page cache cannot hold it next to the
    // pinned experts anyway) in big chunks into a ring of pinned buffers. Workers scatter each chunk: dense matrix bytes
    // straight to the GPU (async from the pinned chunk), expert bytes into their [gate | up | down] RAM slots. When a
    // layer's experts are all in, a third thread pins that layer's region and copies its VRAM-cached experts up.
    void stream_weights(const std::vector<size_t> & region_off) {
        using clock = std::chrono::steady_clock;
        const auto t0_ = clock::now();
        const int NE = c.n_expert;
        struct Dst { uint64_t a, n; void * dev; int il, part, stage; };   // part < 0: a dense matrix (dev, on stage's GPU)
        std::vector<Dst> dst;
        for (const Pending & pd : pending_dense) dst.push_back({g.file_offset(*pd.t), pd.t->nbytes, pd.d, -1, -1, pd.stage});
        std::vector<uint64_t> layer_bytes(c.n_layer, 0);
        for (int il = 0; il < c.n_layer; ++il) {
            const std::string pfx = "blk." + std::to_string(il) + ".";
            const char * parts[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
            for (int part = 0; part < 3; ++part) {
                const Tensor & t = need(pfx + parts[part]);
                dst.push_back({g.file_offset(t), t.nbytes, nullptr, il, part, layer_stage[il]});
                layer_bytes[il] += t.nbytes;
            }
        }
        std::sort(dst.begin(), dst.end(), [](const Dst & x, const Dst & y) { return x.a < y.a; });
        const uint64_t kChunk = 32ull << 20, kAlign = 4096;
        const uint64_t A0 = dst.front().a / kAlign * kAlign;
        uint64_t A1 = 0;
        for (const Dst & d : dst) A1 = std::max(A1, d.a + d.n);
        const int n_chunks = static_cast<int>((A1 - A0 + kChunk - 1) / kChunk);

        const std::string & path = g.shard_path(0);
        int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        const bool direct = fd >= 0;
        if (!direct) fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

        // per stage (GPU): an upload stream, a fill stream, and per buffer an event; the threads below set their own
        // CUDA device (it is per thread) before a copy to a stage's GPU
        constexpr int kBufs = 8;
        const int NS = last() + 1;
        std::vector<int> devs(NS);
        uint8_t * buf[kBufs];
        std::vector<std::vector<cudaEvent_t>> buf_ev(kBufs, std::vector<cudaEvent_t>(NS));
        std::vector<cudaStream_t> upl(NS), fill(NS);
        for (int s = 0; s < NS; ++s) {
            bind(s);
            devs[s] = dev;
            ck(cudaStreamCreateWithFlags(&upl[s], cudaStreamNonBlocking), "stream");
            ck(cudaStreamCreateWithFlags(&fill[s], cudaStreamNonBlocking), "stream");
            for (int b = 0; b < kBufs; ++b) ck(cudaEventCreateWithFlags(&buf_ev[b][s], cudaEventDisableTiming), "event");
        }
        const bool several = multi();
        auto to_dev = [&](int s) { if (several) ck(cudaSetDevice(devs[s]), "cudaSetDevice"); };
        for (int b = 0; b < kBufs; ++b)
            ck(cudaHostAlloc(reinterpret_cast<void **>(&buf[b]), kChunk, cudaHostAllocPortable), "load buffer");

        std::mutex mu;
        std::condition_variable cv;
        std::vector<int> buf_chunk(kBufs, -1);     // the chunk a buffer holds; -1: free
        std::vector<unsigned> buf_copies(kBufs, 0);   // the stages it fed GPU copies: wait for their events before reusing it
        std::deque<int> ready;                     // filled buffers
        std::deque<int> layers_done;
        std::vector<uint64_t> left = layer_bytes;  // expert bytes of a layer still to land
        int chunks_done = 0;
        bool failed = false, pins_over = false;
        double t_wait = 0, t_read = 0, t_scatter = 0, t_pin = 0;   // seconds: the reader waiting / reading, workers, pinner
        std::exception_ptr err;
        auto fail = [&](std::exception_ptr e) {
            std::lock_guard<std::mutex> lk(mu);
            if (!err) err = e;
            failed = true;
            cv.notify_all();
        };

        std::thread reader([&] {
            try {
                for (int i = 0; i < n_chunks; ++i) {
                    const int b = i % kBufs;
                    const auto tw = clock::now();
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&] { return failed || buf_chunk[b] < 0; });
                        if (failed) return;
                    }
                    for (int s = 0; s < NS; ++s)
                        if (buf_copies[b] >> s & 1) ck(cudaEventSynchronize(buf_ev[b][s]), "load buffer");
                    buf_copies[b] = 0;
                    const auto tr = clock::now();
                    t_wait += std::chrono::duration<double>(tr - tw).count();
                    const uint64_t c0 = A0 + static_cast<uint64_t>(i) * kChunk;
                    const uint64_t want = std::min(kChunk, (A1 - c0 + kAlign - 1) / kAlign * kAlign);
                    uint64_t got = 0;
                    while (got < want) {
                        const ssize_t r = ::pread(fd, buf[b] + got, want - got, static_cast<off_t>(c0 + got));
                        if (r < 0) throw std::runtime_error("read failed: " + path);
                        if (r == 0) break;   // the end of the file
                        got += static_cast<uint64_t>(r);
                    }
                    if (c0 + got < std::min(A1, c0 + kChunk)) throw std::runtime_error("short read: " + path);
                    t_read += std::chrono::duration<double>(clock::now() - tr).count();
                    std::lock_guard<std::mutex> lk(mu);
                    buf_chunk[b] = i;
                    ready.push_back(b);
                    cv.notify_all();
                }
            } catch (...) { fail(std::current_exception()); }
        });

        auto worker = [&] {
            try {
                for (;;) {
                    int b;
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&] { return failed || !ready.empty() || chunks_done == n_chunks; });
                        if (failed || (ready.empty() && chunks_done == n_chunks)) return;
                        b = ready.front();
                        ready.pop_front();
                    }
                    const int i = buf_chunk[b];
                    const auto ts_ = clock::now();
                    const uint64_t c0 = A0 + static_cast<uint64_t>(i) * kChunk, c1 = std::min(A1, c0 + kChunk);
                    unsigned copies = 0;
                    std::vector<std::pair<int, uint64_t>> landed;   // (layer, bytes)
                    // the destinations overlapping [c0, c1): dst is sorted by start and the ranges do not overlap
                    auto it = std::upper_bound(dst.begin(), dst.end(), c0, [](uint64_t v, const Dst & d) { return v < d.a + d.n; });
                    for (; it != dst.end() && it->a < c1; ++it) {
                        const uint64_t x0 = std::max(c0, it->a), x1 = std::min(c1, it->a + it->n);
                        if (x0 >= x1) continue;
                        const uint8_t * src = buf[b] + (x0 - c0);
                        if (it->part < 0) {
                            to_dev(it->stage);
                            ck(cudaMemcpyAsync(static_cast<uint8_t *>(it->dev) + (x0 - it->a), src, x1 - x0, cudaMemcpyHostToDevice,
                                               upl[it->stage]), "upload");
                            copies |= 1u << it->stage;
                            continue;
                        }
                        const LayerExperts & L_ = lx[it->il];
                        const size_t piece = it->part < 2 ? L_.gu_bytes : L_.d_bytes, at = static_cast<size_t>(it->part) * L_.gu_bytes;
                        for (uint64_t u = x0 - it->a; u < x1 - it->a;) {
                            const uint64_t e = u / piece, w = u % piece, len = std::min<uint64_t>(piece - w, x1 - it->a - u);
                            std::memcpy(exp_host[static_cast<size_t>(it->il) * NE + e] + at + w, src + (it->a + u - x0), len);
                            u += len;
                        }
                        landed.push_back({it->il, x1 - x0});
                    }
                    for (int s = 0; s < NS; ++s)
                        if (copies >> s & 1) { to_dev(s); ck(cudaEventRecord(buf_ev[b][s], upl[s]), "record"); }
                    std::lock_guard<std::mutex> lk(mu);
                    t_scatter += std::chrono::duration<double>(clock::now() - ts_).count();
                    buf_copies[b] = copies;
                    buf_chunk[b] = -1;
                    ++chunks_done;
                    for (auto [il, n] : landed)
                        if ((left[il] -= n) == 0) layers_done.push_back(il);
                    cv.notify_all();
                }
            } catch (...) { fail(std::current_exception()); }
        };
        std::vector<std::thread> workers;
        for (int w = 0; w < 4; ++w) workers.emplace_back(worker);

        std::thread pinner([&] {   // pin each finished layer, then copy its cached experts to VRAM
            try {
                for (int n = 0; n < c.n_layer; ++n) {
                    int il;
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&] { return failed || !layers_done.empty(); });
                        if (failed) return;
                        il = layers_done.front();
                        layers_done.pop_front();
                    }
                    void * r = host_store + region_off[il];
                    const size_t len = region_off[il + 1] - region_off[il];
                    const auto tp = clock::now();
                    ck(cudaHostRegister(r, len, cudaHostRegisterPortable), "pin experts");
                    t_pin += std::chrono::duration<double>(clock::now() - tp).count();
                    {
                        std::lock_guard<std::mutex> lk(mu);
                        host_regs.push_back({r, len});
                    }
                    const int s = layer_stage[il];
                    to_dev(s);
                    for (int e = 0; e < NE; ++e) {
                        const size_t k = static_cast<size_t>(il) * NE + e;
                        if (exp_dev[k]) ck(cudaMemcpyAsync(exp_dev[k], exp_host[k], lx[il].bytes(), cudaMemcpyHostToDevice, fill[s]), "arena fill");
                    }
                }
            } catch (...) { fail(std::current_exception()); }
            std::lock_guard<std::mutex> lk(mu);
            pins_over = true;
        });

        reader.join();
        for (auto & w : workers) w.join();
        {   // workers are done: wake the pinner if it waits for a layer that will never come (a failure)
            std::lock_guard<std::mutex> lk(mu);
            if (chunks_done != n_chunks) failed = true;
            cv.notify_all();
        }
        pinner.join();
        ::close(fd);
        for (int s = 0; s < NS; ++s) {
            bind(s);
            cudaStreamSynchronize(upl[s]);
            cudaStreamSynchronize(fill[s]);
            for (int b = 0; b < kBufs; ++b) cudaEventDestroy(buf_ev[b][s]);
            cudaStreamDestroy(upl[s]);
            cudaStreamDestroy(fill[s]);
        }
        for (int b = 0; b < kBufs; ++b) cudaFreeHost(buf[b]);
        if (err) std::rethrow_exception(err);
        if (failed) throw std::runtime_error("loading the weights failed");
        pending_dense.clear();
        bind(last());
        fill_draft_head();
        const double secs = std::chrono::duration<double>(clock::now() - t0_).count();
        std::fprintf(stderr, "weights: %.1f GB read in %.1f s (%.2f GB/s%s); reader: %.1f s reading, %.1f s waiting for buffers; "
                     "scatter %.1f s (4 workers); pinning %.1f s\n", (A1 - A0) / 1e9, secs, (A1 - A0) / 1e9 / secs,
                     direct ? ", direct I/O" : "", t_read, t_wait, t_scatter, t_pin);
    }

    void load_experts() {
        const auto t0_ = std::chrono::steady_clock::now();
        const int D = c.n_embd, F = c.n_ff_exp, NE = c.n_expert;
        lx.resize(c.n_layer + 1);
        size_t max_bytes = 0, total = 0;
        for (int il = 0; il < c.n_layer; ++il) {
            const std::string p = "blk." + std::to_string(il) + ".";
            const Tensor & gt = need(p + "ffn_gate_exps.weight");
            const Tensor & ut = need(p + "ffn_up_exps.weight");
            const Tensor & dt = need(p + "ffn_down_exps.weight");
            if (gt.type_id != ut.type_id) throw std::runtime_error(p + ": gate and up formats differ");
            for (const Tensor * t : {&gt, &dt}) {   // every path that computes experts must handle the format, checked now
                const char * why = !cuda::expert_supported(t->type_id) ? "the GPU's expert kernels"
                                 : !CpuExperts::supports(t->type_id) ? "the CPU's (ggml-cpu)" : nullptr;
                if (why) throw std::runtime_error(t->name + ": expert format " + CpuExperts::type_name(t->type_id) + " is not handled by " +
                                                  why + " - this model file is not supported (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0 are)");
            }
            lx[il] = {gt.type_id, dt.type_id, gt.nbytes / NE, dt.nbytes / NE};
            max_bytes = std::max(max_bytes, lx[il].bytes());
            total += lx[il].bytes() * NE;
        }
        if (const char * e = std::getenv("BL_PREFETCH")) pf_max = std::min(std::atoi(e), cuda::kMaxPf);
        if (const char * e = std::getenv("BL_ROUTE_TRACE")) route_trace = std::fopen(e, "ab");
        for (int s = 0; s <= last(); ++s) {   // every stage: its copy slots and expert buffers
            bind(s);
            for (auto & sl : staging) ck(cudaMalloc(&sl, max_bytes), "staging"), allocs.push_back(sl);
            staging_d = dalloc<uint8_t *>(cuda::kMaxSlots);
            ck(cudaMemcpy(staging_d, staging, sizeof staging, cudaMemcpyHostToDevice), "staging table");
            for (auto & par : pf_slot)
                for (auto & sl : par) ck(cudaMalloc(&sl, max_bytes), "prefetch slot"), allocs.push_back(sl);
            pf_slots_d = dalloc<uint8_t *>(2 * cuda::kMaxPf);
            ck(cudaMemcpy(pf_slots_d, pf_slot, sizeof pf_slot, cudaMemcpyHostToDevice), "prefetch slots");
            pred_ids_d = dalloc<int>(cuda::kMaxK);
            pred_w_d = dalloc<float>(cuda::kMaxK);
            exp_act = dalloc<float>(static_cast<size_t>(cuda::kMaxSlots) * kT * F);
            exp_out = dalloc<float>(static_cast<size_t>(cuda::kMaxSlots) * kT * D);
            xq_d = dalloc<uint8_t>(cuda::q8_bytes(D) * kT);
        }
        if (const char * sh_ = std::getenv("BL_CPU_SHARE")) cpu_share = cpu_share_win = static_cast<float>(std::atof(sh_));
        if (const char * sh_ = std::getenv("BL_CPU_SHARE_WIN")) cpu_share_win = static_cast<float>(std::atof(sh_));
        bind(last());
        if (mtp) load_mtp_experts();

        // the VRAM budget per GPU (stages that share a GPU share its budget)
        const char * rs = std::getenv("BL_VRAM_RESERVE_MB");
        // what is allocated after the experts (cuBLAS workspace, the graphs) is only ~0.35 GB, but a smaller reserve made
        // decoding slower: 600 / 800 MB left 0.28 / 0.49 GB free and the corpus ran at ~92 tok/s against ~96 with 1024
        // (the GPU waited longer for the CPU experts; cause not found, 2026-10-05)
        const size_t reserve = static_cast<size_t>(rs ? std::atoll(rs) : 1024) << 20;
        std::unordered_map<int, size_t> budget;   // by device
        for (int s = 0; s <= last(); ++s) {
            const int d = stage(s).dev;
            if (budget.count(d)) continue;
            ck(cudaSetDevice(d), "cudaSetDevice");
            size_t free_b = 0, total_b = 0;
            ck(cudaMemGetInfo(&free_b, &total_b), "meminfo");
            size_t b = free_b > reserve ? free_b - reserve : 0;
            if (const char * cap = std::getenv("BL_EXPERT_CACHE_MB")) b = std::min(b, static_cast<size_t>(std::atoll(cap)) << 20);
            budget[d] = b;
            size_t dense = 0;
            for (const auto & [name, m] : mats) {
                const auto it = owner_of.find(name);
                if (!multi() || (it != owner_of.end() && it->second >= 0 && stage(it->second).dev == d))
                    dense += m.row_bytes * static_cast<size_t>(m.M);
            }
            if (!multi()) dense -= mapped_bytes;   // in host memory
            size_t slots = 0;
            for (int s2 = 0; s2 <= last(); ++s2) if (stage(s2).dev == d) slots += (cuda::kMaxSlots + 2 * cuda::kMaxPf) * max_bytes;
            std::fprintf(stderr, "VRAM%s: %.2f GB total, %.2f GB in use before the experts (dense matrices %.2f GB, %s"
                         "expert staging %.2f GB, the rest - CUDA context, buffers, MTP, cuBLAS - %.2f GB), reserve %.2f GB\n",
                         multi() ? (" of GPU " + std::to_string(d)).c_str() : "", total_b / 1e9, (total_b - free_b) / 1e9, dense / 1e9,
                         multi() ? "" : ("buffers " + std::to_string(dalloc_bytes / 1e9).substr(0, 4) + " GB, ").c_str(), slots / 1e9,
                         (total_b - free_b - dense - (multi() ? 0 : dalloc_bytes) - slots) / 1e9, reserve / 1e9);
        }
        ck(cudaSetDevice(dev), "cudaSetDevice");

        exp_dev.assign(static_cast<size_t>(c.n_layer + 1) * NE, nullptr);
        exp_host.assign(static_cast<size_t>(c.n_layer + 1) * NE, nullptr);
        if (mtp)
            for (int e = 0; e < NE; ++e)
                exp_dev[static_cast<size_t>(c.n_layer) * NE + e] = mtp_arena + e * lx[c.n_layer].bytes();
        std::vector<char> in_vram(exp_dev.size(), 0);
        std::unordered_map<int, size_t> used_dev;
        std::vector<size_t> used_st(last() + 1, 0);
        size_t used = 0;
        for (auto [il, e] : ranking()) {   // greedy by rank; a bigger expert that no longer fits leaves room
            const size_t b = lx[il].bytes();
            const int s = layer_stage[il], d = stage(s).dev;
            if (used_dev[d] + b <= budget[d]) { in_vram[static_cast<size_t>(il) * NE + e] = 1; used_dev[d] += b; used_st[s] += b; used += b; }
        }
        for (int s = 0; s <= last(); ++s) {   // every stage: its arena, and its prefill region
            bind(s);
            arena_bytes = used_st[s];
            if (arena_bytes) ck(cudaMalloc(&arena, arena_bytes), "expert arena"), allocs.push_back(arena);
            if (pf_on) {   // the prefill region: the arena's tail when it is big enough, else memory of its own
                if (arena_bytes >= 2 * loan_bytes) loan_base = arena + ((arena_bytes - loan_bytes) & ~size_t(255));
                else { ck(cudaMalloc(&loan_base, loan_bytes), "prefill region"); allocs.push_back(loan_base); }
                layout_prefill(loan_base);
            }
        }
        {   // every expert in pinned RAM too when it fits beside ~12 GB for the rest (the LRU evicts without copying back)
            const long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGE_SIZE);
            const double ram = static_cast<double>(pages) * psz;
            const char * ra = std::getenv("BL_RAM_ALL");
            ram_all = ra ? std::atoi(ra) != 0 : ram >= total + 12e9;
        }
        if (!ram_all) lru = false;
        if (const char * e = std::getenv("BL_LRU")) lru = lru && std::atoi(e) != 0;
        if (multi() && !lru) throw std::runtime_error("a layer split needs the LRU expert cache (every expert in RAM)");
        if (!ram_all) fast_load = false;   // (decided before the dense weights; this only happens if RAM shrank)

        // where every expert goes: its VRAM slot (rank order) and its pinned RAM slot ([gate | up | down] contiguous).
        // The fast path aligns each layer's RAM region to 2 MiB: it is pinned layer by layer as it fills.
        constexpr size_t kRegion = 2u << 20;
        std::vector<size_t> host_off(exp_dev.size(), SIZE_MAX);
        std::vector<size_t> region_off(c.n_layer + 1, 0);
        size_t a_off = 0, h_off = 0;
        bind(0);
        for (int il = 0; il < c.n_layer; ++il) {
            if (il == l1) { bind(layer_stage[il]); a_off = 0; }   // the next stage's arena
            if (fast_load) h_off = (h_off + kRegion - 1) / kRegion * kRegion;
            region_off[il] = h_off;
            const LayerExperts & L_ = lx[il];
            for (int e = 0; e < NE; ++e) {
                const size_t k = static_cast<size_t>(il) * NE + e;
                if (in_vram[k]) {
                    exp_dev[k] = arena + a_off;
                    if (loan_base && arena + a_off + L_.bytes() > loan_base && loan_base >= arena) loan_slots.push_back({il, arena + a_off});
                    a_off += L_.bytes();
                    ++cached;
                }
                if (!in_vram[k] || ram_all) { host_off[k] = h_off; h_off += L_.bytes(); }
            }
        }
        if (fast_load) h_off = (h_off + kRegion - 1) / kRegion * kRegion;
        region_off[c.n_layer] = h_off;
        host_bytes = h_off;
        lap("expert placement");
        if (host_bytes && fast_load) {   // ordinary memory (huge pages when the kernel gives them), pinned per layer later
            void * p = ::mmap(nullptr, host_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) throw std::runtime_error("mmap of the expert store failed");
            ::madvise(p, host_bytes, MADV_HUGEPAGE);
            host_store = static_cast<uint8_t *>(p);
            host_map_bytes = host_bytes;
        } else if (host_bytes) {
            ck(cudaHostAlloc(reinterpret_cast<void **>(&host_store), host_bytes, cudaHostAllocPortable), "pinned experts");
            host_allocs.push_back(host_store);
        }
        for (size_t k = 0; k < exp_host.size(); ++k)
            if (host_off[k] != SIZE_MAX) exp_host[k] = host_store + host_off[k];
        lap("expert store allocation");

        if (fast_load) {
            stream_weights(region_off);
            lap("weights stream (dense + experts + VRAM fill)");
        } else {   // the slow path: from the mapped file, expert by expert
            std::vector<uint8_t> bounce(max_bytes);
            auto prefetch = [&](int il) {   // ask the kernel to read a layer's experts ahead (mmap faults alone are slow)
                if (il >= c.n_layer) return;
                const std::string p = "blk." + std::to_string(il) + ".";
                for (const char * part : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) g.advise_willneed(g.at(p + part));
            };
            prefetch(0);
            for (int il = 0; il < c.n_layer; ++il) {
                prefetch(il + 1);
                const std::string p = "blk." + std::to_string(il) + ".";
                const uint8_t * gp = need(p + "ffn_gate_exps.weight").data, * up = need(p + "ffn_up_exps.weight").data,
                              * dp = need(p + "ffn_down_exps.weight").data;
                const LayerExperts & L_ = lx[il];
                for (int e = 0; e < NE; ++e) {
                    const size_t k = static_cast<size_t>(il) * NE + e;
                    uint8_t * dst = exp_host[k] ? exp_host[k] : bounce.data();
                    std::memcpy(dst, gp + e * L_.gu_bytes, L_.gu_bytes);
                    std::memcpy(dst + L_.gu_bytes, up + e * L_.gu_bytes, L_.gu_bytes);
                    std::memcpy(dst + 2 * L_.gu_bytes, dp + e * L_.d_bytes, L_.d_bytes);
                    if (exp_dev[k]) {
                        bind(layer_stage[il]);
                        ck(cudaMemcpy(exp_dev[k], dst, L_.bytes(), cudaMemcpyHostToDevice), "arena fill");
                    }
                }
            }
            lap("expert reads + VRAM fill");
        }
        cached_bytes = used;
        const char * ct = std::getenv("BL_CPU_THREADS");
        cpu = std::make_unique<CpuExperts>(ct ? std::atoi(ct) : 8);
        use.assign(exp_dev.size(), 0.f);
        last_use.assign(exp_dev.size(), 0);
        // the restored order goes on as the LRU's: the most recently used before the restart is the newest now
        for (size_t i = 0; i < restored.size(); ++i)
            last_use[static_cast<size_t>(restored[i].first) * NE + restored[i].second] = restored.size() - i;
        use_clock = restored.size();
        max_expert_bytes = max_bytes;
        if (const char * e = std::getenv("BL_LRU_MAX")) lru_max = std::atoi(e);
        if (const char * e = std::getenv("BL_ADAPT_EVERY")) adapt_every = std::atoi(e);
        if (const char * e = std::getenv("BL_ADAPT_MAX")) adapt_max = std::atoi(e);
        if (lru) adapt_every = 0, pf_max = 0;   // the LRU replaces the periodic swaps; prefetch then only costs PCIe
        ck(cudaHostAlloc(reinterpret_cast<void **>(&swap_bounce), static_cast<size_t>(adapt_max) * max_bytes,
                         cudaHostAllocPortable), "swap bounce");
        host_allocs.push_back(swap_bounce);
        for (int s = 0; s <= last(); ++s) {   // every stage's table (it reads only its own layers' entries)
            bind(s);
            table_d = dalloc<uint8_t *>(exp_dev.size());
            ck(cudaMemcpy(table_d, exp_dev.data(), exp_dev.size() * sizeof(uint8_t *), cudaMemcpyHostToDevice), "expert table");
        }
        bind(last());
        std::fprintf(stderr, "experts: %zu of %zu in VRAM (%.2f GB), %.2f GB in pinned RAM (%s); cache: %s; placed in %.1f s\n",
                     cached, exp_dev.size(), used / 1e9, host_bytes / 1e9, ram_all ? "all" : "the rest",
                     lru ? "LRU per layer" : "periodic swaps",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count());
    }

    // ------------------------------------------------------------------------------------------------- forward
    // hyper-connection mix of T tokens; inject (when wanted) into `inj`
    void hc_mix(const std::string & pre, bool with_inject, int il, int T) {
        const int D = c.n_embd;
        cuda::rmsnorm_rows(R, xn, T * c.hc, D, V(pre + "_norm.weight"), c.hc, static_cast<float>(c.rms_eps), 1.f, st);
        cuda::MvGroup gr;
        add(gr, pre + "_down.weight", lo);
        if (with_inject) add(gr, pre + "_inject.weight", inj);
        cuda::matmul_group(gr, xn, T, st);
        cuda::hc_up_mix_t(M(pre + "_up.weight").d, lo, c.hc_low_rank, xn, mixed, D, c.hc, T, st);
        emit(il >= 0 ? L("hc_mixed", il) : "result_norm", mixed, D);
    }

    // the PLE rows of the window's tokens (host: they depend only on token ids) into pinned memory. The 28.8 GB table
    // does not stay in the page cache next to the pinned experts, so the rows are SSD reads (~0.45 ms each): all of
    // the window's pages are requested first (madvise WILLNEED starts them concurrently), then copied.
    void ple_rows(const int * toks, int T) { ple_rows(toks, T, ple_rows_h); }
    void ple_rows(const int * toks, int T, uint8_t * dst) {
        const Tensor & tab = g.at("per_layer_token_embd.weight");
        if (!tab.data) throw std::runtime_error("PLE table (shard 2) not on disk yet");
        std::vector<uint64_t> rows(static_cast<size_t>(T) * c.ple_heads());
        std::vector<int> hist = history;   // committed tokens, then the window's earlier ones
        for (int t = 0; t < T; ++t) {
            std::vector<int64_t> ctx(c.ple_ngram);
            ctx[0] = toks[t];
            bool cut = false;
            const int hn = static_cast<int>(hist.size());
            for (int s = 1; s < c.ple_ngram; ++s) {
                const int64_t tk = cut ? -1 : (hn - s >= 0 ? hist[hn - s] : -1);
                cut = cut || tk < 0 || tk == c.ple_eos;
                ctx[s] = cut ? c.ple_eos : tk;
            }
            for (int n = 2; n <= c.ple_ngram; ++n) {
                uint64_t mx = static_cast<uint64_t>(ctx[0]) * c.ple_multipliers[0];
                for (int j = 1; j < n; ++j) mx ^= static_cast<uint64_t>(ctx[j]) * c.ple_multipliers[j];
                for (int gi = 0; gi < c.ple_heads_per_ngram; ++gi) {
                    const int hh = (n - 2) * c.ple_heads_per_ngram + gi;
                    rows[static_cast<size_t>(t) * c.ple_heads() + hh] = mx % c.ple_head_vocab[hh] + c.ple_head_offsets[hh];
                }
            }
            hist.push_back(toks[t]);
        }
        const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
        for (uint64_t r : rows) {
            const auto a = reinterpret_cast<uintptr_t>(tab.data + r * ple_row_bytes);
            const uintptr_t lo = a & ~(page - 1), hi = (a + ple_row_bytes + page - 1) & ~(page - 1);
            ::madvise(reinterpret_cast<void *>(lo), hi - lo, MADV_WILLNEED);
        }
        for (size_t i = 0; i < rows.size(); ++i)
            std::memcpy(dst + i * ple_row_bytes, tab.data + rows[i] * ple_row_bytes, ple_row_bytes);
    }

    void ple(int T) {
        const int D = c.n_embd, HD = c.hc_dim();
        const Tensor & tab = g.at("per_layer_token_embd.weight");
        const size_t bytes = static_cast<size_t>(T) * c.ple_heads() * ple_row_bytes;
        cuda::copy_words(dev_of(ple_rows_h), ple_rows_d, bytes, st);   // not behind the admissions on the copy engine
        float * emb = t0;   // [T][heads * ple_dim] = [T][D]
        cuda::dequant_rows(tab.type_id, ple_rows_d, ple_row_bytes, T * c.ple_heads(), c.ple_dim, emb, st);
        emit("ple_embd", emb, c.ple_heads() * c.ple_dim);

        const std::string p = "blk." + std::to_string(c.ple_layer) + ".";
        const float eps = static_cast<float>(c.rms_eps);
        float *key = t1, *value = t2, *qn = t3, *gt = t4, *gated = t5;
        mm(p + "ple_key.weight", emb, key, T);
        mm(p + "ple_value.weight", emb, value, T);
        cuda::rmsnorm_rows(key, key, T * c.hc, D, V(p + "ple_norm_key.weight"), c.hc, eps, 1.f, st);
        cuda::rmsnorm_rows(R, qn, T * c.hc, D, V(p + "ple_norm_query.weight"), c.hc, eps, 1.f, st);
        cuda::ple_gate(key, qn, gt, D, T * c.hc, st);
        emit(L("ple_gate", c.ple_layer), gt, c.hc);
        for (int t = 0; t < T; ++t)
            cuda::ple_gated_value(value + static_cast<size_t>(t) * D, gt + t * c.hc, gated + static_cast<size_t>(t) * HD, D, c.hc, st);
        cuda::rmsnorm_rows(gated, w_ple_norm, T * c.hc, D, V(p + "ple_norm_conv.weight"), c.hc, eps, 1.f, st);
        float * conv_out = t3;
        cuda::ple_conv_t(ple_hist, w_ple_norm, V(p + "ple_conv1d.weight"), conv_out, HD, c.ple_conv, c.ple_ngram, T, st);
        emit(L("ple_conv_out", c.ple_layer), conv_out, HD);
        cuda::add(gated, conv_out, T * HD, st);
        cuda::add(R, gated, T * HD, st);
    }

    void gdn(int il, int T) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int S = c.ssm_state, Hk = c.ssm_groups, Hv = c.ssm_v_heads, inner = c.ssm_inner;
        const int conv_dim = 2 * Hk * S + inner;
        const float eps = static_cast<float>(c.rms_eps);
        float *z = t1, *o = t3, *braw = t4, *araw = t5;
        cuda::MvGroup gr;
        add(gr, p + "attn_qkv.weight", w_qkv[il]);
        add(gr, p + "attn_gate.weight", z);
        add(gr, p + "ssm_beta.weight", braw);
        add(gr, p + "ssm_alpha.weight", araw);
        cuda::matmul_group(gr, mixed, T, st);
        cuda::gdn_pre_t(conv_state[il], w_qkv[il], V(p + "ssm_conv1d.weight"), w_conv[il], Hk, Hv, S, eps, braw, araw, Hv,
                        V(p + "ssm_dt.bias"), V(p + "ssm_a"), w_beta[il], w_g[il], T, st);
        cuda::gdn_step_t(gdn_state[il], gdn_scratch, w_conv[il], conv_dim, w_g[il], w_beta[il], Hv, o, Hv, Hk, S, T, nullptr, st);
        emit(L("attn_output", il), o, inner);
        cuda::gdn_post_t(o, V(p + "ssm_norm.weight"), z, Hv, S, eps, T, st);
        mm(p + "ssm_out.weight", o, h, T);
        emit(L("linear_attn_out", il), h, c.n_embd);
    }

    bool qsa_layer(int il) const { return il == c.n_layer || c.is_qsa(il); }

    // which cells each query attends: the indexer's selection (main layers) or the last mtp_window (the MTP layer)
    cuda::AttnSel select(int il, int T, const cuda::TokenState * tsp, const float * iq_raw, const float * ik_raw, float * iq) {
        cuda::AttnSel sel;
        sel.max_cells = kSelCells;
        if (il >= c.n_layer) {
            const int w = std::min(mtp_window > 0 ? mtp_window : max_kv, mtp_window_max());
            sel.window = w;
            sel.max_cells = std::min(w, max_kv);
            return sel;
        }
        static const bool dense_test = std::getenv("BL_IDX_DENSE") != nullptr;   // testing (prefill only): no selection
        if (dense_test) { sel.max_cells = max_kv; return sel; }
        const std::string p = "blk." + std::to_string(il) + ".";
        const float eps = static_cast<float>(c.rms_eps), base = static_cast<float>(c.rope_freq_base);
        cuda::idx_prep(iq_raw, c.idx_heads * c.idx_dim, ik_raw, c.idx_dim, V(p + "indexer.q_norm.weight"), eps, c.rope_dims, base, tsp,
                       c.idx_heads, c.idx_dim, iq, ikraw[il], T, st);
        cuda::idx_blocks(ikraw[il], ikblk[il], V(p + "indexer.k_norm.weight"), eps, c.rope_dims, base, tsp, c.idx_dim, T, st);
        cuda::idx_select(iq, ikblk[il], tsp, c.idx_heads, c.idx_dim, c.idx_top_k / 4, sel_score, max_blocks, sel_list, kSelStride,
                         sel_cnt, T, st);
        sel.list = sel_list;
        sel.cnt = sel_cnt;
        sel.stride = kSelStride;
        return sel;
    }

    void qsa(int il, int T, cuda::TokenState * tsp) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int Dh = c.head_dim, H = c.n_head, Hkv = c.n_head_kv;
        const float eps = static_cast<float>(c.rms_eps);
        float *qf = t0, *q = t1, *k = t3, *v = t4, *att = t5;
        float *iq_raw = t2, *ik_raw = t2 + kT * 512, *iq = t2 + kT * 1024;   // the indexer's (Hq*Di = 512)
        const bool idx = il < c.n_layer;
        cuda::MvGroup gr;
        add(gr, p + "attn_q.weight", qf);
        add(gr, p + "attn_k.weight", k);
        add(gr, p + "attn_v.weight", v);
        if (idx) {
            add(gr, p + "indexer.q_proj.weight", iq_raw);
            add(gr, p + "indexer.k_proj.weight", ik_raw);
        }
        cuda::matmul_group(gr, mixed, T, st);
        cuda::qsa_pre_t(qf, q, H, k, v, Hkv, Dh, V(p + "attn_q_norm.weight"), V(p + "attn_k_norm.weight"), eps, c.rope_dims,
                        static_cast<float>(c.rope_freq_base), tsp, kv[il], T, st);
        const cuda::AttnSel sel = select(il, T, tsp, iq_raw, ik_raw, iq);
        cuda::attention_t(q, kv[il], qf + Dh, 2 * Dh, 2 * H * Dh, att, H, Hkv, Dh, tsp, sel,
                          1.f / std::sqrt(static_cast<float>(Dh)), T, st);
        mm(p + "attn_output.weight", att, h, T);
        emit(L("attn_output", il), h, c.n_embd);
    }

    bool predict_any(int T) const { return T == 1 && pf_max > 0; }   // next-layer prefetch tables get filled

    void ffn(int il, int T, bool inline_service, cuda::TokenState * tsp) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int D = c.n_embd, F = c.n_ff_exp, K = c.n_expert_used;
        const LayerExperts & L_ = lx[il];
        float *rl = t0, *gg = t1, *uu = t2, *rl_next = t3, *sg = t5;
        const bool predict = T == 1 && pf_max > 0 && il + 1 < l1;   // (the next layer on this stage's GPU)
        {   // the router, the next layer's router (a prediction), the shared expert's gate/up and its scalar gate
            cuda::MvGroup gr;
            add(gr, p + "ffn_gate_inp.weight", rl);
            if (predict) add(gr, "blk." + std::to_string(il + 1) + ".ffn_gate_inp.weight", rl_next);
            add(gr, p + "ffn_gate_shexp.weight", gg);
            add(gr, p + "ffn_up_shexp.weight", uu);
            add_vec(gr, p + "ffn_gate_inp_shexp.weight", D, sg);
            cuda::matmul_group(gr, mixed, T, st);
        }
        if (predict) cuda::router_topk2(rl, ids_d, w_d, rl_next, pred_ids_d, pred_w_d, c.n_expert, K, st);
        else cuda::router_topk_t(rl, c.n_expert, K, ids_d, w_d, T, st);
        cuda::plan_experts(il, c.n_expert, K, T, ids_d, w_d, table_d, staging_d, T > 1 ? cpu_share_win : cpu_share, plan_d + il, plan_hd + il, mixed,
                           x_hd + static_cast<size_t>(il) * kT * D, D, bell_hd + il, tsp, pf_hd + il,
                           pf_slots_d + (il % 2) * cuda::kMaxPf, predict ? pred_ids_d : nullptr, st);
        if (inline_service) service(il);
        stamp(kStRoute);
        const cuda::ExpertPlan * pl = plan_d + il;
        cuda::quantize_q8(mixed, D, xq_d, st, T);
        cuda::plan_gate_up(L_.type_gu, pl, 1 << cuda::kHit, T, L_.gu_bytes, F, D, xq_d, exp_act, st);
        cuda::plan_down(L_.type_d, pl, 1 << cuda::kHit, T, L_.gu_bytes, F, D, exp_act, exp_out, st);
        stamp(kStHit);
        {   // the shared expert's down, its SwiGLU taken on load (independent of the routed experts)
            cuda::MvGroup gr;
            add(gr, p + "ffn_down_shexp.weight", sh);
            gr.xf = cuda::Xform::SwiGLU;
            gr.x2 = uu;
            cuda::matmul_group(gr, gg, T, st);
        }
        stamp(kStShared);
        // the phases that can have work in this graph (the MTP layer's experts are all in VRAM: hits only)
        const bool main_layer = il < c.n_layer;
        const float share = T > 1 ? cpu_share_win : cpu_share;
        if (main_layer && predict_any(T)) {
            cuda::wait_flag(pf_flag_hd + il, tsp, pl, cuda::kPf, st);   // prefetched during the previous layer
            cuda::plan_gate_up(L_.type_gu, pl, 1 << cuda::kPf, T, L_.gu_bytes, F, D, xq_d, exp_act, st);
            cuda::plan_down(L_.type_d, pl, 1 << cuda::kPf, T, L_.gu_bytes, F, D, exp_act, exp_out, st);
        }
        if (main_layer && share < 1.f) {   // some misses come over PCIe
            cuda::wait_flag(dma_flag_hd + il, tsp, pl, cuda::kDma, st);
            cuda::plan_gate_up(L_.type_gu, pl, 1 << cuda::kDma, T, L_.gu_bytes, F, D, xq_d, exp_act, st);
            cuda::plan_down(L_.type_d, pl, 1 << cuda::kDma, T, L_.gu_bytes, F, D, exp_act, exp_out, st);
        }
        if (main_layer && share > 0.f) cuda::wait_flag(cpu_flag_hd + il, tsp, pl, cuda::kCpu, st);
        stamp(kStWait);
        cuda::plan_combine(pl, T, exp_out, cpu_out_hd, sh, sg, 1, h, probe ? moe : nullptr, D, st);
        if (probe) emit(L("ffn_moe_out", il), moe, D);
        emit(L("ffn_out", il), h, D);
    }

    // a window's GPU work on the current stage: its layers (the first stage from the embeddings, a later one from the
    // residual the previous one handed over), then the head (the last stage) or the hand-over. With inline_service the
    // host answers each layer's doorbell as it goes (probe path)
    void enqueue_window(int T, bool inline_service) {
        const int D = c.n_embd, HD = c.hc_dim();
        if (stage_prof && !inline_service) {
            stamp_cats = &stamp_cat[T];
            stamp_cats->clear();
            stamp(kStStart);
            stamp(kStCal);   // right after the first: what a stamp itself costs
        }
        cuda::window_begin(ts, tokens_hd, T, st);
        if (l0 == 0) {
            const Mat & te = M("token_embd.weight");
            cuda::embed_tokens(te.type, te.d, te.row_bytes, D, ts->tok, T, emb_t, st);
            emit("model.input_embed", emb_t, D);
            for (int t = 0; t < T; ++t) cuda::repeat(emb_t + static_cast<size_t>(t) * D, R + static_cast<size_t>(t) * HD, D, c.hc, st);
        } else {
            cuda::copy_words(hand_hd, R, sizeof(float) * T * HD, st);
        }
        stamp(kStEmbed);
        for (int il = l0; il < l1; ++il) {
            if (il == c.ple_layer) ple(T);
            hc_mix("blk." + std::to_string(il) + ".hc_attn", true, il, T);
            stamp(kStHc);
            if (c.is_qsa(il)) qsa(il, T, ts);
            else gdn(il, T);
            stamp(c.is_qsa(il) ? kStQsa : kStGdn);
            cuda::hc_combine_t(R, h, inj, D, c.hc, T, st);
            hc_mix("blk." + std::to_string(il) + ".hc_ffn", true, il, T);
            stamp(kStHc);
            ffn(il, T, inline_service, ts);
            cuda::hc_combine_t(R, h, inj, D, c.hc, T, st);
            stamp(kStCombine);
            emit(L("l_last", il), R, HD);
        }
        if (l1 < c.n_layer) {   // to the next stage (its window waits for this one's event)
            cuda::copy_words(R, stage(cur + 1).hand_hd, sizeof(float) * T * HD, st);
            stamp_cats = nullptr;
            return;
        }
        ck(cudaMemcpyAsync(Rwin, R, sizeof(float) * T * HD, cudaMemcpyDeviceToDevice, st), "R rows");   // the MTP's input
        hc_mix("output_hc", false, -1, T);
        mm("output.weight", mixed, logits_d, T);
        emit("result_output", logits_d, c.n_vocab);
        cuda::argmax_rows(logits_d, c.n_vocab, T, argmax_d, st);
        ck(cudaMemcpyAsync(argmax_h, argmax_d, sizeof(int) * T, cudaMemcpyDeviceToHost, st), "argmax");
        stamp(kStHead);
        stamp_cats = nullptr;
    }

    // the MTP layer on Tm rows at positions *mtp_pos_h..: row r takes the main model's final residual mtp_Rin[r]
    // (position p + r) and the token after it (mtp_tok_h[r]); its residual goes to mtp_Rout, the draft after its last
    // row (and that draft's probability) to mtp_id_h / mtp_prob_h
    // one MTP pass over Tm rows: inputs rin [Tm][HD] (copied to mtp_Rin), tokens (device-readable) and the position
    // (*pos); the draft after its last row, as a token id, into dr_tok_d[step] with its probability in dr_prob_d[step]
    void enqueue_mtp(int Tm, const float * rin, const int * toks, const int * pos, int step) {
        const int D = c.n_embd, HD = c.hc_dim(), il = c.n_layer;
        const float eps = static_cast<float>(c.rms_eps);
        ck(cudaMemcpyAsync(mtp_Rin, rin, sizeof(float) * Tm * HD, cudaMemcpyDeviceToDevice, st), "MTP input");
        cuda::window_begin(mts, toks, Tm, st);
        cuda::window_set_pos(mts, pos, st);
        const Mat & te = M("token_embd.weight");
        cuda::embed_tokens(te.type, te.d, te.row_bytes, D, mts->tok, Tm, emb_t, st);
        cuda::rmsnorm_rows(emb_t, t0, Tm, D, V("mtp.pre_fc_norm_embedding.weight"), 1, eps, 1.f, st);
        mm("mtp.fc_embedding.weight", t0, t1, Tm);
        cuda::rmsnorm_rows(mtp_Rin, xn, Tm, HD, V("mtp.pre_fc_norm_hidden.weight"), 1, eps, 1.f, st);
        for (int r = 0; r < Tm; ++r)   // fc_hidden on each of the row's hc streams
            mm("mtp.fc_hidden.weight", xn + static_cast<size_t>(r) * HD, t2 + static_cast<size_t>(r) * HD, c.hc);
        cuda::add_broadcast_streams(t2, t1, R, D, c.hc, Tm, st);
        const std::string p = "blk." + std::to_string(il);
        hc_mix(p + ".hc_attn", true, il, Tm);
        qsa(il, Tm, mts);
        cuda::hc_combine_t(R, h, inj, D, c.hc, Tm, st);
        hc_mix(p + ".hc_ffn", true, il, Tm);
        ffn(il, Tm, false, mts);
        cuda::hc_combine_t(R, h, inj, D, c.hc, Tm, st);
        ck(cudaMemcpyAsync(mtp_Rout, R, sizeof(float) * Tm * HD, cudaMemcpyDeviceToDevice, st), "MTP rows");
        hc_mix("mtp.output_hc", false, -1, Tm);
        const bool sub = !dvocab.empty();
        mm(sub ? "mtp.head" : "output.weight", mixed + static_cast<size_t>(Tm - 1) * D, mtp_logits, 1);
        cuda::argmax_rows(mtp_logits, sub ? static_cast<int>(dvocab.size()) : c.n_vocab, 1, mtp_id_d, st, dr_prob_d + step);
        if (sub) cuda::gather_rows(dvocab_d, 4, mtp_id_d, 1, dr_tok_d + step, st);   // subset index -> token id
        else ck(cudaMemcpyAsync(dr_tok_d + step, mtp_id_d, sizeof(int), cudaMemcpyDeviceToDevice, st), "draft");
    }

    // The drafts for the next round in one graph and one sync: the catch-up over Tm rows (rows Rwin, tokens toks, at
    // MTP position P) gives draft 0 - or with Tm = 0, draft 0 is dr0 (the prefill made it) and the chain starts from
    // mtp_Rout row 0; then `chain` more, each from the previous one's residual and token on the GPU.
    int   draft_tok[kT] = {};
    float draft_prob[kT] = {};
    void draft(int Tm, const int * toks, int P, int chain, int dr0 = -1, float p0 = 0.f) {
        const auto t0_ = std::chrono::steady_clock::now();
        const int HD = c.hc_dim();
        chain = std::clamp(chain, 0, kT - 1);
        if (P + Tm + chain > max_kv) chain = std::max(0, max_kv - P - Tm);   // the chain's cells must fit
        if (Tm > 0) {
            for (int r = 0; r < Tm; ++r) reinterpret_cast<volatile int *>(mtp_tok_h)[r] = toks[r];
        } else {
            reinterpret_cast<volatile int *>(mtp_tok_h)[0] = dr0;
        }
        for (int s = 0; s <= chain; ++s) reinterpret_cast<volatile int *>(mtp_pos_h)[s] = Tm > 0 ? (s == 0 ? P : P + Tm + s - 1) : P + s - 1;
        cudaGraphExec_t & gx = mtp_graph[Tm][chain];
        if (!gx) capture(gx, [&] {
            if (Tm > 0) enqueue_mtp(Tm, Rwin, mtp_tok_hd, mtp_pos_hd, 0);
            else cuda::copy_words(mtp_tok_hd, dr_tok_d, sizeof(int), st);
            for (int s = 1; s <= chain; ++s)
                enqueue_mtp(1, mtp_Rout + (s == 1 && Tm > 0 ? static_cast<size_t>(Tm - 1) * HD : 0), dr_tok_d + s - 1, mtp_pos_hd + s, s);
            ck(cudaMemcpyAsync(dr_tok_h, dr_tok_d, sizeof(int) * (chain + 1), cudaMemcpyDeviceToHost, st), "drafts");
            ck(cudaMemcpyAsync(dr_prob_h, dr_prob_d, sizeof(float) * (chain + 1), cudaMemcpyDeviceToHost, st), "drafts");
        });
        ck(cudaGraphLaunch(gx, st), "MTP launch");
        ck(cudaStreamSynchronize(st), "sync");
        for (int s = 0; s <= chain; ++s) {
            draft_tok[s] = reinterpret_cast<volatile int *>(dr_tok_h)[s];
            draft_prob[s] = reinterpret_cast<volatile float *>(dr_prob_h)[s];
        }
        if (Tm == 0) { draft_tok[0] = dr0; draft_prob[0] = p0; }
        for (int s = chain + 1; s < kT; ++s) draft_prob[s] = -1.f;   // not made
        n_drafts_ready = chain + 1;
        t_draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
        n_mtp_runs += (Tm > 0) + chain;
    }
    int n_drafts_ready = 0;
    long long n_mtp_runs = 0, n_rounds = 0, n_drafted = 0, n_accepted = 0;

    // greedy decoding, speculative with the MTP when there is one: the prompt (prefill, or windows of up to kT tokens
    // when it is short), then rounds of [last token, up to `spec` drafts] verified at once.
    // The session invariant: after a call, the committed state is everything fed and emitted so far except the last
    // emitted token, which is `pending` - the next call feeds it first. Each round commits exactly the tokens up to the
    // last one emitted, so a stop (the callback, max_new, a full context) never leaves unemitted tokens in the state.
    int pending = -1;

    // sampling (temperature > 0): the rows of logits_d judged on the GPU; results in mapped memory
    int *   smp_draft_h = nullptr, * smp_tok_h = nullptr, * smp_acc_h = nullptr;
    float * smp_uni_h = nullptr;
    std::mt19937_64 rng{0x5eed};
    // rows [0, n) of logits (device, [n][V]); row t judges draft[t] (-1: none). Returns the tokens, acceptances in smp_*
    void sample(const float * logits, int n, const int * draft, const Sampling & sp) {
        std::uniform_real_distribution<float> U(0.f, 1.f);
        for (int t = 0; t < n; ++t) {
            reinterpret_cast<volatile int *>(smp_draft_h)[t] = draft ? draft[t] : -1;
            reinterpret_cast<volatile float *>(smp_uni_h)[2 * t] = U(rng);
            reinterpret_cast<volatile float *>(smp_uni_h)[2 * t + 1] = U(rng);
        }
        cuda::SampleParams p;
        p.temperature = sp.temperature;
        p.top_p = sp.top_p;
        p.top_k = sp.top_k;
        cuda::sample_rows(logits, c.n_vocab, n, p, dev_of(smp_draft_h), dev_of(smp_uni_h), dev_of(smp_tok_h), dev_of(smp_acc_h), st);
        ck(cudaStreamSynchronize(st), "sample");
    }

    std::vector<int> generate(const std::vector<int> & prompt_in, int max_new, int spec, float min_p,
                              const std::function<bool(int)> & on_token, const Sampling * smp = nullptr) {
        const bool sampling = smp && smp->temperature > 0.f;
        if (sampling && smp->seed) rng.seed(smp->seed);
        std::vector<int> prompt;
        if (pending >= 0) prompt.push_back(pending);
        prompt.insert(prompt.end(), prompt_in.begin(), prompt_in.end());
        if (prompt.empty()) throw std::runtime_error("generate: empty prompt and nothing pending");
        spec = mtp ? std::clamp(spec, 0, kT - 1) : 0;
        std::vector<int> out;
        if (max_new <= 0) max_new = 1;
        int y = -1;
        const int NP = static_cast<int>(prompt.size());
        if (pos + NP + 1 > max_ctx) throw std::runtime_error("generate: the prompt does not fit the context");
        pending = -1;   // fed below
        if (pf_on && NP > kT) {   // in chunks of up to pf_cap tokens; the MTP's rows for every chunk
            for (int off = 0; off < NP; off += pf_cap) {
                const int n = std::min(pf_cap, NP - off);
                const bool last = off + n == NP;
                y = prefill(prompt.data() + off, n, spec > 0, last ? -1 : prompt[off + n]);
            }
            if (spec) draft(0, nullptr, pos, spec - 1, last_draft.first, last_draft.second);   // the chain after the prefill's draft
        } else
        for (size_t s = 0; s < prompt.size(); s += kT) {   // the prompt; the MTP catches up on each window
            const int T = static_cast<int>(std::min<size_t>(kT, prompt.size() - s));
            const int P = pos;
            verify(&prompt[s], T, false);
            commit(T);
            y = ids_out[T - 1];
            if (spec) {
                int tk[kT];
                for (int r = 0; r < T; ++r) tk[r] = s + r + 1 < prompt.size() ? prompt[s + r + 1] : y;
                draft(T, tk, P, s + T == prompt.size() ? spec - 1 : 0);   // the chain only after the last window
            }
        }
        if (sampling) {   // the first token: drawn from the last prompt row's logits (prefill: row 0; windows: the last row)
            const int last_row = (pf_on && NP > kT) ? 0 : static_cast<int>((NP - 1) % kT);
            sample(logits_d + static_cast<size_t>(last_row) * c.n_vocab, 1, nullptr, *smp);
            y = reinterpret_cast<volatile int *>(smp_tok_h)[0];
        }
        out.push_back(y);
        pending = y;
        if (on_token && !on_token(y)) return out;
        while (static_cast<int>(out.size()) < max_new) {
            const int room = max_ctx - pos;   // cells left: the window [y, drafts] must fit
            if (room < 1) break;
            int W[kT] = {y};
            int k = 0;
            if (spec && room > 1) {   // drafts: the first from the catch-up, the chained ones while the previous is confident
                W[++k] = draft_tok[0];
                while (k < spec && k + 1 < room && k < n_drafts_ready && draft_prob[k - 1] >= min_p) {
                    W[k + 1] = draft_tok[k];
                    ++k;
                }
            }
            const int P = pos;
            verify(W, k + 1, false);
            int n = 1;
            if (sampling) {   // the rejection step row by row: accepted drafts, then a draw where the first one fails
                int dr[kT];
                for (int t = 0; t <= k; ++t) dr[t] = t < k ? W[t + 1] : -1;   // the last row: a plain draw (the bonus token)
                sample(logits_d, k + 1, dr, *smp);
                ids_out.assign(k + 1, -1);
                while (n <= k && reinterpret_cast<volatile int *>(smp_acc_h)[n - 1]) { ids_out[n - 1] = W[n]; ++n; }
                ids_out[n - 1] = reinterpret_cast<volatile int *>(smp_tok_h)[n - 1];
            } else {
                while (n <= k && W[n] == ids_out[n - 1]) ++n;
            }
            ++n_rounds;
            n_drafted += k;
            n_accepted += n - 1;
            // emit ids_out[0..e): stop early on the callback or max_new; then keep W[0..e) = y and the e - 1 emitted
            // drafts, so the state is the emitted text minus its last token (pending again)
            int e = 0;
            bool stop = false;
            while (e < n) {
                const int tok = ids_out[e++];
                out.push_back(tok);
                if ((on_token && !on_token(tok)) || static_cast<int>(out.size()) >= max_new) { stop = true; break; }
            }
            commit(e);
            y = ids_out[e - 1];
            pending = y;
            // the MTP's rows for the kept positions (also when stopping: the next call's drafts attend to them)
            if (spec) draft(e, ids_out.data(), P, stop ? 0 : spec - 1);
            if (stop) break;
        }
        return out;
    }
    std::pair<int, float> last_draft{-1, 0.f};   // the prefill's draft

    // ------------------------------------------------------------------------------------------------- prefill
    // Layer-major over the whole prompt (up to max_kv tokens): the dense parts in sub-chunks of kSc rows (cuBLAS on
    // dequantized weights), each layer's routed experts for all tokens at once, grouped by expert - the cached ones
    // from VRAM, the others copied over PCIe in double-buffered batches while the GPU computes.
    static constexpr int kSc = 256, kPfBatch = cuda::kMaxSlots / 2;
    bool   pf_on = true;
    int    pf_cap = 0;
    // (the buffers are the stage's, Dev::P_*; Dev::pf_ring takes speculative copies of a layer's non-cached experts
    // during its dense part, when the prompt will use most experts)
    double t_prefill_ms = 0;
    long long n_pf_copied = 0;   // experts copied in by prefill
    int    pf_admit = 48;        // per layer: the prompt tail's non-cached experts admitted to the LRU cache (from VRAM)
    int    pf_admit_win = 64;    // the prompt tail: its last this many tokens' expert use ranks the layer's experts
    long long n_pf_admit = 0;

    // The big prefill buffers live in a region "lent" by the expert arena: its tail holds the lowest-ranked experts
    // while decoding; a prefill evicts them (they stay in pinned RAM), uses the region, then refills it in the
    // background with the experts the prompt's tail used. Decode loses no VRAM to prefill.
    // (Dev::loan_base; Dev::loaned: the region is out of the cache, between a prompt's chunks; Dev::loan_slots)
    std::vector<std::vector<int>> tail_rc, all_rc;       // per layer: expert use by the prompt's last tokens / by all
    // the next layer's predicted experts copied during this layer's attention: a layer misses ~300 experts of an 8K
    // chunk, so 384 slots keep PCIe busy through the attention phase instead of stalling the MoE (8K: 4.91 -> 4.43 s)
    int       pf_ring_n = 384;
    size_t    pf_ring_bytes = 0;

    // carves the buffers from base (nullptr: only measures); returns the bytes
    size_t layout_prefill(uint8_t * base) {
        const int D = c.n_embd, HD = c.hc_dim(), NE = c.n_expert, K = c.n_expert_used, F = c.n_ff_exp;
        const int conv_dim = 2 * c.ssm_groups * c.ssm_state + c.ssm_inner;
        const size_t big = std::max<size_t>({static_cast<size_t>(HD), static_cast<size_t>(conv_dim),
                                             2ull * c.n_head * c.head_dim, static_cast<size_t>(NE)});
        const Tensor & gs0 = need("blk.0.ffn_gate_shexp.weight");
        const int Fs = static_cast<int>(gs0.elements() / gs0.shape[0]);
        const size_t cap = pf_cap, A = cap * K;
        size_t off = 0;
        auto take = [&](auto *& ptr, size_t n) {
            using T = std::remove_reference_t<decltype(*ptr)>;
            off = (off + 255) & ~size_t(255);
            ptr = base ? reinterpret_cast<T *>(base + off) : nullptr;
            off += n * sizeof(T);
        };
        auto take_v = [&](void *& ptr, size_t bytes) {
            uint8_t * q = nullptr;
            take(q, bytes);
            ptr = q;
        };
        take(P_R, cap * HD); take(P_mixed, cap * D); take(P_h, cap * D); take(P_sh, cap * D);
        take(P_inj, cap * c.hc); take(P_rl, cap * NE); take(P_w, cap * K);
        take(P_ids, cap * K); take(P_tok, A); take(P_aw, A);
        take(P_rl2, cap * NE); take(P_w2, cap * K); take(P_ids2, cap * K);
        take_v(P_xq, cap * cuda::q8_bytes(D)); take_v(P_xg, A * cuda::q8_bytes(D));
        take(P_act, A * F); take_v(P_aq, A * cuda::q8_bytes(F));
        take(P_g, cap * Fs); take(P_u, cap * Fs); take(P_sg, cap);
        take(P_xn, static_cast<size_t>(kSc) * HD); take(P_lo, static_cast<size_t>(kSc) * c.hc_low_rank);
        take(P_a, kSc * big); take(P_b, kSc * big); take(P_c, kSc * big); take(P_d, kSc * big); take(P_e, kSc * big);
        take(P_pn, static_cast<size_t>(kSc) * HD);
        take(P_ix, static_cast<size_t>(kSc) * 1152);   // the indexer's q (raw, normed) and raw k
        P_scr_n = 4u << 20;
        take(P_scr, P_scr_n);
        P_w16_n = 64u << 20;   // a layer's quantized projections as bf16 (the largest needs ~63M values)
        take_v(P_w16, P_w16_n * 2);
        P_x16_n = std::max(cap * static_cast<size_t>(D), static_cast<size_t>(kSc) * HD * 2);
        take_v(P_x16, P_x16_n * 2);
        take(P_tokd, cap);
        P_items_cap = static_cast<int>(A / 16 + NE + 64);   // enough at the smallest item size
        take(P_items_d, static_cast<size_t>(P_items_cap));
        take_v(P_ple_d, cap * c.ple_heads() * ple_row_bytes);
        pf_ring.assign(pf_ring_n, nullptr);
        for (auto & r : pf_ring) take(r, pf_ring_bytes);
        return (off + 255) & ~size_t(255);
    }

    void alloc_prefill() {
        if (const char * e = std::getenv("BL_PREFILL")) pf_on = std::atoi(e) != 0;
        if (const char * e = std::getenv("BL_PF_ADMIT")) pf_admit = std::atoi(e);
        if (const char * e = std::getenv("BL_PF_ADMIT_WIN")) pf_admit_win = std::atoi(e);
        if (const char * e = std::getenv("BL_PF_RING")) pf_ring_n = std::atoi(e);
        if (!pf_on) return;
        const int NE = c.n_expert, K = c.n_expert_used;
        // longer prompts go in chunks of pf_cap tokens: each chunk streams every layer's non-cached experts over PCIe
        // once, so bigger chunks mean fewer passes (the buffers, ~1.75 GB at 8192, are lent by the expert arena)
        pf_cap = 8192;
        if (const char * e = std::getenv("BL_PF_CHUNK")) pf_cap = std::max(kSc, std::atoi(e));
        pf_cap = std::min(pf_cap, max_kv);
        const size_t cap = pf_cap, A = cap * K;
        for (int il = 0; il < c.n_layer; ++il) {   // ring slots fit the largest expert
            const std::string p = "blk." + std::to_string(il) + ".";
            pf_ring_bytes = std::max(pf_ring_bytes, (2 * need(p + "ffn_gate_exps.weight").nbytes + need(p + "ffn_down_exps.weight").nbytes) / NE);
        }
        for (int s = 0; s <= last(); ++s) {   // every stage: its own buffers (the big ones lent by its arena)
            bind(s);
            loan_bytes = layout_prefill(nullptr);
            P_nsc = dalloc<int>(cap / kSc + 2);
            P_ts = dalloc<cuda::TokenState>(cap / kSc + 2);
            P_ids_h = host_mapped<int>(cap * K, host_allocs); P_w_h = host_mapped<float>(cap * K, host_allocs);
            P_ids2_h = host_mapped<int>(cap * K, host_allocs);
            P_tok_h = host_mapped<int>(A, host_allocs); P_aw_h = host_mapped<float>(A, host_allocs);
            P_tokd_h = host_mapped<int>(cap, host_allocs);
            P_items_h = host_mapped<cuda::MoeItem>(P_items_cap, host_allocs);
            P_ple_h = host_mapped<uint8_t>(cap * c.ple_heads() * ple_row_bytes, host_allocs);
            ck(cudaEventCreateWithFlags(&ev_pre, cudaEventDisableTiming), "event");
            ck(cudaEventCreateWithFlags(&ev_moe_done, cudaEventDisableTiming), "event");
            for (int i = 0; i < 2; ++i) {
                ck(cudaEventCreateWithFlags(&ev_ready[i], cudaEventDisableTiming), "event");
                ck(cudaEventCreateWithFlags(&ev_free[i], cudaEventDisableTiming), "event");
            }
        }
        bind(last());
        tail_rc.assign(c.n_layer, std::vector<int>(NE, 0));
        all_rc.assign(c.n_layer, std::vector<int>(NE, 0));
    }

    // before a prefill: the region's experts leave the cache (the decode table forgets them)
    void loan_out() {
        ck(cudaStreamSynchronize(adm), "admissions");   // a refill or admission may still write into the region
        const int NE = c.n_expert;
        for (auto [il, slot] : loan_slots)
            for (int e = 0; e < NE; ++e) {
                const size_t k = static_cast<size_t>(il) * NE + e;
                if (exp_dev[k] == slot) { exp_dev[k] = nullptr; --cached; break; }
            }
        ck(cudaMemcpy(table_d, exp_dev.data(), exp_dev.size() * sizeof(uint8_t *), cudaMemcpyHostToDevice), "expert table");
    }

    // after: the region's slots take each layer's most used non-cached experts of the prompt's tail, copied in on the
    // admission stream; the table learns each one after its bytes arrived
    void loan_back() {
        if (loan_slots.empty()) return;
        const int NE = c.n_expert;
        ck(cudaStreamSynchronize(st), "prefill");   // the region is free
        ++use_clock;
        std::vector<std::vector<int>> cand(c.n_layer);
        std::vector<size_t> next(c.n_layer, 0);
        for (int il = 0; il < c.n_layer; ++il) {
            for (int e = 0; e < NE; ++e)
                if (!exp_dev[static_cast<size_t>(il) * NE + e]) cand[il].push_back(e);
            auto & tr = tail_rc[il], & ar = all_rc[il];
            std::stable_sort(cand[il].begin(), cand[il].end(), [&](int x, int y) {
                return tr[x] != tr[y] ? tr[x] > tr[y] : ar[x] > ar[y];
            });
        }
        cuda::PtrSet in;
        auto ** tab = const_cast<const uint8_t **>(table_d);
        for (auto [il, slot] : loan_slots) {
            if (next[il] >= cand[il].size()) continue;
            const int e = cand[il][next[il]++];
            const size_t k = static_cast<size_t>(il) * NE + e;
            ck(cudaMemcpyAsync(slot, exp_host[k], lx[il].bytes(), cudaMemcpyHostToDevice, adm), "refill");
            exp_dev[k] = slot;
            last_use[k] = use_clock;
            ++cached;
            in.idx[in.n] = static_cast<int>(k);
            in.val[in.n++] = slot;
            if (in.n == cuda::kMaxPtrSet) { cuda::set_ptrs(tab, in, adm); in.n = 0; }
        }
        if (in.n) cuda::set_ptrs(tab, in, adm);
    }

    // bf16 GEMMs: a BF16 matrix in place; any other dequantized once per layer into P_w16 (a bump region reset by
    // w16_reset), or, when it does not fit, the fp32 path in row blocks
    void w16_reset() { w16.clear(); w16_used = 0; }
    void gemm(const std::string & n, const float * x, int N, float * y, int ldy = 0) {
        const Mat & m = M(n);
        const void * W = nullptr;
        if (m.type == 30) W = m.d;
        else if (auto it = w16.find(n); it != w16.end()) W = it->second;
        else if (w16_used + static_cast<size_t>(m.M) * m.K <= P_w16_n) {
            void * dst = static_cast<uint16_t *>(P_w16) + w16_used;
            cuda::dequant_rows_bf16(m.type, m.d, m.row_bytes, m.M, m.K, dst, st);
            w16_used += static_cast<size_t>(m.M) * m.K;
            w16[n] = W = dst;
        }
        if (W && static_cast<size_t>(N) * m.K <= P_x16_n) cuda::gemm_bf16(W, m.M, m.K, x, N, y, ldy ? ldy : m.M, P_x16, st);
        else cuda::gemm_w(m.type, m.d, m.row_bytes, m.M, m.K, x, N, y, ldy ? ldy : m.M, P_scr, P_scr_n, st);
    }

    // hyper-connection mix of T rows of R (sub-chunk); the inject into inj_out ([T][hc]) when wanted
    void hc_mix_pf(const std::string & pre, const float * Rs, float * inj_out, float * mixed_out, int T) {
        const int D = c.n_embd;
        cuda::rmsnorm_rows(Rs, P_xn, T * c.hc, D, V(pre + "_norm.weight"), c.hc, static_cast<float>(c.rms_eps), 1.f, st);
        gemm(pre + "_down.weight", P_xn, T, P_lo);
        if (inj_out) gemm(pre + "_inject.weight", P_xn, T, inj_out);
        cuda::silu_scale(P_lo, P_lo, static_cast<size_t>(T) * c.hc_low_rank, 1.f / c.hc, st);
        gemm(pre + "_up.weight", P_lo, T, P_a);
        cuda::hc_apply(P_a, P_xn, mixed_out, D, c.hc, T, st);
    }

    void ple_pf(float * Rs, int T, int j, int off) {
        const int D = c.n_embd, HD = c.hc_dim();
        const float eps = static_cast<float>(c.rms_eps);
        const Tensor & tab = g.at("per_layer_token_embd.weight");
        const std::string p = "blk." + std::to_string(c.ple_layer) + ".";
        const uint8_t * rows = static_cast<const uint8_t *>(P_ple_d) + static_cast<size_t>(off) * c.ple_heads() * ple_row_bytes;
        float *emb = P_a, *key = P_b, *value = P_c, *qn = P_d, *gt = P_e, *gated = P_xn;
        cuda::dequant_rows(tab.type_id, rows, ple_row_bytes, T * c.ple_heads(), c.ple_dim, emb, st);
        gemm(p + "ple_key.weight", emb, T, key);
        gemm(p + "ple_value.weight", emb, T, value);
        cuda::rmsnorm_rows(key, key, T * c.hc, D, V(p + "ple_norm_key.weight"), c.hc, eps, 1.f, st);
        cuda::rmsnorm_rows(Rs, qn, T * c.hc, D, V(p + "ple_norm_query.weight"), c.hc, eps, 1.f, st);
        cuda::ple_gate(key, qn, gt, D, T * c.hc, st);
        for (int t = 0; t < T; ++t)
            cuda::ple_gated_value(value + static_cast<size_t>(t) * D, gt + t * c.hc, gated + static_cast<size_t>(t) * HD, D, c.hc, st);
        cuda::rmsnorm_rows(gated, P_pn, T * c.hc, D, V(p + "ple_norm_conv.weight"), c.hc, eps, 1.f, st);
        float * conv_out = P_d;
        cuda::ple_conv_t(ple_hist, P_pn, V(p + "ple_conv1d.weight"), conv_out, HD, c.ple_conv, c.ple_ngram, T, st);
        cuda::add(gated, conv_out, T * HD, st);
        cuda::add(Rs, gated, T * HD, st);
        cuda::ple_commit(ple_hist, P_pn, HD, (c.ple_conv - 1) * c.ple_ngram, P_nsc + j, st);
    }

    void gdn_pf(int il, int T, int j) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int S = c.ssm_state, Hk = c.ssm_groups, Hv = c.ssm_v_heads;
        const int conv_dim = 2 * Hk * S + c.ssm_inner;
        const float eps = static_cast<float>(c.rms_eps);
        float *qkv = P_a, *z = P_b, *braw = P_c, *araw = P_c + static_cast<size_t>(kSc) * Hv, *cv = P_d, *beta = P_e,
              *gg = P_e + static_cast<size_t>(kSc) * Hv;
        gemm(p + "attn_qkv.weight", P_mixed_sc, T, qkv);
        gemm(p + "attn_gate.weight", P_mixed_sc, T, z);
        gemm(p + "ssm_beta.weight", P_mixed_sc, T, braw);
        gemm(p + "ssm_alpha.weight", P_mixed_sc, T, araw);
        cuda::gdn_pre_t(conv_state[il], qkv, V(p + "ssm_conv1d.weight"), cv, Hk, Hv, S, eps, braw, araw, Hv, V(p + "ssm_dt.bias"),
                        V(p + "ssm_a"), beta, gg, T, st);
        float * o = P_c;   // the raw gates are consumed
        cuda::gdn_step_t(gdn_state[il], gdn_state[il], cv, conv_dim, gg, beta, Hv, o, Hv, Hk, S, T, nullptr, st);
        cuda::conv_commit(conv_state[il], qkv, conv_dim, c.ssm_conv - 1, P_nsc + j, st);
        cuda::gdn_post_t(o, V(p + "ssm_norm.weight"), z, Hv, S, eps, T, st);
        gemm(p + "ssm_out.weight", o, T, P_b);
    }

    void qsa_pf(int il, int T, int j, int pos_abs) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int Dh = c.head_dim, H = c.n_head, Hkv = c.n_head_kv;
        const float eps = static_cast<float>(c.rms_eps);
        float *qf = P_a, *k = P_b, *v = P_c, *q = P_d, *att = P_e;
        float *iq_raw = P_ix, *ik_raw = P_ix + kSc * 512, *iq = P_ix + kSc * 640;
        gemm(p + "attn_q.weight", P_mixed_sc, T, qf);
        gemm(p + "attn_k.weight", P_mixed_sc, T, k);
        gemm(p + "attn_v.weight", P_mixed_sc, T, v);
        if (il < c.n_layer) {
            gemm(p + "indexer.q_proj.weight", P_mixed_sc, T, iq_raw);
            gemm(p + "indexer.k_proj.weight", P_mixed_sc, T, ik_raw);
        }
        cuda::qsa_pre_t(qf, q, H, k, v, Hkv, Dh, V(p + "attn_q_norm.weight"), V(p + "attn_k_norm.weight"), eps, c.rope_dims,
                        static_cast<float>(c.rope_freq_base), P_ts + j, kv[il], T, st);
        const cuda::AttnSel sel = select(il, T, P_ts + j, iq_raw, ik_raw, iq);
        if (const char * e = std::getenv("BL_DUMP_SEL"); e && std::atoi(e) == il && sel.list) {   // testing: the selection
            std::vector<int> cnt(T), lst(static_cast<size_t>(T) * kSelStride);
            std::vector<float> qv(static_cast<size_t>(T) * 512);
            ck(cudaMemcpyAsync(cnt.data(), sel_cnt, sizeof(int) * T, cudaMemcpyDeviceToHost, st), "dump");
            ck(cudaMemcpyAsync(lst.data(), sel_list, sizeof(int) * lst.size(), cudaMemcpyDeviceToHost, st), "dump");
            ck(cudaMemcpyAsync(qv.data(), iq, sizeof(float) * qv.size(), cudaMemcpyDeviceToHost, st), "dump");
            ck(cudaStreamSynchronize(st), "dump");
            const char * dir = std::getenv("BL_DUMP_SEL_DIR");
            std::ofstream f(std::string(dir ? dir : ".") + "/sel_dump_" + std::to_string(pos_abs) + ".bin", std::ios::binary);
            f.write(reinterpret_cast<const char *>(&T), 4);
            f.write(reinterpret_cast<const char *>(cnt.data()), 4 * T);
            f.write(reinterpret_cast<const char *>(lst.data()), 4 * lst.size());
            f.write(reinterpret_cast<const char *>(qv.data()), 4 * qv.size());
        }
        cuda::attention_prefill(q, kv[il], qf + Dh, 2 * Dh, 2 * H * Dh, att, H, Hkv, Dh, pos_abs, T,
                                1.f / std::sqrt(static_cast<float>(Dh)), sel, st);
        gemm(p + "attn_output.weight", att, T, P_b);
    }

    // the routed + shared experts of layer il for all N tokens: P_mixed -> P_h
    void moe_pf(int il, int N) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int D = c.n_embd, F = c.n_ff_exp, K = c.n_expert_used, NE = c.n_expert;
        const LayerExperts & L_ = lx[il];
        gemm(p + "ffn_gate_inp.weight", P_mixed, N, P_rl);
        cuda::router_topk_t(P_rl, NE, K, P_ids, P_w, N, st);
        ck(cudaMemcpyAsync(P_ids_h, P_ids, sizeof(int) * N * K, cudaMemcpyDeviceToHost, st), "ids");
        const bool predict = il + 1 < l1 && !pf_ring.empty();   // (the next layer on this stage's GPU)
        if (predict) {   // the next layer's router on this input: which of its experts to copy early
            gemm("blk." + std::to_string(il + 1) + ".ffn_gate_inp.weight", P_mixed, N, P_rl2);
            cuda::router_topk_t(P_rl2, NE, K, P_ids2, P_w2, N, st);
            ck(cudaMemcpyAsync(P_ids2_h, P_ids2, sizeof(int) * N * K, cudaMemcpyDeviceToHost, st), "ids");
        }
        ck(cudaMemcpyAsync(P_w_h, P_w, sizeof(float) * N * K, cudaMemcpyDeviceToHost, st), "w");
        // meanwhile on the GPU: the shared expert and the activations as q8_1
        const Mat & gs = M(p + "ffn_gate_shexp.weight");
        gemm(p + "ffn_gate_shexp.weight", P_mixed, N, P_g);
        gemm(p + "ffn_up_shexp.weight", P_mixed, N, P_u);
        cuda::swiglu(P_g, P_u, P_g, static_cast<size_t>(N) * gs.M, st);
        gemm(p + "ffn_down_shexp.weight", P_g, N, P_sh);
        cuda::gemm_w(0, V(p + "ffn_gate_inp_shexp.weight"), static_cast<size_t>(D) * 4, 1, D, P_mixed, N, P_sg, 1, P_scr, P_scr_n, st);
        cuda::quantize_q8(P_mixed, D, P_xq, st, N);
        ck(cudaMemsetAsync(P_h, 0, sizeof(float) * N * D, st), "h");
        ck(cudaStreamSynchronize(st), "router");

        next_pre.clear();
        if (predict) {
            std::vector<int> pc(NE, 0);
            for (int i = 0; i < N * K; ++i) ++pc[P_ids2_h[i]];
            for (int e = 0; e < NE; ++e)
                if (pc[e] && !exp_dev[static_cast<size_t>(il + 1) * NE + e]) next_pre.push_back(e);
            std::stable_sort(next_pre.begin(), next_pre.end(), [&](int x, int y) { return pc[x] > pc[y]; });
        }
        // group the assignments by expert: cached experts first, then the others in copy order
        std::vector<int> cnt(NE, 0), order;
        for (int i = 0; i < N * K; ++i) ++cnt[P_ids_h[i]];
        if (il < c.n_layer) {
            all_rc[il] = cnt;
            std::fill(tail_rc[il].begin(), tail_rc[il].end(), 0);
            for (int i = std::max(0, N - 64) * K; i < N * K; ++i) ++tail_rc[il][P_ids_h[i]];
        }
        std::vector<int> cached_e, pre_e, miss_e, pre_slot(NE, -1);
        for (size_t i = 0; i < pre_ids.size(); ++i) pre_slot[pre_ids[i]] = static_cast<int>(i);
        for (int e = 0; e < NE; ++e)
            if (cnt[e]) (exp_dev[static_cast<size_t>(il) * NE + e] ? cached_e : pre_slot[e] >= 0 ? pre_e : miss_e).push_back(e);
        std::vector<int> off(NE, 0);
        int a = 0;
        for (int e : cached_e) { off[e] = a; a += cnt[e]; }
        const int a_cached = a;
        for (int e : pre_e) { off[e] = a; a += cnt[e]; }
        const int a_pre = a;
        for (int e : miss_e) { off[e] = a; a += cnt[e]; }
        for (int i = 0; i < N * K; ++i) {
            const int e = P_ids_h[i], ai = off[e]++;
            P_tok_h[ai] = i / K;
            P_aw_h[ai] = P_w_h[i];
        }
        int ni = 0;
        const int moe_tile = cuda::moe_tile();
        auto add_items = [&](int e, const uint8_t * base) {
            const int a1 = off[e], a0 = a1 - cnt[e];
            for (int x = a0; x < a1; x += moe_tile) P_items_h[ni++] = {base, x, std::min(moe_tile, a1 - x)};
        };
        for (int e : cached_e) add_items(e, exp_dev[static_cast<size_t>(il) * NE + e]);
        const int ni_cached = ni;
        for (int e : pre_e) add_items(e, pf_ring[pre_slot[e]]);
        const int ni_pre = ni;
        const int nb = (static_cast<int>(miss_e.size()) + kPfBatch - 1) / kPfBatch;
        std::vector<int> bi0(nb + 1), ba0(nb + 1);
        for (int b = 0; b < nb; ++b) {
            bi0[b] = ni;
            ba0[b] = off[miss_e[b * kPfBatch]] - cnt[miss_e[b * kPfBatch]];
            for (int j = 0; j < kPfBatch && b * kPfBatch + j < static_cast<int>(miss_e.size()); ++j)
                add_items(miss_e[b * kPfBatch + j], staging[(b % 2) * kPfBatch + j]);
        }
        bi0[nb] = ni;
        ba0[nb] = a;
        // LRU: the experts the prompt tail (its last pf_admit_win tokens) uses most should be cached when decoding
        // starts. Up to pf_admit of them that are not cached replace the cached ones it uses least (then the least
        // recently used), copied from where the prefill has them in VRAM; the tail's cached experts become the most recent
        std::vector<std::pair<int, uint8_t *>> adm_x;   // (expert, cache slot)
        std::vector<int> adm_at(NE, -1);                 // expert -> index in adm_x
        std::vector<cuda::PtrSet> out_sets, in_sets;
        if (lru && pf_admit > 0) {
            std::vector<int> rc(NE, 0);
            for (int i = std::max(0, N - pf_admit_win) * K; i < N * K; ++i) ++rc[P_ids_h[i]];
            ++use_clock;
            const size_t l0 = static_cast<size_t>(il) * NE;
            std::vector<int> cand, held;
            for (int e = 0; e < NE; ++e) (exp_dev[l0 + e] ? held : cand).push_back(e);
            std::erase_if(cand, [&](int e) { return rc[e] == 0; });
            std::stable_sort(cand.begin(), cand.end(), [&](int x, int y) { return rc[x] > rc[y]; });
            std::stable_sort(held.begin(), held.end(),
                             [&](int x, int y) { return rc[x] != rc[y] ? rc[x] < rc[y] : last_use[l0 + x] < last_use[l0 + y]; });
            for (size_t i = 0; i < cand.size() && i < held.size() && static_cast<int>(i) < pf_admit; ++i) {
                const int x = cand[i], v = held[i];
                if (rc[x] <= rc[v]) break;
                uint8_t * slot = exp_dev[l0 + v];
                exp_dev[l0 + v] = nullptr;
                exp_dev[l0 + x] = slot;
                if (out_sets.empty() || out_sets.back().n == cuda::kMaxPtrSet) { out_sets.emplace_back(); in_sets.emplace_back(); }
                cuda::PtrSet & os = out_sets.back(), & is = in_sets.back();
                os.idx[os.n] = static_cast<int>(l0 + v); os.val[os.n++] = nullptr;
                is.idx[is.n] = static_cast<int>(l0 + x); is.val[is.n++] = slot;
                adm_at[x] = static_cast<int>(adm_x.size());
                adm_x.push_back({x, slot});
            }
            for (int e = 0; e < NE; ++e)
                if (rc[e] && exp_dev[l0 + e]) last_use[l0 + e] = use_clock;
        }
        // after an expert's last use in this layer, its bytes go to its cache slot (the victims' last use was the
        // cached run, which comes first)
        auto admit_from = [&](const std::vector<int> & es, const std::function<const uint8_t *(int)> & src) {
            for (int e : es)
                if (adm_at[e] >= 0)
                    ck(cudaMemcpyAsync(adm_x[adm_at[e]].second, src(e), L_.bytes(), cudaMemcpyDeviceToDevice, st), "admit");
        };
        const int A = N * K;
        // by kernel from mapped memory: a cudaMemcpy would wait behind the expert copies on the copy engine
        cuda::copy_words(dev_of(P_tok_h), P_tok, sizeof(int) * A, st);
        cuda::copy_words(dev_of(P_aw_h), P_aw, sizeof(float) * A, st);
        cuda::copy_words(dev_of(P_items_h), P_items_d, sizeof(cuda::MoeItem) * ni, st);
        cuda::moe_gather(P_xq, P_tok, A, D, P_xg, st);
        const size_t d_rb = L_.d_bytes / D;
        auto run = [&](int i0, int i1, int a0, int a1) {
            if (i1 <= i0) return;
            cuda::moe_gate_up(L_.type_gu, P_items_d + i0, i1 - i0, L_.gu_bytes, F, D, P_xg, P_act, st);
            cuda::quantize_q8(P_act + static_cast<size_t>(a0) * F, F, static_cast<uint8_t *>(P_aq) + a0 * cuda::q8_bytes(F), st, a1 - a0);
            cuda::moe_down(L_.type_d, P_items_d + i0, i1 - i0, L_.gu_bytes, d_rb, F, D, P_aq, P_tok, P_aw, P_h, st);
        };
        // the copies: batch b into half b % 2 of the staging slots, once the GPU is done with batch b - 2
        for (int b = 0; b < nb; ++b) {
            if (b >= 2) ck(cudaStreamWaitEvent(cp, ev_free[b % 2], 0), "wait");
            for (int j = 0; j < kPfBatch && b * kPfBatch + j < static_cast<int>(miss_e.size()); ++j) {
                const size_t k = static_cast<size_t>(il) * NE + miss_e[b * kPfBatch + j];
                ck(cudaMemcpyAsync(staging[(b % 2) * kPfBatch + j], exp_host[k], L_.bytes(), cudaMemcpyHostToDevice, cp), "copy");
            }
            ck(cudaEventRecord(ev_ready[b % 2], cp), "record");
            if (b == 0) {   // the cached and prefetched experts while the first batch is in flight
                run(0, ni_cached, 0, a_cached);
                if (!pre_ids.empty()) ck(cudaStreamWaitEvent(st, ev_pre, 0), "wait");
                run(ni_cached, ni_pre, a_cached, a_pre);
                admit_from(pre_e, [&](int e) { return pf_ring[pre_slot[e]]; });
            }
            ck(cudaStreamWaitEvent(st, ev_ready[b % 2], 0), "wait");
            run(bi0[b], bi0[b + 1], ba0[b], ba0[b + 1]);
            {
                std::vector<int> es;
                std::vector<int> slot_of(NE, -1);
                for (int j = 0; j < kPfBatch && b * kPfBatch + j < static_cast<int>(miss_e.size()); ++j) {
                    es.push_back(miss_e[b * kPfBatch + j]);
                    slot_of[es.back()] = (b % 2) * kPfBatch + j;
                }
                admit_from(es, [&](int e) { return staging[slot_of[e]]; });
            }
            ck(cudaEventRecord(ev_free[b % 2], st), "record");
        }
        if (nb == 0) {
            run(0, ni_cached, 0, a_cached);
            if (!pre_ids.empty()) ck(cudaStreamWaitEvent(st, ev_pre, 0), "wait");
            run(ni_cached, ni_pre, a_cached, a_pre);
            admit_from(pre_e, [&](int e) { return pf_ring[pre_slot[e]]; });
        }
        ck(cudaEventRecord(ev_moe_done, st), "record");
        if (!adm_x.empty()) {   // the decode table: victims out, newcomers in (their bytes are in place by now)
            auto ** tab = const_cast<const uint8_t **>(table_d);
            for (const auto & os : out_sets) cuda::set_ptrs(tab, os, st);
            for (const auto & is : in_sets) cuda::set_ptrs(tab, is, st);
            n_pf_admit += static_cast<long long>(adm_x.size());
        }
        n_pf_copied += static_cast<long long>(miss_e.size() + pre_ids.size());
        cuda::add_gated(P_h, P_sh, P_sg, D, N, st);
    }

    // one layer (il < n_layer: the model's; il == n_layer: the MTP's) over N rows of P_R at positions pos0..
    void prefetch_layer(int il) {
        pre_ids.clear();
        const int NE = c.n_expert;
        if (il >= c.n_layer || pf_ring.empty() || next_pre.empty()) return;
        ck(cudaStreamWaitEvent(cp, ev_moe_done, 0), "wait");   // the previous layer's experts are done with the ring
        for (int e : next_pre) {
            if (pre_ids.size() >= pf_ring.size()) break;
            const size_t k = static_cast<size_t>(il) * NE + e;
            ck(cudaMemcpyAsync(pf_ring[pre_ids.size()], exp_host[k], lx[il].bytes(), cudaMemcpyHostToDevice, cp), "prefetch");
            pre_ids.push_back(e);
        }
        ck(cudaEventRecord(ev_pre, cp), "record");
    }

    void layer_pf(int il, int N, int pos0) {
        const int D = c.n_embd, HD = c.hc_dim();
        w16_reset();
        prefetch_layer(il);
        const std::string p = "blk." + std::to_string(il);
        for (int j = 0, off = 0; off < N; ++j, off += kSc) {
            const int T = std::min(kSc, N - off);
            float * Rs = P_R + static_cast<size_t>(off) * HD;
            if (il == c.ple_layer) ple_pf(Rs, T, j, off);
            float * inj_s = P_inj + static_cast<size_t>(off) * c.hc;
            P_mixed_sc = P_mixed + static_cast<size_t>(off) * D;
            hc_mix_pf(p + ".hc_attn", Rs, inj_s, P_mixed_sc, T);
            if (qsa_layer(il)) qsa_pf(il, T, j, pos0 + off);
            else gdn_pf(il, T, j);
            cuda::hc_combine_t(Rs, P_b, inj_s, D, c.hc, T, st);
            hc_mix_pf(p + ".hc_ffn", Rs, inj_s, P_mixed_sc, T);
        }
        moe_pf(il, N);
        for (int off = 0; off < N; off += kSc) {
            const int T = std::min(kSc, N - off);
            cuda::hc_combine_t(P_R + static_cast<size_t>(off) * HD, P_h + static_cast<size_t>(off) * D,
                               P_inj + static_cast<size_t>(off) * c.hc, D, c.hc, T, st);
        }
    }

    // feeds N prompt tokens at once; returns the argmax after the last. With the MTP: its rows for the prompt too,
    // and the first draft (last_draft) from the last row.
    std::vector<float> * pf_all_logits = nullptr;   // testing: every position's logits ([N][V])
    int prefill(const int * toks, int N, bool draft, int next = -1) {
        const auto t0_ = std::chrono::steady_clock::now();
        const int D = c.n_embd, HD = c.hc_dim();
        const int pos0 = pos;
        if (N > pf_cap || pos0 + N > max_kv) throw std::runtime_error("prefill: prompt too long");
        std::vector<char> lent(last() + 1);
        {   // per sub-chunk: positions and sizes (every stage)
            std::vector<cuda::TokenState> tsv(N / kSc + 1);
            std::vector<int> nsc(N / kSc + 1);
            for (int j = 0, off = 0; off < N; ++j, off += kSc) { tsv[j] = {}; tsv[j].pos = pos0 + off; nsc[j] = std::min(kSc, N - off); }
            for (int s = 0; s <= last(); ++s) {
                bind(s);
                ck(cudaStreamSynchronize(st), "sync");   // the plain cudaMemcpy below runs on the legacy stream
                lent[s] = loan_base >= arena && loan_base < arena + arena_bytes;
                if (lent[s] && !loaned) { loan_out(); loaned = true; }
                ck(cudaMemcpy(P_ts, tsv.data(), sizeof(cuda::TokenState) * tsv.size(), cudaMemcpyHostToDevice), "prefill positions");
                ck(cudaMemcpy(P_nsc, nsc.data(), sizeof(int) * nsc.size(), cudaMemcpyHostToDevice), "prefill sizes");
            }
        }
        bind(layer_stage[c.ple_layer]);   // the PLE rows: for the stage that runs the PLE layer
        ple_rows(toks, N, P_ple_h);
        cuda::copy_words(dev_of(P_ple_h), P_ple_d, static_cast<size_t>(N) * c.ple_heads() * ple_row_bytes, st);
        bind(0);
        for (int t = 0; t < N; ++t) P_tokd_h[t] = toks[t];
        cuda::copy_words(dev_of(P_tokd_h), P_tokd, sizeof(int) * N, st);
        const Mat & te = M("token_embd.weight");
        for (int off = 0; off < N; off += kSc) {
            const int T = std::min(kSc, N - off);
            cuda::embed_tokens(te.type, te.d, te.row_bytes, D, P_tokd + off, T, P_a, st);
            for (int t = 0; t < T; ++t)
                cuda::repeat(P_a + static_cast<size_t>(t) * D, P_R + static_cast<size_t>(off + t) * HD, D, c.hc, st);
        }
        for (int s = 0; s <= last(); ++s) {   // stage by stage; the residual of all N rows handed on between them
            if (s) {   // (the previous stage's values taken before bind: it swaps this object's members)
                const float * src = P_R;
                const int src_dev = dev;
                ck(cudaStreamSynchronize(st), "prefill stage");
                bind(s);
                ck(cudaMemcpyPeerAsync(P_R, dev, src, src_dev, sizeof(float) * N * HD, st), "prefill hand-over");
            }
            for (int il = l0; il < l1; ++il) layer_pf(il, N, pos0);
        }
        if (pf_all_logits) {   // testing: the head on every row, kT rows at a time
            pf_all_logits->resize(static_cast<size_t>(N) * c.n_vocab);
            for (int off = 0; off < N; off += kT) {
                const int T = std::min(kT, N - off);
                hc_mix_pf("output_hc", P_R + static_cast<size_t>(off) * HD, nullptr, P_mixed, T);
                gemm("output.weight", P_mixed, T, logits_d);
                ck(cudaMemcpyAsync(pf_all_logits->data() + static_cast<size_t>(off) * c.n_vocab, logits_d,
                                   sizeof(float) * T * c.n_vocab, cudaMemcpyDeviceToHost, st), "logits");
                ck(cudaStreamSynchronize(st), "logits");
            }
        }
        // the last row's logits
        ck(cudaMemcpyAsync(R, P_R + static_cast<size_t>(N - 1) * HD, sizeof(float) * HD, cudaMemcpyDeviceToDevice, st), "last row");
        hc_mix("output_hc", false, -1, 1);
        mm("output.weight", mixed, logits_d, 1);
        cuda::argmax_rows(logits_d, c.n_vocab, 1, argmax_d, st);
        ck(cudaMemcpyAsync(argmax_h, argmax_d, sizeof(int), cudaMemcpyDeviceToHost, st), "argmax");
        ck(cudaStreamSynchronize(st), "prefill");
        const int y = *reinterpret_cast<volatile int *>(argmax_h);
        // the committed state: positions, histories (the GDN, conv and PLE states were updated in place)
        for (int t = 0; t < N; ++t) history.push_back(toks[t]);
        pos += N;
        steps += N;
        *reinterpret_cast<volatile int *>(ncommit_h) = pos;
        for (int s = 0; s <= last(); ++s) {   // every stage's position
            bind(s);
            cuda::window_set_pos(ts, ncommit_hd, st);
            ck(cudaStreamSynchronize(st), "position");
        }
        bind(last());
        if (mtp && draft) {   // the MTP's rows: main residual at i with token i+1 (the prompt's, then y)
            for (int t = 0; t < N; ++t) P_tokd_h[t] = t + 1 < N ? toks[t + 1] : next >= 0 ? next : y;
            cuda::copy_words(dev_of(P_tokd_h), P_tokd, sizeof(int) * N, st);
            const float eps = static_cast<float>(c.rms_eps);
            for (int off = 0; off < N; off += kSc) {
                const int T = std::min(kSc, N - off);
                float * Rs = P_R + static_cast<size_t>(off) * HD;
                cuda::embed_tokens(te.type, te.d, te.row_bytes, D, P_tokd + off, T, P_a, st);
                cuda::rmsnorm_rows(P_a, P_b, T, D, V("mtp.pre_fc_norm_embedding.weight"), 1, eps, 1.f, st);
                gemm("mtp.fc_embedding.weight", P_b, T, P_c);
                cuda::rmsnorm_rows(Rs, P_xn, T, HD, V("mtp.pre_fc_norm_hidden.weight"), 1, eps, 1.f, st);
                gemm("mtp.fc_hidden.weight", P_xn, T * c.hc, P_d);
                cuda::add_broadcast_streams(P_d, P_c, Rs, D, c.hc, T, st);
            }
            layer_pf(c.n_layer, N, pos0);
            ck(cudaMemcpyAsync(mtp_Rout, P_R + static_cast<size_t>(N - 1) * HD, sizeof(float) * HD, cudaMemcpyDeviceToDevice, st), "MTP row");
            ck(cudaMemcpyAsync(R, mtp_Rout, sizeof(float) * HD, cudaMemcpyDeviceToDevice, st), "MTP row");
            hc_mix("mtp.output_hc", false, -1, 1);
            const bool sub = !dvocab.empty();
            mm(sub ? "mtp.head" : "output.weight", mixed, mtp_logits, 1);
            cuda::argmax_rows(mtp_logits, sub ? static_cast<int>(dvocab.size()) : c.n_vocab, 1, mtp_id_d, st, mtp_prob_d);
            ck(cudaMemcpyAsync(mtp_id_h, mtp_id_d, sizeof(int), cudaMemcpyDeviceToHost, st), "draft");
            ck(cudaMemcpyAsync(mtp_prob_h, mtp_prob_d, sizeof(float), cudaMemcpyDeviceToHost, st), "draft");
            ck(cudaStreamSynchronize(st), "MTP prefill");
            const int id = *reinterpret_cast<volatile int *>(mtp_id_h);
            last_draft = {dvocab.empty() ? id : dvocab[id], *reinterpret_cast<volatile float *>(mtp_prob_h)};
        }
        if (next < 0)   // the prompt's last chunk: refill the regions
            for (int s = 0; s <= last(); ++s)
                if (lent[s]) { bind(s); loan_back(); loaned = false; }
        bind(last());
        t_prefill_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
        return y;
    }

    // keep the first *ncommit tokens of the window: the GDN recurrence replayed in place for them, the conv and PLE
    // histories advanced, the position moved
    void enqueue_commit() {   // the current stage's layers
        const int S = c.ssm_state, Hk = c.ssm_groups, Hv = c.ssm_v_heads;
        const int conv_dim = 2 * Hk * S + c.ssm_inner;
        for (int il = l0; il < l1; ++il) {
            if (c.is_qsa(il)) continue;
            cuda::conv_commit(conv_state[il], w_qkv[il], conv_dim, c.ssm_conv - 1, ncommit_hd, st);
            cuda::gdn_step_t(gdn_state[il], gdn_state[il], w_conv[il], conv_dim, w_g[il], w_beta[il], Hv, gdn_scratch_out(), Hv,
                             Hk, S, kT, ncommit_hd, st);
        }
        if (c.ple_layer >= l0 && c.ple_layer < l1)
            cuda::ple_commit(ple_hist, w_ple_norm, c.hc_dim(), (c.ple_conv - 1) * c.ple_ngram, ncommit_hd, st);
        cuda::window_commit(ts, ncommit_hd, st);
    }
    float * gdn_scratch_out() { return t5; }   // the replay's outputs are not needed

    // answer layer il's doorbell: copy its DMA experts in, compute its CPU experts, raise the flags, admit misses
    void service(int il) {
        const auto t0_ = std::chrono::steady_clock::now();
        const volatile uint32_t * bell = bell_h + il;
        bool told = false;
        for (long long spins = 0; *bell < seq; ++spins) {
            _mm_pause();
            if (!told && (spins & 0xfffff) == 0 && std::chrono::steady_clock::now() - t0_ > std::chrono::seconds(3)) {
                told = true;   // stuck: say where
                std::fprintf(stderr, "stuck waiting for layer %d's doorbell: bell %u, window seq %u, stream: %s\n", il, *bell,
                             seq, cudaGetErrorString(cudaStreamQuery(st)));
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const auto t1_ = std::chrono::steady_clock::now();
        t_bell_ms += std::chrono::duration<double, std::milli>(t1_ - t0_).count();
        const cuda::ExpertPlan pl = plan_h[il];   // a copy: the GPU rewrites it for the next window
        const LayerExperts & L_ = lx[il];
        const int D = c.n_embd, T = pl.T;
        if (route_trace) {   // record per token: int32 seq, layer, K expert ids
            for (int t = 0; t < T; ++t) {
                const int32_t hdr[2] = {static_cast<int32_t>(seq * kT + t), il};
                int32_t ids[cuda::kMaxK];
                for (int k = 0; k < pl.K; ++k) ids[k] = pl.ids[pl.sel[t][k]];
                std::fwrite(hdr, sizeof hdr, 1, route_trace);
                std::fwrite(ids, sizeof(int32_t), pl.K, route_trace);
            }
        }
        CpuExperts::Job jobs[cuda::kMaxSlots];
        int nj = 0;
        for (int s = 0; s < pl.n_slots; ++s) {
            const size_t k = static_cast<size_t>(il) * c.n_expert + pl.ids[s];
            if (pl.kind[s] == cuda::kDma)
                ck(cudaMemcpyAsync(staging[pl.idx[s]], exp_host[k], L_.bytes(), cudaMemcpyHostToDevice, cp), "expert copy");
            else if (pl.kind[s] == cuda::kCpu)
                jobs[nj++] = {exp_host[k], pl.mask[s], cpu_out_h + static_cast<size_t>(pl.idx[s]) * T * D};
            else
                ++n_hit;
        }
        n_dma += pl.n_dma;
        n_cpu += pl.n_cpu;
        n_pf_used += pl.n_pf;
        if (pl.n_dma) cuda::set_flag(dma_flag_hd + il, seq, cp);
        if (nj) {
            const auto tc = std::chrono::steady_clock::now();
            cpu->run(L_.type_gu, L_.type_d, L_.gu_bytes, c.n_ff_exp, D, x_h + static_cast<size_t>(il) * kT * D, T, jobs, nj);
            t_cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
            std::atomic_thread_fence(std::memory_order_release);
            _mm_sfence();
            reinterpret_cast<volatile uint32_t *>(cpu_flag_h)[il] = seq;   // the GPU is spinning for this
        }
        // the background PCIe work (the LRU admissions, the prefetches) comes after the CPU experts' results: the
        // GPU is waiting for those, while issuing copies on a busy copy stream can hold the host thread for tens of
        // microseconds - better spent idle-spinning on the next layer's doorbell than before the CPU's start
        if (lru) admit(il, pl);
        if (pf_max > 0 && T == 1 && il + 1 < l1) prefetch_next(il, pl);   // (the next layer on this stage's GPU)
    }

    void prefetch_next(int il, const cuda::ExpertPlan & pl) {   // the next layer's likely misses, behind this layer's copies
        const int nl = il + 1;
        const LayerExperts & LN = lx[nl];
        cuda::PfTable & tab = pf_h[nl];
        int n = 0;
        for (int i = 0; i < c.n_expert_used && n < pf_max; ++i) {
            const int id = pl.pred[i];
            if (id < 0) break;
            const size_t k = static_cast<size_t>(nl) * c.n_expert + id;
            if (exp_dev[k]) continue;   // cached anyway
            ck(cudaMemcpyAsync(pf_slot[nl % 2][n], exp_host[k], LN.bytes(), cudaMemcpyHostToDevice, cp), "prefetch");
            tab.ids[n++] = id;
        }
        tab.n = n;
        std::atomic_thread_fence(std::memory_order_release);
        _mm_sfence();
        tab.seq = seq;   // the GPU trusts the table from here on
        if (n) cuda::set_flag(pf_flag_hd + nl, seq, cp);
        n_pf_issued += n;
    }

    // LRU admission of layer il's misses: each takes the slot of the layer's least recently used cached expert.
    // On the admission stream, in order: the victim leaves the GPU's table, the newcomer's bytes land in its slot,
    // the newcomer enters the table. A plan reading the table at any point sees only correct entries.
    void admit(int il, const cuda::ExpertPlan & pl) {
        const int NE = c.n_expert;
        ++use_clock;
        for (int s = 0; s < pl.n_slots; ++s) last_use[static_cast<size_t>(il) * NE + pl.ids[s]] = use_clock;
        cuda::PtrSet out, in;
        std::vector<std::pair<uint8_t *, const uint8_t *>> copies;
        for (int s = 0; s < pl.n_slots && static_cast<int>(copies.size()) < std::min(lru_max, cuda::kMaxPtrSet); ++s) {
            if (pl.kind[s] == cuda::kHit) continue;
            const size_t x = static_cast<size_t>(il) * NE + pl.ids[s];
            if (exp_dev[x]) continue;
            size_t v = SIZE_MAX;
            uint64_t oldest = UINT64_MAX;
            for (int e = 0; e < NE; ++e) {
                const size_t k = static_cast<size_t>(il) * NE + e;
                if (exp_dev[k] && last_use[k] < oldest) { oldest = last_use[k]; v = k; }
            }
            if (v == SIZE_MAX || oldest == use_clock) break;   // nothing older than this window's own experts
            uint8_t * slot = exp_dev[v];
            exp_dev[v] = nullptr;
            exp_dev[x] = slot;
            out.idx[out.n] = static_cast<int>(v); out.val[out.n++] = nullptr;
            in.idx[in.n] = static_cast<int>(x);   in.val[in.n++] = slot;
            copies.push_back({slot, exp_host[x]});
        }
        if (copies.empty()) return;
        auto ** tab = const_cast<const uint8_t **>(table_d);
        cuda::set_ptrs(tab, out, adm);
        for (auto [dst, src] : copies)
            ck(cudaMemcpyAsync(dst, src, lx[il].bytes(), cudaMemcpyHostToDevice, adm), "admit");
        cuda::set_ptrs(tab, in, adm);
        n_admit += static_cast<long long>(copies.size());
    }

    // between windows (nothing in flight): the periodic swaps of the non-LRU cache
    void adapt() {
        const auto t0_ = std::chrono::steady_clock::now();
        struct Swap { float gain; int il; size_t in, out; };
        std::vector<Swap> cand;
        const int NE = c.n_expert;
        for (int il = 0; il < c.n_layer; ++il) {
            std::vector<size_t> hot, cold;
            for (int e = 0; e < NE; ++e) {
                const size_t k = static_cast<size_t>(il) * NE + e;
                if (exp_dev[k]) cold.push_back(k);
                else if (use[k] > 0.f) hot.push_back(k);
            }
            std::sort(hot.begin(), hot.end(), [&](size_t a, size_t b) { return use[a] > use[b]; });
            std::sort(cold.begin(), cold.end(), [&](size_t a, size_t b) { return use[a] < use[b]; });
            for (size_t j = 0; j < hot.size() && j < cold.size(); ++j) {
                if (use[hot[j]] <= use[cold[j]] + adapt_margin) break;
                cand.push_back({use[hot[j]] - use[cold[j]], il, hot[j], cold[j]});
            }
        }
        std::sort(cand.begin(), cand.end(), [](const Swap & a, const Swap & b) { return a.gain > b.gain; });
        if (static_cast<int>(cand.size()) > adapt_max) cand.resize(adapt_max);
        for (size_t b = 0; b < cand.size(); ++b)
            ck(cudaMemcpyAsync(swap_bounce + b * max_expert_bytes, exp_dev[cand[b].out], lx[cand[b].il].bytes(),
                               cudaMemcpyDeviceToHost, cp), "swap out");
        for (const Swap & sw : cand)
            ck(cudaMemcpyAsync(exp_dev[sw.out], exp_host[sw.in], lx[sw.il].bytes(), cudaMemcpyHostToDevice, cp), "swap in");
        ck(cudaStreamSynchronize(cp), "swap");
        for (size_t b = 0; b < cand.size(); ++b) {
            const Swap & sw = cand[b];
            uint8_t * slot = exp_dev[sw.out], * hslot = exp_host[sw.in];
            if (!ram_all) std::memcpy(hslot, swap_bounce + b * max_expert_bytes, lx[sw.il].bytes());
            exp_dev[sw.in] = slot;   if (!ram_all) exp_host[sw.in] = nullptr;
            exp_dev[sw.out] = nullptr; if (!ram_all) exp_host[sw.out] = hslot;
        }
        if (!cand.empty())
            ck(cudaMemcpy(table_d, exp_dev.data(), exp_dev.size() * sizeof(uint8_t *), cudaMemcpyHostToDevice), "expert table");
        for (float & u : use) u *= adapt_decay;
        n_swaps += static_cast<long long>(cand.size());
        t_adapt_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
    }

    void capture(cudaGraphExec_t & exec, const std::function<void()> & enqueue) {
        cudaGraph_t gr;
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
        enqueue();
        ck(cudaStreamEndCapture(st, &gr), "capture");
        ck(cudaGraphInstantiate(&exec, gr, 0), "graph instantiate");
        cudaGraphDestroy(gr);
    }

    void verify(const int * toks, int T, bool want) {
        if (T < 1 || T > kT) throw std::runtime_error("verify: window of " + std::to_string(T) + " tokens");
        if (pos + T > max_ctx) throw std::runtime_error("context full");
        if (probe && T != 1) throw std::runtime_error("probes need one-token windows");
        const auto t0_ = std::chrono::steady_clock::now();
        ++seq;
        for (int t = 0; t < T; ++t) reinterpret_cast<volatile int *>(tokens_h)[t] = toks[t];
        const auto tp_ = std::chrono::steady_clock::now();
        ple_rows(toks, T);
        t_ple_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp_).count();
        if (probe) {
            if (multi()) throw std::runtime_error("probes need one GPU");
            enqueue_window(T, true);
        } else if (!multi()) {
            if (!graph[T]) capture(graph[T], [&] { enqueue_window(T, false); });
            const auto tl_ = std::chrono::steady_clock::now();
            ck(cudaGraphLaunch(graph[T], st), "graph launch");
            t_launch_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tl_).count();
            for (int il = 0; il < c.n_layer; ++il) service(il);
        } else {   // a stage's window starts when the previous one's has handed its residual over
            for (int s = 0; s <= last(); ++s) {
                bind(s);
                if (!graph[T]) capture(graph[T], [&] { enqueue_window(T, false); });
                const auto tl_ = std::chrono::steady_clock::now();
                if (s) ck(cudaStreamWaitEvent(st, stage(s - 1).ev_hand, 0), "wait");
                ck(cudaGraphLaunch(graph[T], st), "graph launch");
                if (s < last()) ck(cudaEventRecord(ev_hand, st), "record");
                t_launch_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tl_).count();
            }
            for (int il = 0; il < c.n_layer; ++il) {
                bind(layer_stage[il]);
                service(il);
            }
            bind(last());
        }
        if (want) ck(cudaMemcpyAsync(logits_h, logits_d, sizeof(float) * T * c.n_vocab, cudaMemcpyDeviceToHost, st), "logits");
        ck(cudaStreamSynchronize(st), "sync");
        if (stage_prof && !probe) add_stamps(T);
        ids_out.assign(argmax_h, argmax_h + T);
        logits.clear();
        if (want) logits.assign(logits_h, logits_h + static_cast<size_t>(T) * c.n_vocab);
        win_tokens.assign(toks, toks + T);
        win_T = T;
        ++windows;
        t_token_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
    }

    void commit(int n) {
        if (n < 1 || n > win_T) throw std::runtime_error("commit: " + std::to_string(n) + " of a window of " + std::to_string(win_T));
        *reinterpret_cast<volatile int *>(ncommit_h) = n;
        for (int s = 0; s <= last(); ++s) {   // every stage: ordered before its next work on its stream, no sync
            bind(s);
            if (!commit_graph) capture(commit_graph, [&] { enqueue_commit(); });
            ck(cudaGraphLaunch(commit_graph, st), "commit");
        }
        bind(last());
        for (int t = 0; t < n; ++t) history.push_back(win_tokens[t]);
        pos += n;
        steps += n;
        win_T = 0;
        if (adapt_every > 0 && steps % adapt_every < static_cast<long long>(n)) {
            ck(cudaStreamSynchronize(st), "sync");   // the swaps rewrite the table
            adapt();
        }
    }
};

namespace {
// The pipeline's spin-wait kernels wait for kernels on another stream (and for the host). With lazy module loading
// (CUDA 12+'s default) the first launch of a kernel waits for the running ones to finish: a deadlock. Load every
// kernel up front; this must happen before the first CUDA call creates the context.
const bool kEagerModules = [] { setenv("CUDA_MODULE_LOADING", "EAGER", 0); return true; }();
}  // namespace

GpuSplit parse_gpu_split(const std::string & gpus, const std::string & layer_split) {
    auto ints = [](const std::string & v, const char * what) {
        std::vector<int> r;
        std::stringstream ss(v);
        for (std::string x; std::getline(ss, x, ',');) {
            try { r.push_back(std::stoi(x)); }
            catch (...) { throw std::runtime_error(std::string(what) + ": not a list of numbers: " + v); }
        }
        return r;
    };
    GpuSplit g;
    if (!gpus.empty()) g.gpus = ints(gpus, "--gpus");
    if (!layer_split.empty() && layer_split != "auto") g.first_layer = ints(layer_split, "--layer-split");
    if (!g.first_layer.empty() && g.first_layer.size() + 1 != g.gpus.size())
        throw std::runtime_error("--layer-split: give one first layer per GPU after the first (--gpus lists " +
                                 std::to_string(g.gpus.size()) + ")");
    return g;
}

Engine::Engine(const std::string & shard1, int max_ctx, const std::string & cache_file, const GpuSplit & split)
    : p_(std::make_unique<Impl>(shard1, max_ctx, split)) {
    p_->cache_file = cache_file;
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr || std::string(std::getenv("CUDA_MODULE_LOADING")) != "EAGER")
        std::fprintf(stderr, "warning: CUDA_MODULE_LOADING is not EAGER: the expert pipeline may deadlock\n");
    (void) kEagerModules;
    p_->load();
}

Engine::~Engine() {
    if (!p_) return;
    auto & I = *p_;
    for (int s = 0; s <= I.last(); ++s) {
        I.bind(s);
        if (I.st) cudaStreamSynchronize(I.st);
        for (auto & gx : I.graph) if (gx) cudaGraphExecDestroy(gx);
        if (I.commit_graph) cudaGraphExecDestroy(I.commit_graph);
    }
    if (I.last() >= 0) I.bind(I.last());
    for (auto & row : I.mtp_graph) for (auto & gx : row) if (gx) cudaGraphExecDestroy(gx);
    I.cpu.reset();
    if (I.route_trace) std::fclose(I.route_trace);
    for (void * a : I.allocs) cudaFree(a);
    for (void * a : I.host_allocs) cudaFreeHost(a);
    for (auto [r, len] : I.host_regs) cudaHostUnregister(r);
    if (I.host_map_bytes) ::munmap(I.host_store, I.host_map_bytes);
    for (int s = 0; s <= I.last(); ++s) {
        I.bind(s);
        if (I.st) cudaStreamDestroy(I.st);
        if (I.cp) cudaStreamDestroy(I.cp);
        if (I.adm) cudaStreamDestroy(I.adm);
    }
}

const ModelConfig & Engine::config() const { return p_->c; }

std::string Engine::report() const {
    const auto & I = *p_;
    char buf[1024];
    const long long n = I.n_hit + I.n_dma + I.n_cpu;
    const double per_w = I.windows ? 1.0 / I.windows : 0.0;
    const int len = std::snprintf(
        buf, sizeof buf,
        "experts: %zu cached (%.2f GB); hit rate %.1f%% (%lld hits, %lld over PCIe, %lld on the CPU)\n"
        "%lld windows, %lld tokens: %.2f ms per window; host: %.2f ms PLE rows, %.2f ms graph launch, %.2f ms waiting for doorbells, %.2f ms computing CPU experts; "
        "LRU admissions %lld; swaps %lld; prefetch %lld issued, %lld used",
        I.cached, I.cached_bytes / 1e9, n ? 100.0 * I.n_hit / n : 0.0, I.n_hit, I.n_dma, I.n_cpu, I.windows, I.steps,
        I.t_token_ms * per_w, I.t_ple_ms * per_w, I.t_launch_ms * per_w, I.t_bell_ms * per_w, I.t_cpu_ms * per_w, I.n_admit, I.n_swaps, I.n_pf_issued, I.n_pf_used);
    std::string r(buf, len);
    {   // the GPU's side: how long it spun waiting for the host's experts (every GPU's)
        double w[4] = {};
        for (int d : I.gpu_list()) {
            double x[4];
            cudaSetDevice(d);
            cuda::wait_times(x, false);
            for (int k = 0; k < 4; ++k) w[k] += x[k];
        }
        cudaSetDevice(I.dev);
        char b2[160];
        const int l3 = std::snprintf(b2, sizeof b2, "\nGPU waiting per window: %.2f ms for CPU experts, %.2f ms for PCIe copies, %.2f ms for prefetches",
                                     w[cuda::kCpu] * per_w, w[cuda::kDma] * per_w, w[cuda::kPf] * per_w);
        r.append(b2, l3);
    }
    {
        r += "\nVRAM free now:";
        for (int d : I.gpu_list()) {
            size_t fr = 0, to = 0;
            cudaSetDevice(d);
            cudaMemGetInfo(&fr, &to);
            char b5[64];
            r.append(b5, I.multi() ? std::snprintf(b5, sizeof b5, " GPU %d %.2f GB", d, fr / 1e9)
                                   : std::snprintf(b5, sizeof b5, " %.2f GB", fr / 1e9));
        }
        cudaSetDevice(I.dev);
    }
    if (I.stage_windows > 0) {   // BL_STAGE_PROF: GPU time per window by stage, each stamp's own cost taken off
        static const char * names[] = {"", "", "embed", "hc mix", "GDN attn", "QSA attn", "route+plan", "hit experts",
                                       "shared expert", "wait (CPU)", "combine", "head"};
        const double cal = I.stage_stamps[Impl::kStCal] ? I.stage_ms[Impl::kStCal] / I.stage_stamps[Impl::kStCal] : 0.0;
        const double pw = 1.0 / I.stage_windows;
        double tot = 0, stamps = 0;
        std::string s = "\nGPU stages per window (ms):";
        for (int k = Impl::kStEmbed; k < Impl::kStN; ++k) {
            const double ms = (I.stage_ms[k] - cal * I.stage_stamps[k]) * pw;
            tot += ms;
            stamps += cal * I.stage_stamps[k] * pw;
            char b3[64];
            s.append(b3, std::snprintf(b3, sizeof b3, " %s %.2f,", names[k], ms));
        }
        char b4[128];
        s.append(b4, std::snprintf(b4, sizeof b4, " total %.2f (+%.2f stamps, %.1f us each)", tot, stamps, cal * 1e3));
        r += s;
    }
    if (I.t_prefill_ms > 0) {
        const int l2 = std::snprintf(buf, sizeof buf, "\nprefill: %.1f ms, %lld experts copied in, %lld admitted to the cache",
                                     I.t_prefill_ms, I.n_pf_copied, I.n_pf_admit);
        r.append(buf, l2);
    }
    if (I.n_rounds) {
        const int l2 = std::snprintf(buf, sizeof buf,
            "\nMTP: %lld rounds, %.2f tokens per round; %lld drafts, %lld accepted (%.1f%%); %lld MTP runs, %.2f ms each",
            I.n_rounds, static_cast<double>(I.n_rounds + I.n_accepted) / I.n_rounds, I.n_drafted, I.n_accepted,
            I.n_drafted ? 100.0 * I.n_accepted / I.n_drafted : 0.0, I.n_mtp_runs,
            I.n_mtp_runs ? I.t_draft_ms / I.n_mtp_runs : 0.0);
        r.append(buf, l2);
    }
    return r;
}

int Engine::position() const { return p_->pos; }
void Engine::save_cache(const std::string & path) const { p_->save_cache(path); }
void Engine::set_probe(Probe p) { p_->probe = std::move(p); }

void Engine::set(const std::string & key, double v) {
    auto & I = *p_;
    I.enter();
    auto drop_graphs = [&] {   // a kernel argument changed: capture again
        for (int s = 0; s <= I.last(); ++s) {
            I.bind(s);
            for (auto & gx : I.graph) if (gx) { cudaGraphExecDestroy(gx); gx = nullptr; }
        }
        I.bind(I.last());
        for (auto & row : I.mtp_graph) for (auto & gx : row) if (gx) { cudaGraphExecDestroy(gx); gx = nullptr; }
    };
    if (key == "cpu_share") { I.cpu_share = static_cast<float>(v); drop_graphs(); }
    else if (key == "cpu_share_win") { I.cpu_share_win = static_cast<float>(v); drop_graphs(); }
    else if (key == "prefetch") { I.pf_max = std::min(static_cast<int>(v), cuda::kMaxPf); drop_graphs(); }
    else if (key == "adapt_every") I.adapt_every = static_cast<int>(v);
    else if (key == "adapt_max") I.adapt_max = std::min(static_cast<int>(v), I.adapt_max);   // the bounce buffer's size
    else if (key == "adapt_decay") I.adapt_decay = static_cast<float>(v);
    else if (key == "adapt_margin") I.adapt_margin = static_cast<float>(v);
    else if (key == "lru") I.lru = I.ram_all && v != 0;
    else if (key == "lru_max") I.lru_max = static_cast<int>(v);
    else if (key == "mtp_window") { I.mtp_window = static_cast<int>(v); drop_graphs(); }
    else throw std::runtime_error("unknown engine setting " + key);
}

void Engine::reset_stats() {
    auto & I = *p_;
    I.n_hit = I.n_dma = I.n_cpu = I.n_swaps = I.n_pf_issued = I.n_pf_used = I.n_admit = 0;
    for (int d : I.gpu_list()) {
        double w[4];
        cudaSetDevice(d);
        cuda::wait_times(w, true);
    }
    cudaSetDevice(I.dev);
    std::fill(std::begin(I.stage_ms), std::end(I.stage_ms), 0.0);
    std::fill(std::begin(I.stage_stamps), std::end(I.stage_stamps), 0LL);
    I.stage_windows = 0;
    I.t_token_ms = I.t_bell_ms = I.t_cpu_ms = I.t_adapt_ms = I.t_draft_ms = I.t_ple_ms = I.t_launch_ms = 0;
    I.n_mtp_runs = I.n_rounds = I.n_drafted = I.n_accepted = 0;
    I.t_prefill_ms = 0;
    I.n_pf_copied = 0;
    I.n_pf_admit = 0;
    I.steps = I.windows = 0;
}

void Engine::reset() {
    auto & I = *p_;
    I.enter();
    for (int s = 0; s <= I.last(); ++s) {   // a commit may still be queued (commit does not wait)
        I.bind(s);
        ck(cudaStreamSynchronize(I.st), "sync");
    }
    I.pending = -1;
    const ModelConfig & c = I.c;
    const size_t conv = static_cast<size_t>(2 * c.ssm_groups * c.ssm_state + c.ssm_inner) * (c.ssm_conv - 1);
    const size_t ssm  = static_cast<size_t>(c.ssm_v_heads) * c.ssm_state * c.ssm_state;
    for (int il = 0; il < c.n_layer; ++il) {   // (each on its stage's GPU)
        I.bind(I.layer_stage[il]);
        if (I.conv_state[il]) ck(cudaMemset(I.conv_state[il], 0, conv * 4), "reset");
        if (I.gdn_state[il]) ck(cudaMemset(I.gdn_state[il], 0, ssm * 4), "reset");
    }
    I.bind(I.layer_stage[c.ple_layer]);
    ck(cudaMemset(I.ple_hist, 0, static_cast<size_t>(c.ple_conv - 1) * c.ple_ngram * c.hc_dim() * 4), "reset");
    I.bind(I.last());
    I.pos = 0;   // the KV cells past pos are never read
    I.history.clear();
    I.win_T = 0;
    I.reset_token_state();
}

const std::vector<int> & Engine::verify(const int * tokens, int T, bool want_logits) {
    p_->enter();
    p_->verify(tokens, T, want_logits);
    return p_->ids_out;
}
const std::vector<float> & Engine::logits() const { return p_->logits; }
void Engine::commit(int n) { p_->enter(); p_->commit(n); }

bool Engine::has_mtp() const { return p_->mtp; }
int Engine::pending() const { return p_->pending; }
std::vector<float> Engine::prefill_logits(const std::vector<int> & tokens) {
    std::vector<float> out;   // the last chunk's rows
    auto & I = *p_;
    I.enter();
    const int n = static_cast<int>(tokens.size());
    for (int off = 0; off < n; off += I.pf_cap) {
        const int m = std::min(I.pf_cap, n - off);
        const bool last = off + m == n;
        I.pf_all_logits = last ? &out : nullptr;
        I.prefill(tokens.data() + off, m, false, last ? -1 : tokens[off + m]);
    }
    I.pf_all_logits = nullptr;
    return out;
}
std::vector<int> Engine::generate(const std::vector<int> & prompt, int max_new, int spec, float min_p,
                                  const std::function<bool(int)> & on_token, const Sampling * sampling) {
    p_->enter();
    return p_->generate(prompt, max_new, spec, min_p, on_token, sampling);
}

const std::vector<float> & Engine::step(int token, bool want_logits) {
    p_->enter();
    p_->verify(&token, 1, want_logits);
    p_->commit(1);
    return p_->logits;
}

}  // namespace bl
