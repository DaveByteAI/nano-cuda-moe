#include "bl/cpu_experts.h"
#include "bl/cpu_iq.h"
#include "bl/cpu_q2.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"

namespace bl {

struct CpuExperts::Impl {
    int n_threads;
    std::vector<std::thread> workers;

    // One "phase" = a list of tasks every thread pulls from; the caller waits until all are done. The phase id and
    // the next task index share one 64-bit word, and a task is claimed by compare-exchange only while the id still
    // matches: a thread late from an earlier phase can neither run nor use up a task of the current one.
    std::atomic<uint64_t> ticket{0};   // phase id << 32 | next task index
    std::atomic<int>      done{0};
    std::atomic<bool>     quit{false};
    std::atomic<int>      sleepers{0};   // workers parked in ticket.wait (idle for a while)
    static constexpr int  kSpins = 200000;   // pause()s before an idle worker parks (~ a few hundred us)
    uint32_t              phase_id = 0;
    int                   n_tasks = 0;
    void (*fn)(Impl *, int task) = nullptr;

    // the batch being computed
    const ggml_type_traits_cpu * tr_gu = nullptr, * tr_d = nullptr;
    size_t gu_bytes = 0, gu_rb = 0, d_rb = 0, xq_rb = 0, aq_rb = 0;
    int F = 0, D = 0, n = 0, T = 1;
    bool q2_down = false;            // down is Q2_0: our AVX2 kernel (ggml's is scalar on x86)
    float limit = 0.f;               // > 0: DeepSeek-V4's clamps, gate at most limit, up within +-limit
    float swiglu(float g, float u) const {
        if (limit > 0.f) { g = std::fmin(g, limit); u = std::fmin(std::fmax(u, -limit), limit); }
        return g / (1.f + std::exp(-g)) * u;
    }
    uint32_t type_gu = 0;
    bool iq_gu = false;              // gate/up through cpu_iq (all of a job's tokens per row) instead of ggml's vec_dot
    const float * x = nullptr;
    const Job * jobs = nullptr;
    std::vector<uint8_t> xq, aq;     // x quantized for the gate/up dots [T]; each job's act quantized for down [n][T]
    std::vector<float>   act;        // [n][T][F]
    static constexpr int kGuRows = 32, kDownRows = 64;
    bool   prof = std::getenv("BL_CPU_PROFILE") != nullptr;
    double t_ph[4] = {};   // x quantize, gate/up, act quantize, down (us)
    long   calls = 0;
    // BL_CPU_PROFILE: time inside tasks, summed over threads, per phase (utilization = busy / (wall * threads))
    std::atomic<long long> busy_ns[4] = {};
    int cur_ph = 0;

    explicit Impl(int nt) : n_threads(nt) {
        ggml_cpu_init();
        // one worker per physical core (cpus 0..7 are the cores, 8..15 their hyper-threads on this machine);
        // BL_CPU_PIN=0 leaves placement to the OS
        const char * pin = std::getenv("BL_CPU_PIN");
        const bool do_pin = !pin || std::atoi(pin) != 0;
        for (int i = 1; i < n_threads; ++i) {
            workers.emplace_back([this] { loop(); });
            if (do_pin) {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(i % static_cast<int>(std::thread::hardware_concurrency()), &set);
                pthread_setaffinity_np(workers.back().native_handle(), sizeof set, &set);
            }
        }
    }
    ~Impl() {
        if (prof && calls) {
            std::fprintf(stderr, "cpu experts per call (us): quantize x %.1f, gate/up %.1f, quantize act %.1f, down %.1f\n",
                         t_ph[0] / calls, t_ph[1] / calls, t_ph[2] / calls, t_ph[3] / calls);
            std::fprintf(stderr, "cpu experts thread utilization: gate/up %.0f%%, down %.0f%% (%d threads)\n",
                         100.0 * busy_ns[1] / (t_ph[1] * 1e3 * n_threads), 100.0 * busy_ns[3] / (t_ph[3] * 1e3 * n_threads), n_threads);
        }
        quit = true;
        ticket.fetch_add(uint64_t{1} << 32);   // a new phase id wakes the workers, which then see quit
        ticket.notify_all();
        for (auto & t : workers) t.join();
    }

    // runs tasks of phase `id` until none are left (or the phase is over)
    void work(uint32_t id, int n_t, void (*f)(Impl *, int)) {
        uint64_t v = ticket.load(std::memory_order_acquire);
        while (true) {
            if (static_cast<uint32_t>(v >> 32) != id || static_cast<int>(v & 0xffffffffu) >= n_t) return;
            if (!ticket.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel)) continue;
            if (prof) {
                const auto a = std::chrono::steady_clock::now();
                f(this, static_cast<int>(v & 0xffffffffu));
                busy_ns[cur_ph].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - a).count(),
                                          std::memory_order_relaxed);
            } else {
                f(this, static_cast<int>(v & 0xffffffffu));
            }
            done.fetch_add(1, std::memory_order_release);
            v = ticket.load(std::memory_order_acquire);
        }
    }
    void loop() {
        uint32_t seen = 0;
        while (true) {
            uint64_t v;
            int spins = 0;
            while (static_cast<uint32_t>((v = ticket.load(std::memory_order_acquire)) >> 32) == seen) {
                if (++spins < kSpins) { _mm_pause(); continue; }
                // idle: park until the ticket changes (a phase() that saw no sleeper stored it before we checked)
                sleepers.fetch_add(1, std::memory_order_acq_rel);
                ticket.wait(v, std::memory_order_acquire);
                sleepers.fetch_sub(1, std::memory_order_acq_rel);
                spins = 0;
            }
            seen = static_cast<uint32_t>(v >> 32);
            if (quit) return;
            work(seen, n_tasks, fn);   // n_tasks/fn were written before the ticket that announced the phase
        }
    }
    void phase(int tasks, void (*f)(Impl *, int)) {
        n_tasks = tasks;
        fn = f;
        done.store(0, std::memory_order_relaxed);
        ++phase_id;
        ticket.store(static_cast<uint64_t>(phase_id) << 32, std::memory_order_seq_cst);
        if (sleepers.load(std::memory_order_seq_cst) > 0) ticket.notify_all();
        work(phase_id, tasks, f);
        while (done.load(std::memory_order_acquire) < tasks) _mm_pause();
    }

    static void quant_x_task(Impl * I, int t) {   // token t's activations as the gate/up dots want them
        ggml_get_type_traits_cpu(I->tr_gu->vec_dot_type)->from_float(I->x + static_cast<size_t>(t) * I->D, I->xq.data() + t * I->xq_rb, I->D);
    }
    // rows r0.. of job e's gate and up, for every token in its mask (the rows stay in cache across the tokens)
    static void gate_up_task(Impl * I, int task) {
        const int per = I->F / kGuRows, e = task / per, r0 = (task % per) * kGuRows;
        const uint8_t * g = I->jobs[e].base, * u = g + I->gu_bytes;
        const int mask = I->jobs[e].mask;
        if (I->iq_gu && __builtin_popcount(static_cast<unsigned>(mask)) >= 2) {   // each row decoded once for all the job's tokens
            // (one token: ggml's vec_dot below is as fast or faster)
            const void * xs[cpu::kIqMaxTokens];
            int ts[cpu::kIqMaxTokens], nt = 0;
            for (int t = 0; t < I->T && nt < cpu::kIqMaxTokens; ++t)
                if ((mask >> t) & 1) { xs[nt] = I->xq.data() + t * I->xq_rb; ts[nt++] = t; }
            float gv[cpu::kIqMaxTokens], uv[cpu::kIqMaxTokens];
            for (int r = r0; r < r0 + kGuRows; ++r) {
                cpu::iq_gate_up(I->type_gu, g + r * I->gu_rb, u + r * I->gu_rb, I->D, xs, nt, gv, uv);
                for (int k = 0; k < nt; ++k)
                    I->act[(static_cast<size_t>(e) * I->T + ts[k]) * I->F + r] = I->swiglu(gv[k], uv[k]);
            }
            return;
        }
        for (int r = r0; r < r0 + kGuRows; ++r)
            for (int t = 0; t < I->T; ++t) {
                if (!((mask >> t) & 1)) continue;
                const void * xq = I->xq.data() + t * I->xq_rb;
                float gv, uv;
                I->tr_gu->vec_dot(I->D, &gv, 0, g + r * I->gu_rb, 0, xq, 0, 1);
                I->tr_gu->vec_dot(I->D, &uv, 0, u + r * I->gu_rb, 0, xq, 0, 1);
                I->act[(static_cast<size_t>(e) * I->T + t) * I->F + r] = I->swiglu(gv, uv);
            }
    }
    static void quant_act_task(Impl * I, int task) {   // task = job * T + token
        const int e = task / I->T, t = task % I->T;
        if (!((I->jobs[e].mask >> t) & 1)) return;
        const float * a = I->act.data() + (static_cast<size_t>(e) * I->T + t) * I->F;
        uint8_t * dst = I->aq.data() + static_cast<size_t>(task) * I->aq_rb;
        if (I->q2_down) { cpu::q8planes_quantize(a, I->F, dst); return; }
        ggml_get_type_traits_cpu(I->tr_d->vec_dot_type)->from_float(a, dst, I->F);
    }
    static void down_task(Impl * I, int task) {
        const int per = I->D / kDownRows, e = task / per, r0 = (task % per) * kDownRows;
        const uint8_t * d = I->jobs[e].base + 2 * I->gu_bytes;
        const int mask = I->jobs[e].mask;
        for (int t = 0; t < I->T; ++t) {
            if (!((mask >> t) & 1)) continue;
            const void * aq = I->aq.data() + (static_cast<size_t>(e) * I->T + t) * I->aq_rb;
            float * out = I->jobs[e].out + static_cast<size_t>(t) * I->D;
            if (I->q2_down) {
                for (int r = r0; r < r0 + kDownRows; ++r) out[r] = cpu::q2_0_dot(d + r * I->d_rb, aq, I->F);
            } else {
                for (int r = r0; r < r0 + kDownRows; ++r) I->tr_d->vec_dot(I->F, out + r, 0, d + r * I->d_rb, 0, aq, 0, 1);
            }
        }
    }
};

CpuExperts::CpuExperts(int n_threads) : p_(std::make_unique<Impl>(n_threads)) {}
CpuExperts::~CpuExperts() = default;
int CpuExperts::threads() const { return p_->n_threads; }

bool CpuExperts::supports(uint32_t type) {
    if (type >= GGML_TYPE_COUNT) return false;
    const ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(static_cast<ggml_type>(type));
    return tr && tr->vec_dot && ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float;
}

const char * CpuExperts::type_name(uint32_t type) {
    return type < GGML_TYPE_COUNT ? ggml_type_name(static_cast<ggml_type>(type)) : "unknown";
}

void CpuExperts::run(uint32_t type_gu, uint32_t type_d, size_t gu_bytes, int F, int D, const float * x, int T, const Job * jobs,
                     int n, float swiglu_limit) {
    if (n == 0) return;
    Impl & I = *p_;
    I.limit = swiglu_limit;
    const auto tg = static_cast<ggml_type>(type_gu), td = static_cast<ggml_type>(type_d);
    I.tr_gu = ggml_get_type_traits_cpu(tg);
    I.tr_d  = ggml_get_type_traits_cpu(td);
    if (!I.tr_gu->vec_dot || !I.tr_d->vec_dot) throw std::runtime_error("ggml-cpu has no vec_dot for an expert format");
    if (F % Impl::kGuRows || D % Impl::kDownRows) throw std::runtime_error("expert shape not a multiple of the CPU tiles");
    I.gu_bytes = gu_bytes;
    I.F = F;
    I.D = D;
    I.n = n;
    I.T = T;
    I.jobs = jobs;
    I.gu_rb = ggml_row_size(tg, D);
    I.d_rb  = ggml_row_size(td, F);
    I.xq_rb = ggml_row_size(I.tr_gu->vec_dot_type, D);
    I.q2_down = td == GGML_TYPE_Q2_0;
    I.type_gu = type_gu;
    static const bool iq_off = std::getenv("BL_CPU_IQ") && std::atoi(std::getenv("BL_CPU_IQ")) == 0;
    I.iq_gu = !iq_off && cpu::iq_supported(type_gu) && T <= cpu::kIqMaxTokens && D % 256 == 0;
    I.x = x;
    I.aq_rb = I.q2_down ? cpu::q8planes_bytes(F) : ggml_row_size(I.tr_d->vec_dot_type, F);
    I.xq.resize(I.xq_rb * T);
    I.aq.resize(I.aq_rb * n * T);
    I.act.resize(static_cast<size_t>(n) * T * F);

    using clk = std::chrono::steady_clock;
    auto us = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
    const auto t0 = clk::now();
    if (T > 1) I.phase(T, &Impl::quant_x_task);   // the window's tokens, one per thread
    else Impl::quant_x_task(&I, 0);
    const auto t1 = clk::now();
    I.cur_ph = 1;
    I.phase(n * (F / Impl::kGuRows), &Impl::gate_up_task);
    const auto t2 = clk::now();
    I.cur_ph = 2;
    I.phase(n * T, &Impl::quant_act_task);
    const auto t3 = clk::now();
    I.cur_ph = 3;
    I.phase(n * (D / Impl::kDownRows), &Impl::down_task);
    I.cur_ph = 0;
    if (I.prof) {
        const auto t4 = clk::now();
        I.t_ph[0] += us(t0, t1); I.t_ph[1] += us(t1, t2); I.t_ph[2] += us(t2, t3); I.t_ph[3] += us(t3, t4);
        ++I.calls;
    }
}

}  // namespace bl
