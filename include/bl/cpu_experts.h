// Routed experts computed on the CPU, in place in (pinned) RAM, while the GPU works on the cached ones.
// The dot products are ggml-cpu's (AVX2 for every i-quant); threading and scheduling are ours.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace bl {

class CpuExperts {
public:
    explicit CpuExperts(int n_threads);   // n_threads includes the calling thread
    ~CpuExperts();
    CpuExperts(const CpuExperts &) = delete;
    CpuExperts & operator=(const CpuExperts &) = delete;

    struct Job {
        const uint8_t * base;   // the expert's [gate | up | down] in host memory
        int             mask;   // bit t: token t of the window uses it
        float *         out;    // [T][D]: rows of the tokens in mask are written
    };
    // out[t] = down(silu(gate . x[t]) * (up . x[t])) for each job and each token in its mask; x is [T][D];
    // blocks until all are done. swiglu_limit > 0 clamps first: gate . x at most the limit, up . x within +-limit
    // (DeepSeek-V4's experts)
    void run(uint32_t type_gu, uint32_t type_d, size_t gu_bytes, int F, int D, const float * x, int T, const Job * jobs, int n,
             float swiglu_limit = 0.f);

    int threads() const;

    static bool supports(uint32_t type);           // ggml-cpu has the dot product for this expert format
    static const char * type_name(uint32_t type);  // ggml's name ("iq3_xxs")

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace bl
