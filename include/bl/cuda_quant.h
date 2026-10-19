// Quantized weight kernels (CUDA). A weight matrix is ggml-shaped: M rows of K values, each row a run of blocks.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace bl::cuda {

bool supported(uint32_t type_id);
// a routed expert format the GPU's expert kernels (decode windows and prefill) handle
bool expert_supported(uint32_t type_id);

// out[r*K + k] = W[r][k] as float, rows [0, M)
void dequant_rows(uint32_t type_id, const void * W, size_t row_bytes, int M, int K, float * out, cudaStream_t s);

// the same as bf16 (round to nearest even); K % 8 == 0
void dequant_rows_bf16(uint32_t type_id, const void * W, size_t row_bytes, int M, int K, void * out, cudaStream_t s);

// y[r] = sum_k W[r][k] * x[k]   (one token; fp32 activations and accumulation)
void matvec(uint32_t type_id, const void * W, size_t row_bytes, int M, int K, const float * x, float * y,
            cudaStream_t s);
// a block-diagonal matrix in one launch: G groups of M rows; group g reads x[g*K ..) and writes y[g*M ..)
void matvec_blocks(uint32_t type_id, const void * W, size_t row_bytes, int M, int K, int G, const float * x, float * y,
                   cudaStream_t s);

}  // namespace bl::cuda
