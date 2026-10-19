cmake_minimum_required(VERSION 3.24)
project(nano-cuda-moe-qwen38 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(BL_ENABLE_CUDA "Build the CUDA engine (off: only the GGUF inspector)" ON)

# ---- llama.cpp (MIT), pinned: ggml-cpu's dot products compute the experts that run on the CPU, libllama is the
# tokenizer (vocab only), and its vendor/ tree has cpp-httplib and nlohmann/json for the server. Built here as a
# static, CPU-only subproject. -DBL_LLAMA_SRC=<dir> uses a local checkout of the same commit instead of the download.
set(BL_LLAMA_COMMIT 3cf03257f219afbe7334045ff7c6a06ac68c627d)
set(BL_LLAMA_SRC "" CACHE PATH "a local llama.cpp source tree at commit ${BL_LLAMA_COMMIT} (default: download it)")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
include(FetchContent)
if(BL_LLAMA_SRC)
  FetchContent_Declare(llama SOURCE_DIR ${BL_LLAMA_SRC})
else()
  FetchContent_Declare(llama
    URL https://github.com/ggml-org/llama.cpp/archive/${BL_LLAMA_COMMIT}.tar.gz
    URL_HASH SHA256=c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
endif()
FetchContent_MakeAvailable(llama)
set(BL_VENDOR ${llama_SOURCE_DIR}/vendor)

# Our own targets: tuned for the machine that builds them (AVX2 at least for the CPU experts)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native> $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

add_library(bl_core STATIC src/gguf.cpp src/config.cpp)
target_include_directories(bl_core PUBLIC include)

add_executable(bl-inspect tools/bl_inspect.cpp)
target_link_libraries(bl-inspect PRIVATE bl_core)

# text: the tokenizer (libllama, vocab only) and the chat template
add_library(bl_text STATIC src/text.cpp)
target_include_directories(bl_text PUBLIC include)
target_link_libraries(bl_text PUBLIC llama ggml)
add_executable(bl-tok tools/bl_tok.cpp)
target_link_libraries(bl-tok PRIVATE bl_text)

# the MTP layer from the official checkpoint's tensors (tools/fetch_mtp.py) -> GGUF
add_executable(bl-mtp-pack tools/mtp_pack.cpp)
target_include_directories(bl-mtp-pack PRIVATE ${BL_VENDOR})
target_link_libraries(bl-mtp-pack PRIVATE ggml-base pthread)

if(BL_ENABLE_CUDA)
  # tested on sm_86 (RTX 3090); the kernels need sm_80 or newer (mma.sync, cp.async). Other cards:
  # -DCMAKE_CUDA_ARCHITECTURES=native
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
  set(CMAKE_CUDA_STANDARD 20)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_library(bl_cuda STATIC src/cuda/quant.cu src/cuda/ops.cu src/cuda/experts.cu src/cuda/fused.cu src/cuda/pipeline.cu
              src/cuda/window.cu src/cuda/prefill.cu src/cuda/sample.cu src/engine.cpp src/cpu_experts.cpp src/cpu_q2.cpp
              src/cpu_iq.cpp)
  target_include_directories(bl_cuda PUBLIC include PRIVATE third_party/ggml)
  target_link_libraries(bl_cuda PUBLIC CUDA::cudart CUDA::cublas bl_core ggml-cpu ggml-base)

  # the programs
  add_library(bl_chat STATIC src/chat.cpp src/chat_models.cpp)
  target_link_libraries(bl_chat PUBLIC bl_cuda bl_text bl_dsv41)
  add_executable(bl-chat tools/bl_chat.cpp)
  target_link_libraries(bl-chat PRIVATE bl_chat)
  add_library(bl_httplib STATIC ${BL_VENDOR}/cpp-httplib/httplib.cpp)
  target_include_directories(bl_httplib PUBLIC ${BL_VENDOR}/cpp-httplib ${BL_VENDOR})
  target_compile_options(bl_httplib PRIVATE -w)
  add_executable(bl-server tools/bl_server.cpp)
  target_link_libraries(bl-server PRIVATE bl_chat bl_httplib pthread)
  add_executable(bl-run tools/bl_run.cpp)
  target_link_libraries(bl-run PRIVATE bl_cuda)

  # tests and benchmarks
  foreach(t test_window test_session)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE bl_cuda)
  endforeach()
  foreach(t test_sample test_mmq test_quant bench_dense bench_idx bench_experts)
    add_executable(${t} tests/${t}.cu)
    target_link_libraries(${t} PRIVATE bl_cuda)
  endforeach()
  add_executable(bench_cpu tests/bench_cpu.cpp)
  target_link_libraries(bench_cpu PRIVATE bl_cuda)

  # the reference dumper: llama.cpp's own forward pass, layer by layer (CPU-only here; slow on the full model)
  add_executable(bl-ref-dump tools/ref_dump.cpp)
  target_link_libraries(bl-ref-dump PRIVATE llama ggml)

  # DeepSeek-V4.1-Flash (deepseek41), phase A: docs/DSV41.md
  add_library(bl_dsv41 STATIC src/dsv41/config.cpp src/dsv41/ops.cu src/dsv41/engine.cpp src/dsv41/ram_tier.cpp)
  target_include_directories(bl_dsv41 PUBLIC include)
  target_link_libraries(bl_dsv41 PUBLIC bl_cuda)
  add_executable(bl-dsv41 tools/bl_dsv41.cpp)
  target_link_libraries(bl-dsv41 PRIVATE bl_dsv41)
  foreach(t test_dsv41_ops bench_dsv41_experts bench_dsv41_dense)
    add_executable(${t} tests/${t}.cu)
    target_link_libraries(${t} PRIVATE bl_dsv41)
  endforeach()
  add_executable(bench_dsv41_ram tests/bench_dsv41_ram.cpp)
  target_link_libraries(bench_dsv41_ram PRIVATE bl_dsv41)
  add_executable(bench_dsv41_cpu tests/bench_dsv41_cpu.cpp)
  target_link_libraries(bench_dsv41_cpu PRIVATE bl_dsv41)
endif()
