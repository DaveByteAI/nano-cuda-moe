<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-en-dark.svg">
  <img alt="nano-cuda-moe: Qwen3.8-Flash-Next, a 125B-parameter MoE with 6B active, on one 24 GB RTX 3090. About 97 tokens per second." src="docs/assets/hero-en-light.svg" width="100%">
</picture>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#deepseek-v41-flash">DeepSeek-V4.1-Flash</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

Run large Mixture-of-Experts models on **one 24 GB RTX 3090 and 64 GB of RAM**. The engine is C++ and CUDA written
for each model: one process, no Python or ML framework at inference time, an OpenAI-compatible server and a chat page
included. Two models so far; the same `bl-server` and `bl-chat` run both and pick the engine from the GGUF file:

| Model | Model file | Where the experts live | Decode on a 3090 + 64 GB | |
|---|---|---|---|---|
| [**Qwen3.8-Flash-Next**](#highlights) (125B, 6B active) | 76 GB, IQ3_XXS | VRAM + RAM | **~97 tok/s** | `main` and `dsv41` branches |
| [**DeepSeek-V4.1-Flash-1017**](#deepseek-v41-flash) | 359 GB, 2-bit experts (IQ2_XXS) | VRAM + RAM + NVMe | **8-12 tok/s** (~20 on repetitive output) | this `dsv41` branch |

DeepSeek-V4.1-Flash's file is five times the RAM: most of it streams from the NVMe while the model runs. On the same
machine and file, llama.cpp decodes it at about 1 token/s.

This is the CUDA counterpart of [**nano-metal-moe-qwen36**](https://github.com/DaveByteAI/nano-metal-moe-qwen36),
which runs Qwen3.6-35B-A3B on a 16 GB Mac mini with Metal. Both start from the same observation: a MoE model needs
all of its experts somewhere, but each token touches only a few, so the expensive memory only has to hold the
experts that are used most.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-en-dark.svg">
  <img alt="Architecture: the dense weights and a cache of the most used experts in the GPU's 24 GB; all 25,088 experts pinned in RAM, where the CPU computes the ones the GPU does not hold; the n-gram embedding table on the SSD." src="docs/assets/architecture-en-light.svg" width="100%">
</picture>

## Highlights

Qwen3.8-Flash-Next (the next sections up to [Usage](#usage); DeepSeek-V4.1 has [its own](#deepseek-v41-flash)):


- **125B model, 24 GB card.** The 42.9 GB of experts live in pinned RAM. The GPU keeps the dense weights (4.1 GB) and
  a per-layer LRU cache of about 10,000 of the 25,088 experts (~17.5 GB), which serves ~90% of the experts a token
  needs.
- **The CPU computes the misses where they are.** No PCIe round trip: while the GPU runs the cached experts of a
  layer, the CPU runs the missing ones from RAM (AVX2), and the GPU waits on a flag in mapped memory. A whole
  decode step is one CUDA graph.
- **Speculative decoding with the model's own MTP layer.** It drafts up to 3 tokens; one pass over the 48 layers
  checks them, about 2.5-3 tokens per pass. With sampling, the drafts go through rejection sampling on the GPU, so the
  output distribution is unchanged.
- **Long context, up to the model's full 256K.** The model's sparse attention (an indexer picks 2048 positions per
  query) with an int8 KV cache: a 128K-token prompt is read in 78 s, a 256K one in 165 s (~1,550 tokens/s).
- **Usable.** `bl-server` speaks the OpenAI chat API (streaming, reasoning content, multi-turn prefix reuse) and serves
  a chat page; `bl-chat` is the terminal version.

## Performance

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/performance-en-dark.svg">
  <img alt="Decode speed on a 7-text corpus: 97 tokens/s against 65 for Strata on the same machine and file; prompt processing about 1,600-1,800 tokens/s at 32K and 128K tokens." src="docs/assets/performance-en-light.svg" width="100%">
</picture>

RTX 3090 24 GB (PCIe Gen3), Core i7-10700 (8 cores, AVX2), 64 GB DDR4, Linux. Model: ISTA-DASLab's GSQ-RCO IQ3_XXS
GGUF. Greedy decoding, 3 MTP drafts per round.

| | |
|---|---|
| Decode, 7 short texts (chat, code, docs; 200 tokens each) | **97 tok/s** |
| Decode, chat in the web page (temperature 0.7) | 92-104 tok/s |
| Decode after a 32K / 128K-token prompt | 82 / 75 tok/s |
| Prompt processing, 32K / 128K tokens | 18.0 s / 78 s |
| A 256K-token prompt (`--ctx 262144`): read / decode after it | 165 s / 66 tok/s |
| (`--ctx 262144` keeps 3.9 GB for the KV cache: the expert cache drops from 17.5 to 14 GB, short chats decode 5-12% slower) | |
| First token for a short question | ~0.4 s |
| Startup (75.8 GB read from NVMe) | ~25 s |
| (The first long prompt after a start reads the n-gram table from the SSD: ~2 s slower at 32K) | |
| Accuracy against llama.cpp on the same GGUF (4K prompt) | median KL 0.003-0.005, top-1 90-93% |

Compared with [Strata](https://github.com/eddoursul/Strata), another engine for this model, on the same machine with
the same GGUF file and inputs: decode is **50% faster here (97 vs 65 tok/s)** on the corpus; Strata reads long prompts faster (the 32K prompt in 13.3 s against 18.0 s here), and decodes after it at about the same speed (85 vs 82 tok/s). The [optimization log](docs/LOG.md) (in Chinese) has every step and its measurement.

## Requirements

- An NVIDIA GPU with 24 GB, compute capability 8.0 or newer (developed and measured on an RTX 3090, sm_86)
- 64 GB of RAM (the experts are pinned: 43 GB), an x86-64 CPU with AVX2
- 80 GB of disk, ideally NVMe (the 28.8 GB n-gram table is read a few rows per token)
- Linux, CUDA 12 or 13, CMake 3.24+, GCC 11+ (C++20), Python 3 (standard library only, for the download scripts)

## Quick start

### 1. Build

```bash
git clone https://github.com/DaveByteAI/nano-cuda-moe
cd nano-cuda-moe
cmake -B build                      # downloads llama.cpp (pinned) for ggml-cpu and the tokenizer
cmake --build build -j
```

Not an RTX 30-series card? Add `-DCMAKE_CUDA_ARCHITECTURES=native`. A CUDA toolkit outside `PATH`:
`-DCMAKE_CUDA_COMPILER=/path/to/nvcc`.

### 2. Get the model

```bash
scripts/download_model.sh          # into models/: 75.8 GB of GGUF, then the MTP layer (~5.5 GB fetched, 0.9 GB packed)
```

The model files come from [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(SHA-256 checked, resumable). The GGUF has no MTP layer, so `tools/fetch_mtp.py` reads just its 31 tensors from the
official [BF16 checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) with HTTP Range requests (not the 360 GB),
and `bl-mtp-pack` quantizes its experts to Q2_0. In mainland China: `HF_ENDPOINT=https://hf-mirror.com scripts/download_model.sh`.

The repository has four quantizations; `scripts/download_model.sh QUANT` picks one (IQ3_XXS by default). The second
file, the 28.8 GB n-gram table, is the same in all four, so the script links one already present instead of fetching it.

| Quantization | First file | Routed experts | RAM for the LRU cache (experts + 12 GB) | Status |
|---|---|---|---|---|
| **IQ3_XXS** | 47.0 GB | 43.0 GB | 55 GB | tested (all numbers here) |
| IQ2_XS | 39.2 GB | ≈ 35 GB | ≈ 47 GB | untested |
| Q2_0 | 37.6 GB | ≈ 33.5 GB | ≈ 46 GB | untested |
| IQ3_S | 54.8 GB | 50.3 GB | 62 GB | runs: ~82 tok/s (IQ3_XXS: ~100 on the same texts); accuracy against llama.cpp not measured |

A name is only a label: each file mixes formats per layer (the IQ3_XXS one has IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and
IQ3_S experts, with IQ4_NL or Q2_0 for `down`; the IQ3_S one also IQ4_XS, and IQ4_XS token embeddings). The engine checks every expert tensor at start and stops with the
tensor's name if a format is not handled. A bigger file leaves a smaller share of its experts in VRAM, so it decodes
slower; a smaller one decodes faster at some cost in accuracy.

### 3. Run

```bash
build/bl-server --model models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --host 0.0.0.0 --port 8080
```

Open `http://<this machine>:8080` for the chat page, or use any OpenAI client with `base_url=http://<host>:8080/v1`.
In the terminal: `build/bl-chat --model models/...-00001-of-00002.gguf`.

## Usage

`bl-server --model SHARD1 [--host 127.0.0.1] [--port 8080] [--ctx 32768] [--name NAME] [--web DIR|none] [--expert-cache FILE]`

| | |
|---|---|
| `--ctx N` | context length, in tokens or as `64k`, `128k`, `256k` (default 32768; up to 262144, the model's maximum, tested). A longer context leaves less VRAM for the expert cache: 262144 takes 3.9 GB of it |
| `--expert-cache FILE` | save the VRAM expert cache after each reply and start from it next time |
| `POST /v1/chat/completions` | `messages`, `stream`, `temperature` (0.7), `top_p` (0.8), `top_k` (20), `max_tokens`, `seed`, `reasoning_effort` or `chat_template_kwargs.enable_thinking`; thinking comes back as `reasoning_content`; `spec` (MTP drafts, 0-3) |
| `GET /v1/models`, `GET /health` | |
| `GET /bl/models`, `POST /bl/models {"id": "IQ3_S"}` | list the quantizations next to `--model` (and the server's `ctx`) and switch to one (also in the chat page's settings); the reload takes 30-60 s, chat requests meanwhile get 503, and a model that fails to load gives way to the previous one |

One request is served at a time; a request that continues the previous conversation only processes the new part
(`usage.prompt_tokens_details.cached_tokens`). The chat page draws SVG code blocks as pictures, so "draw a cat
fishing" works.

`bl-chat --model SHARD1 [--ctx N] [--temp 0.7] [--top-p 0.8] [--top-k 20] [--spec 3] [--think off|low|medium|xhigh] [--system TEXT]`;
in the chat: `/reset`, `/think low`, `/temp 0.3`, `/stats`, `/exit`.

**Several GPUs (experimental)**: on the [`multi-gpu`](https://github.com/DaveByteAI/nano-cuda-moe/tree/multi-gpu)
branch, `--gpus 0,1` splits the layers between the cards. So far checked only as two stages on one GPU, not yet on two
real GPUs; reports are welcome.

## DeepSeek-V4.1-Flash

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) through acrogenesis'
[IQ2_XXS GGUF](https://huggingface.co/acrogenesis/DeepSeek-V4.1-Flash-1017-IQ2_XXS-GGUF): 40 layers of 384 experts
(6 per token) with hyper-connections, compressed sparse attention and two Engram n-gram tables. The file is 359 GB:
140 GB of 2-bit experts (15,360 of them, 9.1 MB each), 7.6 GB of Q8_0 dense weights, 209 GB of Engram tables. The
model description and the forward pass, checked against the official code: [docs/DSV41.md](docs/DSV41.md) (Chinese).

### How it fits

- **Three tiers for the experts.** VRAM keeps ~1,700 of them (LRU), a pinned RAM pool ~5,700 more (52 GB on a 64 GB
  machine), and the rest are read from the NVMe when a token needs them (O_DIRECT, two threads, 1.5 MB requests:
  this SSD gets slower with more in flight). VRAM and RAM hold different experts, so together they cover half the model.
- **GPU and CPU at the same time.** In each layer the GPU computes the experts it holds while 8 CPU threads compute the
  misses straight from RAM; copying a 9 MB expert over PCIe would cost more than computing it on the CPU.
- **Reading ahead.** The router of layer *i*+2 applied to layer *i*'s input predicts its experts well enough that the
  top 2 are fetched from disk two layers early. An expert enters VRAM only on its second miss within the last 1,000, which
  halves the 9 MB-in, 9 MB-out traffic of admissions.
- **Small copies never queue behind big ones.** Router results and flags move through a kernel on mapped host memory,
  not the copy engines, which are busy with expert copies (+12%).
- **Engram on the NVMe.** Its rows depend only on the token ids, so all 48 reads of a token start before layer 0.
- **Prompts** go in one chunk of up to 4,096 tokens, borrowing VRAM cache segments for the activations; the RAM tier is
  frozen meanwhile, since a scan over every expert is LRU's worst case.

### Performance

RTX 3090 24 GB, Core i7-10700, 64 GB DDR4, an NVMe SSD; greedy unless noted.

| | |
|---|---|
| Chat in the web page (temperature 0.7, an 883-token reply) | 8.3 tok/s |
| New text fed token by token (Chinese / English / CUDA code, 512 tokens each) | 9.9 / 9.6 / 11.7 tok/s |
| Repetitive output | ~20 tok/s |
| llama.cpp (the DeepSeek-V4.1 PR), same machine and file, experts on the CPU | 0.97 tok/s |
| A 1,000-token prompt | 59 s (~17 tok/s) |
| Startup, warming the VRAM and RAM tiers from a usage profile | ~40 s |

Each token reads about 13 experts from the disk; more RAM means fewer reads (simulated: ~14 tok/s with 96 GB, ~17
with 128 GB). The [optimization log](docs/LOG.md) has every step from 4.9 to 10 tok/s, and the ones that did not work.

### Run it

Needs, besides the build requirements above: 64 GB of RAM and ~360 GB of disk, NVMe for the experts (the Engram files
can be on a slower disk: `ENGRAM_DIR`).

```bash
git checkout dsv41 && cmake -B build && cmake --build build -j
scripts/download_dsv41.sh all        # 359 GB into models/DSV41-IQ2_XXS (resumable, SHA-256 checked)
build/bl-server --model models/DSV41-IQ2_XXS/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Experts-Q8-Dense-Engram-00001-of-00010.gguf --ctx 16k
build/bl-chat   --model models/DSV41-IQ2_XXS/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Experts-Q8-Dense-Engram-00001-of-00010.gguf
```

Thinking (`--think`, `enable_thinking`) uses the release's `<think>` prompt. `bl-dsv41 --tokens-file F --force 64
--stats-after 128` is the speed benchmark (`bench/dsv41/`); the tuning variables are in [docs/DSV41.md](docs/DSV41.md).

### Limits

- **2-bit experts.** IQ2_XXS is what makes 140 GB of experts possible here; answers are below the official model's.
- **Context up to 16K** for now (the two-level candidate selection past 16K positions is not implemented).
- **No tool calls** (the GGUF's chat template has none).
- **No speculative decoding.** The model's own draft (DSpark, 3 layers, not in this GGUF) is packed from the official
  weights by `bl-dspark-pack` and guesses well (3.25 of 5 tokens accepted), but checking 6 tokens at once touches
  ~21 different experts per layer instead of 6 and costs 8.5 single steps here. It pays off only when the model fits in
  VRAM.

## Checking accuracy and speed

```bash
build/bl-run --model models/...-00001-of-00002.gguf --tokens-file bench/corpus/chat_zh.tokens --gen 200 --spec 3
BL_STAGE_PROF=1 build/bl-run ...      # the decode step's GPU time per stage
build/bench_dense models/...gguf 4    # kernel microbenchmarks: bench_dense, bench_experts, bench_idx, bench_cpu
BL_CPU_SHARE=0 build/test_window models/...gguf bench/prompts/mixed128.tokens   # a verify window = token by token, bit for bit
```

`bench/` holds the fixed inputs behind every number here (see [bench/README.md](bench/README.md)). Accuracy is
measured against llama.cpp's own forward pass on the same GGUF: `bl-ref-dump` writes its logits and per-layer
activations, `bl-run --ref` and `tools/logits_compare.py` (needs numpy) compare.

## How it works

- **Three tiers.** GPU: attention, the DeltaNet layers, the hyper-connection weights, routers, shared experts, the
  output head, the MTP layer and its 512 experts, the KV cache, and the expert cache in the rest of VRAM. RAM: all
  experts, pinned in huge pages. SSD: the n-gram embedding table, a few rows per token through the page cache.
- **The GPU plans each layer itself.** A kernel reads the router's choice, sorts the experts into cached and missing,
  writes the missing ones to mapped host memory ("rings the doorbell") and runs the cached ones; a host thread
  computes the missing ones on the CPU and raises a flag the GPU spins on. Each miss is then copied into the layer's
  LRU cache over PCIe for next time.
- **Verify windows.** The last token and up to 3 MTP drafts run through the 48 layers at once; the GDN state is only
  committed for the accepted tokens. Multi-token kernels (dp4a on int8 activations, tensor-core prefill) make a window
  of 4 cost little more than 1.
- **Prompts** go layer by layer over 8K-token chunks: cuBLAS GEMMs, tensor-core MoE and attention kernels, and the
  next layer's experts copied over PCIe while the current layer's attention runs.
- **Sparse attention.** The model's QSA layers attend to the 512 blocks of 4 positions an indexer scores highest (plus
  the tail); the KV cache is int8 with a scale per 32 values.

The [optimization log](docs/LOG.md) (Chinese) records each change with its measurement, including the ones that did not
work; [docs/MODEL.md](docs/MODEL.md) describes the model.

## Repository layout

```
src/            the Qwen3.8 engine: engine.cpp (scheduling, caches, prefill, MTP), cuda/ (kernels), cpu_*.cpp (AVX2 experts);
                chat_models.cpp (bl-chat / bl-server pick the engine by the GGUF's architecture)
src/dsv41/      the DeepSeek-V4.1 engine: engine.cpp, ops.cu, ram_tier.cpp (pinned RAM pool, NVMe reads)
include/bl/     its interfaces
tools/          bl-server, bl-chat, bl-run, bl-tok, bl-ref-dump, bl-mtp-pack; fetch_mtp.py, make_rank.py, make_draft_vocab.py;
                bl-dsv41, bl-dspark-pack, dsv41_*.py, dspark_ref.py
tests/          correctness tests and kernel benchmarks
web/            the chat page (served by bl-server)
data/           expert-rank.bin (the cache's starting order), draft-vocab.bin (the MTP head's vocabulary)
bench/          the benchmark inputs and their generators; dsv41/ (DeepSeek texts, expert usage profile)
scripts/        download_model.sh, download_dsv41.sh, qwen_regression.sh
docs/           LOG.md (optimization log), MODEL.md, DSV41.md, assets/ (figures: make_figures.py)
```

## Credits

- **Model:** [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team; the GSQ-RCO
  quantization by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF). Their licenses
  apply to the model files, which are not part of this repository.
- **Model:** [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) by DeepSeek; the
  IQ2_XXS GGUF by [acrogenesis](https://huggingface.co/acrogenesis/DeepSeek-V4.1-Flash-1017-IQ2_XXS-GGUF); the llama.cpp
  DeepSeek-V4.1 pull request (#28696) as the numerical reference.
- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)** (MIT): the quantization formats and codebooks, the CPU
  dot products, the tokenizer, and the reference implementation every result is checked against.

## License

[Apache License 2.0](LICENSE).
