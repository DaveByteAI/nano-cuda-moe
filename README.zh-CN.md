<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-zh-dark.svg">
  <img alt="nano-cuda-moe：Qwen3.8-Flash-Next，1250 亿参数、每个 token 激活 60 亿的混合专家模型，跑在一张 24 GB 的 RTX 3090 上，每秒约 97 个 token。" src="docs/assets/hero-zh-light.svg" width="100%">
</picture>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#原理">原理</a> ·
  <a href="#性能">性能</a> ·
  <a href="#deepseek-v41-flashdsv41-分支">DeepSeek-V4.1-Flash</a> ·
  <a href="README.md">English</a>
</p>

在**一张 24 GB 的 RTX 3090、64 GB 内存**的普通电脑上运行大型混合专家（MoE）模型。引擎是为每个模型专门写的 C++ 和
CUDA：推理时只有一个进程，不需要 Python 或任何机器学习框架；自带 OpenAI 兼容的服务和聊天网页。目前支持两个模型：

| 模型 | 模型文件 | 专家放在哪 | 3090 + 64 GB 上的生成速度 | 分支 |
|---|---|---|---|---|
| [**Qwen3.8-Flash-Next**](#亮点)（125B，激活 6B） | 76 GB，IQ3_XXS | 显存 + 内存 | **约 97 tok/s** | `main`（本页） |
| [**DeepSeek-V4.1-Flash-1017**](#deepseek-v41-flashdsv41-分支) | 359 GB，专家 2 bit（IQ2_XXS） | 显存 + 内存 + NVMe | **8–12 tok/s**（重复性输出约 20） | [`dsv41`](https://github.com/DaveByteAI/nano-cuda-moe/tree/dsv41) |

它是 [**nano-metal-moe-qwen36**](https://github.com/DaveByteAI/nano-metal-moe-qwen36) 的 CUDA 版本。那个项目用
Metal 在 16 GB 的 Mac mini 上运行 Qwen3.6-35B-A3B。两者出发点相同：MoE 模型的全部专家都得有地方放，但每个 token
只用到其中几个，所以昂贵的那层存储只需要装下最常用的专家。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-zh-dark.svg">
  <img alt="架构：显卡的 24 GB 里放稠密权重和最常用专家的缓存；全部 25,088 个专家锁在内存里，显卡没有的由 CPU 直接在内存里算；n-gram 嵌入表放在 SSD 上。" src="docs/assets/architecture-zh-light.svg" width="100%">
</picture>

## 亮点

以下讲的是 Qwen3.8-Flash-Next；DeepSeek-V4.1 见[后面](#deepseek-v41-flashdsv41-分支)。


- **125B 的模型，24 GB 的卡。** 42.9 GB 的专家放在锁页内存里。显卡上放稠密权重（4.1 GB），再加上每层一个 LRU
  专家缓存，装得下 25,088 个专家中的约 1 万个（约 17.5 GB），能满足一个 token 所需专家的约 90%。
- **未命中的专家由 CPU 就地计算。** 不走 PCIe 来回搬运：显卡算这一层缓存里的专家时，CPU 同时在内存里算缺的那些
  （AVX2），显卡等映射内存里的一个标志。整个 decode 步骤是一个 CUDA graph。
- **用模型自带的 MTP 层做投机解码。** 一次起草最多 3 个 token，主模型一遍 48 层把它们一起验证，平均每遍出 2.5–3 个
  token。采样时草稿在 GPU 上做拒绝采样，输出分布和不起草时完全相同。
- **长上下文，支持模型的完整 256K。** 用模型自带的稀疏注意力（索引器为每个位置挑出 2048 个位置），KV cache 用 int8。
  128K token 的提示词读 78 秒，256K 读 165 秒（每秒约 1,550 个 token）。
- **拿来就能用。** `bl-server` 兼容 OpenAI 的对话接口（流式输出、思考内容、多轮对话复用前缀），并提供聊天网页；
  `bl-chat` 是终端版本。

## 性能

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/performance-zh-dark.svg">
  <img alt="7 段文本上的生成速度：97 tok/s，同一台机器、同一个模型文件上 Strata 为 65；32K 和 128K 提示词的处理速度约每秒 1,600–1,800 个 token。" src="docs/assets/performance-zh-light.svg" width="100%">
</picture>

RTX 3090 24 GB（PCIe Gen3）、Core i7-10700（8 核，AVX2）、64 GB DDR4、Linux。模型是 ISTA-DASLab 的 GSQ-RCO IQ3_XXS
GGUF。贪心解码，每轮 3 个 MTP 草稿。

| | |
|---|---|
| 生成速度，7 段短文本（对话、代码、文档，各生成 200 个 token） | **97 tok/s** |
| 生成速度，网页聊天（温度 0.7） | 92–104 tok/s |
| 32K / 128K token 提示词之后的生成速度 | 82 / 75 tok/s |
| 提示词处理，32K / 128K token | 18.0 s / 78 s |
| 256K token 提示词（`--ctx 262144`）：读取 / 之后的生成速度 | 165 s / 66 tok/s |
| （`--ctx 262144` 要给 KV cache 留 3.9 GB 显存：专家缓存从 17.5 GB 降到 14 GB，短对话的生成慢 5–12%） | |
| 短问题的首字延迟 | 约 0.4 s |
| 启动（从 NVMe 读 75.8 GB） | 约 25 s |
| （启动后的第一个长提示词要从 SSD 读 n-gram 表：32K 时慢约 2 s） | |
| 精度，与 llama.cpp 在同一个 GGUF 上对比（4K 提示词） | KL 中位数 0.003–0.005，top-1 一致率 90–93% |

和另一个运行这个模型的引擎 [Strata](https://github.com/eddoursul/Strata) 在同一台机器、同一个 GGUF 文件、同样的
输入上对比：这里的生成速度**快 50%（97 对 65 tok/s）**；长提示词 Strata 读得更快（32K 提示词它用 13.3 s，这里 18.0 s），读完之后的生成速度两边差不多（85 对 82 tok/s）。[优化日志](docs/LOG.md)记录了每一步改动和它的测量。

## 环境要求

- NVIDIA 显卡，24 GB 显存，计算能力 8.0 或更新（开发和测量都在 RTX 3090 上，sm_86）
- 64 GB 内存（专家要锁在内存里，43 GB），支持 AVX2 的 x86-64 CPU
- 80 GB 磁盘空间，最好是 NVMe（28.8 GB 的 n-gram 表每个 token 读几行）
- Linux，CUDA 12 或 13，CMake 3.24+，GCC 11+（C++20），Python 3（只用标准库，下载脚本用）

## 快速开始

### 1. 编译

```bash
git clone https://github.com/DaveByteAI/nano-cuda-moe
cd nano-cuda-moe
cmake -B build                      # 会自动下载固定版本的 llama.cpp（用它的 ggml-cpu 和分词器）
cmake --build build -j
```

不是 RTX 30 系列的显卡，加上 `-DCMAKE_CUDA_ARCHITECTURES=native`。CUDA 不在 `PATH` 里：
`-DCMAKE_CUDA_COMPILER=/path/to/nvcc`。

### 2. 下载模型

```bash
scripts/download_model.sh          # 下载到 models/：75.8 GB 的 GGUF，然后是 MTP 层（下载约 5.5 GB，打包后 0.9 GB）
```

模型文件来自 [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
（校验 SHA-256，支持断点续传）。这个 GGUF 里没有 MTP 层，所以 `tools/fetch_mtp.py` 用 HTTP Range 请求只从官方
[BF16 原版权重](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)里取它的 31 个张量（不用下载 360 GB），再由
`bl-mtp-pack` 把其中的专家量化成 Q2_0。国内网络：`HF_ENDPOINT=https://hf-mirror.com scripts/download_model.sh`。

仓库里有四种量化，`scripts/download_model.sh 量化名` 选择其中一种（默认 IQ3_XXS）。第二个文件（28.8 GB 的
n-gram 表）四种完全相同，脚本发现本地已有就直接链接，不再下载。

| 量化 | 第一个文件 | 路由专家 | LRU 缓存需要的内存（专家 + 12 GB） | 状态 |
|---|---|---|---|---|
| **IQ3_XXS** | 47.0 GB | 43.0 GB | 55 GB | 已测试（本文所有数字） |
| IQ2_XS | 39.2 GB | ≈ 35 GB | ≈ 47 GB | 未测试 |
| Q2_0 | 37.6 GB | ≈ 33.5 GB | ≈ 46 GB | 未测试 |
| IQ3_S | 54.8 GB | 50.3 GB | 62 GB | 能运行：约 82 tok/s（同样文本上 IQ3_XXS 约 100）；未和 llama.cpp 对比精度 |

量化名只是个标签：每个文件都按层混用多种格式（IQ3_XXS 这个文件的专家用了 IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S，
`down` 用 IQ4_NL 或 Q2_0；IQ3_S 文件还用到 IQ4_XS，token 嵌入表也是 IQ4_XS）。引擎启动时会检查每个专家张量，遇到不支持的格式就停下来并报出张量名。文件越大，能放进
显存的专家比例越小，解码越慢；文件越小解码越快，但精度会有损失。

### 3. 运行

```bash
build/bl-server --model models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --host 0.0.0.0 --port 8080
```

浏览器打开 `http://<这台机器>:8080` 就是聊天网页；也可以用任何 OpenAI 客户端，`base_url=http://<host>:8080/v1`。
终端里聊天：`build/bl-chat --model models/...-00001-of-00002.gguf`。

## 用法

`bl-server --model SHARD1 [--host 127.0.0.1] [--port 8080] [--ctx 32768] [--name NAME] [--web DIR|none] [--expert-cache FILE]`

| | |
|---|---|
| `--ctx N` | 上下文长度（默认 32768，最长测到 262144，即模型的上限）。越长，留给专家缓存的显存越少：262144 要占 3.9 GB |
| `--expert-cache FILE` | 每次回复后保存显存里的专家缓存，下次启动时从它开始 |
| `POST /v1/chat/completions` | `messages`、`stream`、`temperature`（0.7）、`top_p`（0.8）、`top_k`（20）、`max_tokens`、`seed`、`reasoning_effort` 或 `chat_template_kwargs.enable_thinking`；思考内容在 `reasoning_content` 里返回；`spec`（MTP 草稿数，0–3） |
| `GET /v1/models`、`GET /health` | |
| `GET /bl/models`、`POST /bl/models {"id": "IQ3_S"}` | 列出 `--model` 同目录下的量化版本并切换（聊天网页的设置面板里也能切）；重新加载需要 30–60 秒，期间聊天请求返回 503；新模型加载失败会自动换回原来的 |

一次处理一个请求；如果新请求是在上一次对话后面接着说，只处理新增的部分（见 `usage.prompt_tokens_details.cached_tokens`）。
聊天网页会把 SVG 代码块直接画成图，所以可以说"画一只钓鱼的小猫"。

`bl-chat --model SHARD1 [--ctx N] [--temp 0.7] [--top-p 0.8] [--top-k 20] [--spec 3] [--think off|low|medium|xhigh] [--system TEXT]`；
对话中可以用 `/reset`、`/think low`、`/temp 0.3`、`/stats`、`/exit`。

**多块 GPU（实验性）**：在 [`multi-gpu`](https://github.com/DaveByteAI/nano-cuda-moe/tree/multi-gpu) 分支，
加 `--gpus 0,1` 把层分给几张卡。目前只在单卡上模拟验证过，还没在两张真卡上测过，欢迎反馈。

## DeepSeek-V4.1-Flash（dsv41 分支）

[`dsv41` 分支](https://github.com/DaveByteAI/nano-cuda-moe/tree/dsv41)在同一台机器上、用同一套 `bl-server` 和 `bl-chat` 运行
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)（acrogenesis 的
[IQ2_XXS GGUF](https://huggingface.co/acrogenesis/DeepSeek-V4.1-Flash-1017-IQ2_XXS-GGUF)）。文件 359 GB，是内存的 5 倍多：
2 bit 的专家 140 GB，Engram n-gram 表 209 GB。专家分三层存（显存约 1,700 个、锁页内存约 5,700 个、其余在 token 需要时从
NVMe 读）；GPU 算显存里有的专家，CPU 同时算没命中的，后面两层要用的专家提前从盘里读。

| | |
|---|---|
| 网页聊天（温度 0.7） | 8.3 tok/s |
| 新内容逐 token 喂入（中文 / 英文 / CUDA 代码） | 9.9 / 9.6 / 11.7 tok/s |
| llama.cpp（DeepSeek-V4.1 的 PR），同一台机器、同一个文件 | 0.97 tok/s |
| 1,000 个 token 的提示词 / 启动 | 59 s / 约 40 s |

限制：专家是 2 bit（回答质量比官方原版低），上下文最多 16K，没有工具调用。安装、设计和每一步的测量见分支上的
[README](https://github.com/DaveByteAI/nano-cuda-moe/tree/dsv41/README.zh-CN.md#deepseek-v41-flash) 和[优化日志](https://github.com/DaveByteAI/nano-cuda-moe/tree/dsv41/docs/LOG.md)。

```bash
git checkout dsv41 && cmake -B build && cmake --build build -j
scripts/download_dsv41.sh all        # 359 GB
build/bl-server --model models/DSV41-IQ2_XXS/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Experts-Q8-Dense-Engram-00001-of-00010.gguf --ctx 16k
```

## 测速度和精度

```bash
build/bl-run --model models/...-00001-of-00002.gguf --tokens-file bench/corpus/chat_zh.tokens --gen 200 --spec 3
BL_STAGE_PROF=1 build/bl-run ...      # decode 一轮里各阶段的 GPU 时间
build/bench_dense models/...gguf 4    # kernel 微基准：bench_dense、bench_experts、bench_idx、bench_cpu
BL_CPU_SHARE=0 build/test_window models/...gguf bench/prompts/mixed128.tokens   # 验证窗口与逐 token 结果逐位一致
```

`bench/` 里是这里所有数字用到的固定输入（见 [bench/README.md](bench/README.md)）。精度的参照是 llama.cpp 自己在同一个
GGUF 上的前向计算：`bl-ref-dump` 导出它的 logits 和逐层激活，再用 `bl-run --ref` 和 `tools/logits_compare.py`（需要 numpy）对比。

## 原理

- **三层存储。** 显卡：注意力、DeltaNet 层、hyper-connection 权重、路由器、共享专家、输出头、MTP 层和它的 512 个专家、
  KV cache，剩下的显存都给专家缓存。内存：全部专家，锁在大页内存里。SSD：n-gram 嵌入表，每个 token 经页缓存读几行。
- **每一层由 GPU 自己调度。** 一个 kernel 读取路由结果，把专家分成"缓存里有"和"缺的"，把缺的写进映射的主机内存（"按门铃"），
  然后算缓存里的；主机线程在 CPU 上算缺的那些，算完设一个标志，GPU 在那里自旋等待。缺的专家随后经 PCIe 拷进这一层的
  LRU 缓存，下次就命中了。
- **验证窗口。** 最后一个 token 和最多 3 个 MTP 草稿一起走完 48 层；GDN 的状态只为被接受的 token 提交。多 token kernel
  （int8 激活上的 dp4a、张量核心）让一次处理 4 个 token 的开销只比 1 个多一点。
- **提示词**按 8K token 一块、逐层处理：cuBLAS 矩阵乘、张量核心的专家和注意力 kernel；当前层算注意力时，下一层要用的
  专家同时经 PCIe 拷进来。
- **稀疏注意力。** 模型的 QSA 层只看索引器打分最高的 512 个块（每块 4 个位置）加上末尾；KV cache 是 int8，每 32 个值一个缩放。

[优化日志](docs/LOG.md)按时间记录了每个改动和它的测量，包括没有成功的尝试；[docs/MODEL.md](docs/MODEL.md) 介绍模型结构。

## 目录结构

```
src/            引擎：engine.cpp（调度、缓存、prefill、MTP）、cuda/（kernel）、cpu_*.cpp（AVX2 专家）
include/bl/     接口
tools/          bl-server、bl-chat、bl-run、bl-tok、bl-ref-dump、bl-mtp-pack；fetch_mtp.py、make_rank.py、make_draft_vocab.py
tests/          正确性测试和 kernel 基准
web/            聊天网页（由 bl-server 提供）
data/           expert-rank.bin（专家缓存的初始顺序）、draft-vocab.bin（MTP 草稿头的词表）
bench/          测试输入和生成它们的脚本
scripts/        download_model.sh
docs/           LOG.md（优化日志）、MODEL.md、assets/（插图，由 make_figures.py 生成）
```

## 致谢

- **模型：** Qwen 团队的 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)；GSQ-RCO 量化来自
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。模型文件不在本仓库里，适用它们各自的许可。
- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)**（MIT）：量化格式和码本、CPU 点积、分词器，以及所有结果对照的参考实现。

## 许可证

[Apache License 2.0](LICENSE)。
