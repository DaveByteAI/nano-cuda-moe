# boundless 开发与优化记录

> 这是开发过程的原始记录，按时间顺序写成，包括没有成功的尝试。引擎在开发时叫 boundless（程序名前缀 `bl-`）；
> 文中的路径（`../models` 等）是当时工作区的布局，现在的用法见 [README](../README.zh-CN.md)。

机器：RTX 3090 24 GB（PCIe Gen3 x16）、i7-10700（AVX2）、62 GB DDR4、Ubuntu 24.04、驱动 595 / CUDA 13.2。
模型：ISTA-DASLab GSQ-RCO IQ3_XXS（revision `ed59f92`）。参照：llama.cpp `3cf0325`。

## 2026-10-04 本机实测

| 项目 | 数值 | 测法 |
|---|---|---|
| 显存带宽 | 约 840 GB/s | cudaMemcpy D2D，1 GiB，按读写合计 |
| PCIe 锁页拷贝（主机到显存） | 12.5 GB/s | cudaMemcpy H2D，1 GiB pinned |
| 内存读带宽 | 37 GB/s（4 线程峰值），单线程 23 GB/s | AVX2 顺序读，1 GiB |
| NVMe 顺序读 | 约 2.0 GB/s | dd O_DIRECT，4 MB 块 |

## 2026-10-04 阶段 1：量化 kernel

12 种权重格式的 GPU 反量化（`src/cuda/dequant.cuh`）按 ggml 的 `dequantize_row_*` 逐个移植，在真实张量上
**与 ggml 的 `to_float` 逐位一致**（`tests/test_quant`，每种格式整张张量，最多 8.4 亿个值，0 ulp）。

朴素 matvec（一个 warp 负责一行）在整张张量上的有效带宽（`test_quant --bench`）：

| 格式 | 张量 | GB/s |
|---|---|---|
| BF16 | output_hc_down（6.6 MB） | 571 |
| Q5_K | output（437 MB） | 473 |
| IQ3_S | token_embd（273 MB） | 471 |
| IQ4_NL | ffn_down_exps（472 MB） | 399 |
| IQ4_XS | attn_qkv（14 MB） | 383 |
| IQ2_S | ffn_gate_exps（269 MB） | 335 |
| Q2_0 | ffn_down_exps（236 MB） | 334 |
| IQ2_XS | ffn_gate_exps（243 MB） | 316 |
| IQ2_XXS | ffn_gate_exps（216 MB） | 259 |
| Q6_K | attn_gate（13 MB） | 226 |

离实测上限 840 GB/s 还有 1.5–3 倍空间，属于阶段 3 的工作。

## 2026-10-04 阶段 1：完整前向跑通，精度对齐

`src/engine.cpp`：稠密权重常驻显存；每层选中的 10 个专家从 mmap 临时拷进显存；QSA 先按稠密注意力实现
（上下文不超过索引器的 2048 + 3 个位置时，结果完全相同）；全部 fp32，逐 token 计算。

**正确性**（`bl-run --ref` 与 `tools/logits_compare.py`，对照 llama.cpp 的 `bl-ref-dump`）：

| 测试 | 结果 |
|---|---|
| 短问答（24 token，"What is the capital of France?"） | top-1 为 "Paris"，与参照的 top-5 顺序完全一致，最后位置 KL 0.0002 |
| mixed.txt 前 128 token（中英文、代码、JSON，teacher-forced） | **平均 KL 0.0065 nats，最大 0.069，top-1 一致率 95.3%** |

参照尺度：Strata 论文报告它对 llama.cpp 的平均 KL 是 0.022，llama.cpp 自己 CPU 与 GPU 两个后端之间是 0.058。

注意：llama.cpp 的参照并非纯 fp32。BF16 矩阵乘的激活会被取整成 bf16，量化矩阵乘的激活会被量化成 Q8；
batch 较大时它还会把矩阵乘卸载到 GPU。所以逐节点的相对误差（中间层残差最高 0.4）主要反映参照本身的噪声，
以及近似并列时专家路由的翻转。判断正确性看最终分布。

**速度**（未优化）：prompt 约 7 tok/s，逐 token 计算。瓶颈是每个专家从可分页内存拷贝
（约 1.75 MB × 10 个/层），以及每个专家之后的一次同步。还没有做分段计时。

## 2026-10-04 阶段 2–3：三层存储、设备驱动流水线、本机调优

测法：`bl-run --warm 2`。同一段 128 token 的混合文本先跑两遍（让自适应缓存适应文本），然后冻结缓存再计时，
逐 token 喂入（等同于解码）。"第 1 遍"的数字里包含 graph 录制、锁页内存首次访问等一次性开销，以后只报预热后的数字。

| 步骤 | ms/token | tok/s | 说明 |
|---|---|---|---|
| 阶段 1 朴素版 | 143 | 7 | 每个专家从可分页内存拷贝后同步 |
| 显存专家仓库（18.9 GB，按一份专家使用频率统计排序）+ 锁页内存 DMA | 47 | 21 | 命中率 66% |
| + CPU 计算一半未命中（ggml vec_dot） | 40 | 25 | |
| + kernel 融合、GDN 递推放进寄存器 | 38 | 26 | |
| + 设备驱动流水线（GPU 自己判定命中，按门铃交给主机）+ 整个 token 一个 CUDA graph | 30.5（预热） | 33 | 修了两个 bug：线程池阶段切换的竞态；CUDA 懒加载与自旋 kernel 死锁（改为 `CUDA_MODULE_LOADING=EAGER`） |
| + 自适应专家缓存（每 32 token 交换最多 64 对） | 24.8 | 40 | 新文本首遍命中率约 88% |
| + 稠密 matvec：向量化读取、split-K、x 交错存放 | 22.5 | 44 | |
| + AVX2 版 Q2_0 点积（ggml 在 x86 上是标量实现，单线程 0.72 → 5.75 GB/s） | 19.8 | 50 | CPU 分担比例改为 0.7 |
| + 下一层专家预测与预取（每层 2 个，召回率约 69%） | 19.1 | 52 | |
| + 专家 kernel 改为 dp4a（从 llama.cpp 移植），路由 top-k 重写，IQ 格式符号判断改为位测试 | **18.3** | **55** | 专家 kernel 带宽 298 → 484 GB/s |

精度（mixed128，对照 llama.cpp）：平均 KL 0.0070，top-1 一致率 96.9%。

关键测量：
- GPU 专家 kernel 原来受指令吞吐限制，不受带宽限制：gate/up 不论格式都要约 37 µs，
  改成 dp4a 后降到 20–25 µs。
- CPU：ggml 的 i-quant AVX2 点积单线程 2.4–4.1 GB/s（IQ3_S 只有 0.9–1.7），8 线程线程池约 18 GB/s。
  gate/up 占 CPU 时间的约 80%。
- 预取：每层 2 个最好。更多会让 PCIe 队列（先进先出）堵在下一层真正需要的拷贝前面。

每 token 的 GPU kernel 时间分布（nsys）：稠密 matvec 约 7.5 ms，等待未命中约 3 ms，命中专家约 2 ms，
其余各项都在 1 ms 以下。

## 2026-10-04 阶段 4–5：LRU 缓存、验证窗口、MTP 投机解码

### LRU 与验证窗口

- 每层 LRU 专家缓存（全部专家常驻锁页内存，淘汰时不用写回）：命中率 96–97.5%，逐 token 生成 40.5 tok/s。之前的自适应交换是 32.8 tok/s。
- 验证窗口：T ≤ 4 个 token 一次前向。`verify()` 不改动已提交的状态，`commit(n)` 只对接受的 n 个 token 重放 GDN 递推。
  在 `BL_CPU_SHARE=0` 下，窗口与逐 token 的结果逐位一致（各种 T 以及回滚，见 `test_window`）。
- PLE 行原来是逐行的 SSD 读，每行约 0.45 ms，成本被掩盖了。改成先对整个窗口的行做 madvise WILLNEED，再拷贝：
  T=1 从 26.6 降到 20.4 ms，T=4 从 67.4 降到 40.2 ms/窗口。

### MTP 草稿层

- 先用一份现成的 MTP 打包文件（`mtp-q2_0.gguf`，后来换成自己的 `bl-mtp-pack`，见 2026-10-06）。把 MTP 层当作主模型的"第 48 层"（QSA 类型），拥有自己的 KV 和位置；
  512 个 Q2_0 专家全部放在显存。所有 F32 norm 权重都要 +1（Gemma 风格）。
- 输入：位置 i 取主模型最终残差 R_i（4×2560）和 token i+1 的 embedding。
  先算 fc_embedding(norm(e)) 和 fc_hidden(norm(R)) 并逐流广播相加，再过 QSA、MoE、HC，
  然后经 MTP 自己的 mixer 接到共享的 head。链式草稿使用上一步 MTP 的输出残差。
- `generate()`：每轮验证 [y, d1..dk]（k ≤ 3），接受到第一个不一致为止。MTP 对已接受的行补算（catch-up），
  最后一行给出下一轮的 d1；若上一个草稿的概率 ≥ min_p（0.5），继续链式起草。
- 正确性：`BL_CPU_SHARE=0` 时 spec 0 与 spec 3 的贪心输出完全一致。默认设置下 CPU 与 GPU 专家的数值差异
  会让近似并列处在约 100 token 后分叉。
- 接受率：草稿 60–87%，每轮 2.5–3.4 个 token，因文本而异。

### 第一版之后的优化（7 个语料 × 200 token，spec 3 / spec 0，tok/s）

| 步骤 | spec 3 | spec 0 | 说明 |
|---|---|---|---|
| MTP 第一版（chat_zh 300 token，冷启动） | 76.1 | — | Strata 在本机同条件为 75.9 |
| + 稠密多 token matmul 用 dp4a × q8_1 激活（新增 Q4_K/Q5_K/Q6_K/IQ4_XS 的 dot32m） | 70.0 | 45.2 | 第一次全语料测量。T=4 窗口 35→29 ms；KL 0.0069，与之前相同 |
| + 注意力按位置切块（flash-decoding：一个块处理共享同一 kv 头的全部 q 头和全部 token）；hc_up_mix 改 16 字节读取；plan_k 并行化 | 78.5 | 52.8 | 1800 token 上下文从 62–70 提到 75–81 |
| + MTP 草稿 head 只算 10.6 万个 token（一份现成的草稿词表，后来换成自己生成的 `data/draft-vocab.bin`）；MTP 投影在加载时转成 Q8_0 | 82.6 | — | MTP 每次 1.85 → 1.07 ms |
| + 窗口的 CPU 分担改为 1.0（未命中的 PCIe 拷贝会和 LRU 准入抢带宽）；argmax 两遍化 | **83.8** | **53.9** | chat_zh 86.0 |

- 对照：Strata 在本机 chat_zh 上是 75.87 tok/s（每轮 2.56 个 token，接受率 0.663）。
- profile（spec 3，默认设置）：wait_k（等 CPU 和 DMA 的未命中）约占 GPU 时间的 27%；稠密 matmul 约 20%；
  专家 kernel 约 18%。
- 测量噪声：同一设置 ±4%。每组设置的第一次运行是冷缓存，会慢 5–8%。

### 与 Strata 的全语料对比（同机、同模型文件，各 200 token）

| 语料 | Strata（每个文件单独进程，冷启动） | boundless spec 3 |
|---|---|---|
| chat_en | 79.4 | 87.1 |
| chat_zh | 68.9 | 84.3 |
| code_cuda（1800 token 提示词） | 53.3 | 79.7（单独冷启动 83.9） |
| code_llama | 57.1 | 87.7 |
| en_docs | 79.2 | 92.8 |
| mixed | 62.3 | 75.9（单独冷启动 76.9） |
| zh_readme | 68.1 | 82.8 |
| **合计** | **65.6** | **84.0** |

## 2026-10-04 批量 prefill（逐层处理整段提示词）

原来的提示词路径是长度 4 的窗口：1800 token 要 16–20 s（110 tok/s）。Strata 需要 2.5 s。

新的路径（`prefill()`）：
- 逐层处理：一层先算完整段提示词，再进入下一层。这样每层每个专家最多经 PCIe 拷贝一次。
  稠密部分按 256 行一段处理（GDN 递推和注意力要按顺序）。
- 稠密投影：cuBLAS BF16 GEMM。每层把量化权重解量化成 bf16 一次；hyper-connection 的 BF16 权重直接用。
- 路由专家：按专家把 token 分组，用 dp4a 的 `dot32m<F,16>`（16 个 token 一块），down 的输出用 atomicAdd 散回各 token。
  未命中的专家用双缓冲批次经 PCIe 拷入，同时计算已缓存的专家。
- 预取：用下一层的路由器作用在本层输入上（对整段提示词取并集），预测下一层会用到哪些未缓存的专家，
  在下一层的稠密部分进行时拷入环形区（128 个槽）。
- 注意力：每个 warp 处理一行查询，在线 softmax。GDN 沿用逐 token 的递推 kernel。
- MTP：对整段提示词也逐层算一遍（同一个函数，第 48 层），最后一行给出第一个草稿。

| 步骤 | 1800 token |
|---|---|
| 窗口路径 | 16–20 s |
| 逐层 + TF32 GEMM | 3.4 s |
| BF16 GEMM，每层只解量化一次 | 2.8 s |
| 预取环 + 按预测排序 | 2.6 s |
| 每层的小元数据改由 kernel 从映射内存读（cudaMemcpy 会排在专家 DMA 后面，每层白等约 10 ms） | 2.3 s |

同样的问题在 decode 里也有：PLE 行的 H2D 拷贝会排在 LRU 准入的拷贝后面，现在也改用 kernel 读取。

显存：prefill 的大缓冲区和环（约 1.1 GB）放在专家仓库的尾部。decode 时这段放的是排名最低的专家；
prefill 开始时把它们逐出（它们仍在锁页内存里），结束后在后台用提示词末尾最常用的专家填回。
prefill 时还会把提示词最后 64 个 token 用到的未缓存专家准入 LRU（每层最多 48 个，从显存里的暂存位置做 D2D 拷贝）。
结果是 decode 不因 prefill 损失显存，命中率与原先的窗口路径相同。

| 提示词长度 | Strata prefill | boundless prefill（含 MTP 行） |
|---|---|---|
| 1800 | 2.44–2.59 s | 2.35–2.55 s |
| 359 | 1.54 s | 1.45 s |
| 47–48 | 0.65 s | 0.40–0.52 s |

- 精度：mixed128 逐位置 logits 对照 llama.cpp，平均 KL 0.0107，top-1 一致率 95.3%。
  窗口路径是 0.0063，差距来自 GEMM 里激活取整成 bf16。
- 调试中发现的坑：`cudaMemcpy` 走 legacy 默认流，不会等非阻塞流 `st` 上的 kernel。
  调试路径读到了还没算完的 logits，看上去 KL 有 10。
- 第一次调用 cuBLAS 约 1.5 s，改为在加载时预热。
- 加入 prefill 后的全语料 decode：spec 3 84.0 tok/s，spec 0 52.1 tok/s。

## 2026-10-04 长上下文：QSA 稀疏注意力（indexer）

原来 QSA 用稠密注意力，只在 ≤ 2051 token 时精确。按 llama.cpp 的 qwen4exp 图（`build_qsa_top_k`）实现了 indexer：
- 每 4 个位置一块。块键 = 4 个原始 `indexer.k_proj` 输出取均值 → RMSNorm(k_norm) → 在块起点位置做 RoPE。
  原始键按位置缓存；每个窗口重算它补齐的块，所以投机草稿被拒绝时自动正确。
- 位置 q 的查询：`indexer.q_proj` 得到 4 头 × 128 → RMSNorm(q_norm) → RoPE(q)。块分数 = Σ_h relu(q_h · 块键)。
- 选取：末尾不完整的块（[⌊(q+1)/4⌋·4, q]）总是选中，再加分数最高的 512 个完整块。
  参照的 top-k 宽度是 2051 个单元：末尾不足 3 个时，用第 513 名块的最低几个位置补齐（与 llama.cpp 一致）。
  分数并列时取低位置。
- 实现：idx_prep / idx_blocks / idx_score（每个块键一个线程，分散到多个线程块）/ idx_select（4 趟 8 位 radix select + 前缀和，输出升序位置表）。
  注意力 kernel（decode 的分块版和 prefill 版）改为按位置表读取。块数 ≤ 512 时位置表就是 0..q，等价于原来的稠密注意力。
- MTP 层：只看最近 2051 个位置。
- KV cache 改为 fp16（32K 时 13 层约 0.83 GB）。prefill 超过 2048 token 时分块做，MTP 行也逐块补算。

精度（4000 token 提示词，最后 256 个位置，对照 llama.cpp）：

| 对比 | 平均 KL | 中位数 KL | top-1 一致率 |
|---|---|---|---|
| llama.cpp ‖ 我们的 prefill | 0.14–0.16 | 0.0044 | 93.4% |
| llama.cpp ‖ 我们的窗口路径 | 0.19 | 0.0062 | 91.4% |
| 我们的窗口路径 ‖ 我们的 prefill | 0.18 | 0.0016 | 92.2% |
| 对照：不做选取（稠密） | 0.29 | 0.0058 | 90.6% |

- 平均 KL 被少数位置拉高（每 256 个位置里约 12 个 KL > 0.5）。我们自己的两条路径之间、同一程序的两次运行之间
  （prefill 里 MoE down 用 atomicAdd，求和顺序不固定）也有同样的长尾。
  原因是 top-k 阈值附近的块对微小数值差异敏感：一个块被选中与否，会改变复制类位置的预测。这不是实现错误。
- 选取与 llama.cpp 逐单元对比（第 3 层）：每个查询约 2050 个单元中平均相差 12.7 个（约 3 个块，都是阈值附近的并列）。
- 回归：test_window 逐位一致；4K 上下文 spec 0 与 spec 3 的输出 80/80 一致（稀疏选取 + 回滚都覆盖到了）。

32K token 提示词（代码审查任务），`--ctx 33000`：

| | 结果 |
|---|---|
| prefill | 46.6–51 s（约 650 tok/s，16 块 × 2048） |
| decode spec 3 | 45.7 → **62.9 tok/s**（idx_select 改成并行打分 + radix select 之前占 GPU 时间 17.5%） |
| decode spec 0 | 35.6 → **46.7 tok/s** |
| 输出 | 连贯，能指出提示词中具体文件、函数里的 bug |

短上下文全语料不受影响：spec 3 81.4，spec 0 52.4（在 ±4% 噪声内）。

## 2026-10-04 会话状态修正（问题 1）

`generate()` 原来的问题：
- 最后输出的 token 不送进模型；
- 回调中途停止时，KV 里可能留着没输出的草稿 token。

多轮对话会因此丢 token 或多 token。

现在的约定：一次调用之后，已提交的状态 = 所有喂入和输出过的 token，减去最后一个输出 token。
那个 token 是 `pending()`，下一次 `generate()` 先喂它，所以轮次可以直接串联：`generate(第 1 轮)`、`generate(第 2 轮)`……

实现：每轮验证后，先确定实际输出到第几个（回调停止、max_new、context 满都在这里决定），再只 `commit` 这么多。
context 满时正常停止，不再抛异常。

验证（`tests/test_session.cpp`，确定性模式 `BL_CPU_SHARE=0 BL_NO_Q8=1 BL_PREFILL=0`，spec 3 和 spec 0 各一遍，17 项全部通过）：
- 两轮串联的输出 = 一次调用 [p1, out1, p2] 的输出，逐 token 相同；
- 回调中途停止后，第二轮同样一致；
- generate(p1, 5) + generate({}, 20) = generate(p1, 25)；
- context 填满时正常停止，状态正确。

对照：模拟旧行为（丢掉第 1 轮最后一个 token）时，第 13 个 token 起与正确结果不同。
默认模式（prefill + q8 + CPU 专家）下所有状态检查都通过；输出在第 8–13 个 token 后因数值差异分叉，属于预期。

## 2026-10-04 启动提速（问题 3）

原来的分解（总计 131 s）：

| 阶段 | 用时 |
|---|---|
| 43 GB 专家：经 mmap 缺页读取，约 0.43 GB/s | 100.8 s |
| `cudaHostAlloc` 43 GB | 14.5 s |
| 稠密权重：mmap + 同步拷贝 | 11.8 s |

新的加载方式：
- shard 1 的数据区按顺序整段读一遍（O_DIRECT，32 MB 一块，8 个 pinned 环形缓冲）。
  这块消费级 NVMe 单流能跑 2.2–2.3 GB/s，多个并发流在不同位置读时合计只有 1.3 GB/s，所以只用一个读线程。
- 4 个线程分发每一块：稠密矩阵直接从 pinned 块异步拷进显存；专家字节拷进各自的 [gate | up | down] 位置。
  稠密矩阵在 `load()` 里只做 cudaMalloc，草稿 head 改为加载后在 GPU 上按行收集。
- 专家仓库改为普通匿名内存 + MADV_HUGEPAGE，每层 2 MiB 对齐。
  一层读完后由第三个线程做 `cudaHostRegister`（先写入后注册几乎不额外花时间），并把该层要缓存的专家异步拷进显存。
- 内存不够放下全部专家，或设置 `BL_FAST_LOAD=0` 时，走原来的路径。

| | 原来 | 现在 |
|---|---|---|
| 稠密权重 | 11.8 s | 0.8 s |
| 锁页分配 | 14.5 s | 0（与读取重叠，注册共 1.6 s） |
| 专家读取 + 显存填充 | 100.8 s | 19.8–21.3 s（读线程 96% 时间在等磁盘：已到磁盘上限） |
| **总计** | **131 s** | **25.9 s** |

- 正确性：新旧加载路径在 mixed128 的 128 个位置上 logits 逐位相同，生成内容相同。
- 系统刚释放大量内存（page cache 回收）时，第一次加载可能慢到约 32 s。
- 附带收益：专家内存用上了大页，CPU 算专家时 TLB 未命中更少，每窗口 9.1 → 7.2–8.2 ms。
  全语料 decode：spec 3 88.3 tok/s（原 81–84），spec 0 53.2 tok/s。

## 2026-10-04 提速项 1、2：实现了，端到端收益约为 0

### 1a. 不录制空跑的 kernel
LRU 模式下没有预取；窗口里未命中全部交给 CPU，所以 DMA 阶段为空；MTP 层只有命中。
这些阶段的 wait/gate_up/down 不再录进 graph，每层少 4–6 个 kernel。

### 1b. 每轮从约 5 次同步降到 2 次
- commit 之后不再同步。
- MTP 补算和草稿链放进同一个 graph：子集索引→token 的映射在 GPU 上做（`gather_rows`），
  下一步直接读上一步的 token 和残差。每轮只在读草稿时同步一次。
- 代价：链总是算满 2 步，每轮 MTP 从约 2.6 次变成 3 次。
- 回归：test_session、test_window 通过。1800 token 提示词下 spec 0 与 spec 3 的输出在第 96 个 token 后分叉；
  两次相同的 spec 0 运行在第 58 个 token 就分叉，所以这是 prefill 的 atomicAdd 不确定性，与投机无关。
- 结论：之前 profile 显示的每轮 4.4 ms 空隙，大部分是 nsys 自己的开销
  （有 profiler 时 verify graph 的 launch 要 1.9 ms，不开 profiler 时只有 0.03 ms）。
  不开 profiler 时，一轮 = verify 25.8 ms + 草稿 3.5 ms，已经没有未解释的空隙。
  少掉的同步和多算的链步大致相互抵消。

### 2. CPU 专家的多 token AVX2 kernel（`src/cpu_iq.cpp`）
- 对 IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S，每个 32 值块只解码一次（grid 查表、±1 符号向量、整数 scale），
  窗口里每个 token 用 sign + maddubs + madd 累加。与 ggml 的 vec_dot 结果逐位相同（bench_cpu 检查）。
- 单线程（GB/s，按权重字节）：

| | ggml（每个 token） | 我们，1 个 token | 我们，4 个 token（折算到每 token） |
|---|---|---|---|
| 吞吐 | 2.8–4.1 | 2.6–3.7 | 5.6–10.7 |

- 引擎里 CPU 侧的任务大多只服务 1 个 token，这时解码速度（约 3 GB/s/线程）才是瓶颈，和 ggml 打平。
  所以只在 ≥ 2 个 token 时用新 kernel。
- x 的量化改成并行，每次调用 17 → 5 µs。
- 用超线程（12 或 16 线程）没有收益。
- 全语料 A/B（spec 3，各 2 次）：新 kernel 87.9 / 87.1，只用 ggml 87.8 / 88.9 —— 没有差别。

### 现在的看法
decode 每轮约 29 ms，其中 GPU 等 CPU 算专家约 5 ms，主要受命中率限制。CPU 单 token 解码已接近 AVX2 的实际上限。
接下来真正有效的方向：
1. GPU kernel 向带宽靠拢：稠密约 9.7 ms/轮、命中专家约 6.4 ms/轮，都只跑到约 50% 带宽；
2. 提高命中率，减少 CPU 工作量。

## 2026-10-04 GPU kernel：稠密 matmul 与命中专家

### 稠密（第 1 项）
- **BF16 hc_down + inject**（K=10240，324 行）：每行都从 L2 重读 T×K 的 fp32 x（160 KB），x 的流量是权重的 8 倍。
  改为 split-K：每个 block 负责 16 行（每个 warp 2 行）和 K 的一段，这段 x 只在共享内存放一份。
  各段的部分和由最后完成的 block 按固定顺序相加（不用 atomic）。切分数只取决于形状，所以窗口和单 token 仍逐位一致。
  冷 L2：约 27 µs → T=4 14.5 µs、T=1 12.8 µs。路由矩阵（BF16 512 行）也走这条路：17.5 → 7.5 µs。
- **hc_up_mix**：原来一个 warp 一行一行算，40 个 16 字节块分两轮，第二轮只有 8 个 lane 干活。
  改为每 8 个 lane 负责一个流的一行，4 组正好覆盖 4 个流，每个 lane 5 次加载一次发出。约 20 → 16.6 µs（T=4）。
- **量化矩阵**（q8 路径）：K=2560 有 80 个 32 值子块，32 个 lane 要分 2.5 轮。
  改为 16 lane/行、一个 warp 2 行。qkv 31.6 → 24.7 µs，q 36.8 → 29.2 µs，ssm_out 19.8 → 17.0 µs，
  output.weight 760 → 587 µs（745 GB/s）。8 lane/行更慢。fp32 路径（BL_NO_Q8）保留一个 warp 一行，以保持和单 token 逐位一致。

### 命中专家（第 2 项）
- `dot32m` 加了 token 掩码：T=4 窗口里每个专家平均只被约 1.3 个 token 选中，没选中的 token 不再计算。
- gate/up 改 16 lane/行（80 个子块，每 lane 5 个）；down 改 4 lane/行（20 个子块，每 lane 5 个）。
- bench_experts（T=4、30 个专家）：gate/up 106–140 → 54–69 µs，down 57–71 → 32–44 µs，平均 302 → 543 GB/s。

### 结果
- 正确性：test_window 逐位一致，test_session 通过；mixed128 KL：窗口路径 0.0063，prefill 0.0097。
- 中途一个 bug：掩码默认值写成 0xff，prefill 一次算 16 个 token，后 8 个被屏蔽，KL 变成 0.40。已改为全 1。
- 全语料：spec 3 87–89 → **96.5 / 98.1 tok/s**，spec 0 53.2 → **57.8 tok/s**；每个窗口约 26 → 21.7 ms。
- 新的每轮分布（25.4 ms GPU）：等 CPU 专家 6.1 ms（24%），稠密约 10 ms，命中专家 3.9 ms（原 6.4），其余约 5 ms。

## 2026-10-04 能用了：采样、分词、对话模板、bl-chat、bl-server

- **采样**（`src/cuda/sample.cu`）：每行一个 block。radix select 选出 top-k（≤ 256）→ 按 (logit 降序, id 升序) 排序 →
  temperature softmax → top-p 截断。对 MTP 的确定性草稿 d：以 p(d) 接受，否则从去掉 d 后重新归一化的分布里采样，
  输出分布与不投机时相同。`test_sample`：无草稿、高概率草稿、低概率草稿、被截掉的草稿四种情况，
  20 万次采样，最大偏差 ≤ 2.4σ，接受率等于 p(d)。
- **分词**：llama.cpp 的 libllama 以 vocab_only 方式加载，0.7 s，中英文往返完全一致；EOG 用 `llama_vocab_is_eog`。
- **对话模板**（`Text::render`）：把模型的 Jinja 模板中纯文本、无工具的部分写成 C++
  （system 合并、推理强度说明、`<think>` 处理、tool 消息）。
- **Chat 层**（`src/chat.cpp`）：渲染整段对话；会话文本是新文本的前缀时，只喂后缀（多轮不重复 prefill）。
  思考内容和回答以 `</think>` 为界分开；流式输出做 UTF-8 完整性处理。
- **bl-chat**：终端聊天，支持命令 /reset /think /temp /stats。
  实测：第 1 轮 18 个提示 token，首字 0.39 s，70 tok/s；第 2 轮复用 96 个 token，首字 0.21 s，92 tok/s；
  思考模式下答案正确，推理以暗色显示。
- **bl-server**：OpenAI 兼容的 `/v1/chat/completions`，支持流式 SSE、`reasoning_content`、`usage.cached_tokens`。
  多轮请求复用会话（第 2 轮 `cached_tokens: 27`）。
- 采样（温度 0.7）时草稿接受率约 54%（贪心约 70%）。
- 用法见 `docs/USAGE.md`。

## 2026-10-04 长提示词 prefill：8K 分块 + 张量核心专家

- **分块 2048 → 8192**：每块都要把各层未缓存的专家经 PCIe 流式传一遍，块越大传的次数越少。
  32K 提示词的专家拷贝次数从 19.6 万降到 5.9 万；工作区约 1.75 GB，从专家仓库临时借用。
  50.0 → 39.8 s。之后 PCIe 只忙 2 s，瓶颈变成 GPU 计算：8K 单块 9.6 s 里，专家 gate/up 3.5 s + down 1.4 s。
- **专家 kernel 的 lane 利用率**（16 lane/行、4 lane/行）：8K 9.63 → 8.36 s。32 token 的 tile 反而更慢。
- **专家改用张量核心**（`mma.sync m16n8k16` int8）：每个 block 处理一个专家的 64 行 × 64 个 token。
  每个 32 值子块解码一次（`decode32.cuh`：int8 加两个 16 值半块的缩放），所有 token 共用；
  半块的 int32 积乘以（权重缩放 × 激活缩放）后累加成浮点。
  8K 单块：专家 4.9 → 1.15 s，总计 8.36 → 6.0 s。
- **精度**：mixed128 prefill 的 KL 为 0.00555（dp4a 版 0.00559）。`BL_MOE_DP4A=1` 可切回旧 kernel 做对照。
- **结果**：32K 提示词 **50 s → 25.9 s**（约 1240 tok/s）。
- 8K 单块里现在剩下：cuBLAS 1.19 s、注意力 1.0 s、GDN 递推 1.0 s、专家 1.15 s。
  下一步可做张量核心 flash attention 和分块并行的 GDN。

## 2026-10-05 128K 上下文：int8 KV、索引器原始键环形缓冲、MTP 环形 KV

128K 时显存的大头（改之前）：13 层 fp16 KV 3.4 GB、索引器原始键（fp32，每层每位置 512 B）0.8 GB。
它们都从专家缓存里扣。

- **索引器原始键改成 512 行的环**（`kIdxRing`）：一个块只需要自己的 4 个原始键，prefill 一次最多写 256 行。
  无损，128K 时省 0.8 GB。
- **MTP 层的 KV 改成 4096 格的环**：MTP 只看最近 2051 个位置。128K 时省 0.27 GB。
  验证（确定性模式 `BL_CPU_SHARE=0 BL_PREFILL=0`，4K 提示词，环会绕回）：环 4096 与不绕回的环（`BL_MTP_RING=16384`）
  草稿接受数完全相同（127/180），输出相同。
- **int8 KV**：每 32 个值一个 fp16 缩放（`src/cuda/kv.cuh`），每行 256 值占 272 B（原 512 B）。
  写入（qsa_pre_t）、decode 注意力、prefill 注意力都改为模板，`BL_KV_F16=1` 切回 fp16。
- 精度（4K 提示词最后 256 个位置，同一测试程序，一次加载同时测 prefill 和窗口路径）：

| | fp16 KV | int8 KV |
|---|---|---|
| prefill 对 llama.cpp：中位数 KL / top-1 | 0.0063 / 90.2% | 0.0081 / 90.6% |
| 窗口路径对 llama.cpp：中位数 KL / top-1 | 0.0032 / 91.4% | 0.0022 / 89.8% |
| int8 对 fp16（直接比）：中位数 KL | — | prefill 0.0009，窗口 0.0016 |

  int8 带来的差异远小于与 llama.cpp 原有的差距；平均 KL 仍由阈值附近块的长尾主导（两种 KV 都是 0.16–0.21）。
- 测试程序：`bl-run --prefill-logits A --save-logits B --save-last 256 --window 4` 一次加载跑两条路径；
  `logits_compare.py` 增加中位数。

**128K 实测**（`bench/prompts/128k.tokens`，代码审查任务，`--ctx 128300`，spec 3）：

| | 结果 |
|---|---|
| 专家缓存 | 15.77 GB（改之前估计约 13 GB；4K 上下文时 17.65 GB） |
| prefill | 148 s（约 860 tok/s）。32K 是 25.9 s，按线性外推约 104 s，多出来的主要是索引器打分（随长度平方增长） |
| decode | **68.8 tok/s**；每轮 30 ms，其中等 CPU 专家 15.5 ms（命中率只有 90%：长 prefill 之后缓存还是冷的） |
| 输出 | 连贯，能指出提示词里具体文件、函数中的问题 |

- 128K decode 的主要成本不是注意力或索引器，而是 prefill 刚结束时专家缓存命中率低。

## 2026-10-05 长上下文 decode 变慢的原因排查（两个假设都不成立）

现象：同一个代码审查任务，1K / 32K / 128K 提示词的 decode 是 100 / 81 / 69 tok/s。

**假设 1：长 prefill 之后专家缓存是"冷"的。** prefill 结束时按提示词末尾的专家使用频率决定缓存内容
（`BL_PF_ADMIT_WIN` 统计最后多少个 token，`BL_PF_ADMIT` 每层最多换入多少个；每次表更新超过 64 个时分批）。
32K 提示词，每组单独进程：

| 策略 | decode 命中率 | tok/s |
|---|---|---|
| 原来：最后 64 token，每层最多 48 个 | 88.5% | 80.5 |
| 最后 1024 token，每层最多 512 个 | 86.8% | 76.9 |
| 最后 8192 token，每层最多 512 个 | 86.8% | 73.9 |
| prefill 不换入 | 89.2% | 81.5 |
| 对照：1K 提示词 | 89.0% | 100.3 |

按提示词换入反而更差：回答（文字、markdown）用的专家和提示词（代码）不同。1K 提示词的命中率也是 89%，
所以前 200 个 token 命中率低是缓存初始状态的普遍问题，与上下文长度无关。默认值保持不变。

**假设 2：MTP 只看最近 2051 个位置。** 也可以看更长，比如 32768。
decode 注意力 kernel 改为一个 block 可以依次处理多个 32 位置的块（在线 softmax 合并；每 block 一块时结果逐位不变），
`mtp_window` 可设（`BL_MTP_WINDOW`，或 `set("mtp_window")`），MTP 的环按窗口大小取 2 的幂。
32K 提示词：窗口 2051 接受率 65.6%、MTP 每次 1.12 ms；窗口 32768 接受率 66.7%、每次 2.14 ms。没有收益，默认保持 2051。
同一提示词多次运行的接受率在 60–66% 之间波动（prefill 不确定，有时生成文本不同）；
1K 提示词的 78% 是因为生成的文本不同，不是上下文长度造成的。

**结论**：长上下文的真实额外成本是每轮多 2–4 ms（32K / 128K，索引器和注意力）；其余差异来自生成内容。
decode 的最大成本在所有长度下都一样：前几百个 token 命中率约 89%，每轮等 CPU 专家约 13 ms（预热后命中率 96–97%）。

## 2026-10-05 重启时恢复专家缓存（机制可用，没有收益，默认关闭）

- `Engine(model, ctx, cache_file)` 启动时把保存的缓存列表排在初始统计之前，并按原顺序恢复 LRU 时钟；
  `save_cache(path)` 按最近使用从新到旧写出（先写 .tmp 再 rename）。文件约 40 KB。
  `bl-server --expert-cache FILE` 每次回复后保存；`bl-run --cache-file / --save-cache` 用于测试。
- 实验：先用 3 段语料（chat_zh、en_docs、code_llama）各生成 200 token 并保存，再在 1K 代码审查提示词上恢复：
  命中率 88.4%，冷启动（初始统计）89.0%。没有收益。
- 原因：新文本前 200 个 token 的命中率本来就在 88–94% 之间，与缓存冷热关系不大（语料基准 `ref/corpus_v11.log`
  同一进程连续 7 段：88.7–94.4%；本次 3 段：92.2、90.6、89.3%）。早先记录的"96–97%"来自先在同一段文本上预热两遍的测试，
  不代表新文本。每层显存只放得下约 38% 的专家，每段文本用到的组合不同。

## 2026-10-05 索引器打分：每个块键只读一次

- 微基准 `bench_idx [T] [ctx]`（不加载模型，几秒）：在不同位置调用 idx_select，并估算整个 prefill 的索引器时间；
  同时把新打分 kernel 与原 kernel（`idx_select_ref`）的分数和选择逐位比较。
- 测量：128K prefill 里索引器约 29.3 s。nsys 显示 95% 在打分（`idx_score_k`），不是选块。
  原因：一个线程算一个（查询，块键）对，每个查询都把全部块键读一遍。128K 时块键 16.7 MB，超过 6 MB 的 L2，
  一次调用（256 个查询）要读 4.3 GB。32K 时块键 4 MB 还在 L2 里，所以以前没暴露。
- 新 kernel `idx_score_tiled_k`：一个线程把一个块键（128 维）放在寄存器里，查询 16 个一批放共享内存（广播读），
  查询再按 64 个一组分到 grid.y（短上下文时 block 太少）。求和顺序与原 kernel 相同：分数、选择逐位一致。

| 每次调用（256 个查询） | 原来 | 现在 |
|---|---|---|
| 8K 处 | 0.30 ms | 0.19 ms |
| 64K 处 | 3.91 ms | 0.52 ms |
| 128K 处 | 13.7 ms | 1.44 ms（打分 0.69 + 选块 0.80） |
| 128K prefill 合计（估算） | 29.3 s | 3.4 s |

- 端到端 128K：prefill **148.4 → 113.9 s**（约 1120 tok/s，接近按 32K 线性外推的 104 s）。
- decode 每轮没有变化（30.0 → 29.9 ms）：微基准里每层省 0.08 ms，但 decode 的关键路径是等 CPU 专家（每轮 15 ms），
  GPU 上省下的时间只是变成了等待。

## 2026-10-05 prefill：张量核心注意力、GDN 并行化

8K 单块 prefill（32K 提示词的前 8192 token，nsys 按 kernel 汇总）原来 6.43 s：注意力 1.42 s、GDN 递推 0.79 s、
cuBLAS 约 1.1 s、专家约 1.1 s、GDN 前处理 0.24 s、to_bf16 0.22 s。

- **注意力**（`attention_prefill_mma_k`）：原来一个 warp 处理一个（token，头），每个位置一次 warp 归约，
  同一 token 的 12 个 q 头各读一遍 K/V。索引器按 token 选位置，所以一个 block 处理一个（token，kv 头）：
  12 个头补成 16 行，做 mma.sync m16n8k16（fp16 输入、fp32 累加）；每批 32 个位置的 K/V 只取一次（反量化成 fp16
  放共享内存），在线 softmax，P·V 的 B 片段用 ldmatrix.trans 取。共享内存行宽补 8 个 half 避免 bank 冲突。
  1.42 → 0.41 s。`BL_ATTN_OLD=1` 切回旧 kernel。
- **GDN 前处理**：卷积只读前 3 个 token 的输入（都在显存里），改成按 token 并行（网格 头 × token），结果逐位相同。
  0.24 s → 可以忽略。
- **GDN 递推**：原来 192 个 block × 1 个 warp，每线程一列 128 行，两条 128 长的乘加依赖链，每 token 每层约 2.6 µs。
  改为每列 4 个线程（交错分行，避免 bank 冲突），部分和用 shuffle 合并；下一个 token 的 q、k 双缓冲预取。0.79 → 0.37 s。
  decode 用同一个 kernel。
- 精度（4K 提示词，对 llama.cpp）：prefill 中位数 KL 0.0046、top-1 93.0%；窗口路径 0.0030、92.6%（都在原来的波动范围内）。
  改动前后直接比较：注意力 0.0012，GDN 0.0010（prefill）/ 0.0024（窗口）。

| | 原来 | 现在 |
|---|---|---|
| 8K 单块 | 6.43 s | 4.96 s |
| 32K 提示词 | 25.9 s | **20.0 s** |
| 语料 decode spec 3 / spec 0 | 96.5–98.1 / 57.8 | 97.3 / 57.4（不变） |

- 8K 单块里还剩：cuBLAS 1.07 s、专家 1.0 s、注意力 0.41 s、GDN 0.37 s、to_bf16 0.22 s；
  GPU kernel 合计 3.99 s，墙钟 4.96 s，约 1 s 是 GPU 空闲（主机在等路由结果、组织专家任务）。

## 2026-10-05 decode 关键路径的重新测量

- **纠正一个误读**：`report()` 里的"waiting for doorbells"是**主机**在等 GPU 的门铃（GPU 在算），不是 GPU 在等 CPU。
  之前据此说"GPU 每轮等 CPU 12–13 ms"是错的。
- 新增 GPU 侧计时：`wait_k` 用 `%globaltimer` 把自旋时长按类型累加（`cuda::wait_times`），`report()` 打印
  "GPU waiting per window"。同一 stream 上 wait 排在命中专家和共享专家之后，所以这就是 GPU 因 CPU 未算完而空闲的时间。
- chat_zh，spec 3：每轮 20.65 ms，**GPU 等 CPU 专家 4.58 ms（22%）**；主机算 CPU 专家 6.78 ms/轮
  （每层约 1.5 个未命中，约 140 µs：量化 5 µs、gate/up 88 µs、量化激活 2 µs、down 32 µs）。
  即使 CPU 专家不花时间，decode 最多也只能快约 28%。
- **CPU 线程数**：`bench_cpu` 线程池 8 → 12 线程吞吐 +30%（16 与 12 相同，超线程有效）；引擎里只有 +3%
  （语料 97.3 → 98.1，chat_zh 每轮 21.6 → 20.8 ms，噪声范围内）：每层任务小，阶段间同步占比大。默认仍为 8。
- **LRU 下的下一层预取**（`set("prefetch", 2)`，spec 0）：63.1 / 62.8 → 58.7 / 60.6 tok/s。约 6400 次预取只用上约 1600 次（25%）：
  LRU 已经缓存了常用专家，"预测会用但不在缓存里"的多是冷门专家。预取还和准入争 PCIe。确认关闭是对的，不扩展到窗口。
- nsys 在 `--cuda-graph-trace=node` 下投机解码几乎不出草稿（199 轮出 199 个 token），这种 profile 不代表真实 decode。

## 2026-10-05 decode 的 GPU 时间分解（BL_STAGE_PROF）

- 方法：`BL_STAGE_PROF=1` 时，窗口 graph 在每层各阶段边界插一个 1 线程的 `stamp_k`，写入 `%globaltimer`；
  `verify()` 同步后读回，按阶段累加相邻打卡点的差。开头两个紧挨着的打卡点量出打卡本身的开销（1.1 µs/个），从各阶段扣除。
  每轮多约 0.4 ms，接受率与总速度和平时一致（不像 nsys 那样干扰投机解码）。
- chat_zh，spec 3，每轮（ms）：

| 阶段 | 每轮 | 说明 |
|---|---|---|
| 等 CPU 专家 | 5.19 | 每层约 108 µs |
| 命中专家 | 3.66 | 约 2 GB/轮，约 550 GB/s |
| hc 混合（每层 2 次） | 3.35 | 每次读 13 MB 的 BF16（10240×320 + 320×10240），每轮约 1.25 GB，约 370 GB/s |
| GDN 层注意力 | 3.19 | 36 层，89 µs/层 |
| 路由 + 规划 | 1.88 | 39 µs/层，主要是三个串行小 kernel 的延迟 |
| QSA 层注意力 | 1.84 | 12 层，153 µs/层 |
| 输出头 | 0.65 | 437 MB，接近带宽上限 |
| 共享专家 / 合并 / 嵌入 | 0.43 / 0.43 / 0.01 | |
| 合计 | 20.6 | 另有 MTP 草稿每轮约 3 × 1.05 ms，不在窗口内 |

  en_docs 的比例相同（合计 20.3 ms，等 CPU 5.4 ms）。
- 意外：hc 混合是每轮读得最多的权重（比命中专家还多），而效率只有带宽的约 44%。

## 2026-10-05 hc 混合：两个尝试

- **降维 split-K 改为每块 512 个值 + 先发出全部读取**：T=4 16.8 → 18.0 µs（更慢），撤销。
  同样 6.5 MB 的权重，T=1 约 12 µs、T=4 约 17 µs：T=4 时多出的 4.5 µs 来自每个权重 × 4 个 token 的乘加、BF16 转换和共享内存读取，
  不是纯带宽问题，所以只增加在途访存没有用。
- **hc 权重用 Q8_0 的精度**（`BL_HC_Q8_SIM=1`：加载后把 298 个 hc 矩阵经 Q8_0 舍入再写回 BF16，kernel 不变；
  比真正的 Q8_0 多一次 BF16 舍入，偏悲观）。4K 提示词最后 256 个位置：
  对 llama.cpp 的中位数 KL prefill 0.0046 → 0.0020、窗口 0.0030 → 0.0024；top-1 93.0 → 92.6%、92.6 → 89.5%；
  与当前版本直接比较：中位数 KL prefill 0.0058、窗口 0.0028（int8 KV 0.0009、注意力 0.0012、GDN 0.0010 的 3–5 倍）。

## 2026-10-05 CPU 专家：阶段合并没有空间

- `BL_CPU_PROFILE=1` 增加线程利用率（任务内时间之和 ÷（阶段墙钟 × 线程数））。chat_zh，spec 3，每层：

| | 8 线程 | 12 线程 |
|---|---|---|
| gate/up | 120.6 µs，利用率 93% | 130.1 µs，87% |
| down | 56.5 µs，95% | 58.6 µs，95% |
| 量化输入 / 激活 | 4.6 / 2.1 µs | 5.5 / 2.5 µs |

- 利用率 93–95%：同步和分配不均只占 5–7%，合并阶段每轮最多省约 0.2–0.3 ms，不做。
- 12 线程在引擎里更慢：权重每次从内存读（gate/up 约 15 GB/s），同一核心的两个超线程争 AVX2 单元。
  `bench_cpu` 显示 12 线程 +30%，是因为它反复算同 2 个专家（3.4 MB，在 16 MB 的 L3 里）——和 GPU 微基准的 L2 热缓存同一个坑。
- 计算受限：每个核心约 2 GB/s。IQ 格式每 32 个权重要 12 次标量查表（网格 + 符号）再拼向量，AVX2 没有更快的替代
  （这代 CPU 的 gather 更慢）。ggml 的实现也是这个思路，没有便宜的提速办法。

## 2026-10-05 prefill 的 GPU 空闲：是在等 PCIe，不是等主机

- 测法：nsys 录 8K prefill（prefill 不用 graph，nsys 不干扰），导出 GPU trace，统计计算 stream 上的空隙及其前后的操作。
  span 4683 ms，计算 stream 忙 3919 ms，空闲 764 ms：
  - 671 ms 在专家 kernel 之前（等 staging 里的专家拷完），313 次；
  - 76 ms 是 7.7 万个 < 20 µs 的启动间隙；主机组织任务（memset → copy_words）只有 17 ms。
  - 同期 H2D 拷贝 stream 忙 1974 ms，24.5 GB，12.4 GB/s（PCIe 跑满）。
- 原因：每层约 300 个未命中专家。预取环（128 个，17.6 ms 拷完）在本层注意力阶段（约 57 ms）就拷完了，PCIe 剩约 40 ms 空闲；
  其余约 160 个到 MoE 阶段才分批拷（每个 137 µs，比算它的约 83 µs 慢），专家 kernel 等拷贝。
- 预取环加大（`BL_PF_RING`，默认 128 → 384；prefill 期间多借约 440 MB 显存，结束后归还）：

| 环 | 8K 块 | 拷入专家 |
|---|---|---|
| 128 | 4.91 s | 13894 |
| 256 | 4.48 s | 14467 |
| 384 | 4.43 s | 14811（多出的是预测错误，用的是本来空闲的 PCIe） |

- 32K 提示词：20.0 → **17.9 s**。计算本身不变（权重只是从预取环而不是 staging 槽读），不需要重测精度。

## 2026-10-05 128K 复测（今天全部改动之后）

| 128K 代码审查提示词，spec 3 | 早上（int8 KV 后） | 索引器优化后 | 现在 |
|---|---|---|---|
| prefill | 148.4 s | 113.9 s | **74.2 s（约 1720 tok/s）** |
| decode | 68.8 tok/s | 66.6 tok/s | **74.3 tok/s** |

- 生成内容与之前相同（前 40 多个 token 一致）。
- decode 每轮 29.5 ms，其中 GPU 等 CPU 专家 9.6 ms（短上下文约 4–5 ms）：128K 时专家缓存 15.8 GB（4K 时 17.5 GB），
  命中率 88.1%。长上下文下给专家缓存腾显存的价值更大。

## 2026-10-05 给专家缓存腾显存

- 显存账（ctx 32768）：总 25.29 GB；专家之前 7.01 GB = 稠密矩阵 4.41 + 缓冲区 1.36（其中 MTP 的 512 个 Q2_0 专家 0.71、
  int8 KV 0.40）+ staging 0.13 + CUDA 上下文 / cuBLAS 等 1.12；专家缓存 17.20；预留 1.07。跑完 prefill 和 decode 后还剩 0.72 GB
  （预留只用了约 0.35 GB）。加载时打印这份账，`report()` 打印当前空闲显存。
- **token_embd（273 MB）放到映射的锁页主机内存**（`map_mat`；每轮只读 T 行，MTP 每次 1 行）：嵌入阶段每轮 0.01 → 0.02 ms。
  `BL_EMBD_VRAM=1` 放回显存。
- **降预留没有成功**。7 段语料，spec 3，ctx 32768：

| 配置 | 专家缓存 | 跑完剩余 | 平均命中率 | GPU 等 CPU | tok/s |
|---|---|---|---|---|---|
| 预留 1024，嵌入在显存（原来） | 17.20 GB | 0.72 GB | 90.2% | 9.37 ms | 95.3 / 95.7 |
| 预留 1024，嵌入在主机（采用） | 17.48 GB | — | 91.1% | 9.67 ms | 96.8 |
| 预留 800，嵌入在主机 | 17.71 GB | 0.49 GB | 90.6% | 10.74 ms | 92.4 |
| 预留 600，嵌入在显存 | 17.65 GB | — | 90.9% | 10.65 ms | 91.9 |
| 预留 600，嵌入在主机 | 17.92 GB | 0.28 GB | 91.4% | 10.01 ms | 93.7 / 92.9 |

  预留低于 1 GB 时稳定变慢（同一配置两次运行相差 < 1%），而且慢在 GPU 等 CPU 上，CPU 端代码没变。
  可能是显存紧张时驱动的某种行为，也可能与缓存超过某个大小有关，数据还不能区分。待查；预留保持 1024 MB。

## 2026-10-06 整理成可以公开的仓库（nano-cuda-moe-qwen38）

- **构建**：CMake 自动下载官方 llama.cpp `3cf0325`（源码包，校验 SHA-256），作为静态、只编 CPU 的子项目，
  提供 ggml-cpu（CPU 专家）、libllama（分词）、vendor/ 下的 httplib 和 json。之前依赖工作区里的另一份源码、
  一个预编译的库目录，连 ninja 都来自别的项目的虚拟环境。全新 clone 到编译完成 1 分 33 秒。
  分词结果、CPU kernel 与 ggml 的对比、窗口与逐 token 的逐位一致性都和改动前相同。
- **MTP 层**：`tools/fetch_mtp.py` 用 HTTP Range 只从官方 BF16 权重（360 GB）里取 31 个 `mtp.*` 张量（5.21 GB）；
  `bl-mtp-pack` 把专家量化成 Q2_0：每 64 个值一组，缩放在 amax × 0.30…1.00 的 36 个候选里取均方误差最小的，
  每个候选再用最小二乘修正一次。与之前用的那份 MTP 文件相比：其余 29 个张量逐字节相同；
  专家的相对均方根误差 0.388 / 0.385（gate_up / down），之前那份是 0.429 / 0.424。
- **草稿词表**（`tools/make_draft_vocab.py`，只读词表）：中日韩文字的 token 全部收录，ASCII token 取编号 < 40960 的
  （BPE 的编号就是合并顺序），空白、数字和特殊 token 全收，其他文字不收。106,166 个，覆盖之前那份的 93.7%。
- **专家初始排名**（`tools/make_rank.py`）：7 段语料以窗口路径运行时记录的路由（16,641 个 token），按次数排序。
- **默认路径**：MTP 在模型文件旁边（`models/mtp-q2_0.gguf`），排名和词表在仓库的 `data/`。
- A/B（7 段语料，spec 3）：

| | 自己的 MTP、词表、排名 | 之前用的那套 |
|---|---|---|
| decode | 95.48 tok/s | 95.10 tok/s |
| 草稿接受率 | 69.6%（874/1256） | 71.9%（902/1254） |
| 平均命中率 | 91.2% | 91.1% |

- 32K 提示词 18.0 s（启动后第一次 20.2 s：n-gram 表要从 SSD 读，页缓存还是空的）。
- 文档：中英双语 README、Apache-2.0、第三方声明、插图脚本、下载脚本（模型分片的 SHA-256 来自 Hugging Face，
  本机的两个文件已核对一致）。

## 2026-10-06 测试输入换来源，重新测量

- 语料 `code_cuda` 换成本仓库的 `src/cuda/window.cu`，`en_docs` 换成 llama.cpp 的 `docs/build.md`；
  1K–128K 长提示词改由 llama.cpp `3cf0325` 的源码树生成（`bench/make_corpus.py`、`bench/make_prompts.py`）。
- 同一台机器、同一个 GGUF、新的输入：

| | 本引擎 | Strata（它安装时写的配置：spec 4、int8 KV、32K） |
|---|---|---|
| 7 段语料 decode（各 200 token） | **97.2 tok/s** | 65.0 tok/s（每段一个新进程；1400 token / 21.5 s） |
| 32K 提示词 | 18.0 s（1775 tok/s） | **13.3 s（2410 tok/s）** |
| 32K 之后的 decode | 82.1 tok/s | 84.6 tok/s |
| 128K 提示词 | 78.3 s，之后 decode 74.8 tok/s | （它的配置最长 32K） |

- 长提示词 prefill 还落后：Strata 的 32K 是 4 块、6.6 万次专家 DMA，13.3 s。值得再看本引擎 prefill 的流水线。

## 2026-10-07 256K 上下文（模型的上限）

- GGUF 的 `context_length` 是 262144，RoPE 没有外推：256K 是模型原生的长度。引擎里没有写死的上限，256K 时 KV 下标最大约
  1.3 亿，离 int32 很远；注意力每次最多看 2051 个位置，与长度无关。
- 显存：12 个 QSA 层的 int8 KV 每 token 1088 字节，256K 共 3.42 GB，加索引器约 0.47 GB。`--ctx 262144` 下专家缓存
  14.07 GB（8061 个；32K 时 17.48 GB、10027 个），剩 0.74 GB。
- 一个进程、`--ctx 262144`，提示词由一棵 llama.cpp 源码树生成（`bench/make_prompts.py`，不进仓库）：

| | 读提示词 | 之后 decode | 命中率 |
|---|---|---|---|
| 短对话 34 token | 0.6 s | 99.8 tok/s（32K 设置下 105） | 91.1% |
| 代码 1800 token | 2.4 s | 124.5 tok/s（32K 设置下 142） | 89.8% |
| 128K | 78.7 s | 68.1 tok/s | 88.5% |
| 256K | **165.1 s**（1550 tok/s） | **66.3 tok/s** | 87.7% |

- 长度翻倍、时间 2.1 倍：索引器平方增长的那部分比预想的小。进程最大内存 48 GB。
- 256K 的回答指向 `llama_context::decode`、`embeddings_nextn_masked`、`embd_nextn`，都在提示词里，位置约在 35% 处、离结尾
  约 16 万 token：远处的内容取得到。
- 代价是短对话：缓存少了 2 千个专家，慢 5–12%。所以平时用 32K，需要时再用 `--ctx 262144` 启动。

## 2026-10-13 一个偶发崩溃（未解决）

- 复现：`bl-run --window 4` 连续喂 7 段语料（`bench/eval.py` 的窗口测量），大约三次一次在第 5 段之后崩溃，
  `free(): invalid size`（glibc 堆被写坏）；网页聊天和生成测试没有遇到过。
- 已排除：AddressSanitizer 版本（主机代码全部插桩）完整跑通、没有报告；GPU 到主机的异步拷贝全部写锁页内存。
  还没查：GPU 核函数通过映射指针写主机内存时的越界、CPU 专家线程之间的竞争。
- 考官遇到这种崩溃会重试一次，报告里记 `retried`。

## 2026-10-18 dsv41 分支：DeepSeek-V4.1-Flash-1017（阶段 A，正确性优先）

- 模型：acrogenesis 的 IQ2_XXS 专家 / Q8_0 稠密 / Q8_0 Engram 的 GGUF，revision `f5221b0`，10 个文件 359 GB
  （`scripts/download_dsv41.sh`）。规格与公式全部写在 [DSV41.md](DSV41.md)，依据是官方 `inference/model.py`、`kernel.py`、
  `engram.py`，与 llama.cpp PR #28696 逐条核对过。
- 下载：两个代理都连不上（SSL 失败）。最快的办法是 hf-mirror 只请求一次 302 拿到 HF CDN 的签名地址，再对 CDN 开 16 个连接：
  34–40 MiB/s；单连接约 6 MiB/s；对镜像本身开 16 个连接会被 429。镜像的 302 偶尔也回 429，脚本退避重试，不把镜像地址交给 aria2。
- 参考实现：GitHub 直连约 25 KB/s，`git fetch` 反复卡死；通过 ghfast.top 取 PR commit `2076412` 的源码包（8 s），
  tar 头的 commit id 一致，5 个关键文件与独立来源逐字节相同。CPU 版编出 `bl-ref-dump-dsv41`。
- 校验：GGUF 里 Engram 的素数、偏移、乘数与 `engram.py` 的构造方法重算结果完全相同；`tests/test_dsv41_ops` 每个 kernel
  对照参考公式的 CPU 实现，误差都在 fp32 舍入范围内（相对 1e-7 量级）。
- 结构上值得记的点：RoPE 只转最后 64 维、相邻配对（llama.cpp `LLAMA_ROPE_TYPE_NORM`，转换时不重排权重）；YaRN 不乘
  mscale；hc 系数滞后一个子层；只有 1 个 KV 头且 K = V；压缩 KV 与 indexer 只在源层计算、后面的层共用。
- 字节预算：路由专家 140.14 GB（每 token 读 2.19 GB）、稠密加 output 8.94 GB（每 token 全读，840 GB/s 下约 10.6 ms 的下限）、
  Engram 表 208.90 GB（每 token 48 行，几乎每行一个 4 KB 页）。

## 2026-10-19 dsv41：带 Engram 的解码优化（混合解码，`BL_DSV41_HYBRID=1`）

- 测量方法：`bl-dsv41 --force 64` 先 prefill 64 个 token，再逐 token 喂真实文本（teacher-forced，每次运行工作量相同），
  `bench/dsv41/real_{zh,cuda,en}.tokens` 各 512 token，统计第 128 步之后的 320 步。原来的 `code.tokens` 生成会重复（不带 Engram
  时陷进死循环），专家复用率虚高，不再用来比速度。prefill 的 logits 每次运行有 ±0.2 的差（原子加），自由生成长了会分岔。
- `BL_DSV41_TRACE` 记录每步每层的专家选择，`tools/dsv41_cache_sim.py` 离线重放缓存策略（与实测的 VRAM 未命中数吻合）。

| 改动 | zh ms/token | 依据 |
|---|---|---|
| 起点（Engram 打开） | 203 | 每 token 读盘 16.7 次、每次约 8 ms |
| IO 线程 4 → 1 | 176 | 这块 NVMe 并发多了反而慢（3 MB 块：1 线程 1.6 GB/s，4 线程 1.3） |
| VRAM / RAM 两层互斥（进 VRAM 就移出 RAM，淘汰时拷回） | 150 | 模拟器读盘 −28%，实测 −39% |
| 早拷入：一层 ≥2 个未命中时 1 个拷进 VRAM 由 GPU 算 + 预留空闲槽 | 141 | 这次拷贝本来就要做 |
| 2 个 IO 线程、读拆成 1.5 MB | 137 | `tests/bench_dsv41_ram.cpp`：一个专家 3 段分给 2 线程时尾段只有 1 路 |
| 层内流水：先算已在内存的未命中，同时等盘 | 132 | `ExpertRam::start/wait` |
| hc_fn 每行一个 block 并顺带求平方和、wo_a 一次发射 | 129 | hc 部分每步 3.1 → 1.1 ms |
| 小拷贝改用 kernel 读写锁页内存（不走拷贝引擎） | 116 | 路由结果、plan、CPU 输出排在 9 MB 专家拷贝后面，每层约 0.38 ms |
| 预读第 il+2 层路由的 top-2；共享专家与 CPU 专家并行 | 112 | |
| split-K 注意力（8 个头共用一次 key 读取） | 112.7 | 路由前 GPU 等待 14.9 → 12.5 ms（总时间被读盘波动盖住） |
| logits 与 step 开头的上传走锁页缓冲 | 110 | |
| q_a+kv、共享专家 gate+up 各合成一次发射（同一块分配，不复制） | 109.9 | 显存碎片少了，缓存多 13 个槽 |
| 准入过滤：一个专家在最近 1000 次未命中里第二次出现才进 VRAM | 106.3 | 每次拷入 = 9 MB 上 + 9 MB 回，占 PCIe 和内存带宽；回拷减半 |
| matvec 循环 `#pragma unroll 4`（共用 kernel，Qwen3.8 logits 逐字节不变） | 107 | wo_b 70 → 58 us |
| 从盘读的专家：gate/up 两段到了就先算，down 到了再算 | 107 | 等盘 49.8 → 48.2 ms |
| 自己的单行 rmsnorm（1024 线程）、q_a/kv 两个 norm 一次发射；路由阶段 4 个小拷贝一次发射；Sinkhorn 并进 hc_fn 的最后一个 block；注意力 16 个 key 一块 | **104.8** | 路由前等待 12.1 → 10.0 ms |

- 三段文本：zh 203 → 105、cuda 158 → 89、en 222 → 113 ms/token（8.8–11.2 tok/s）。
- CPU 专家：IQ2_XXS 在 K = 5120 上每线程 3.1 GB/s，冷热数据一样，线程数线性（`tests/bench_dsv41_cpu.cpp`），是解码算力的
  上限，不是内存；gather 取码本没有变化。`bench_cpu` 的尺寸是 Qwen 的，对这个模型的数字不能用。
- 没用的：直接 O_DIRECT 读进槽位（去掉中转 memcpy，中转缓冲本来就在 L3）；把一部分未命中经临时槽交给 GPU
  （按 0.77 / 0.49 ms 的代价平衡也一样：拷贝串行，GPU 端等待抵掉 CPU 的节省）；x 放 shared memory 的 matvec（更慢）；
  LFU / SLRU / ARC / LRU-2 都不如 LRU（模拟）；RAM 层也加准入过滤（模拟：读盘 12.9 → 14.6–19.5 次/步）；预读 top-16 把盘塞满（257 ms）；CPU 线程 8/12/16 无差别（IQ2_XXS
  解码算力受限，利用率 96%）；3 个 IO 线程；自适应早拷入个数；`BL_CPU_SPINS`。
- 事故：`BL_DSV41_RAM_GB=56` 的锁页池把内存用尽，机器死机重启。现在池子最多 MemAvailable − 8 GB；默认留 12 GB
  （运行中 MemAvailable 最低 11.4 GB）。
- 现在的构成（zh）：等盘约 51 ms、CPU 算未命中约 30 ms、路由前 GPU 约 13 ms、GPU 收尾约 9 ms。

## 2026-10-19 dsv41：prefill

- prefill 卡在读盘：64 token 一块时，每块每层用到约 250 个专家，40 层约 1 万个，超过 VRAM + RAM（约 7400），下一块又要重读。
  `BL_DSV41_CHUNK`：64 → 5.9 tok/s，512 → 9.1，1024 → 13.7（1024 token 的提示词）。
- 长提示词一块跑完（上限 `BL_DSV41_PREFILL_MAX` = 4096）：专家缓存分成 64 槽一段，prefill 时借出末尾的段（里面的专家先拷回 RAM），
  缓冲区放得下为止，完了再还回来。512 token 86 → 53 s，1023 token 约 173 → 69 s（14.8 tok/s）。check-prefill（512 个位置）：
  KL 0.0065、top-1 497/512（64 一块：0.0075、486/512）。
- 长 prefill 期间 RAM 层冻结：会挤掉常驻专家的读放进 3 层的临时槽。一块之内没有复用，读盘不变；紧接着的解码前 223 步
  150.7 vs 176.4 ms/token；超过 4096 token 的多块提示词跨块复用。
- 没用的：读下一层的全部专家（不冻结时挤掉后面要用的，冻结时也多读 28%：1023 个 token 也只用到每层约 3/4 的专家）。
- 输出头 BF16 → Q8_0（加载时转换，KL 2e-5）、显存预留 1.5 → 0.5 GB：专家缓存 1542 → 1728 槽；解码 zh/cuda/en 101/86/104 ms/token。

## 2026-10-19 dsv41：DSpark（官方的 MTP）测了接受率，结论是这台机器上不做

- 这个 GGUF 没有 DSpark 的 3 层；官方权重里在 `mtp.0-2`（分片 44-46，7.9 GB）。`bl-dspark-pack` 打成 GGUF：注意力等 FP8 → Q8_0，
  专家 MXFP4 → IQ4_XS（相对 MXFP4 的 RMS 误差 7.6%），路由器和 Markov 头保留 BF16，共 7.97 GB，其中专家 7.2 GB。
- 结构（官方 `model.py` DSparkBlock / forward_spec）：主模型第 37–39 层注意力输入的 4 路均值拼成 15360 维 → `main_proj` →
  `main_norm`；每层的窗口里只存主模型各位置的 KV；草稿块是 `[刚出的 token, noise×4]`，块内双向注意力；
  输出头和主模型共用，另加 Markov 偏置（依赖上一个草稿 token，所以 5 个草稿依次取）。
- 接受率：`tools/dspark_ref.py`（numpy，直接读官方 FP8/MXFP4 权重，无重量化）在主模型（IQ2_XXS）的贪心生成轨迹上
  （`bl-dsv41 --gen 256 --dump --filter 'l_last-(36|37|38)'`，提示词是三段测试文本的前 128 个 token）：

| 文本 | 平均接受（满分 5） | 每次验证产出 |
|---|---|---|
| zh | 3.06 | 4.06 |
| en | 2.32 | 3.32 |
| cuda | 4.35 | 5.35（生成内容重复） |

  逐位置接受率 86 / 74 / 65 / 54 / 45%。贪心轨迹偏重复，数字偏乐观。
- 但验证不便宜：`bl-dsv41 --force 64 --chunk 6`（6 个 token 一次走 prefill）861 ms 一块，单步 101 ms，即 8.5 步。
  专家轨迹：连续 6 个 token 每层用到 21.4 个不同专家（单个 6 个），而 CPU 算未命中按（token，专家）计、读盘按不同专家计。
  就算写一条理想的小批量混合验证路径，上限也只有约 1.1–1.2 倍，再减去草稿自己 7.2 GB 专家占掉的 VRAM / RAM 缓存。
- 结论：卸载场景下投机解码赚不到；模型整个放得进显存（多卡、96 GB 的卡）时才值得做。打包工具和参考实现留着。

## 2026-10-20 dsv41：YOCO prefill 截断（照 DeepInfer 的 decoder SWA bounded replay）

- DeepSeek 开源了推理框架 DeepInfer（只支持 SM100 和昇腾 950）。它的配置转换（`hugging_face/config.py`）把第一个
  ratio=1 的层（第 20 层）当作 YOCO 的分界线 `n_prefill_layer`：第 20–39 层的压缩 KV 和 indexer K 只来自第 20 层的输入，
  各层自己的窗口 KV 才用本层的 hidden（`yoco_swa_share_hidden: False`）。所以 prefill 时后 20 层只要跑提示词最后 128 个 token
  （官方 Q&A："总 prefill 计算量近乎减半"）。
- 实现（`prefill_chunk`）：到第 20 层先对整块算 hc 混合、norm、压缩器和 indexer K，再把 hc 流 `R` 和待用的 `pre` 缩到回放段的行，
  后面的层照常跑；回放段之前的位置没有后 20 层的窗口 KV，这些层的窗口从 `dec_from` 起（即官方的 `max(s, i-127)`）。
  只在不要求每个位置 logits 时生效（`--check-prefill`、`--save-logits` 不受影响）；`BL_DSV41_YOCO_KEEP`（默认 128，0 关）。
- 质量（`--force 1024`，再逐 token 喂 128 个真实 token，对比每步 logits）：截断 vs 不截断平均 KL 0.011，top-1 相同 124/128；
  两次都不截断之间就有 KL 0.009（混合解码的 CPU/GPU 累加顺序不固定），截断的误差和这个噪声差不多。猜中真实下一个 token：
  73/127 vs 75、72/127。
- 速度（交替各跑两次，同一 4096 token 提示词）：

| | prefill | 其中第 20 层起 | 读盘 |
|---|---|---|---|
| 不截断 | 81.6 s（50 tok/s） | 38.6 s | 13008 |
| 截断 | **55.1 s（74 tok/s）** | **12.1 s** | 9080 |

  1024 token：57 → 41.6 s。后 20 层就算只跑 128 个 token，每层也要用到大多数专家，所以没有到一半。
- 顺带发现：Engram 表的页缓存被挤掉时，4096 token 的 prefill 光读 Engram 行就要 43 s（热的时候 0.2 s），单次测量的方差主要来自这里。
- `bl-dsv41 --prefill` 的 bug：判断要不要存 logits 用的是没打开的 `ofstream` 的布尔值（总是 true），每次都申请全位置 logits
  （4096 token 是 2.1 GB，直接 OOM），截断也不会生效。改成 `is_open()`；`--force` 也能 `--save-logits` 了（逐步的 logits）。

## 2026-10-20 dsv41：思考模式的提示词对齐官方编码（deepseek-recipe）

- DeepSeek 同时开源了 deepseek-recipe（官方 API 的提示词编码，Rust + Python，PyPI `deepseek-recipe`）。拿它的
  `DeepseekV41Encoding.render_conversation` 对照：非思考模式和我们逐字节一致；思考模式官方在第一条消息前多一行
  `<｜System｜>Reasoning Effort: N (range 1-100, the higher the value, the more thorough the reasoning)\n\n`
  （low 50，max 100，其余 75；第一条是 system 时它的内容直接接在后面，不再写 `<｜System｜>`），我们原来没有。
- `render_dsv41` 改为接收 effort（`ChatParams::think`），移到 `bl` 命名空间以便测试；`tests/test_dsv41_template.cpp`：
  6 组对话 × 5 种 effort 共 30 个用例，期望值由 deepseek-recipe 0.2.0 生成，全部逐字节一致。
- 端到端：`bl-chat --think xhigh` 问"17 乘以 23"，思考后答 391，思考和正文分开正常。

## 2026-10-20 dsv41：工具调用（DSML，照 deepseek-recipe）；修 chat/server 长提示词的 cublasCreate 失败

- 修 bug：bl-chat / bl-server 里第一个超过 64 token 的提示词报 `cublasCreate failed`（之后每个请求都报 `cublasSgemm failed`）。
  cuBLAS 句柄是第一次 gemm 时才建的，那时专家缓存已经拿走剩下的显存、长 prefill 又借走了缓存段，句柄没显存可用。
  引擎加载时、算缓存大小之前先建好（`cuda::init_blas`）。bl-dsv41 碰巧没撞上（留的 0.5 GB 余量没被预热占满）。
- 工具调用：`render_dsv41` 改成按官方逐条消息的结构写（`<｜Assistant｜>` 属于助手消息），加上工具提示词（Python 风格的
  JSON，保持客户端的 key 顺序）、DSML 调用块、`<tool_result>`、带工具时保留历史思考内容。`Text::ToolCall` 加 `arguments`
  （客户端原样的 JSON，DSML 要靠它区分 `string="true"` 和 `"false"`）。`ChatModel::tool_call_tag()` 告诉 server 用哪种
  解析和在哪里截住流式输出；server 加 DSML 解析。
- 测试：`test_dsv41_template` 扩到 43 个用例（含工具定义、并行调用、嵌套参数和 `1e-05`、工具结果、思考），期望值由
  deepseek-recipe 0.2.0 生成，全部逐字节一致。
- 端到端（bl-server，OpenAI 格式）：问北京明天天气 → `get_weather({"city":"北京","days":1})`；回传结果后正常作答，
  会话复用 350 token（重新渲染的 DSML 和模型生成的一致）。思考 + 流式：问上海和广州后天 → 两个并行调用（days=2），
  content 为空、没有 DSML 漏出；回传后复用 496 token。295 token 的工具提示词 prefill 22 s。

## 2026-10-20 dsv41：两级候选选择，上下文不再限 16K

- 照官方 `model.py` 的 `select_candidate_blocks`：第 20 层（候选源）的 indexer 分数按 8 个位置一块取最大，最新的那块固定入选
  （它没填满，会被老块挤掉），留 top-2048 块做成位掩码；第 24、28、32、36 层（之后的 top-k 源）在 top-512 之前把块外的位置
  设成 -inf。块数不到 2048 时全选，掩码不用（16K 以内和原来一样）。三个参数 GGUF 里没有，写在 `Dsv41Config`（取自 config.json）。
- 新 kernel：`candidate_mask`（块最大值 → 复用 `topk_sorted` 选块 → 位掩码）、`mask_scores`。`test_dsv41_ops`：块集合和掩码后的
  top-512 与 CPU 版 `select_candidate_blocks` 完全一致（n = 16384 / 16392 / 40000 / 131072）。
- prefill 的打分缓冲区 `pf.isc` 原来是 T × max_ctx（4096 行 × 128K = 2 GB），改成按行分片、最多 256 MB；16K 时一片就是整块。
- 回归（16K 以内）：`--check-prefill` 512 个位置，改前 KL 0.048、top-1 479/512，改后 0.049、485/512（`BL_DSV41_CACHE_GB=14`，
  改前的版本不调小缓存时 OOM）。这项指标本身波动大（混合解码的累加顺序）。
- 大海捞针：第一行写"保险柜密码是 7391-2846"，接整份 LOG.md，最后问密码，共 20629 token、`--ctx 32768`：答"仓库管理员的保险柜
  密码是 7391-2846"。prefill 230 s（90 tok/s，第 20 层起 14 s）。
- 显存：压缩 KV + indexer key 每 token 约 6.4 KB（fp32），32K 0.2 GB、128K 0.84 GB，从专家缓存里扣。bl-chat / bl-server
  不再把上下文压到 16K（默认 32K）。
- 64K（`--ctx 65536`）：仓库文档和源码拼成 63486 token，第一行藏密码、正中间藏会议室编号，最后两个一起问：答"1. 7391-2846
  2. B 座 1407 会议室"。prefill 588 s（108 tok/s：按 4096 一块读专家，块越多摊得越薄；indexer 的平方项在 64K 还不是大头）。
  decode（`--gen 160 --stats-after 64`，没有预热专家）：170 ms/token，同条件 4K 对照 139；命中率 63.7% vs 74.3%（长 prefill 后
  缓存冷），两者分不开。`bl-dsv41 --prefill` 现在打印 prefill 耗时。
