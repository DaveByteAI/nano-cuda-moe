# DeepSeek-V4.1-Flash-1017（GGUF 架构名 `deepseek41`）模型规格

boundless `dsv41` 分支的实现依据。来源：
- GGUF：acrogenesis `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Experts-Q8-Dense-Engram`，revision `f5221b0`，10 个文件 359 GB
  （`scripts/download_dsv41.sh`）
- 数学依据：官方 `inference/model.py`、`kernel.py`、`engram.py`（HF 仓库 `deepseek-ai/DeepSeek-V4.1-Flash-1017`）
- 数值参照：llama.cpp PR #28696（vcruz305/llama.cpp `2076412`）的 `src/models/deepseek41.cpp` 和 `deepseek4.cpp`

## 超参数

| 项目 | 值 |
|---|---|
| 层数 | 40（另有 3 层 DSpark 投机解码层，这个 GGUF 里**没有**） |
| `n_embd` | 5120 |
| 词表 | 129,280；BOS 0，EOS 1 |
| hyper-connection | 4 路残差流（`hc_dim` = 20480），mHC：`fn` 20480 → 24，Sinkhorn 20 次，eps 1e-6 |
| MoE | 384 个路由专家，每 token 选 6 个，`n_ff_exp` = 2304；1 个共享专家（同样 2304，**没有门**） |
| 路由 | `sqrt(softplus(W·x))`；选专家用 分数 + 偏置（noaux_tc），权重用不带偏置的分数，归一化后 × 1.5 |
| SwiGLU 截断 | up 截到 [-10, 10]，gate 只截上限 10；路由专家和共享专家都截 |
| 注意力 | 64 个 q 头 × 512；**只有 1 个 KV 头，512 维，K 和 V 是同一个向量**；q 低秩 5120 → 1280 → 32768 |
| 输出投影 | 分 8 组的低秩：每组 8 头 × 512 = 4096 → 1024，拼起来 8192 → 5120 |
| RoPE | 只转最后 64 维，**相邻两个数配成一对**；纯滑窗层 base 1e4、不用 YaRN；压缩层 base 1.6e5 + YaRN（因子 16，原长 65536，β 32/1），**不乘 mscale** |
| 滑动窗口 | 128（每层自己保存原始 KV） |
| 压缩比 | 第 0、1 层 0；第 2–19 层 2；第 20–39 层 1（列表长 43，后 3 个属于 DSpark 层） |
| KV 源层 | 2、8、14、20：只有它们带压缩器；后面的层共用最近一个源层的压缩 KV |
| indexer 源层 | 2、8、14、20、24、28、32、36：32 头 × 128，每个 query 选 top-512 个压缩位置；后面的层共用最近一个源层的选择结果 |
| 两级候选 | 第 20 层以后先按 8 个一块选 2048 块；压缩位置不到 16384 个时等于全选，**先不实现，上下文限 16K** |
| Engram | 第 1 层和第 14 层；4-gram，每个 n 长度 8 个头，每 token 查 (4−1)×8 = 24 行 × 256 维 |
| RMSNorm eps | **1e-20**（hc 和 Engram 的 rsqrt 同样用这个 eps） |
| 上下文 | 1,048,576（这个分支先做到 16K） |

## 前向（单 token；prefill 时按 token 批量做同样的计算）

```
x = token_embd[tok]                                      # BF16 的一行
R = repeat(x, 4)                                          # [4, 5120]
pre = [1, 0, 0, 0]                                        # 第 0 层没有上一个子层，取第一路
for il in 0..39:
    if il in (1, 14): R = ENGRAM(R)                       # 在本层最前面，直接写进 4 路残差流
    # 注意力子层：用的是"上一个子层"算出来的 pre
    res = R
    a_pre, a_post, a_comb = HC_MIXES(R, hc_attn_*)
    h = rmsnorm(Σ_j pre[j]·R[j]) * attn_norm
    h = ATTN(h)
    R = HC_POST(h, res, a_post, a_comb)
    # FFN 子层：用注意力子层算出来的 a_pre
    res = R
    pre, f_post, f_comb = HC_MIXES(R, hc_ffn_*)           # 这个 pre 留给下一层的注意力
    h = rmsnorm(Σ_j a_pre[j]·R[j]) * ffn_norm
    h = MoE(h) + SharedExpert(h)
    R = HC_POST(h, res, f_post, f_comb)
h = rmsnorm(Σ_j pre[j]·R[j]) * output_norm               # 最后一层 FFN 的 pre 负责合并 4 路；没有单独的 hc 输出头
logits = output · h                                       # BF16，[129280, 5120]
```

**系数滞后一个子层**是 V4.1 和 V4 的关键区别：一个子层算出来的 pre，给下一个子层用。

**HC_MIXES**（`model.py` Block.hc_mixes 与 `kernel.py` hc_split_sinkhorn）：
```
m    = (fn · flat(R)) * rsqrt(mean(flat(R)²) + eps)        # flat(R) 是 20480 维；整体一个统计量，没有 norm 权重
pre  = sigmoid(m[0:4]  * scale[0] + base[0:4]) + 1e-6
post = 2 * sigmoid(m[4:8] * scale[1] + base[4:8])
C    = m[8:24] * scale[2] + base[8:24]，看成 4×4（C[j][k]，j 是行）
C    = softmax_按行(C) + 1e-6；C /= (列和 + 1e-6)
重复 19 次：C /= (行和 + 1e-6)；C /= (列和 + 1e-6)
```

**HC_POST**：`R'[k] = post[k] * h + Σ_j C[j][k] * res[j]`

**ENGRAM**（`model.py` Engram 与 `engram.py`）：
- 行号：token id 先经过 `engram.token_map` 映射成压缩 id（大小写、重音、空白归一化后相同的 token 合并）。
  第 p 个位置取 `t[0] = 压缩 id(tok[p])`，`t[s] = 压缩 id(tok[p−s])`（s = 1..3，序列开头之前用 pad_id 2 的映射值）。
  `rolling = t[0]·mult[0]`，对 s = 1..3：`rolling ^= t[s]·mult[s]`，然后对每个头 h：
  `row[(s−1)·8 + h] = rolling % prime[s−1][h] + offset[s−1][h]`。
  mult、prime、offset 都在 GGUF 元数据 `deepseek41.engram.*` 里（每个 Engram 层各一组）。
  **行号只依赖 token id**：下一个 token 一确定，这 24 行就能开始读，第 14 层的那份有前 13 层的时间去读盘。
- `emb` = 24 行拼成 6144 维，`kv = engram_wkv · emb`（6144 → 25600 = 4×5120 的 key + 5120 的 value）。
- 对每一路 c：`dot = Σ (R[c]·q_w[c]·k_w[c]·key[c]) * rsqrt(mean(R[c]²)+eps) * rsqrt(mean(key[c]²)+eps) / √5120`，
  `gate = sigmoid(sign(dot)·√max(|dot|, 1e-6))`，`R[c] += gate · value`。

**ATTN**（`model.py` Attention）：
```
qr = rmsnorm(Wq_a · h) * q_a_norm                         # 1280
q  = Wq_b · qr → [64][512]，最后 64 维做 RoPE（位置 p）
kv = rmsnorm(Wkv · h) * kv_a_norm → 512，最后 64 维做 RoPE（位置 p），写进本层的 128 窗口
压缩比 > 0 的层，额外的 KV 是共享的压缩 KV（见下）里 indexer 选中的位置
o[h] = Σ_t softmax(q_h·k_t / √512 ; 分母多加 exp(sink_h)) · k_t     # K = V
o 最后 64 维做逆 RoPE（位置 p）
o 按 8 个头一组：oa[g] = Wo_a[g] · o[g]（4096 → 1024），out = Wo_b · concat(oa)（8192 → 5120）
```
窗口：当前位置和它之前的 127 个位置。

**压缩器**（只在 KV 源层 2、8、14、20）：
- 比例 1（第 20 层）：`latent = rmsnorm(Wc_kv · h) * c_norm`，每个位置一条。
- 比例 2（第 2、8、14 层）：每两个相邻位置一组，`kv = Wc_kv·h`、`s = Wc_gate·h`（都按 fp32 算），
  对每一维在组内做 softmax 加权：`latent = rmsnorm(Σ_i softmax_i(s)·kv_i) * c_norm`。组满才产生一条；
  未满的那一个位置存在状态里。
- 这条 latent（RoPE 之前）先给 indexer 算 key，然后最后 64 维按**组内第一个位置** `j·ratio` 做 RoPE，存进压缩 KV。
- 位置 p 的 query 能看到的压缩条数 = `(p+1) // ratio`。

**INDEXER**（只在 indexer 源层）：
- key（只在 KV 源层生成，写进共享的 indexer key 缓存）：`k = rmsnorm(Wi_k · latent) * k_norm`（512 → 128），
  最后 64 维按 `j·ratio` 做 RoPE。
- query：`iq = Wi_qb · qr → [32][128]`，最后 64 维按位置 p 做 RoPE（和本层注意力一样用压缩层的 base 与 YaRN）。
- 头权重：`w = Wi_proj · h`（5120 → 32）× `1/√(128·32)`。
- 打分：`score[t] = Σ_h relu(iq_h · k_t) · w_h`，看不到的位置是 −∞，取前 min(512, 可见条数) 个。

**MoE**：上面"路由"和"SwiGLU 截断"两行；每个专家 `down(silu(min(gate·h, 10)) * clamp(up·h, ±10))`；
输出 = Σ 权重 × 专家输出 + 共享专家输出。

## 和官方参考实现的数值差异（按 llama.cpp 的做法）

官方 `model.py` 在几处模拟训练时的低精度：窗口 KV 量化成 FP8 再反量化，压缩 KV 用 FP4（16 个一组），
indexer 的 q、k 用 FP4，Wo_a 之前的 o 用 FP8。llama.cpp 都没有模拟，直接用浮点算。
boundless 先对齐 llama.cpp（它是逐张量对拍的参照），这些模拟以后再单独评估。

## 字节预算（GGUF 头部实测）

| 部分 | 大小 | 每个 token 读多少 |
|---|---|---|
| 路由专家（40 层 × 384 个，全部 IQ2_XXS） | **140.14 GB** | 6 个 / 层，每个 9.12 MB，共 **2.19 GB** |
| 稠密部分（注意力 Q8_0、共享专家 Q8_0、hc F32、路由器 BF16、indexer、压缩器、Engram 投影） | **7.62 GB** | 全部 |
| output（BF16） | 1.32 GB | 全部 |
| token_embd（BF16） | 1.32 GB | 1 行 |
| Engram 表（Q8_0，2 × [256, 3.84 亿]） | **208.90 GB** | 48 行 × 272 B，几乎每行一个 4 KB 页 |

每层稠密部分大约 190 MB：Wq_b 44.6 MB、Wo_b 44.6 MB、Wo_a 35.7 MB、共享专家 37.6 MB、Wq_a 7.0 MB、
Wkv 2.8 MB、hc_fn 3.9 MB、路由器 3.9 MB；源层再加压缩器 10.5 MB 和 indexer 6 MB。

**结构性结论：**
1. 稠密部分加 output 共 8.9 GB，每个 token 都要全部读一遍。按显存实测 840 GB/s 算，**仅这一项就有约 10.6 ms 的下限**，
   约 94 tok/s 封顶；投机解码能把这一项摊到多个 token 上。把稠密部分换成 4 bit 格式能减半，代价是精度。
2. 显存 24 GB 减去稠密的 8.9 GB、KV 和工作区，约剩 13 GB 给专家，只占 140 GB 的 9%。
   内存 62 GB 大约还能放 45 GB，其余约 80 GB 在 NVMe 上。**专家的三层存储（显存、内存、NVMe）是这个模型最重要的工程。**
3. Engram 表必须放在 NVMe 上：第 1 层没有时间去藏读盘的延迟，prefill 一次要随机读上万行。
