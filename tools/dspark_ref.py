#!/usr/bin/env python3
"""DeepSeek-V4.1's DSpark draft in numpy, straight from the official checkpoint (mtp.* in shards 44-46), for measuring
how many draft tokens the main model would accept before building it into the engine.

  dspark_ref.py SRC_DIR MAIN_GGUF_DIR TRACE_DIR...

Each TRACE_DIR comes from `bl-dsv41 --tokens-file DIR/prompt.tokens --gen N --dump DIR --filter 'l_last-(36|37|38)'`:
the hc stream after layers 36-38 (= the attention input of the target layers 37-39) at every position, and the
greedy continuation in DIR/out.txt. Greedy speculative decoding accepts a draft token exactly when it equals the main
model's greedy one, so on a greedy trace the accepted length of the block drafted at position i is the number of
leading draft tokens matching seq[i+2..]. Follows ref/model.py (DSparkBlock, DSparkAttention, forward_spec); the fp8
activation round trips are left out. Needs numpy and gguf-py (GGUF_PY); DSPARK_MAX_BLOCKS=N stops each trace after N.

Measured 2026-10-19 (docs/LOG.md): 3.25 of 5 drafts accepted on average, but on the offloaded 3090 a 6-token verify costs
8.5 single steps, so speculative decoding does not pay there.
"""
import json
import mmap
import os
import re
import sys

import numpy as np

# gguf-py (llama.cpp's, needs numpy and pyyaml) for the main model's embedding and head: GGUF_PY=.../llama.cpp/gguf-py
if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])

D, H, HD, RD, QL, OG, OL, WIN, HC = 5120, 64, 512, 64, 1280, 8, 1024, 128, 4
EPS, HC_EPS, SINKHORN, ROUTE_SCALE, LIMIT = 1e-20, 1e-6, 20, 1.5, 10.0
BLOCK, NOISE, TOPK = 5, 128799, 3

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def e4m3_table():
    t = np.zeros(256, np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        v = np.nan if (e == 15 and m == 7) else (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
        t[b] = -v if s else v
    return t


E4M3 = e4m3_table()


def e8m0(a):
    return np.ldexp(np.float32(1), a.astype(np.int32) - 127).astype(np.float32)


class Safetensors:
    def __init__(self, paths):
        self.t = {}
        for p in paths:
            f = open(p, "rb")
            m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            hl = int.from_bytes(m[:8], "little")
            for k, v in json.loads(m[8:8 + hl]).items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.t[k] = (v["dtype"], v["shape"], m, 8 + hl + a, b - a)

    def raw(self, k):
        dt, shape, m, off, n = self.t[k]
        np_dt = {"F8_E4M3": np.uint8, "I8": np.uint8, "F8_E8M0": np.uint8, "BF16": np.uint16, "F32": np.float32}[dt]
        return dt, np.frombuffer(m, np_dt, n // np.dtype(np_dt).itemsize, off).reshape(shape)

    def get(self, k):
        """weight k as float32, dequantized with its .scale where it has one"""
        dt, a = self.raw(k)
        if dt == "BF16":
            return (a.astype(np.uint32) << 16).view(np.float32)
        if dt == "F32":
            return a.astype(np.float32)
        s = e8m0(self.raw(k[:-len("weight")] + "scale")[1])
        if dt == "F8_E4M3":   # 32x32 blocks
            return E4M3[a] * np.repeat(np.repeat(s, 32, 0), 32, 1)[:a.shape[0], :a.shape[1]]
        if dt == "I8":   # MXFP4: e2m1 pairs, low nibble first, a scale per 32
            w = np.empty((a.shape[0], 2 * a.shape[1]), np.float32)
            w[:, 0::2], w[:, 1::2] = FP4[a & 15], FP4[a >> 4]
            return w * np.repeat(s, 32, 1)
        raise ValueError(dt)


def rmsnorm(x, w):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS) * w


def rope(x, pos, inverse=False):
    """the last RD dims of x [..., T, (heads,) HD] rotated as adjacent pairs at positions pos [T]"""
    f = 1.0 / (10000.0 ** (np.arange(0, RD, 2, dtype=np.float64) / RD))
    ang = np.outer(np.asarray(pos, np.float64), f)
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    if inverse:
        s = -s
    if x.ndim == 3:
        c, s = c[:, None, :], s[:, None, :]
    y = x.copy()
    a, b = x[..., HD - RD::2], x[..., HD - RD + 1::2]
    y[..., HD - RD::2], y[..., HD - RD + 1::2] = a * c - b * s, a * s + b * c
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hc_mixes(h, fn, scale, base):
    """h [T, HC, D] -> pre [T,HC], post [T,HC], comb [T,HC,HC] (kernel.py hc_split_sinkhorn)"""
    x = h.reshape(h.shape[0], -1)
    m = (x @ fn.T) / np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS)
    pre = sigmoid(m[:, :HC] * scale[0] + base[:HC]) + HC_EPS
    post = 2 * sigmoid(m[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
    comb = (m[:, 2 * HC:] * scale[2] + base[2 * HC:]).reshape(-1, HC, HC)
    comb = np.exp(comb - comb.max(-1, keepdims=True))
    comb = comb / comb.sum(-1, keepdims=True) + HC_EPS
    comb = comb / (comb.sum(-2, keepdims=True) + HC_EPS)
    for _ in range(SINKHORN - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdims=True) + HC_EPS)
    return pre, post, comb


def hc_pre(h, pre):
    return np.einsum("thd,th->td", h, pre)


def hc_post(x, res, post, comb):
    return post[:, :, None] * x[:, None, :] + np.einsum("tij,tid->tjd", comb, res)


class Stage:
    def __init__(self, st, s):
        p = f"mtp.{s}."
        g = lambda k: st.get(p + k)
        self.st, self.p = st, p
        self.wq_a, self.q_norm, self.wq_b = g("attn.wq_a.weight"), g("attn.q_norm.weight"), g("attn.wq_b.weight")
        self.wkv, self.kv_norm = g("attn.wkv.weight"), g("attn.kv_norm.weight")
        self.wo_a = g("attn.wo_a.weight").reshape(OG, OL, -1)
        self.wo_b, self.sink = g("attn.wo_b.weight"), g("attn.attn_sink")
        self.attn_norm, self.ffn_norm = g("attn_norm.weight"), g("ffn_norm.weight")
        self.gate, self.bias = g("ffn.gate.weight"), g("ffn.gate.bias")
        self.sh = [g(f"ffn.shared_experts.{w}.weight") for w in ("w1", "w3", "w2")]
        self.hc_attn = [g("hc_attn_fn"), g("hc_attn_scale"), g("hc_attn_base")]
        self.hc_ffn = [g("hc_ffn_fn"), g("hc_ffn_scale"), g("hc_ffn_base")]
        self.experts = {}
        self.window = {}   # position -> main_kv (post-RoPE)

    def expert(self, e):
        if e not in self.experts:
            self.experts[e] = [self.st.get(f"{self.p}ffn.experts.{e}.{w}.weight") for w in ("w1", "w3", "w2")]
        return self.experts[e]

    def seed(self, main_x, pos):
        """the window KV the main model's positions contribute (DSparkAttention, the start_pos == 0 path / main_kv)"""
        kv = rope(rmsnorm(main_x @ self.wkv.T, self.kv_norm), pos)
        for p, k in zip(pos, kv):
            self.window[p] = k

    def attn(self, x, start):
        T = x.shape[0]
        pos = np.arange(start + 1, start + 1 + T)
        q = (rmsnorm(x @ self.wq_a.T, self.q_norm) @ self.wq_b.T).reshape(T, H, HD)
        q = rope(q, pos)
        kv = rope(rmsnorm(x @ self.wkv.T, self.kv_norm), pos)
        keys = np.concatenate([np.stack([self.window[p] for p in range(max(0, start - WIN + 1), start + 1)]), kv])
        sc = np.einsum("thd,nd->thn", q, keys) * HD ** -0.5
        mx = np.maximum(sc.max(-1, keepdims=True), self.sink[None, :, None])
        e = np.exp(sc - mx)
        o = np.einsum("thn,nd->thd", e, keys) / (e.sum(-1) + np.exp(self.sink[None, :] - mx[..., 0]))[..., None]
        o = rope(o, pos, inverse=True).reshape(T, OG, -1)
        oa = np.einsum("tgd,grd->tgr", o, self.wo_a).reshape(T, -1)
        return (oa @ self.wo_b.T).astype(np.float32)

    def ffn_one(self, x, w):
        gate, up = x @ w[0].T, x @ w[1].T
        up = np.clip(up, -LIMIT, LIMIT)
        gate = np.minimum(gate, LIMIT)
        return (gate / (1 + np.exp(-gate)) * up) @ w[2].T

    def moe(self, x):
        sc = np.sqrt(np.logaddexp(0, x @ self.gate.T))
        idx = np.argsort(-(sc + self.bias), -1)[:, :TOPK]
        wt = np.take_along_axis(sc, idx, -1)
        wt = wt / (wt.sum(-1, keepdims=True) + 1e-20) * ROUTE_SCALE
        y = self.ffn_one(x, self.sh)
        for t in range(x.shape[0]):
            for e, w in zip(idx[t], wt[t]):
                y[t] += w * self.ffn_one(x[t:t + 1], self.expert(int(e)))[0]
        return y

    def forward(self, h, pre_mix, start):
        res = h
        a_pre, post, comb = hc_mixes(h, *self.hc_attn)
        x = self.attn(rmsnorm(hc_pre(h, pre_mix), self.attn_norm), start)
        h = hc_post(x, res, post, comb)
        res = h
        f_pre, post, comb = hc_mixes(h, *self.hc_ffn)
        x = self.moe(rmsnorm(hc_pre(h, a_pre), self.ffn_norm))
        return hc_post(x, res, post, comb).astype(np.float32), f_pre.astype(np.float32)


class Main:
    """the main model's embedding and output head (BF16 in the GGUF)"""

    def __init__(self, gguf_dir):
        import glob
        import gguf
        for f in sorted(glob.glob(os.path.join(gguf_dir, "*.gguf"))):
            for t in gguf.GGUFReader(f).tensors:
                if t.name == "token_embd.weight":
                    self.embd = np.asarray(t.data).view(np.uint16).reshape(-1, D)
                elif t.name == "output.weight":
                    self.head = (np.asarray(t.data).view(np.uint16).reshape(-1, D).astype(np.uint32) << 16).view(np.float32)

    def embed(self, ids):
        return (self.embd[ids].astype(np.uint32) << 16).view(np.float32)


def load_trace(d):
    idx = [l.split("\t") for l in open(os.path.join(d, "index.tsv")).read().splitlines()[1:]]
    data = np.fromfile(os.path.join(d, "data.bin"), np.float32)
    by = {}
    for name, _, n, _, _, _, off, nb in idx:
        by.setdefault(name, []).append(data[int(off) // 4:(int(off) + int(nb)) // 4])
    hid = np.concatenate([np.stack(by[f"l_last-{l}"]).reshape(-1, HC, D).mean(1) for l in (36, 37, 38)], -1)
    prompt = [int(v) for v in open(os.path.join(d, "prompt.tokens")).read().strip().split(",")]
    gen = [int(v) for v in re.search(r"^generated: (.*)$", open(os.path.join(d, "out.txt")).read(), re.M).group(1).split()]
    return hid, prompt, prompt + gen


def main():
    src, gguf_dir, traces = sys.argv[1], sys.argv[2], sys.argv[3:]
    st = Safetensors([os.path.join(src, f"model-000{n}-of-00048.safetensors") for n in (44, 45, 46)])
    stages = [Stage(st, s) for s in range(3)]
    main_proj, main_norm = st.get("mtp.0.main_proj.weight"), st.get("mtp.0.main_norm.weight")
    norm = st.get("mtp.2.norm.weight")
    mk_embed, mk_head = st.get("mtp.2.markov_head.embed.weight"), st.get("mtp.2.markov_head.head.weight")
    conf_w = st.get("mtp.2.confidence_head.proj.weight")[0]
    mm = Main(gguf_dir)
    print("loaded", flush=True)
    total = {}
    for d in traces:
        hid, prompt, seq = load_trace(d)
        n = hid.shape[0]
        main_x = rmsnorm(hid @ main_proj.T, main_norm)
        for s in stages:
            s.window = {}
            s.seed(main_x, np.arange(n))
        acc, confs = [], []
        for i in range(len(prompt) - 1, min(n - BLOCK - 1, len(prompt) - 1 + int(os.environ.get("DSPARK_MAX_BLOCKS", 1 << 30)))):
            o = seq[i + 1]
            h = np.repeat(mm.embed(np.array([o] + [NOISE] * (BLOCK - 1)))[:, None, :], HC, 1)
            pre = np.zeros((BLOCK, HC), np.float32)
            pre[:, 0] = 1
            for s in stages:
                h, pre = s.forward(h, pre, i)
            x = rmsnorm(hc_pre(h, pre), norm)
            assert x.dtype == np.float32, x.dtype
            logits = x @ mm.head.T
            out = [o]
            for j in range(BLOCK):
                logits[j] += mk_head @ mk_embed[out[j]]
                out.append(int(np.argmax(logits[j])))
            conf = np.concatenate([x, mk_embed[out[:BLOCK]]], -1) @ conf_w
            k = 0
            while k < BLOCK and out[1 + k] == seq[i + 2 + k]:
                k += 1
            acc.append(k)
            confs.append(conf)
            if len(acc) % 32 == 0:
                print(f"  {d}: {len(acc)} blocks, mean accepted {np.mean(acc):.2f}", flush=True)
        acc = np.array(acc)
        total[d] = acc
        hist = np.bincount(acc, minlength=BLOCK + 1)
        print(f"{d}: {len(acc)} blocks, mean accepted {acc.mean():.3f} -> {1 + acc.mean():.3f} tokens per main step;"
              f" histogram {hist.tolist()}", flush=True)
        np.save(os.path.join(d, "accepted.npy"), acc)
        np.save(os.path.join(d, "confidence.npy"), np.array(confs))
    allacc = np.concatenate(list(total.values()))
    print(f"all: mean accepted {allacc.mean():.3f}; per-position acceptance "
          f"{[round(float((allacc > j).mean()), 3) for j in range(BLOCK)]}")


if __name__ == "__main__":
    main()
