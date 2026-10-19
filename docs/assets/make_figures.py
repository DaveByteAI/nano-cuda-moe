#!/usr/bin/env python3
"""Generate the README figures as SVG (light + dark, English + Chinese).

    python3 docs/assets/make_figures.py

Writes docs/assets/<figure>-<lang>-<theme>.svg; the READMEs embed them with <picture>, so GitHub shows the variant
that matches the viewer's theme. The numbers are in DATA (measured, see docs/LOG.md); edit them and run again.
The visual language is nano-metal-moe-qwen36's, the Metal sibling of this project.
"""

import os
from xml.sax.saxutils import escape

OUT = os.path.dirname(os.path.abspath(__file__))
FONT = ("system-ui, -apple-system, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', "
        "'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif")
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

# RTX 3090 24 GB, i7-10700, 64 GB DDR4; IQ3_XXS GGUF; greedy, 3 MTP drafts (docs/LOG.md)
DATA = {
    "decode": 97.2,          # tok/s, the 7-text corpus (bench/corpus), 200 tokens each
    "decode_strata": 65.0,   # Strata on the same machine, file and inputs (each text a fresh process, as it runs)
    "prefill": [("32K", 32000 / 18.0), ("128K", 128000 / 78.3)],                         # tok/s
    "decode_ctx": [("short", 97.2), ("32K", 82.1), ("128K", 74.8)],                       # tok/s after a prompt of
    "vram": [("cache", 17.3), ("dense", 4.1), ("mtp_kv", 1.5), ("runtime", 2.4)],        # GB of the 25.3 GB card
}

THEMES = {
    "light": {
        "surface": "#fcfcfb", "plane": "#f4f3ef", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "#e4e3dd",
        "gpu": "#2a78d6", "ssd": "#eb6834", "cpu": "#1baf7a", "mem": "#4a3aa7",
        "base": "#b9b7af", "good": "#006300", "dot": "#dcdad3", "wash": 0.11, "seg": 0.55,
    },
    "dark": {
        "surface": "#1a1a19", "plane": "#222220", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "border": "#2e2e2b",
        "gpu": "#3987e5", "ssd": "#d95926", "cpu": "#199e70", "mem": "#9085e9",
        "base": "#5c5b56", "good": "#0ca30c", "dot": "#34342f", "wash": 0.18, "seg": 0.8,
    },
}

T = {
    "en": {
        "eyebrow": "LOCAL LLM INFERENCE  ·  NVIDIA  ·  CUDA",
        "title": "nano-cuda-moe",
        "subtitle": "Qwen3.8 and DeepSeek-V4.1 on one 24 GB RTX 3090",
        "tiles": [(f"{DATA['decode']:.0f}", "tok/s", "Qwen3.8 · 125B"), ("~10", "tok/s", "DeepSeek-V4.1 · 2-bit"),
                  ("24 GB", "", "one RTX 3090"), ("64 GB", "", "RAM")],
        "grid_caption": "Qwen3.8: each layer has 512 experts; a token uses 10.",
        "grid_caption2": "About 9 come from the GPU cache, 1 runs on the CPU.",

        "arch_title": "How a 125B model runs on a 24 GB card",
        "arch_sub": "The GPU holds the dense weights and the experts used most. Every expert is in RAM, where the CPU computes the misses.",
        "vram_title": "GPU · 24 GB",
        "vram_rows": [("Expert cache", "≈ 10,000 of 25,088 experts"), ("Dense weights", "attention, DeltaNet, routers, head"),
                      ("MTP layer, KV cache", "int8 KV, 512 MTP experts"), ("CUDA runtime, headroom", "")],
        "flow_title": "Per token, × 48 layers",
        "flow": [("GPU", "Attention or DeltaNet, router", "dense weights, always in VRAM"),
                 ("GPU", "Plan: cached or missing", "the GPU rings a doorbell in host memory"),
                 ("split",),
                 ("GPU", "Combine, next layer", "misses then enter the LRU cache over PCIe")],
        "hit": ("GPU", "Cached", "≈ 90% of experts"),
        "miss": ("CPU", "Missing", "≈ 10%, AVX2 in RAM"),
        "token": "token", "next": "next token",
        "ram_title": "RAM · 64 GB",
        "ram_sub": "All 25,088 experts, pinned",
        "ram_sub2": "42.9 GB, 2.8 bits per weight",
        "ssd_title": "SSD",
        "ssd_sub": "n-gram embedding table",
        "ssd_sub2": "28.8 GB, a few rows per token",
        "mtp": "Each round the MTP layer drafts up to 3 tokens; one pass over the 48 layers verifies all 4.",

        "perf_title": "Decode 50% faster than Strata on the same machine and file",
        "perf_sub": "RTX 3090 24 GB, i7-10700, 64 GB. Greedy decoding with 3 MTP drafts.",
        "panels": [("Decode, 7 short texts", "tok/s, higher is better"),
                   ("Prompt processing", "tok/s by prompt length"),
                   ("Decode after a long prompt", "tok/s by prompt length")],
        "names": ["Strata", "nano-cuda-moe"],
        "ctx_labels": {"short": "short"},
    },
    "zh": {
        "eyebrow": "本地大模型推理  ·  NVIDIA  ·  CUDA",
        "title": "nano-cuda-moe",
        "subtitle": "一张 24 GB 的 RTX 3090 跑 Qwen3.8 和 DeepSeek-V4.1",
        "tiles": [(f"{DATA['decode']:.0f}", "tok/s", "Qwen3.8 · 125B"), ("~10", "tok/s", "DeepSeek-V4.1 · 2 bit"),
                  ("24 GB", "", "一张 RTX 3090"), ("64 GB", "", "内存")],
        "grid_caption": "Qwen3.8：每层 512 个专家，每个 token 用 10 个。",
        "grid_caption2": "约 9 个来自显卡缓存，1 个由 CPU 计算。",

        "arch_title": "125B 的模型怎么跑在 24 GB 的显卡上",
        "arch_sub": "显卡放稠密权重和最常用的专家；全部专家都在内存里，显卡没有的由 CPU 直接计算。",
        "vram_title": "显卡 · 24 GB",
        "vram_rows": [("专家缓存", "25,088 个专家中的约 1 万个"), ("稠密权重", "注意力、DeltaNet、路由、输出头"),
                      ("MTP 层、KV cache", "int8 KV，512 个 MTP 专家"), ("CUDA 运行时、余量", "")],
        "flow_title": "每个 token，× 48 层",
        "flow": [("GPU", "注意力或 DeltaNet，路由", "稠密权重常驻显存"),
                 ("GPU", "规划：缓存里有，还是缺", "在主机内存里按门铃"),
                 ("split",),
                 ("GPU", "合并，进入下一层", "缺的专家随后经 PCIe 进入 LRU 缓存")],
        "hit": ("GPU", "缓存命中", "约 90% 的专家"),
        "miss": ("CPU", "未命中", "约 10%，在内存里算"),
        "token": "token", "next": "下一个 token",
        "ram_title": "内存 · 64 GB",
        "ram_sub": "全部 25,088 个专家，锁页",
        "ram_sub2": "42.9 GB，每个权重约 2.8 bit",
        "ssd_title": "SSD",
        "ssd_sub": "n-gram 嵌入表",
        "ssd_sub2": "28.8 GB，每个 token 读几行",
        "mtp": "每一轮 MTP 层起草最多 3 个 token，主模型一遍 48 层同时验证 4 个。",

        "perf_title": "同一台机器、同一个模型文件，生成速度比 Strata 快 50%",
        "perf_sub": "RTX 3090 24 GB、i7-10700、64 GB 内存。贪心解码，每轮 3 个 MTP 草稿。",
        "panels": [("生成速度，7 段短文本", "tok/s，越高越好"),
                   ("提示词处理", "tok/s，按提示词长度"),
                   ("长提示词之后的生成", "tok/s，按提示词长度")],
        "names": ["Strata", "nano-cuda-moe"],
        "ctx_labels": {"short": "短"},
    },
}


# --------------------------------------------------------------------------
# tiny SVG helpers (as in nano-metal-moe)
# --------------------------------------------------------------------------

def text(x, y, s, size, fill, weight=400, anchor="start", family=FONT, spacing=None, opacity=None):
    extra = ""
    if spacing is not None:
        extra += f' letter-spacing="{spacing}"'
    if opacity is not None:
        extra += f' opacity="{opacity}"'
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>{escape(s)}</text>')


def rect(x, y, w, h, fill, rx=0, opacity=None, stroke=None, sw=1):
    extra = f' fill-opacity="{opacity}"' if opacity is not None else ""
    if stroke:
        extra += f' stroke="{stroke}" stroke-width="{sw}"'
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{extra}/>'


def bar_h(x, y, w, h, fill, r=4):
    r = min(r, w / 2, h / 2)
    return (f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z" fill="{fill}"/>')


def arrow_defs(c):
    out = ['<defs>']
    for name, color in (("ink", c["muted"]), ("gpu", c["gpu"]), ("cpu", c["cpu"]), ("ssd", c["ssd"]), ("mem", c["mem"])):
        out.append(f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
                   f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0.8 L9,5 L0,9.2 Z" fill="{color}"/></marker>')
    out.append('</defs>')
    return "".join(out)


def line(x1, y1, x2, y2, color, width=1.6, head=None, dash=None):
    extra = f' marker-end="url(#ah-{head})"' if head else ""
    if dash:
        extra += f' stroke-dasharray="{dash}"'
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round"{extra}/>')


def svg(w, h, body, c, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{escape(title)}"><title>{escape(title)}</title>'
            f'{arrow_defs(c)}'
            f'{rect(0.5, 0.5, w - 1, h - 1, c["surface"], rx=18, stroke=c["border"], sw=1)}'
            f'{body}</svg>\n')


# 512 experts as 32 x 16; the 10 a token uses: 9 cached (GPU), 1 computed on the CPU
HITS = {37, 70, 141, 166, 203, 260, 318, 377, 452}
MISS = {421}


def expert_grid(x0, y0, cell, gap, c, faded=False, cols=32):
    out = []
    for i in range(512):
        r, col = divmod(i, cols)
        cx, cy = x0 + col * (cell + gap) + cell / 2, y0 + r * (cell + gap) + cell / 2
        hue = c["gpu"] if i in HITS else c["cpu"] if i in MISS else None
        if hue and not faded:
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{cell * 0.95:.1f}" fill="{hue}" fill-opacity="0.18"/>')
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{cell / 2:.1f}" fill="{hue}"/>')
        else:
            fade = ' fill-opacity="0.6"' if faded else ""
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{cell / 2 * 0.78:.1f}" fill="{c["dot"]}"{fade}/>')
    return "".join(out)


def card(x, y, w, h, role, color, title, sub, c, tag_w=46):
    return "".join([rect(x, y, w, h, color, rx=12, opacity=c["wash"]),
                    rect(x, y + 12, 4, h - 24, color, rx=2),
                    rect(x + 18, y + h / 2 - 11, tag_w, 22, color, rx=11),
                    text(x + 18 + tag_w / 2, y + h / 2 + 4.5, role, 11.5, "#ffffff", 700, anchor="middle", spacing=0.6),
                    text(x + 18 + tag_w + 14, y + h / 2 - 3, title, 16, c["ink"], 650),
                    text(x + 18 + tag_w + 14, y + h / 2 + 17, sub, 12.5, c["ink2"], 450)])


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def hero(c, t):
    w, h = 1200, 380
    b = [f'<defs><radialGradient id="glow" cx="0.78" cy="0.45" r="0.55">'
         f'<stop offset="0" stop-color="{c["gpu"]}" stop-opacity="{c["wash"] * 0.9}"/>'
         f'<stop offset="1" stop-color="{c["gpu"]}" stop-opacity="0"/></radialGradient></defs>',
         rect(1, 1, w - 2, h - 2, "url(#glow)", rx=18),
         text(56, 74, t["eyebrow"], 13, c["muted"], 600, spacing=1.6),
         text(54, 132, t["title"], 50, c["ink"], 750, spacing=-1),
         text(56, 172, t["subtitle"], 22, c["ink2"], 500)]
    tx, ty, tw, th, tg = 56, 214, 156, 112, 14
    accents = [c["gpu"], c["mem"], c["cpu"], c["ssd"]]
    for i, (big, unit, label) in enumerate(t["tiles"]):
        x = tx + i * (tw + tg)
        b.append(rect(x, ty, tw, th, c["plane"], rx=12))
        b.append(rect(x + 16, ty + 18, 22, 4, accents[i], rx=2))
        b.append(f'<text x="{x + 16}" y="{ty + 66}" font-family="{FONT}" fill="{c["ink"]}">'
                 f'<tspan font-size="32" font-weight="700">{escape(big)}</tspan>'
                 f'<tspan font-size="15" font-weight="500" fill="{c["ink2"]}" dx="5">{escape(unit)}</tspan></text>')
        b.append(text(x + 16, ty + 92, label, 13, c["ink2"], 500))
    cell, gap = 7, 3.6
    spanx, spany = 32 * cell + 31 * gap, 16 * cell + 15 * gap
    gx, gy = w - 104 - spanx, 92
    b.append(f'<g opacity="0.35">{expert_grid(gx + 24, gy - 20, cell, gap, c, faded=True)}</g>')
    b.append(f'<g opacity="0.6">{expert_grid(gx + 12, gy - 10, cell, gap, c, faded=True)}</g>')
    b.append(rect(gx - 14, gy - 14, spanx + 28, spany + 28, c["surface"], rx=14, opacity=0.86))
    b.append(expert_grid(gx, gy, cell, gap, c))
    b.append(text(gx + spanx / 2, gy + spany + 46, t["grid_caption"], 14, c["ink2"], 500, anchor="middle"))
    b.append(text(gx + spanx / 2, gy + spany + 67, t["grid_caption2"], 14, c["ink2"], 500, anchor="middle"))
    return svg(w, h, "".join(b), c, f'{t["title"]}: {t["subtitle"]}')


def architecture(c, t):
    w, h = 1200, 680
    b = [text(48, 62, t["arch_title"], 26, c["ink"], 720),
         text(48, 90, t["arch_sub"], 15, c["ink2"], 450)]

    # ---- the GPU's 24 GB as a stacked bar ----
    mx, my, mw, mh = 64, 168, 60, 380
    b.append(text(48, 146, t["vram_title"], 15, c["ink"], 650))
    total = sum(v for _, v in DATA["vram"])
    colors = {"cache": (c["gpu"], 1.0), "dense": (c["gpu"], c["seg"] * 0.75), "mtp_kv": (c["mem"], 1.0),
              "runtime": (c["base"], c["seg"])}
    y = my
    centers = []
    for i, (key, gb) in enumerate(DATA["vram"]):
        hh = mh * gb / total
        col, op = colors[key]
        top_r = 8 if i == 0 else 0
        bot_r = 8 if i == len(DATA["vram"]) - 1 else 0
        y0, y1 = y, y + hh - (2 if i < len(DATA["vram"]) - 1 else 0)
        d = (f"M{mx},{y0 + top_r} Q{mx},{y0} {mx + top_r},{y0} H{mx + mw - top_r} Q{mx + mw},{y0} {mx + mw},{y0 + top_r} "
             f"V{y1 - bot_r} Q{mx + mw},{y1} {mx + mw - bot_r},{y1} H{mx + bot_r} Q{mx},{y1} {mx},{y1 - bot_r} Z")
        b.append(f'<path d="{d}" fill="{col}" fill-opacity="{op}"/>')
        centers.append(((y0 + y1) / 2, gb))
        y += hh
    for (name, sub), (cy, gb), key in zip(t["vram_rows"], centers, [k for k, _ in DATA["vram"]]):
        dy = {"dense": -16, "mtp_kv": 2, "runtime": 18}.get(key, 0)
        b.append(text(mx + mw + 18, cy - 4 + dy, f"{name}", 13.5, c["ink"], 600))
        b.append(text(mx + mw + 18, cy + 14 + dy, f"{gb:.1f} GB" + (f" · {sub}" if sub else ""), 12, c["ink2"], 450))
    cache_y = centers[0][0]

    # ---- the per-layer flow ----
    fx, fw, fh = 420, 400, 64
    b.append(text(fx, 146, t["flow_title"], 15, c["ink"], 650))
    tok_y = 168
    b.append(rect(fx, tok_y, 92, 30, c["plane"], rx=15))
    b.append(text(fx + 46, tok_y + 20, t["token"], 13, c["ink2"], 600, anchor="middle", family=MONO))
    ys = [tok_y + 50, tok_y + 50 + (fh + 20)]
    split_y = ys[1] + fh + 26
    ys += [split_y, split_y + fh + 30]
    b.append(line(fx + 46, tok_y + 30, fx + 46, ys[0] - 3, c["muted"], head="ink"))
    for i in (0, 1):
        role, title, sub = t["flow"][i]
        b.append(card(fx, ys[i], fw, fh, role, c["gpu"], title, sub, c))
    b.append(line(fx + 46, ys[0] + fh, fx + 46, ys[1] - 3, c["muted"], head="ink"))
    hw = (fw - 16) / 2   # the split: GPU (hits) | CPU (misses)
    b.append(card(fx, split_y, hw, fh, *t["hit"][:1], c["gpu"], t["hit"][1], t["hit"][2], c, tag_w=40))
    b.append(card(fx + hw + 16, split_y, hw, fh, *t["miss"][:1], c["cpu"], t["miss"][1], t["miss"][2], c, tag_w=40))
    b.append(line(fx + 46, ys[1] + fh, fx + 46, split_y - 3, c["muted"], head="ink"))
    b.append(line(fx + 46, ys[1] + fh + 10, fx + hw + 16 + 46, ys[1] + fh + 10, c["muted"]))
    b.append(line(fx + hw + 16 + 46, ys[1] + fh + 10, fx + hw + 16 + 46, split_y - 3, c["muted"], head="ink"))
    role, title, sub = t["flow"][3]
    b.append(line(fx + 46, split_y + fh, fx + 46, ys[3] - 3, c["muted"], head="ink"))
    b.append(line(fx + hw + 16 + 46, split_y + fh, fx + hw + 16 + 46, split_y + fh + 14, c["muted"]))
    b.append(line(fx + hw + 16 + 46, split_y + fh + 14, fx + 52, split_y + fh + 14, c["muted"]))
    b.append(card(fx, ys[3], fw, fh, role, c["gpu"], title, sub, c))
    nt_y = ys[3] + fh + 14
    b.append(line(fx + 46, ys[3] + fh, fx + 46, nt_y + 14, c["muted"], head="ink"))
    b.append(text(fx + 64, nt_y + 12, t["next"], 13, c["ink2"], 600, family=MONO))
    # the cache feeds the hits
    b.append(line(mx + mw + 6, cache_y + 34, fx - 8, split_y + fh / 2, c["gpu"], 1.8, head="gpu", dash="5 5"))

    # ---- RAM and SSD ----
    rx, rw = 880, 280
    ry = split_y - 10
    b.append(text(rx, ry - 16, t["ram_title"], 15, c["ink"], 650))
    b.append(rect(rx, ry, rw, 84, c["mem"], rx=12, opacity=c["wash"]))
    b.append(rect(rx, ry + 12, 4, 60, c["mem"], rx=2))
    b.append(text(rx + 20, ry + 36, t["ram_sub"], 15, c["ink"], 650))
    b.append(text(rx + 20, ry + 58, t["ram_sub2"], 12.5, c["ink2"], 450))
    b.append(line(rx - 6, ry + 42, fx + fw + 8, split_y + fh / 2, c["cpu"], 1.8, head="cpu"))
    sy = ys[0] - 4
    b.append(text(rx, sy - 16, t["ssd_title"], 15, c["ink"], 650))
    b.append(rect(rx, sy, rw, 84, c["ssd"], rx=12, opacity=c["wash"]))
    b.append(rect(rx, sy + 12, 4, 60, c["ssd"], rx=2))
    b.append(text(rx + 20, sy + 36, t["ssd_sub"], 15, c["ink"], 650))
    b.append(text(rx + 20, sy + 58, t["ssd_sub2"], 12.5, c["ink2"], 450))
    b.append(line(rx - 6, sy + 42, fx + fw + 8, ys[0] + fh / 2, c["ssd"], 1.6, head="ssd", dash="5 5"))

    b.append(rect(48, h - 58, w - 96, 40, c["plane"], rx=12))
    b.append(text(w / 2, h - 33, t["mtp"], 14, c["ink2"], 550, anchor="middle"))
    return svg(w, h, "".join(b), c, t["arch_title"])


def performance(c, t):
    w, h = 1200, 360
    b = [text(48, 62, t["perf_title"], 26, c["ink"], 720),
         text(48, 90, t["perf_sub"], 15, c["ink2"], 450)]
    pw, pg, px0, py, ph = 352, 24, 48, 120, 210
    panels = [
        [(t["names"][0], DATA["decode_strata"], c["base"]), (t["names"][1], DATA["decode"], c["gpu"])],
        [(n, v, c["gpu"]) for n, v in DATA["prefill"]],
        [(t["ctx_labels"].get(n, n), v, c["gpu"]) for n, v in DATA["decode_ctx"]],
    ]
    vmax = [110, 2000, 110]
    delta = f"+{100 * (DATA['decode'] / DATA['decode_strata'] - 1):.0f}%"
    for i, ((title, sub), rows) in enumerate(zip(t["panels"], panels)):
        x0 = px0 + i * (pw + pg)
        b.append(rect(x0, py, pw, ph, c["plane"], rx=14))
        b.append(text(x0 + 22, py + 34, title, 16, c["ink"], 680))
        b.append(text(x0 + 22, py + 56, sub, 12.5, c["muted"], 500))
        if i == 0:
            b.append(text(x0 + pw - 22, py + 36, delta, 20, c["good"], 750, anchor="end"))
        lab_w = 124 if i == 0 else 52
        bx, bw_max = x0 + 22 + lab_w, pw - 22 - lab_w - 70
        step = 112 / max(len(rows), 1)
        for j, (lab, v, col) in enumerate(rows):
            yy = py + 84 + j * step
            b.append(text(x0 + 22, yy + 15, lab, 13, c["ink2"], 650, family=MONO if i else FONT))
            ww = bw_max * min(v, vmax[i]) / vmax[i]
            b.append(bar_h(bx, yy, ww, 22, col))
            b.append(text(bx + ww + 8, yy + 16, f"{v:,.0f}" if v >= 100 or i == 1 else f"{v:.1f}", 14, c["ink"], 650))
        b.append(line(bx, py + 78, bx, py + 84 + 112, c["axis"], 1))
    return svg(w, h, "".join(b), c, t["perf_title"])


FIGURES = {"hero": hero, "architecture": architecture, "performance": performance}


def main():
    for lang, strings in T.items():
        for theme, colors in THEMES.items():
            for name, fn in FIGURES.items():
                with open(os.path.join(OUT, f"{name}-{lang}-{theme}.svg"), "w", encoding="utf-8") as f:
                    f.write(fn(colors, strings))
    print(f"wrote {len(T) * len(THEMES) * len(FIGURES)} figures to {OUT}")


if __name__ == "__main__":
    main()
