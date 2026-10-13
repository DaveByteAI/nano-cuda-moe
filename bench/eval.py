#!/usr/bin/env python3
"""The examiner: build a checkout of the engine, check it against a reference, measure it; one JSON report.

    python3 bench/eval.py --repo PATH [--out report.json]      # PATH: the checkout to judge (default: this repository)
    python3 bench/eval.py --make-ref                           # once, from a build known to be right

1. build      cmake --build PATH/build (configured on first use, with this repository's llama.cpp source)
2. correct    a 1K prompt's last 32 positions through the prefill path and the window path: their logits against the
              reference's, KL median <= 2e-3 and top-1 agreement >= 90% on both (one GPU against itself: ~1e-4, 97%)
3. speed      the 7 corpus texts fed in verify windows of 4 tokens (window_tok_s: the decode kernels on fixed text, the
              number to compare by); then in one process the same texts generating 200 tokens each (MTP 3) and a 32K
              prompt (twice, the second counted): decode tok/s (free text: noisy), first-token latency, the 32K
              prompt's time, experts in VRAM; against the reference's as
              fractions (vs_reference: + is faster for tok/s, slower for seconds). The tokens that match the reference's
              before the first difference are information only: near-ties flip with the prefill's float order
The inputs (bench/) and the reference (logs/eval-ref/) are this repository's, never the judged checkout's. Every run
has a time limit; a crash, a hang, a non-zero exit or a failed gate makes the verdict "fail". Needs the GPU to itself
(stop bl-server first). Python standard library only.
"""
import argparse
import array
import json
import math
import operator
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "models" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
REF = ROOT / "logs" / "eval-ref"
V = 248320
POS = 32
KL_MEDIAN_MAX, TOP1_MIN = 2e-3, 0.90


def run(cmd, timeout, cwd=None, env=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr, time.time() - t0
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return -9, out, err + f"\n[timed out after {timeout} s]", time.time() - t0


def tail(s, n=40):
    return "\n".join(s.strip().splitlines()[-n:])


def build(repo, jobs=8):
    b = repo / "build"
    if not (b / "CMakeCache.txt").exists():   # configured like this repository's build: its nvcc, its llama.cpp source
        args = ["cmake", "-B", str(b), "-S", str(repo)]
        src = ROOT / "build" / "_deps" / "llama-src"
        if src.exists(): args.append(f"-DBL_LLAMA_SRC={src}")
        cache = ROOT / "build" / "CMakeCache.txt"
        for line in cache.read_text().splitlines() if cache.exists() else []:
            for key in ("CMAKE_CUDA_COMPILER:", "CMAKE_CUDA_ARCHITECTURES:"):
                if line.startswith(key): args.append("-D" + line)
        rc, out, err, _ = run(args, 900)
        if rc: return {"ok": False, "log": tail(out + err)}
    rc, out, err, secs = run(["cmake", "--build", str(b), f"-j{jobs}"], 1800)
    errors = [l for l in (out + err).splitlines() if "error" in l.lower()]
    return {"ok": rc == 0, "seconds": round(secs, 1), "log": tail("\n".join(errors) if rc and errors else out + err, 40 if rc else 5)}


def read_logits(path):
    a = array.array("f")
    with open(path, "rb") as f: a.frombytes(f.read())
    return a


def compare(x, y):
    """KL(x || y) per position and top-1 agreement, rows of V logits"""
    n = min(len(x), len(y)) // V
    kls, agree = [], 0
    for r in range(n):
        a, b = x[r * V:(r + 1) * V], y[r * V:(r + 1) * V]
        ma, mb = max(a), max(b)
        agree += a.index(ma) == b.index(mb)
        ea = list(map(math.exp, map(operator.sub, a, [ma] * V)))
        za = math.fsum(ea)
        lse_a = ma + math.log(za)
        lse_b = mb + math.log(math.fsum(map(math.exp, map(operator.sub, b, [mb] * V))))
        # KL = sum p_a (a - b) - (lse_a - lse_b)
        kls.append(math.fsum(map(operator.mul, ea, map(operator.sub, a, b))) / za - (lse_a - lse_b))
    kls.sort()
    return {"positions": n, "kl_median": kls[n // 2] if n else None, "kl_max": kls[-1] if n else None,
            "top1": agree / n if n else 0.0}


def correctness(binary, model, ref, workdir):
    pf, win = workdir / "prefill.logits", workdir / "window.logits"
    cmd = [str(binary), "--model", str(model), "--tokens-file", str(HERE / "prompts" / "1k.tokens"), "--ctx", "4096",
           "--save-last", str(POS), "--window", "4", "--prefill-logits", str(pf), "--save-logits", str(win)]
    rc, out, err, secs = run(cmd, 600)
    res = {"seconds": round(secs, 1), "rc": rc}
    if rc or not pf.exists() or not win.exists():
        res.update(ok=False, log=tail(out + err))
        return res
    if ref is None:   # making the reference
        res["ok"] = True
        return res
    res["prefill"] = compare(read_logits(pf), read_logits(ref / "prefill.logits"))
    res["window"] = compare(read_logits(win), read_logits(ref / "window.logits"))
    res["ok"] = all(r["positions"] == POS and r["kl_median"] <= KL_MEDIAN_MAX and r["top1"] >= TOP1_MIN
                    for r in (res["prefill"], res["window"]))
    if not res["ok"]: res["why"] = f"needs KL median <= {KL_MEDIAN_MAX} and top-1 >= {TOP1_MIN:.0%} on both paths"
    return res


def windows(binary, model):
    """the decode kernels on fixed text: each corpus text fed in verify windows of 4 tokens (what a decode round
    verifies). Unlike free generation (whose text, and with it the experts and the drafts' acceptance, changes with
    the prefill's float order) it reads the same tokens every time: the number to compare versions by"""
    files = sorted((HERE / "corpus").glob("*.tokens"))
    cmd = [str(binary), "--model", str(model), "--ctx", "8192", "--window", "4"]
    for f in files: cmd += ["--tokens-file", str(f)]
    for attempt in range(2):   # a known, rare crash of this run ("free(): invalid size", docs/LOG.md 2026-10-13): once more
        rc, out, err, secs = run(cmd, 900)
        rows = re.findall(r"prompt: (\d+) tokens in ([\d.]+) s \(", err)
        if not rc and len(rows) == len(files): break
    if rc or len(rows) != len(files):
        return {"ok": False, "rc": rc, "texts_done": len(rows), "log": tail(out + err)}
    n, t = sum(int(r[0]) for r in rows), sum(float(r[1]) for r in rows)
    return {"ok": True, "window_tok_s": round(n / t, 2), "seconds": round(secs, 1), "retried": attempt}


def speed(binary, model):
    w = windows(binary, model)
    if not w["ok"]: return w
    # the 32K prompt twice, the second counted: the first reads the n-gram table's rows from the SSD when the page cache
    # does not hold them (seconds of difference between runs otherwise)
    files = sorted((HERE / "corpus").glob("*.tokens")) + [HERE / "prompts" / "32k.tokens"] * 2
    cmd = [str(binary), "--model", str(model), "--gen", "200", "--spec", "3", "--ctx", "40960", "--no-prepass"]
    for f in files: cmd += ["--tokens-file", str(f)]
    rc, out, err, secs = run(cmd, 1200)
    res = {"seconds": round(secs, 1), "rc": rc}
    rows = re.findall(r"prompt: (\d+) tokens in ([\d.]+) s; decode: (\d+) tokens in ([\d.]+) s \(([\d.]+) tok/s\)", out)
    if rc or len(rows) != len(files) or "stuck waiting" in err:
        res.update(ok=False, log=tail(out + err))
        return res
    corpus = rows[:-2]
    toks = sum(int(r[2]) for r in corpus)
    secs_dec = sum(float(r[3]) for r in corpus)
    res["ok"] = True
    res["window_tok_s"] = w["window_tok_s"]
    res["seconds"] = round(res["seconds"] + w["seconds"], 1)
    res["decode_tok_s"] = round(toks / secs_dec, 2)                           # the 7 texts, as one number (free text: noisy)
    res["decode_per_text"] = [float(r[4]) for r in corpus]
    res["first_token_s"] = float(corpus[0][1])                                # a 34-token chat turn
    res["prompt_32k_s"] = float(rows[-1][1])
    res["decode_after_32k_tok_s"] = float(rows[-1][4])
    if m := re.search(r"experts: (\d+) of \d+ in VRAM \(([\d.]+) GB\)", err):
        res["experts_in_vram"], res["expert_cache_gb"] = int(m.group(1)), float(m.group(2))
    res["generated"] = re.findall(r"^generated:(.*)$", out, re.M)
    return res


def speed_avg(binary, model, runs):
    """speed() `runs` times (a process each): the means, and every run's numbers (between processes the corpus decode
    moves by up to ~3%)"""
    rs = []
    for _ in range(runs):
        r = speed(binary, model)
        if not r["ok"]: return r
        rs.append(r)
    if runs == 1: return rs[0]
    out = dict(rs[0])
    for k in ("window_tok_s", "decode_tok_s", "first_token_s", "prompt_32k_s", "decode_after_32k_tok_s"):
        out[k] = round(sum(r[k] for r in rs) / runs, 3)
        out[k + "_runs"] = [r[k] for r in rs]
    out["seconds"] = round(sum(r["seconds"] for r in rs), 1)
    return out


def prompt32(binary, model):
    """the 32K prompt's time (run twice in one process, the second counted: the first may read the n-gram table from the SSD)"""
    f = str(HERE / "prompts" / "32k.tokens")
    for _ in range(2):   # (once more after a crash, as in windows())
        rc, out, err, secs = run([str(binary), "--model", str(model), "--ctx", "40960", "--gen", "1", "--spec", "0", "--no-prepass",
                                  "--tokens-file", f, "--tokens-file", f], 900)
        rows = re.findall(r"prompt: (\d+) tokens in ([\d.]+) s;", out)
        if not rc and len(rows) == 2: break
    if rc or len(rows) != 2: return None
    return float(rows[-1][1])


def ab(base, cand, model, rounds, prompt_rounds):
    """the base build and the candidate measured in turns, the order swapped every round (the machine drifts: one process
    against another an hour apart differs by up to ~5% on the windows); per pair, candidate / base"""
    w = {"base": [], "cand": []}
    p = {"base": [], "cand": []}
    bins = {"base": base, "cand": cand}
    for r in range(max(rounds, prompt_rounds)):
        for who in (("base", "cand") if r % 2 == 0 else ("cand", "base")):
            if r < rounds:
                x = windows(bins[who], model)
                if not x["ok"]: return {"ok": False, "who": who, "log": x.get("log", "")}
                w[who].append(x["window_tok_s"])
            if r < prompt_rounds:
                t = prompt32(bins[who], model)
                if t is None: return {"ok": False, "who": who, "log": "the 32K prompt run failed"}
                p[who].append(t)
    wr = sorted(c / b - 1 for b, c in zip(w["base"], w["cand"]))
    pr = sorted(b / c - 1 for b, c in zip(p["base"], p["cand"]))   # + is faster
    med = lambda v: v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
    gain = lambda v: {"median": round(med(v), 4), "min": round(v[0], 4), "max": round(v[-1], 4)} if v else None
    return {"ok": True, "windows": w, "prompt_32k_s": p, "windows_gain": gain(wr), "prompt_gain": gain(pr)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(ROOT), help="the checkout to judge")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--ref", default=str(REF))
    ap.add_argument("--make-ref", action="store_true", help="store this checkout's results as the reference")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--speed-runs", type=int, default=1, help="speed measurements (a process each), averaged")
    ap.add_argument("--out", help="also write the report here")
    ap.add_argument("--base", help="A/B instead of speed: this checkout's build/bl-run against the judged one, in turns")
    ap.add_argument("--rounds", type=int, default=3, help="A/B: window runs of each")
    ap.add_argument("--prompt-rounds", type=int, default=2, help="A/B: 32K prompt runs of each")
    ap.add_argument("--no-speed", action="store_true", help="build and correctness only")
    a = ap.parse_args()
    repo, ref = pathlib.Path(a.repo).resolve(), pathlib.Path(a.ref)
    if subprocess.run(["pgrep", "-x", "bl-server"], capture_output=True).returncode == 0:
        sys.exit("bl-server is running: the examiner needs the GPU to itself")
    if not a.make_ref and not (ref / "prefill.logits").exists():
        sys.exit(f"no reference in {ref}: run with --make-ref on a build known to be right first")
    work = ROOT / "logs" / "eval-work"
    work.mkdir(parents=True, exist_ok=True)
    if a.make_ref: ref.mkdir(parents=True, exist_ok=True)
    report = {"repo": str(repo), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    report["build"] = {"ok": True, "skipped": True} if a.skip_build else build(repo)
    binary = repo / "build" / "bl-run"
    if report["build"]["ok"]:
        report["correct"] = correctness(binary, a.model, None if a.make_ref else ref, ref if a.make_ref else work)
        if report["correct"]["ok"]:
            if a.base:
                report["ab"] = ab(pathlib.Path(a.base) / "build" / "bl-run", binary, a.model, a.rounds, a.prompt_rounds)
            elif not a.no_speed:
                report["speed"] = speed_avg(binary, a.model, a.speed_runs)
    measured = report.get("ab", report.get("speed", {"ok": a.no_speed}))
    ok = report["build"]["ok"] and report.get("correct", {}).get("ok") and measured.get("ok")
    report["verdict"] = "pass" if ok else "fail"
    report["seconds"] = round(time.time() - t0, 1)
    if a.make_ref and ok:
        (ref / "speed.json").write_text(json.dumps(report["speed"], indent=1) + "\n")
        report["note"] = f"reference stored in {ref}"
    elif ok and "speed" in report and (ref / "speed.json").exists():   # against the reference's speed
        base = json.loads((ref / "speed.json").read_text())
        s = report["speed"]
        s["vs_reference"] = {k: round(s[k] / base[k] - 1, 4) for k in ("window_tok_s", "decode_tok_s", "prompt_32k_s", "decode_after_32k_tok_s") if k in base}
        gen_ref = base.get("generated", [])
        same = [len(os.path.commonprefix([x.split(), y.split()])) for x, y in zip(s["generated"], gen_ref)]
        s["same_tokens_as_reference"] = same   # per text: how many tokens match before the first difference (greedy)
    report.get("speed", {}).pop("generated", None)   # (kept in the reference only)
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if a.out: pathlib.Path(a.out).write_text(text + "\n")
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
