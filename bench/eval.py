#!/usr/bin/env python3
"""The examiner: build a checkout of the engine, check it against a reference, measure it; one JSON report.

    python3 bench/eval.py --repo PATH [--out report.json]      # PATH: the checkout to judge (default: this repository)
    python3 bench/eval.py --make-ref                           # once, from a build known to be right

1. build      cmake --build PATH/build (configured on first use, with this repository's llama.cpp source)
2. correct    a 1K prompt's last 32 positions through the prefill path and the window path: their logits against the
              reference's, KL median <= 2e-3 and top-1 agreement >= 90% on both (one GPU against itself: ~1e-4, 97%)
3. speed      the 7 corpus texts (200 tokens each, MTP 3) and a 32K prompt (twice, the second counted) in one process:
              decode tok/s, first-token latency, the 32K prompt's time, experts in VRAM; against the reference's as
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


def build(repo):
    b = repo / "build"
    if not (b / "CMakeCache.txt").exists():
        src = ROOT / "build" / "_deps" / "llama-src"
        rc, out, err, _ = run(["cmake", "-B", str(b), "-S", str(repo)] + ([f"-DBL_LLAMA_SRC={src}"] if src.exists() else []), 900)
        if rc: return {"ok": False, "log": tail(out + err)}
    rc, out, err, secs = run(["cmake", "--build", str(b), "-j8"], 1800)
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


def speed(binary, model):
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
    res["decode_tok_s"] = round(toks / secs_dec, 2)                           # the 7 texts, as one number
    res["decode_per_text"] = [float(r[4]) for r in corpus]
    res["first_token_s"] = float(corpus[0][1])                                # a 34-token chat turn
    res["prompt_32k_s"] = float(rows[-1][1])
    res["decode_after_32k_tok_s"] = float(rows[-1][4])
    if m := re.search(r"experts: (\d+) of \d+ in VRAM \(([\d.]+) GB\)", err):
        res["experts_in_vram"], res["expert_cache_gb"] = int(m.group(1)), float(m.group(2))
    res["generated"] = re.findall(r"^generated:(.*)$", out, re.M)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(ROOT), help="the checkout to judge")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--ref", default=str(REF))
    ap.add_argument("--make-ref", action="store_true", help="store this checkout's results as the reference")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--out", help="also write the report here")
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
            report["speed"] = speed(binary, a.model)
    ok = report["build"]["ok"] and report.get("correct", {}).get("ok") and report.get("speed", {}).get("ok")
    report["verdict"] = "pass" if ok else "fail"
    report["seconds"] = round(time.time() - t0, 1)
    if a.make_ref and ok:
        (ref / "speed.json").write_text(json.dumps(report["speed"], indent=1) + "\n")
        report["note"] = f"reference stored in {ref}"
    elif ok and (ref / "speed.json").exists():   # against the reference's speed
        base = json.loads((ref / "speed.json").read_text())
        s = report["speed"]
        s["vs_reference"] = {k: round(s[k] / base[k] - 1, 4) for k in ("decode_tok_s", "prompt_32k_s", "decode_after_32k_tok_s") if k in base}
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
