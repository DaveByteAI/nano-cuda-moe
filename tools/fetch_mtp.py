#!/usr/bin/env python3
"""Download the MTP (multi-token prediction) layer of Qwen3.8-Flash-Next - and nothing else.

The quantized GGUF has no MTP layer; the official BF16 checkpoint (Qwen/Qwen3.8-Flash-Next, 360 GB in 131 files) does.
Its 31 `mtp.*` tensors (~5.5 GB) are spread over 28 of the files. A safetensors file starts with an 8-byte header
length and a JSON header giving every tensor's byte range, so each tensor is fetched with HTTP Range requests alone.

    python3 tools/fetch_mtp.py [--out models/mtp-src]

then `build/bl-mtp-pack models/mtp-src models/mtp-q2_0.gguf` quantizes the experts and writes the GGUF the engine
loads. Python standard library only. Resumable (per tensor, byte-exact). Mirrors: HF_ENDPOINT=https://hf-mirror.com.
A proxy is taken from https_proxy as usual.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"   # main on 2026-10-06: every install reads the same bytes
ENDPOINT = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")

# SHA-256 of each tensor's bytes at REVISION (filled in from the first verified download); a mismatch stops the run
PINNED_SHA256 = {}


def url(fname):
    return f"{ENDPOINT}/{REPO}/resolve/{REVISION}/{fname}"


def get(fname, start=None, end=None):
    """The bytes [start, end] (inclusive) of a file of the repository, or the whole file."""
    req = urllib.request.Request(url(fname), headers={"User-Agent": "nano-cuda-moe"})
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    resp = urllib.request.urlopen(req, timeout=60)
    if start is not None and resp.status != 206:
        sys.exit(f"{fname}: the server ignored the Range request (HTTP {resp.status}); try another HF_ENDPOINT")
    return resp


def header(fname):
    n = int.from_bytes(get(fname, 0, 7).read(), "little")
    return 8 + n, json.loads(get(fname, 8, 8 + n - 1).read())


def fetch(fname, begin, end, path):
    """Bytes [begin, end) of fname into path, resuming a partial file."""
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have > end - begin:
        os.remove(path)
        have = 0
    if have == end - begin:
        return
    resp = get(fname, begin + have, end - 1)
    done, total = have, end - begin
    with open(path, "ab") as f:
        while chunk := resp.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {os.path.basename(path)}: {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    if done != total:
        sys.exit(f"{path}: got {done} bytes of {total}; run again to resume")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models/mtp-src", help="directory for the raw tensors and manifest.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    index = json.loads(get("model.safetensors.index.json").read())
    wanted = {k: v for k, v in index["weight_map"].items() if k.startswith("mtp.")}
    print(f"{REPO}@{REVISION[:7]}: {len(wanted)} MTP tensors in {len(set(wanted.values()))} files")
    manifest = []
    for fname in sorted(set(wanted.values())):
        base, hdr = header(fname)
        for name in sorted(k for k, v in wanted.items() if v == fname):
            t = hdr[name]
            b, e = t["data_offsets"]
            path = os.path.join(a.out, name + ".bin")
            fetch(fname, base + b, base + e, path)
            manifest.append({"name": name, "dtype": t["dtype"], "shape": t["shape"], "file": os.path.basename(path)})
    bad = 0
    for m in manifest:
        m["sha256"] = sha256(os.path.join(a.out, m["file"]))
        want = PINNED_SHA256.get(m["name"])
        if want and want != m["sha256"]:
            print(f"  {m['name']}: SHA-256 mismatch - delete {m['file']} and run again")
            bad += 1
    if bad:
        sys.exit(f"{bad} tensor(s) corrupt")
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump({"repo": REPO, "revision": REVISION, "tensors": manifest}, f, indent=1)
    total = sum(os.path.getsize(os.path.join(a.out, m["file"])) for m in manifest)
    print(f"{len(manifest)} tensors, {total / 1e9:.2f} GB in {a.out}"
          f"{' (all SHA-256 checked)' if PINNED_SHA256 else ''}")


if __name__ == "__main__":
    main()
