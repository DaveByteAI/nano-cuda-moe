#!/usr/bin/env python3
"""Tensor-by-tensor comparison of bl-dsv41 --dump against bl-ref-dump (llama.cpp's deepseek41 graph), in forward order.
   dsv41_compare.py REF_DIR OURS_DIR [--pos P] [--all]
The reference evaluates the whole prompt in one graph: each tensor holds every position, the last dimension being
the token. Ours has one entry per tensor per step. For each name both have, prints the relative error
|ours - ref| / |ref| (L2) at position P (default: every position, worst shown) and the cosine similarity; the first
name that goes wrong is where to look. Tensors whose per-position size differs (idx_score: the reference row is padded
to the longest prefix) are compared on our length."""
import sys
import numpy as np

def load(d):
    rows = [l.split('\t') for l in open(f'{d}/index.tsv').read().splitlines()[1:]]
    return rows, d

def read(d, r):
    n = int(r[7]) // 4
    return np.fromfile(f'{d}/data.bin', dtype=np.int32 if r[1] == 'i32' else np.float32, count=n, offset=int(r[6]))

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    ref_dir, ours_dir = args[:2]
    pos = int(args[args.index('--pos') + 1]) if '--pos' in args else None
    show_all = '--all' in args
    ref_rows, _ = load(ref_dir)
    ours_rows, _ = load(ours_dir)

    ref = {}
    for r in ref_rows:   # the last graph that has the name (a prompt in one ubatch: the only one)
        ref[r[0]] = r
    ours = {}
    order = []
    for r in ours_rows:
        if r[0] not in ours:
            ours[r[0]] = []
            order.append(r[0])
        ours[r[0]].append(r)
    T = max(len(v) for v in ours.values())

    first_bad = None
    print(f'{"tensor":28s} {"rel err (worst pos)":>22s} {"cos":>9s}  n')
    for name in order:
        if name not in ref:
            continue
        rr = ref[name]
        ne = [int(x) for x in rr[2:6]]
        full = read(ref_dir, rr).astype(np.float64)
        steps = ours[name]
        n_ours = int(steps[0][2])
        T_ref = len(full) // n_ours if len(full) % n_ours == 0 else None
        per = None
        if T_ref == len(steps):
            per = n_ours
        else:   # padded rows (idx_score): the last dim is the token, the row the longest prefix
            last = [x for x in ne if x > 1]
            if last and last[-1] == len(steps):
                per = len(full) // len(steps)
        if per is None:
            continue
        worst, worst_p, cos_w = -1.0, -1, 1.0
        for p, s in enumerate(steps):
            if pos is not None and p != pos:
                continue
            o = read(ours_dir, s).astype(np.float64)
            r = full[p * per: p * per + len(o)]
            ok = np.isfinite(r)
            o, r = o[ok], r[ok]
            if r.size == 0:
                continue
            den = np.linalg.norm(r)
            err = np.linalg.norm(o - r) / (den if den > 0 else 1.0)
            cos = float(o @ r / (np.linalg.norm(o) * den)) if den > 0 and np.linalg.norm(o) > 0 else 1.0
            if err > worst:
                worst, worst_p, cos_w = err, p, cos
        if worst < 0:
            continue
        flag = ''
        if worst > 2e-2 and first_bad is None:
            first_bad = name
            flag = '   <-- first above 2%'
        if show_all or worst > 1e-3 or flag or name.startswith(('l_last', 'result')):
            print(f'{name:28s} {worst:14.3e} @{worst_p:<6d} {cos_w:9.6f}  {n_ours}{flag}')
    print('first tensor above 2%:', first_bad or 'none')

if __name__ == '__main__':
    main()
