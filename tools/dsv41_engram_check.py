#!/usr/bin/env python3
"""The Engram row ids of a token list, computed the way the official engram.py NgramHashState does (numpy, int64),
from the GGUF metadata, against bl-dsv41 --engram-rows.
   dsv41_engram_check.py SHARD1.gguf TOKENS_FILE BL_DSV41
Also checks the metadata itself against engram.py's construction (primes, offsets, multipliers)."""
import struct
import subprocess
import sys

import numpy as np

SC = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def gguf_meta(path):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    _, _, nkv = struct.unpack('<IQQ', f.read(20))

    def s():
        n, = struct.unpack('<Q', f.read(8))
        return f.read(n).decode('utf-8', 'replace')

    def val(t):
        if t in SC:
            return struct.unpack('<' + SC[t], f.read(struct.calcsize(SC[t])))[0]
        if t == 8:
            return s()
        et, n = struct.unpack('<IQ', f.read(12))
        if et in SC:
            return list(struct.unpack('<%d%s' % (n, SC[et]), f.read(n * struct.calcsize(SC[et]))))
        return [val(et) for _ in range(n)]

    out = {}
    for _ in range(nkv):
        k = s()
        t, = struct.unpack('<I', f.read(4))
        out[k] = val(t)
    return out


def isprime(x):
    if x < 2:
        return False
    i = 2
    while i * i <= x:
        if x % i == 0:
            return False
        i += 1
    return True


def main():
    shard1, tokens_file, exe = sys.argv[1:4]
    m = gguf_meta(shard1)
    a = 'deepseek41.engram.'
    layers, n, heads = m[a + 'layer_ids'], m[a + 'max_ngram_size'], m[a + 'head_count']
    token_map = np.array(m[a + 'token_map'], dtype=np.int64)
    compressed = int(token_map.max()) + 1

    # the metadata against engram.py's construction (engram_vocab_size 16,000,000 in the official config)
    seen, primes = set(), []
    for _ in layers:
        per = []
        for _ in range(n - 1):
            cur = 16000000 - 1
            for _ in range(heads):
                cur += 1
                while not isprime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                per.append(cur)
        primes.append(per)
    offsets = [np.cumsum([0] + p[:-1]) for p in primes]
    bound = max(1, (np.iinfo(np.int64).max // compressed) // 2)
    mult = np.stack([np.random.default_rng(10007 * l).integers(0, bound, size=(n,), dtype=np.int64) * 2 + 1 for l in layers])
    ok = True
    for name, want in (('primes', np.array(primes).ravel()), ('offsets', np.array(offsets).ravel()), ('multipliers', mult.ravel())):
        got = np.array(m[a + name], dtype=np.uint64).astype(np.int64)
        same = np.array_equal(got, want)
        ok &= same
        print(f'metadata {name:12s} {"matches engram.py" if same else "DIFFERS"}')

    # NgramHashState.forward (text only, start_pos 0, one sequence)
    ids = np.array([int(x) for x in open(tokens_file).read().replace(',', ' ').split()], dtype=np.int64)
    pad = token_map[m[a + 'pad_id']]
    comp = token_map[ids]
    L = len(ids)
    pos = np.arange(L)
    toks, blocked = [], np.zeros(L, dtype=bool)
    for shift in range(n):
        src = comp[np.clip(pos - shift, 0, None)]
        blocked = blocked | (pos < shift)
        toks.append(np.where(blocked, pad, src))
    toks = np.stack(toks, axis=-1)                                   # [L, n]
    P = np.array(primes).reshape(len(layers), n - 1, heads)
    O = np.array(offsets).reshape(len(layers), n - 1, heads)
    rows = []
    for e in range(len(layers)):
        prod = toks * mult[e]                                        # int64, no overflow by construction
        rolling = prod[:, 0]
        per = []
        for i in range(1, n):
            rolling = np.bitwise_xor(rolling, prod[:, i])
            per.append((rolling[:, None] % P[e, i - 1]) + O[e, i - 1])
        rows.append(np.concatenate(per, axis=-1))                    # [L, (n-1) * heads]
    want = np.concatenate(rows, axis=-1)

    out = subprocess.run([exe, '--model', shard1, '--tokens-file', tokens_file, '--engram-rows'], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(out.stderr)
    got = np.array([[int(x) for x in l.split(',')] for l in out.stdout.strip().splitlines()], dtype=np.int64)
    same = got.shape == want.shape and np.array_equal(got, want)
    ok &= same
    print(f'row ids {got.shape}: {"identical to engram.py" if same else "DIFFER"}')
    if not same and got.shape == want.shape:
        bad = np.argwhere(got != want)[:5]
        print('first differences (pos, col):', bad.tolist())
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
