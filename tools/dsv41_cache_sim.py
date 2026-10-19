#!/usr/bin/env python3
"""Expert cache policy simulation on BL_DSV41_TRACE files (one line per decode step and layer:
"pos layer: e0 .. e5 | next layer's top-24 prediction"). The picks do not depend on the cache, so one trace replays
every policy. Counts per step after --warm steps: VRAM misses (computed on the CPU) and disk reads (RAM tier misses,
demanded or prefetched).

    dsv41_cache_sim.py TRACE [TRACE ...] [--vram 1529] [--ram 5714] [--warm 128]

Several traces are concatenated (one long session). Standard library only.
"""
import argparse
import heapq
from collections import OrderedDict, defaultdict

N_EXPERT = 384


def load(paths):
    steps = []   # [ [ (layer, picks, pred), ... per layer ], ... per step ]
    for p in paths:
        cur, last_pos = None, None
        for line in open(p):
            head, rest = line.split(':', 1)
            pos, il = map(int, head.split())
            picks, pred = rest.split('|')
            picks = [int(v) for v in picks.split()]
            pred = [int(v) for v in pred.split()]
            if pos != last_pos:
                cur = []
                steps.append(cur)
                last_pos = pos
            cur.append((il, picks, pred))
    return steps


class Lru:
    def __init__(self, n):
        self.n, self.d = n, OrderedDict()

    def __contains__(self, k):
        return k in self.d

    def touch(self, k):
        self.d.move_to_end(k)

    def add(self, k):   # returns the evicted key or None
        if k in self.d:
            self.d.move_to_end(k)
            return None
        self.d[k] = 1
        if len(self.d) > self.n:
            return self.d.popitem(last=False)[0]
        return None

    def discard(self, k):
        self.d.pop(k, None)

    def to_tail(self, k):   # first in line for eviction
        if k in self.d:
            self.d.move_to_end(k, last=False)


class Slru:
    """2Q-like: new keys enter a probation LRU; a second use promotes into the protected LRU (its eviction demotes
    back into probation)"""
    def __init__(self, n, frac=0.75):
        self.prot, self.prob = Lru(int(n * frac)), Lru(n - int(n * frac))

    def __contains__(self, k):
        return k in self.prot or k in self.prob

    def touch(self, k):
        if k in self.prot:
            self.prot.touch(k)
        elif k in self.prob:
            self.prob.discard(k)
            ev = self.prot.add(k)
            if ev is not None:
                self.prob.add(ev)

    def add(self, k):
        if k in self:
            self.touch(k)
            return None
        return self.prob.add(k)

    def discard(self, k):
        self.prot.discard(k)
        self.prob.discard(k)

    def to_tail(self, k):
        if k in self.prot:
            self.prot.discard(k)
            self.prob.add(k)
        self.prob.to_tail(k)


class Lfu:
    """evicts the lowest (use count, last use) among the resident keys; a heap with stale entries skipped"""
    def __init__(self, n, cnt):
        self.n, self.s, self.cnt, self.t, self.h = n, {}, cnt, 0, []

    def __contains__(self, k):
        return k in self.s

    def touch(self, k):
        self.t += 1
        self.s[k] = (self.cnt[k], self.t)
        heapq.heappush(self.h, (self.cnt[k], self.t, k))

    def add(self, k):
        self.touch(k)
        if len(self.s) > self.n:
            while True:
                c, t, v = heapq.heappop(self.h)
                if self.s.get(v) == (c, t):
                    del self.s[v]
                    return v
        return None

    def discard(self, k):
        self.s.pop(k, None)


def simulate(steps, vram, ram, warm, admit=1, exclusive=False, prefetch=0, ram_policy='lru', vram_policy='lru',
             tail=False):
    cnt = defaultdict(float)
    V = Lru(vram) if vram_policy == 'lru' else Lfu(vram, cnt)
    R = {'lru': Lru, 'slru': Slru}[ram_policy](ram) if ram_policy != 'lfu' else Lfu(ram, cnt)
    vm = dr = pf = pf_used = 0
    n = 0
    pending = set()   # prefetched keys not used yet
    for si, layers in enumerate(steps):
        measure = si >= warm
        if measure:
            n += 1
        for il, picks, pred in layers:
            misses = []
            for e in picks:
                k = il * N_EXPERT + e
                cnt[k] += 1
                if k in V:
                    V.touch(k)
                    continue
                misses.append(k)
            if measure:
                vm += len(misses)
            for k in misses:   # through the RAM tier
                if k in R:
                    R.touch(k)
                    if k in pending:
                        pending.discard(k)
                        if measure:
                            pf_used += 1
                else:
                    if measure:
                        dr += 1
                    R.add(k)
            for k in misses[:admit]:   # into VRAM
                ev = V.add(k)
                if exclusive:
                    R.discard(k)
                    if ev is not None:
                        R.add(ev)   # demoted to RAM (a device-to-host copy)
                elif tail:
                    R.to_tail(k)   # its RAM copy goes first
            if prefetch and il + 1 < 40:
                for e in pred[:prefetch]:
                    k = (il + 1) * N_EXPERT + e
                    if k in V or k in R:
                        continue
                    R.add(k)
                    pending.add(k)
                    if measure:
                        pf += 1
                        dr += 1
    n = max(n, 1)
    return dict(vram_miss=vm / n, disk=dr / n, prefetch=pf / n, prefetch_used=pf_used / n)


def belady(steps, cap, warm):
    """the optimal replacement for one cache of `cap` (the disk-read floor of a single tier of that size)"""
    seq = []
    for si, layers in enumerate(steps):
        for il, picks, _ in layers:
            for e in picks:
                seq.append((si, il * N_EXPERT + e))
    nxt = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        k = seq[i][1]
        nxt[i] = last.get(k, 1 << 60)
        last[k] = i
    res, heap, misses = {}, [], 0
    nmeas = sum(1 for si in range(len(steps)) if si >= warm)
    for i, (si, k) in enumerate(seq):
        if k in res:
            res[k] = nxt[i]
            heapq.heappush(heap, (-nxt[i], k))
            continue
        if si >= warm:
            misses += 1
        if len(res) >= cap:
            while True:
                nn, v = heapq.heappop(heap)
                if v in res and res[v] == -nn:
                    del res[v]
                    break
        res[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return misses / max(nmeas, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('trace', nargs='+')
    ap.add_argument('--vram', type=int, default=1529)
    ap.add_argument('--ram', type=int, default=5714)
    ap.add_argument('--warm', type=int, default=128)
    a = ap.parse_args()
    steps = load(a.trace)
    uniq = len({il * N_EXPERT + e for s in steps for il, p, _ in s for e in p})
    print(f'{len(steps)} steps, {uniq} distinct experts used ({100 * uniq / (40 * N_EXPERT):.1f}% of all)')
    rows = [
        ('now: inclusive LRU, admit 1', {}),
        ('admit 2', dict(admit=2)),
        ('admit 6', dict(admit=6)),
        ('exclusive tiers', dict(exclusive=True)),
        ('exclusive, admit 2', dict(exclusive=True, admit=2)),
        ('RAM LFU', dict(ram_policy='lfu')),
        ('exclusive, RAM LFU', dict(exclusive=True, ram_policy='lfu')),
        ('VRAM LFU, admit 6', dict(vram_policy='lfu', admit=6)),
        ('admit 0', dict(admit=0)),
        ('admitted -> RAM tail', dict(tail=True)),
        ('admitted -> RAM tail, admit 2', dict(tail=True, admit=2)),
        ('RAM SLRU', dict(ram_policy='slru')),
        ('exclusive, RAM SLRU', dict(exclusive=True, ram_policy='slru')),
        ('exclusive, RAM SLRU, admit 2', dict(exclusive=True, ram_policy='slru', admit=2)),
        ('exclusive, prefetch 2', dict(exclusive=True, prefetch=2)),
        ('prefetch 2', dict(prefetch=2)),
        ('prefetch 6', dict(prefetch=6)),
    ]
    print(f'{"policy":32s} {"VRAM miss/step":>14s} {"disk/step":>10s} {"prefetch/step":>14s} {"pf used/step":>13s}')
    for name, kw in rows:
        r = simulate(steps, a.vram, a.ram, a.warm, **kw)
        print(f'{name:32s} {r["vram_miss"]:14.2f} {r["disk"]:10.2f} {r["prefetch"]:14.2f} {r["prefetch_used"]:13.2f}')
    print(f'Belady floor, one tier of RAM {a.ram}: {belady(steps, a.ram, a.warm):.2f} disk/step; '
          f'of RAM+VRAM {a.ram + a.vram}: {belady(steps, a.ram + a.vram, a.warm):.2f}')


if __name__ == '__main__':
    main()
