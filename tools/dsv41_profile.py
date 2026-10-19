#!/usr/bin/env python3
"""Expert usage profile from BL_DSV41_TRACE files: how often each (layer, expert) was picked, most used first.
The engine warms its RAM tier from it at start (Dsv41Engine::warm; bl-chat / bl-server find it in bench/dsv41/).

    dsv41_profile.py TRACE [TRACE ...] > bench/dsv41/expert_profile.txt

Lines "layer expert count". Standard library only.
"""
import collections
import sys

cnt = collections.Counter()
for path in sys.argv[1:]:
    for line in open(path):
        head, rest = line.split(':', 1)
        il = int(head.split()[1])
        for e in rest.split('|')[0].split():
            cnt[(il, int(e))] += 1
print(f'# {sum(cnt.values())} picks from {len(sys.argv) - 1} traces; layer expert count')
for (il, e), n in sorted(cnt.items(), key=lambda kv: -kv[1]):
    print(il, e, n)
