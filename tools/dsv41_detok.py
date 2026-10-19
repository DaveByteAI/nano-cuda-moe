#!/usr/bin/env python3
"""Token ids <-> text with DeepSeek-V4.1's official tokenizer.json (the `tokenizers` package).
   dsv41_detok.py decode 671,6102,294        (or ids on stdin, any separators)
   dsv41_detok.py encode "some text"         (prints BOS 0 then the ids, comma separated)
The tokenizer file: data/dsv41/tokenizer.json, or $DSV41_TOKENIZER."""
import os
import re
import sys

from tokenizers import Tokenizer

here = os.path.dirname(os.path.abspath(__file__))
path = os.environ.get('DSV41_TOKENIZER', os.path.join(here, '..', 'data', 'dsv41', 'tokenizer.json'))
tok = Tokenizer.from_file(path)
mode = sys.argv[1] if len(sys.argv) > 1 else 'decode'
arg = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else sys.stdin.read()
if mode == 'decode':
    ids = [int(x) for x in re.findall(r'\d+', arg)]
    print(tok.decode(ids, skip_special_tokens=False))
elif mode == 'encode':
    print(','.join(map(str, [0] + tok.encode(arg, add_special_tokens=False).ids)))
else:
    sys.exit(__doc__)
