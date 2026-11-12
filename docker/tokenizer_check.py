#!/usr/bin/env python3
"""docker/tokenizer_check.py - the C++ toks tokenizer against the Python reference, on the same strings.

  tokenizer_check.py <pack tokenizer dir>

Same items, both encoders: ids must match item by item, and the C++ side is timed against the Python one.
"""
import json
import pathlib
import struct
import subprocess
import sys
import time

from strata_tokenizer import Tokenizer

ITEMS = [
    "hello world",
    "",
    "  spaces\tand\nnewlines\r\n",
    "\x01\x02\x1f control bytes",
    "café naïve — Berlin ✓",
    "日本語のテキストと中文",
    "emoji 🙂🚀🧑‍💻 and Z̈́x́ combining marks",
    "e\u0301combining\u0301mark",
    "a very long ASCII run: " + "x" * 4000,
    "\xef\xbf\xbd replacement byte",
    '{"json": "quotes and \\backslash"}',
    "The quick brown fox jumps over the lazy dog. " * 40,
]


def load_pack(d: pathlib.Path) -> Tokenizer:
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    tokens = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    merges = (d / "merges.txt").read_text(encoding="utf-8").split("\n")
    merges = [m for m in merges if m]
    types = json.loads((d / "token_type.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    return Tokenizer(tokens, merges, types, meta.get("pre", "qwen35"), meta.get("special_ids"))


def main() -> int:
    pack = pathlib.Path(sys.argv[1])
    tok = load_pack(pack)
    # The pack is mounted read-only, so the corpus and the ids go to /tmp.
    corpus = pathlib.Path("/tmp/corpus.bin")
    with corpus.open("wb") as f:
        for s in ITEMS:
            b = s.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)

    py_ids = [tok.encode(s) for s in ITEMS]
    t0 = time.perf_counter()
    py_ids = [tok.encode(s) for s in ITEMS]
    py_ms = (time.perf_counter() - t0) * 1000.0

    ids_out = corpus.with_suffix(".ids")
    t0 = time.perf_counter()
    subprocess.run(["strata-toks-tokenizer", str(pack), str(corpus), str(ids_out)], check=True)
    cpp_ms = (time.perf_counter() - t0) * 1000.0

    cpp_ids = [[int(x) for x in line.split()] for line in ids_out.read_text().splitlines()]
    bad = 0
    for s, p, c in zip(ITEMS, py_ids, cpp_ids):
        if p != c:
            bad += 1
            print(f"MISMATCH {s[:32]!r}: python {len(p)} ids, cpp {len(c)} ids")
    total = sum(len(x) for x in py_ids)
    nbytes = sum(len(s.encode("utf-8")) for s in ITEMS)
    print(f"items {len(ITEMS)}, bytes {nbytes}, ids {total}, mismatch {bad}")
    # The cpp figure is the whole CLI process: loading the pack, then encoding.  The python figure is the
    # encode loop alone with the pack already loaded, so the two ns/byte numbers are not comparable.
    print(f"python encode {py_ms:.1f} ms ({py_ms * 1e6 / nbytes:.0f} ns/byte), "
          f"cpp load+encode {cpp_ms:.1f} ms ({cpp_ms * 1e6 / nbytes:.0f} ns/byte)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
