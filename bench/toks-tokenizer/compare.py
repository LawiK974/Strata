#!/usr/bin/env python3
"""Compare Strata's tokenizer (tools/strata_tokenizer.py) with toks (github.com/actual-computer/toks)
on this machine, on the same corpus and the same vocab/merges.

Both sides encode plain text (no special tokens, no template): Strata's encode(text) with
parse_special=False, toks with TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS.  Ids must match exactly; the
script reports the first mismatch if they do not.

usage: compare.py <pack tokenizer dir> <tokenizer.json for toks>
"""
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
import strata_tokenizer as ST  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"


def load_strata(tdir: pathlib.Path) -> ST.Tokenizer:
    if tdir.is_file():                      # a real HF tokenizer.json: its model.vocab / model.merges
        doc = json.loads(tdir.read_text(encoding="utf-8"))
        vocab = doc["model"]["vocab"]
        merges = doc["model"]["merges"]
        tokens = [None] * (max(vocab.values()) + 1)
        for t, i in vocab.items():
            tokens[i] = t
        return ST.Tokenizer(tokens, merges, None, "qwen35", None)
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * (max(vocab.values()) + 1)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tdir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tdir / "token_type.json").read_text())
    meta = json.loads((tdir / "tokenizer.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, merges, types, meta["pre"], meta.get("special_ids"))


def corpus() -> list[str]:
    items = [p["p"] for p in json.loads((HERE.parents[0] / "profile_corpus.json").read_text())]
    items += ["", "hello world", "café", "e\u0301", "你好世界", "مرحبا", "a\u00a0b",
              "\x00\x01\x7f", "😀🚀", "x" * 5000, "for i in range(10):\n    print(i)\n"]
    items += [" ".join(items)]                      # one long blob
    return items


def write_corpus(items: list[str]) -> pathlib.Path:
    path = OUT / "corpus.bin"
    with open(path, "wb") as f:
        for s in items:
            b = s.encode("utf-8")
            f.write(len(b).to_bytes(4, "little"))
            f.write(b)
    return path


def strata_run(tok: ST.Tokenizer, items: list[str], cold: bool) -> tuple[list[list[int]], float, float]:
    t0 = time.perf_counter()
    ids = []
    for s in items:
        if cold:
            tok._piece_ids.clear()                 # no piece cache: every item is a first sight
        ids.append(tok.encode(s))
    t1 = time.perf_counter()
    return ids, t1 - t0, t0


def toks_run(harness: pathlib.Path, tj: pathlib.Path, corpus: pathlib.Path, mode: str) -> list[list[int]]:
    r = subprocess.run([str(harness), str(tj), str(corpus), str(OUT / ("toks_%s.ids" % mode)), mode],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("harness failed: %s %s" % (r.stdout, r.stderr))
    print(" ".join(r.stdout.splitlines()))
    lines = (OUT / ("toks_%s.ids" % mode)).read_text().splitlines()
    return [[int(x) for x in l.split()] for l in lines]


def diff(a: list[list[int]], b: list[list[int]], items: list[str]) -> None:
    if len(a) != len(b):
        raise SystemExit("different item counts: %d vs %d" % (len(a), len(b)))
    bad = 0
    first = None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            bad += 1
            if first is None:
                first = (i, items[i][:60], x[:12], y[:12])
    if bad == 0:
        print("ids identical on all %d items (%d ids)" % (len(a), sum(len(x) for x in a)))
    else:
        print("MISMATCH on %d of %d items; first: item %d %r" % (bad, len(a), first[0], first[1]))
        print("  strata %s\n  toks   %s" % (first[2], first[3]))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare.py <pack tokenizer dir> <toks tokenizer.json>")
    OUT.mkdir(exist_ok=True)
    harness = OUT / "harness"
    if not harness.exists():
        raise SystemExit("build the harness first (run.sh)")
    items = corpus()
    corpus_path = write_corpus(items)
    bytes_total = sum(len(s.encode("utf-8")) for s in items)

    t0 = time.perf_counter()
    tok = load_strata(pathlib.Path(sys.argv[1]))
    load_s = time.perf_counter() - t0
    print("strata load %.0f ms" % (load_s * 1e3))

    for cold in (True, False):
        ids, secs, _ = strata_run(tok, items, cold)
        print("strata encode %.0f ms (%s), %.2f ns/byte" %
              (secs * 1e3, "cold" if cold else "warm", secs * 1e9 / bytes_total))
        (OUT / ("strata_%s.ids" % ("cold" if cold else "warm"))).write_text(
            "\n".join(" ".join(map(str, x)) for x in ids))
        toks_ids = toks_run(harness, pathlib.Path(sys.argv[2]), corpus_path, "cold" if cold else "warm")
        diff(ids, toks_ids, items)


if __name__ == "__main__":
    main()
