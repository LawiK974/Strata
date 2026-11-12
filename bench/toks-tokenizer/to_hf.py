#!/usr/bin/env python3
"""Write an HF-format tokenizer.json from a Strata pack's tokenizer/ folder.

toks reads HuggingFace tokenizer.json; a Strata pack ships vocab.json + merges.txt + token_type.json +
its own tokenizer.json (model "gpt2", pre "qwen35", pre_pattern).  This builds the HF file with the same
vocab, the same merges in the same rank order, and the pack's own pre_pattern, so the two tokenizers do
the same job.  normalizer is null: Strata does not NFC-normalize, so a null normalizer is what makes the
ids comparable.  Pass --nfc to write NFC instead, which is what a real HF file carries.
"""
import json
import pathlib
import sys

def convert(tdir: pathlib.Path, out: pathlib.Path, nfc: bool = False) -> None:
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    merges = [m for m in (tdir / "merges.txt").read_text(encoding="utf-8").split("\n") if m.strip()]
    meta = json.loads((tdir / "tokenizer.json").read_text(encoding="utf-8"))
    doc = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "normalizer": {"type": "NFC"} if nfc else None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": meta["pre_pattern"]}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        ]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
    }
    out.write_text(json.dumps(doc), encoding="utf-8")
    print("%s: %d ids, %d merges, normalizer %s" %
          (out, len(vocab), len(merges), "NFC" if nfc else "null"))

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--nfc"]
    if len(args) != 2:
        raise SystemExit("usage: to_hf.py <pack tokenizer dir> <out tokenizer.json> [--nfc]")
    convert(pathlib.Path(args[0]), pathlib.Path(args[1]), "--nfc" in sys.argv)
