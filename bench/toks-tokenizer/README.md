# Strata's tokenizer vs toks

A small experiment, not a feature. It compares the tokenizer Strata actually uses today
(`tools/strata_tokenizer.py`) with the C library [toks](https://github.com/actual-computer/toks) driven
directly from C++, on the same vocab, the same merges and the same text. Nothing here is wired into the
engine, the build or the server.

## What each side is

Strata has no C++ tokenizer. `src/program/generate.cpp` only calls the Python server, and that server
tokenizes with `tools/strata_tokenizer.py`. So the comparison is Python vs C, not C++ vs C.

- Strata side: `Tokenizer(tokens, merges, token_types, "qwen35", special_ids)` from the pack's
  `tokenizer/` folder, `encode(text)` with `parse_special=False` (plain text, no chat template).
- toks side: `harness.cpp`, 100 lines, `toks_load` → `toks_scratch_init` → `toks_encode` with
  `TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS`. It reads a corpus as a flat binary (uint32 length + bytes) so
  the harness parses no JSON of its own.

toks reads a HuggingFace `tokenizer.json`, which a Strata pack does not have. `to_hf.py` writes one from
the pack's `vocab.json`, `merges.txt` and the pack's own `pre_pattern` (`Split` + `ByteLevel`, and
`normalizer: null` because Strata does not NFC-normalize). `--nfc` writes the NFC normalizer instead,
which is what a real HuggingFace file carries.

## Run it

```sh
bash bench/toks-tokenizer/run.sh                # build toks, convert, compile harness, compare
PACK=<pack tokenizer dir> bash bench/toks-tokenizer/run.sh
```

`build_toks.sh` clones toks to `/tmp/toks` if it is not there, and builds the lib with clang (toks' default)
when it is installed, gcc otherwise — no toks source is changed. `make test` still fails: its `asmcheck`
target audits the `.S` files with `llvm-objdump`, which is not installed here; all 54 objects assemble.

## Measured on wikles-pc

Linux 7.3.3-300.fc45.x86_64, 8 cores, clang 23.1.5 and gcc 16.2.1, python 3.15.2, toks 0.3.3 tier 3 (AVX2), paths 2.
Pack `coder-iq1_m` (248 320 ids, 247 587 merges). Corpus: 57 items, 21 321 bytes, 3 782 ids — the 45
prompts of `bench/profile_corpus.json` plus edge cases (empty, accents, combining accent, CJK, Arabic,
nbsp, NUL, emoji, 5 000 x's, a long blob). Cold means a fresh scratch / an empty piece cache per item,
warm means one scratch and cache reused.

| | load | encode cold | encode warm |
|---|---|---|---|
| Strata (Python) | 190–196 ms | 21–22 ms, 986–1025 ns/byte | 5 ms, 234–257 ns/byte |
| toks, clang | 241–267 ms | 1 ms, 48–51 ns/byte (18.6–19.7 MB/s) | 1 ms, 49.2–49.7 ns/byte |
| toks, gcc | 261–288 ms | 1 ms, 46–49 ns/byte (19.5–20.8 MB/s) | 1 ms, 46.9–47.0 ns/byte |

The spread between runs of the same build is the same size as the difference between the two compilers
(~±10%), so the compiler is not what decides the comparison: encode is 45–51 ns/byte either way, and the ids
are identical either way.

- **Ids are identical on all 57 items, cold and warm.** The generated HF file reproduces Strata's BPE
  exactly, including the 5 000-byte word split by `pre_pattern` and the non-ASCII cases.
- toks is about **22× faster cold** and **5× faster warm** than the Python tokenizer, and does not speed
  up when warm — its caches are already the fast path, Strata's piece cache is what makes warm Python
  cheap.
- Load is slower on toks (262 ms vs 193 ms): it parses 3.5 MB of JSON, validates and certifies it once.

## The same algorithm in plain C

`strata_bpe.c` (642 lines) is a port of `tools/strata_tokenizer.py` — the qwen35 split, the byte map, the
merge heap and the piece cache — driven by `harness_strata.c`. It is an experiment, not the engine's
tokenizer; nothing in `src/` uses it. It reads the pack's own `vocab.json` and `merges.txt`, so there is no
converted file in this path.

Same corpus as above (`out/corpus.bin`, 21 321 bytes, 57 items, 3 782 ids), gcc 16.2.1, `-O2`:

| | load | encode cold | encode warm |
|---|---|---|---|
| strata ported to C | 7 154–7 173 ms | 1 036–1 042 ms, 48 612–48 885 ns/byte (0.02 MB/s) | 824–827 ms, 38 637–38 801 ns/byte (0.026 MB/s) |

Re-measured Python on that exact `out/corpus.bin`: cold 22.4 ms, 1 049 ns/byte; warm 4.8 ms, 224 ns/byte.
So the C port is **46× slower than Python cold** and **173× slower warm**, and about **950× slower than
toks cold / 820× warm**. `-O3` changes nothing worth keeping (warm 824.6 ms, 38 673 ns/byte).

The ids are identical to Python on all 57 items, cold and warm (3 782 ids, 0 differences), and the pieces
are identical too (2 363 pieces, empty diff once the harness's `#ITEM#` separators are stripped). Load is
7 s because it parses 5.7 MB of `vocab.json` with a hand-written scanner instead of a JSON library.

Why the port loses to Python is reading, not measurement (unconfirmed static analysis): Python's `_bpe`
looks up dicts and merges whole words, while the port re-scans the node list on every merge. That is the
shape of the algorithm, not of the language — the point of the experiment was that a naive C port of
Strata's algorithm is nowhere near toks, so an engine build would want toks' algorithm, not a translation
of the Python one.

## Where they do not agree

With the real HuggingFace file `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/.../tokenizer.json`
(248 070 ids, its own NFC normalizer) on the same corpus, 2 of 57 items differ — `café` and
`e` + combining accent: toks NFC-normalizes, Strata does not. The same 2 items differ when `--nfc` is
written for the pack file. That is a tokenizer choice, not a toks bug; a Strata build using toks would
have to keep `normalizer: null` to stay byte-identical.

Not compared: the type-4 markup path (`<tool_call>` JSON re-encoding, `split_special`/greedy match in
`strata_tokenizer.py`), which toks does not model at all.

## Not run

- toks' own test suite (`make test`) — does not compile with gcc on this machine.
- Multi-thread encode (`toks_par_*`), streaming decode, and the tier 1/2 paths.
- Any tokenization inside the engine; this stays a standalone experiment.
