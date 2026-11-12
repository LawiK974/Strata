#!/usr/bin/env bash
# Run the whole experiment: build toks, convert the pack's tokenizer to HF format, compile the harness,
# compare ids and timings.  TOKS_DIR defaults to /tmp/toks; PACK to the coder pack on this machine.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TOKS_DIR="${TOKS_DIR:-/tmp/toks}"
PACK="${PACK:-/home/wikle/work-rx6000/strata-data/Strata-data/packs/coder-iq1_m/tokenizer}"
mkdir -p "$HERE/out"
TOKS_DIR="$TOKS_DIR" bash "$HERE/build_toks.sh"
python3 "$HERE/to_hf.py" "$PACK" "$HERE/out/tokenizer.hf.json"
g++ -O2 -I "$TOKS_DIR/include" "$HERE/harness.cpp" "$TOKS_DIR/build/linux-x86_64/libtoks.a" -pthread -o "$HERE/out/harness"
python3 "$HERE/compare.py" "$PACK" "$HERE/out/tokenizer.hf.json"

# The same algorithm ported to plain C (strata_bpe.c), timed on the same corpus.
# -std=gnu11, not c11: clock_gettime needs POSIX, which c11 hides on glibc.
gcc -O2 -std=gnu11 -Wall -Wextra "$HERE/strata_bpe.c" "$HERE/harness_strata.c" -o "$HERE/out/harness_strata"
for mode in cold warm; do
  "$HERE/out/harness_strata" "$PACK" "$HERE/out/corpus.bin" "$HERE/out/strata_c_$mode.ids" "$mode"
done
