#!/usr/bin/env bash
# Build toks (https://github.com/actual-computer/toks) as a static lib for the harness.
# Its Makefile defaults to clang; this machine has only gcc, and gcc's -Werror rejects one
# always_inline warning in src/core/spm_c.c, so CSTRICT is overridden (no source change).
set -euo pipefail
TOKS_DIR="${TOKS_DIR:-/tmp/toks}"
if [ ! -d "$TOKS_DIR" ]; then
  git clone --depth 1 https://github.com/actual-computer/toks.git "$TOKS_DIR"
fi
cd "$TOKS_DIR"
# clang is toks' default compiler; gcc needs one relaxed warning (always_inline in spm_c.c).
if command -v clang >/dev/null 2>&1; then
  make CC=clang lib -j8 >/dev/null
else
  make CC=gcc lib -j8 >/dev/null 2>&1 || \
    make CC=gcc CSTRICT='-std=c17 -O3 -fno-strict-aliasing -fwrapv -Wall -Wextra -Wconversion -Wsign-conversion -Wno-error=attributes -fno-builtin-strlen -fno-builtin-bcmp' lib -j8
fi
  make CC=gcc CSTRICT='-std=c17 -O3 -fno-strict-aliasing -fwrapv -Wall -Wextra -Wconversion -Wsign-conversion -Wno-error=attributes -fno-builtin-strlen -fno-builtin-bcmp' lib -j8
ls -l "$TOKS_DIR"/build/*/libtoks.a
