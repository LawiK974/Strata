#!/usr/bin/env bash
# Build the image with the toks-backed pack tokenizer inside it.  podman, no docker and no root.
#
#   docker/build_tokenizer.sh [--pack <pack tokenizer dir>] [--toks <toks checkout>] [--tag <name>] [--no-build]
#
# --pack runs the image against that pack right after building it.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
TOKS=/tmp/toks
PACK=""
TAG=strata-toks-tokenizer:local
RUN_BUILD=1

while [ $# -gt 0 ]; do
  case "$1" in
    --pack) PACK=$2; shift 2 ;;
    --toks) TOKS=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --no-build) RUN_BUILD=0; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ ! -f "$TOKS/include/toks.h" ]; then
  echo "no toks checkout at $TOKS (clone https://github.com/actual-computer/toks and pass --toks)" >&2
  exit 1
fi

# toks' sources go into the build context, and out of it again once the image is built.
CONTEXT_SRC="$REPO/docker/toks-src"
cp -a "$TOKS" "$CONTEXT_SRC"
rm -rf "$CONTEXT_SRC/build"   # so the image compiles toks itself rather than reusing a host build
trap 'rm -rf "$CONTEXT_SRC"' EXIT

if [ "$RUN_BUILD" = 1 ]; then
  podman build -t "$TAG" -f "$REPO/docker/Dockerfile.tokenizer" "$REPO"
fi

if [ -n "$PACK" ]; then
  podman run --rm -v "$PACK:/pack:ro" "$TAG"
fi
