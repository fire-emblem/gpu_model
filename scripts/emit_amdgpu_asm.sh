#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <input.ll> <output.s> [gfx_target]" >&2
  exit 1
fi

INPUT_LL="$1"
OUTPUT_S="$2"
MCU="${3:-gfx900}"

if [[ ! -f "$INPUT_LL" ]]; then
  echo "missing input LLVM IR: $INPUT_LL" >&2
  exit 1
fi

if ! command -v llc >/dev/null 2>&1; then
  echo "missing llc in PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_S")"
llc -march=amdgcn -mcpu="$MCU" -filetype=asm "$INPUT_LL" -o "$OUTPUT_S"
