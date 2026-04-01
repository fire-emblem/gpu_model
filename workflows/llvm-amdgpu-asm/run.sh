#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CASE_DIR/../.." && pwd)"
source "$ROOT/examples/common.sh"

OUT_DIR="$CASE_DIR/results"
mkdir -p "$OUT_DIR"

gpu_model_require_cmd llc

"$ROOT/scripts/emit_amdgpu_asm.sh" \
  "$CASE_DIR/min_amdgpu.ll" \
  "$OUT_DIR/min_amdgpu.s" \
  gfx900

{
  echo "llvm_toolchain=$(llc --version | head -n 1)"
  echo "output=$OUT_DIR/min_amdgpu.s"
  sed -n '1,80p' "$OUT_DIR/min_amdgpu.s"
} | tee "$OUT_DIR/stdout.txt"
