#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CASE_DIR/../.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$CASE_DIR/results"
mkdir -p "$OUT_DIR"

gpu_model_ensure_targets "$BUILD_DIR" fma_loop_cycle_trace_main

"$BUILD_DIR/fma_loop_cycle_trace_main" \
  --mode cycle \
  --grid 2 \
  --block 65 \
  --n 8 \
  --iterations 2 \
  --mul0 2 \
  --add0 1 \
  --mul1 3 \
  --add1 2 \
  --latency 9 \
  --timeline-columns 40 \
  --group-by block \
  --out-dir "$OUT_DIR" | tee "$OUT_DIR/stdout.txt"
