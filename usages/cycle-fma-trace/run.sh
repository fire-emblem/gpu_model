#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/usages/cycle-fma-trace/results"
mkdir -p "$OUT_DIR"

if [[ ! -x "$ROOT/build/fma_loop_cycle_trace_main" ]]; then
  echo "missing executable: $ROOT/build/fma_loop_cycle_trace_main" >&2
  exit 1
fi

"$ROOT/build/fma_loop_cycle_trace_main" \
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
