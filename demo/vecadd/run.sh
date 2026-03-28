#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEMO_DIR="$ROOT/demo/vecadd"
OUT_DIR="$DEMO_DIR/results"
mkdir -p "$OUT_DIR"

SO_PATH="$ROOT/build/libgpu_model_hip_interposer.so"
if [[ ! -f "$SO_PATH" ]]; then
  echo "missing interposer library: $SO_PATH" >&2
  echo "build first with: cmake -S . -B build && cmake --build build -j8" >&2
  exit 1
fi

if ! command -v hipcc >/dev/null 2>&1; then
  echo "missing tool: hipcc" >&2
  exit 1
fi

SRC="$DEMO_DIR/vecadd.hip"
EXE="$OUT_DIR/vecadd.out"
LOG="$OUT_DIR/stdout.txt"

hipcc "$SRC" -o "$EXE"

ROCM_LIB=""
if [[ -d /opt/rocm/lib ]]; then
  ROCM_LIB="/opt/rocm/lib"
elif [[ -d /opt/rocm/lib64 ]]; then
  ROCM_LIB="/opt/rocm/lib64"
fi

if [[ -n "$ROCM_LIB" ]]; then
  LD_LIBRARY_PATH="$ROCM_LIB:${LD_LIBRARY_PATH:-}" \
  GPU_MODEL_HIP_INTERPOSER_DEBUG=1 \
  LD_PRELOAD="$SO_PATH" \
    "$EXE" | tee "$LOG"
else
  GPU_MODEL_HIP_INTERPOSER_DEBUG=1 \
  LD_PRELOAD="$SO_PATH" \
    "$EXE" | tee "$LOG"
fi
