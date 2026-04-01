#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CASE_DIR/../.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$CASE_DIR/results"
mkdir -p "$OUT_DIR"

gpu_model_require_cmd hipcc
gpu_model_ensure_targets "$BUILD_DIR" gpu_model_hip_interposer

SO_PATH="$BUILD_DIR/libgpu_model_hip_interposer.so"
SRC="$CASE_DIR/vecadd.hip"
EXE="$OUT_DIR/vecadd.out"

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
    "$EXE" | tee "$OUT_DIR/stdout.txt"
else
  GPU_MODEL_HIP_INTERPOSER_DEBUG=1 \
  LD_PRELOAD="$SO_PATH" \
    "$EXE" | tee "$OUT_DIR/stdout.txt"
fi
