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
SRC="$CASE_DIR/mma_gemm.hip"
EXE="$OUT_DIR/mma_gemm.out"

if ! hipcc --offload-arch=gfx90a "$SRC" -o "$EXE"; then
  echo "STATUS: unsupported_yet (gfx90a mfma compile unavailable)" | tee "$OUT_DIR/stdout.txt"
  exit 0
fi

GPU_MODEL_HIP_INTERPOSER_DEBUG=1 LD_PRELOAD="$SO_PATH" \
  "$EXE" 2>&1 | tee "$OUT_DIR/stdout.txt"

grep -q "mma_gemm out=4.000000 expected=4.000000" "$OUT_DIR/stdout.txt"
