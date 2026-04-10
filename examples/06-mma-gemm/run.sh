#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CASE_DIR/../.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$(gpu_model_detect_results_dir "$ROOT" "$CASE_DIR")"
mkdir -p "$OUT_DIR"

gpu_model_require_cmd hipcc
gpu_model_ensure_targets "$BUILD_DIR" gpu_model_hip_runtime_abi

SO_PATH="$BUILD_DIR/libgpu_model_hip_runtime_abi.so"
SRC="$CASE_DIR/mma_gemm.hip"
EXE="$OUT_DIR/mma_gemm.out"

if ! gpu_model_compile_hip_source "$ROOT" --offload-arch=gfx90a "$SRC" -o "$EXE"; then
  for mode in mt; do
    mode_dir="$(gpu_model_mode_dir "$OUT_DIR" "$mode")"
    mkdir -p "$mode_dir"
    echo "STATUS: unsupported_yet (gfx90a mfma compile unavailable)" | tee "$mode_dir/stdout.txt"
  done
  exit 0
fi

for mode in mt; do
  mode_dir="$(gpu_model_mode_dir "$OUT_DIR" "$mode")"
  gpu_model_run_interposed_mode "$SO_PATH" "$EXE" "$mode_dir" "$mode"
  gpu_model_assert_mode_success "$mode_dir" "mma_gemm out=4.000000 expected=4.000000"
done
