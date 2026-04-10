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
SRC="$CASE_DIR/fma_loop.hip"
EXE="$OUT_DIR/fma_loop.out"

gpu_model_compile_hip_source "$ROOT" "$SRC" -o "$EXE"

for mode in mt; do
  mode_dir="$(gpu_model_mode_dir "$OUT_DIR" "$mode")"
  gpu_model_run_interposed_mode "$SO_PATH" "$EXE" "$mode_dir" "$mode"
  gpu_model_assert_mode_success "$mode_dir" "fma_loop host path ok"
done
