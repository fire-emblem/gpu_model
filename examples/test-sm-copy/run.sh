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

gpu_model_compile_hip_source "$ROOT" "$CASE_DIR/sm_copy.hip" -o "$OUT_DIR/sm_copy.out"
gpu_model_run_interposed_mode "$SO_PATH" "$OUT_DIR/sm_copy.out" "$OUT_DIR/cycle/sm_copy" "cycle" || echo "sm_copy failed (expected)"

# Also test 1D version for comparison
gpu_model_compile_hip_source "$ROOT" "$CASE_DIR/sm_copy_1d.hip" -o "$OUT_DIR/sm_copy_1d.out"
gpu_model_run_interposed_mode "$SO_PATH" "$OUT_DIR/sm_copy_1d.out" "$OUT_DIR/cycle/sm_copy_1d" "cycle"
