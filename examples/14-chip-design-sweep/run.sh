#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CASE_DIR/../.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$(gpu_model_detect_results_dir "$ROOT" "$CASE_DIR")"
mkdir -p "$OUT_DIR"

env GPU_MODEL_EXAMPLE_OUT_DIR="$OUT_DIR" \
  "$BUILD_DIR/gpu_model_chip_design_sweep_demo" | tee "$OUT_DIR/stdout.txt"
