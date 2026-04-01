#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$BUILD_DIR/scaling-regression"
mkdir -p "$OUT_DIR"

gpu_model_ensure_targets "$BUILD_DIR" gpu_model_tests
"$BUILD_DIR/tests/gpu_model_tests" \
  --gtest_filter='RequestedShapes/*:RequestedThreadScales/*' | tee "$OUT_DIR/stdout.txt"
