#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/usages/scaling-regression/results"
mkdir -p "$OUT_DIR"

if [[ ! -x "$ROOT/build/tests/gpu_model_tests" ]]; then
  echo "missing test binary: $ROOT/build/tests/gpu_model_tests" >&2
  exit 1
fi

"$ROOT/build/tests/gpu_model_tests" \
  --gtest_filter='RequestedShapes/*:RequestedThreadScales/*' | tee "$OUT_DIR/stdout.txt"
