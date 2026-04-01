#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CASE_DIR/../.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$CASE_DIR/results"
mkdir -p "$OUT_DIR"

gpu_model_require_cmd hipcc
gpu_model_ensure_targets "$BUILD_DIR" gpu_model_tests code_object_dump_main

TEST_OUT="$OUT_DIR/gtest_stdout.txt"
"$BUILD_DIR/tests/gpu_model_tests" \
  --gtest_filter='EncodedGcnInstructionArrayParserTest.*:AmdgpuCodeObjectDecoderTest.DecodesRawInstructionsFromAmdgpuObject:AmdgpuCodeObjectDecoderTest.DecodesRawInstructionsFromHipExecutable' \
  | tee "$TEST_OUT"

SRC="$CASE_DIR/hip_vecadd.cpp"
EXE="$OUT_DIR/hip_vecadd.out"
VALIDATION_OUT="$OUT_DIR/validation.txt"

hipcc "$SRC" -o "$EXE"
"$BUILD_DIR/code_object_dump_main" "$EXE" vecadd | tee "$OUT_DIR/stdout.txt"

{
  echo "check: parser/object pipeline test output present"
  grep -q "EncodedGcnInstructionArrayParserTest.ParsesTextBytesIntoInstructionArrays" "$TEST_OUT"
  echo "ok"

  echo "check: dump output contains op_type field"
  grep -q " op_type=" "$OUT_DIR/stdout.txt"
  echo "ok"

  echo "check: dump output contains instantiated class field"
  grep -q " class=" "$OUT_DIR/stdout.txt"
  echo "ok"

  echo "check: vecadd contains scalar-memory front-end instruction object"
  grep -q "class=s_load_dword" "$OUT_DIR/stdout.txt"
  echo "ok"
} | tee "$VALIDATION_OUT"
