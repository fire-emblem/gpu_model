#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/usages/raw-code-object-decode/results"
mkdir -p "$OUT_DIR"

if ! command -v hipcc >/dev/null 2>&1; then
  echo "missing tool: hipcc" >&2
  exit 1
fi

cmake --build "$ROOT/build" --target gpu_model_tests code_object_dump_main -j 8

TEST_OUT="$OUT_DIR/gtest_stdout.txt"
"$ROOT/build/tests/gpu_model_tests" \
  --gtest_filter='EncodedGcnInstructionArrayParserTest.*:AmdgpuCodeObjectDecoderTest.DecodesRawInstructionsFromAmdgpuObject:AmdgpuCodeObjectDecoderTest.DecodesRawInstructionsFromHipExecutable' \
  | tee "$TEST_OUT"

SRC="$OUT_DIR/hip_vecadd.cpp"
EXE="$OUT_DIR/hip_vecadd.out"
VALIDATION_OUT="$OUT_DIR/validation.txt"

cat > "$SRC" <<'EOF'
#include <hip/hip_runtime.h>

extern "C" __global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  return 0;
}
EOF

hipcc "$SRC" -o "$EXE"
"$ROOT/build/code_object_dump_main" "$EXE" vecadd | tee "$OUT_DIR/stdout.txt"

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
