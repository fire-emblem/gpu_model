#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/usages/hip-fatbin-launch/results"
mkdir -p "$OUT_DIR"

for tool in hipcc clang-offload-bundler llvm-objcopy llvm-objdump readelf; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing tool: $tool" >&2
    exit 1
  fi
done

if [[ ! -x "$ROOT/build/tests/gpu_model_tests" ]]; then
  echo "missing executable: $ROOT/build/tests/gpu_model_tests" >&2
  exit 1
fi

SRC="$OUT_DIR/hip_empty_kernel.cpp"
OBJ="$OUT_DIR/hip_empty_kernel.o"
EXE="$OUT_DIR/hip_empty_kernel.out"
FATBIN="$OUT_DIR/hip_empty_kernel.hip_fatbin"
DEVICE="$OUT_DIR/hip_empty_kernel.gfx.co"

cat > "$SRC" <<'EOF'
#include <hip/hip_runtime.h>

extern "C" __global__ void empty_kernel() {}

int main() {
  return 0;
}
EOF

hipcc -c "$SRC" -o "$OBJ"
hipcc "$SRC" -o "$EXE"
llvm-objcopy --dump-section .hip_fatbin="$FATBIN" "$EXE"
clang-offload-bundler --list --type=o --input="$FATBIN" > "$OUT_DIR/bundles.txt"

TARGET="$(grep 'amdgcn-amd-amdhsa' "$OUT_DIR/bundles.txt" | head -n 1)"
if [[ -z "$TARGET" ]]; then
  echo "missing AMDGPU bundle target in $FATBIN" >&2
  exit 1
fi

clang-offload-bundler --unbundle --type=o --input="$FATBIN" --targets="$TARGET" --output="$DEVICE"
readelf -h "$DEVICE" > "$OUT_DIR/device_readelf_header.txt"
llvm-objdump -d "$DEVICE" > "$OUT_DIR/device_objdump.txt"

"$ROOT/build/tests/gpu_model_tests" \
  --gtest_filter=RuntimeHooksTest.LaunchesHipExecutableWithEmbeddedFatbin | tee "$OUT_DIR/stdout.txt"
