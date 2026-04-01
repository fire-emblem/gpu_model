#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CASE_DIR/../.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$CASE_DIR/results"
mkdir -p "$OUT_DIR"

for tool in hipcc clang-offload-bundler llvm-objcopy llvm-objdump readelf; do
  gpu_model_require_cmd "$tool"
done

gpu_model_ensure_targets "$BUILD_DIR" gpu_model_tests

SRC="$CASE_DIR/empty_kernel.cpp"
OBJ="$OUT_DIR/hip_empty_kernel.o"
EXE="$OUT_DIR/hip_empty_kernel.out"
FATBIN="$OUT_DIR/hip_empty_kernel.hip_fatbin"
DEVICE="$OUT_DIR/hip_empty_kernel.gfx.co"

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

"$BUILD_DIR/tests/gpu_model_tests" \
  --gtest_filter=RuntimeHooksTest.LaunchesHipExecutableWithEmbeddedFatbin | tee "$OUT_DIR/stdout.txt"
