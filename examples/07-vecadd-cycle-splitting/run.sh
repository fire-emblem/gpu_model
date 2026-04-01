#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CASE_DIR/../.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$CASE_DIR/results"
mkdir -p "$OUT_DIR"

gpu_model_require_cmd hipcc
gpu_model_ensure_targets "$BUILD_DIR" gpu_model_hip_interposer
SO_PATH="$BUILD_DIR/libgpu_model_hip_interposer.so"

for name in vecadd_direct vecadd_grid_stride vecadd_chunk2; do
  hipcc "$CASE_DIR/${name}.hip" -o "$OUT_DIR/${name}.out"
  GPU_MODEL_HIP_INTERPOSER_DEBUG=1 LD_PRELOAD="$SO_PATH" \
    "$OUT_DIR/${name}.out" 2>&1 | tee "$OUT_DIR/${name}.stdout.txt"
  grep -q "${name} validation ok" "$OUT_DIR/${name}.stdout.txt"
done

cat <<'EOF' | tee "$OUT_DIR/stdout.txt"
STATUS: compiled and validated three different vecadd HIP programs
TODO: wire real cycle comparison on compiled .out programs
EOF
