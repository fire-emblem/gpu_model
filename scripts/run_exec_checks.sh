#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/build/exec-checks"
mkdir -p "$OUT_DIR"

source "$ROOT/examples/common.sh"
BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
JOBS="${JOBS:-8}"

echo "[exec-check] build targets"
cmake --build "$BUILD_DIR" --target gpu_model_tests gpu_model_hip_interposer -j "$JOBS"

echo "[exec-check] hip command line interposer usage"
"$ROOT/examples/01-hip-command-line-interposer/run.sh" 2>&1 | tee "$OUT_DIR/hip_command_line_interposer.log"

echo "[exec-check] validate combined outputs"
grep -q "vecadd validation ok" "$OUT_DIR/hip_command_line_interposer.log"

cat <<'EOF' | tee "$OUT_DIR/summary.txt"
[exec-check] ok
- hip executable interposer usage passed
EOF
