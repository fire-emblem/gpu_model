#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/build/exec-checks"
mkdir -p "$OUT_DIR"

JOBS="${JOBS:-8}"

echo "[exec-check] build targets"
cmake --build "$ROOT/build" --target gpu_model_tests code_object_dump_main gpu_model_hip_interposer -j "$JOBS"

echo "[exec-check] raw code object decode usage"
"$ROOT/usages/raw-code-object-decode/run.sh" 2>&1 | tee "$OUT_DIR/raw_code_object_decode.log"

echo "[exec-check] hip command line interposer usage"
"$ROOT/usages/hip-command-line-interposer/run.sh" 2>&1 | tee "$OUT_DIR/hip_command_line_interposer.log"

echo "[exec-check] validate combined outputs"
grep -q "class=s_load_dwordx4" "$OUT_DIR/raw_code_object_decode.log"
grep -q "class=v_lshlrev_b64" "$OUT_DIR/raw_code_object_decode.log"
grep -q "hipLaunchKernel result ok=1" "$OUT_DIR/hip_command_line_interposer.log"
grep -q "vecadd host path ok" "$OUT_DIR/hip_command_line_interposer.log"

cat <<'EOF' | tee "$OUT_DIR/summary.txt"
[exec-check] ok
- raw decode/object usage passed
- hip executable interposer usage passed
EOF
