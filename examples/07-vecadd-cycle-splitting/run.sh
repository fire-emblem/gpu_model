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

declare -A cycle_totals=()

for name in vecadd_direct vecadd_grid_stride vecadd_chunk2; do
  hipcc "$CASE_DIR/${name}.hip" -o "$OUT_DIR/${name}.out"
  for mode in st mt cycle; do
    mode_dir="$(gpu_model_mode_dir "$OUT_DIR" "$mode" "$name")"
    gpu_model_run_interposed_mode "$SO_PATH" "$OUT_DIR/${name}.out" "$mode_dir" "$mode"
    gpu_model_assert_mode_success "$mode_dir" "${name} validation ok"
  done
  cycle_totals["$name"]="$(gpu_model_summary_field "$OUT_DIR/cycle/$name/launch_summary.txt" total_cycles)"
done

for name in vecadd_direct vecadd_grid_stride vecadd_chunk2; do
  value="${cycle_totals[$name]}"
  [[ "$value" =~ ^[0-9]+$ ]]
  (( value > 0 ))
done

if [[ "${cycle_totals[vecadd_direct]}" == "${cycle_totals[vecadd_grid_stride]}" &&
      "${cycle_totals[vecadd_grid_stride]}" == "${cycle_totals[vecadd_chunk2]}" ]]; then
  echo "cycle totals unexpectedly identical across all vecadd variants" >&2
  exit 1
fi

cat >"$OUT_DIR/cycle_comparison.txt" <<EOF
vecadd_direct total_cycles=${cycle_totals[vecadd_direct]}
vecadd_grid_stride total_cycles=${cycle_totals[vecadd_grid_stride]}
vecadd_chunk2 total_cycles=${cycle_totals[vecadd_chunk2]}
EOF

cat <<EOF | tee "$OUT_DIR/stdout.txt"
STATUS: compiled and validated three different vecadd HIP programs
vecadd_direct cycle_total=${cycle_totals[vecadd_direct]}
vecadd_grid_stride cycle_total=${cycle_totals[vecadd_grid_stride]}
vecadd_chunk2 cycle_total=${cycle_totals[vecadd_chunk2]}
EOF
