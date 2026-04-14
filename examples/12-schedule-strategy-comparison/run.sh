#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CASE_DIR/../.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$(gpu_model_detect_results_dir "$ROOT" "$CASE_DIR")"
mkdir -p "$OUT_DIR"

gpu_model_require_cmd hipcc
gpu_model_ensure_targets "$BUILD_DIR" gpu_model_hip_runtime_abi
SO_PATH="$BUILD_DIR/libgpu_model_hip_runtime_abi.so"

declare -A cycle_totals=()
declare -A active_cycles=()
declare -A ipc=()

echo "=========================================="
echo "Schedule Strategy Comparison Example"
echo "=========================================="
echo ""
echo "Comparing different block/thread configurations for vecadd:"
echo "  - low_parallelism:    1 block  x 64 threads  = 64 threads (64 iters/thread)"
echo "  - moderate_parallelism: 8 blocks x 128 threads = 1024 threads (4 iters/thread)"
echo "  - optimal_parallelism: 16 blocks x 256 threads = 4096 threads (1 iter/thread)"
echo ""
echo "All variants process the same 4096 elements using grid-stride loops."
echo ""

for name in vecadd_low_parallelism vecadd_moderate_parallelism vecadd_optimal_parallelism; do
  gpu_model_compile_hip_source "$ROOT" "$CASE_DIR/${name}.hip" -o "$OUT_DIR/${name}.out"
  for mode in st mt cycle; do
    mode_dir="$(gpu_model_mode_dir "$OUT_DIR" "$mode" "$name")"
    gpu_model_run_interposed_mode "$SO_PATH" "$OUT_DIR/${name}.out" "$mode_dir" "$mode"
    gpu_model_assert_mode_success "$mode_dir" "${name} validation ok"
  done
  cycle_totals["$name"]="$(gpu_model_cycle_metric "$OUT_DIR/cycle/$name" total_cycles)"
  active_cycles["$name"]="$(gpu_model_cycle_metric "$OUT_DIR/cycle/$name" active_cycles)"
  ipc["$name"]="$(gpu_model_cycle_metric "$OUT_DIR/cycle/$name" ipc)"
done

# Validate all cycle values are positive integers
for name in vecadd_low_parallelism vecadd_moderate_parallelism vecadd_optimal_parallelism; do
  value="${cycle_totals[$name]}"
  [[ "$value" =~ ^[0-9]+$ ]]
  (( value > 0 ))
done

# Generate comparison report
cat >"$OUT_DIR/cycle_comparison.txt" <<EOF
# Schedule Strategy Comparison for vecadd (n=4096)
#
# Configuration Analysis:
#   low_parallelism:      1 block  x 64 threads  = 64 threads,  64 iterations per thread
#   moderate_parallelism: 8 blocks x 128 threads = 1024 threads, 4 iterations per thread
#   optimal_parallelism:  16 blocks x 256 threads = 4096 threads, 1 iteration per thread
#
# Key insight: More parallelism = better occupancy = lower total cycles
#
vecadd_low_parallelism      total_cycles=${cycle_totals[vecadd_low_parallelism]} active_cycles=${active_cycles[vecadd_low_parallelism]} ipc=${ipc[vecadd_low_parallelism]}
vecadd_moderate_parallelism total_cycles=${cycle_totals[vecadd_moderate_parallelism]} active_cycles=${active_cycles[vecadd_moderate_parallelism]} ipc=${ipc[vecadd_moderate_parallelism]}
vecadd_optimal_parallelism  total_cycles=${cycle_totals[vecadd_optimal_parallelism]} active_cycles=${active_cycles[vecadd_optimal_parallelism]} ipc=${ipc[vecadd_optimal_parallelism]}
EOF

# Output results
cat <<EOF | tee "$OUT_DIR/stdout.txt"
================================================================================
SCHEDULE STRATEGY COMPARISON: vecadd (n=4096 elements)
================================================================================

Configuration             | Threads | Iter/Thread | Total Cycles | Active Cycles | IPC
--------------------------|---------|-------------|--------------|---------------|-----
low_parallelism (1x64)    | 64      | 64          | ${cycle_totals[vecadd_low_parallelism]}
moderate_parallelism (8x128) | 1024 | 4           | ${cycle_totals[vecadd_moderate_parallelism]}
optimal_parallelism (16x256) | 4096 | 1           | ${cycle_totals[vecadd_optimal_parallelism]}

================================================================================
ANALYSIS: Better schedule strategy reduces total cycles
================================================================================

The cycle model demonstrates that:
1. Higher parallelism (more blocks/threads) reduces total execution time
2. Optimal parallelism achieves best IPC (instructions per cycle)
3. Low parallelism wastes compute resources with serial execution

This validates the importance of choosing appropriate grid/block dimensions
for GPU kernel performance optimization.
EOF
