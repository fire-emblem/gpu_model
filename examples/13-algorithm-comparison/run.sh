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
echo "Algorithm Comparison Example"
echo "=========================================="
echo ""
echo "Comparing different algorithms for matrix transpose (256x256):"
echo "  - transpose_naive:    Direct transpose (strided writes)"
echo "  - transpose_shared:   Shared memory tile buffer (coalesced R/W)"
echo "  - transpose_diagonal: Shared memory + diagonal reordering"
echo ""

for name in transpose_naive transpose_shared transpose_diagonal; do
  hipcc "$CASE_DIR/${name}.hip" -o "$OUT_DIR/${name}.out"
  for mode in st mt cycle; do
    mode_dir="$(gpu_model_mode_dir "$OUT_DIR" "$mode" "$name")"
    gpu_model_run_interposed_mode "$SO_PATH" "$OUT_DIR/${name}.out" "$mode_dir" "$mode"
    gpu_model_assert_mode_success "$mode_dir" "${name} validation ok"
  done
  cycle_totals["$name"]="$(gpu_model_summary_field "$OUT_DIR/cycle/$name/launch_summary.txt" total_cycles)"
  active_cycles["$name"]="$(gpu_model_summary_field "$OUT_DIR/cycle/$name/launch_summary.txt" active_cycles)"
  ipc["$name"]="$(gpu_model_summary_field "$OUT_DIR/cycle/$name/launch_summary.txt" ipc)"
done

# Validate all cycle values are positive integers
for name in transpose_naive transpose_shared transpose_diagonal; do
  value="${cycle_totals[$name]}"
  [[ "$value" =~ ^[0-9]+$ ]]
  (( value > 0 ))
done

# Generate comparison report
cat >"$OUT_DIR/cycle_comparison.txt" <<EOF
# Algorithm Comparison for Matrix Transpose (256x256)
#
# Algorithm Analysis:
#   transpose_naive:    Strided writes cause bank conflicts, poor memory throughput
#   transpose_shared:   Uses shared memory as buffer, coalesced reads and writes
#   transpose_diagonal: Shared memory + diagonal reordering to avoid partition camping
#
# Key insight: Algorithm choice dramatically impacts memory throughput
#
transpose_naive    total_cycles=${cycle_totals[transpose_naive]} active_cycles=${active_cycles[transpose_naive]} ipc=${ipc[transpose_naive]}
transpose_shared   total_cycles=${cycle_totals[transpose_shared]} active_cycles=${active_cycles[transpose_shared]} ipc=${ipc[transpose_shared]}
transpose_diagonal total_cycles=${cycle_totals[transpose_diagonal]} active_cycles=${active_cycles[transpose_diagonal]} ipc=${ipc[transpose_diagonal]}
EOF

# Output results
cat <<EOF | tee "$OUT_DIR/stdout.txt"
================================================================================
ALGORITHM COMPARISON: Matrix Transpose (256x256 elements)
================================================================================

Algorithm             | Memory Pattern      | Total Cycles | Active Cycles | IPC
----------------------|---------------------|--------------|---------------|-----
transpose_naive       | Strided writes      | ${cycle_totals[transpose_naive]}
transpose_shared      | Coalesced R/W       | ${cycle_totals[transpose_shared]}
transpose_diagonal    | Coalesced + diagonal| ${cycle_totals[transpose_diagonal]}

================================================================================
ANALYSIS: Algorithm choice is critical for GPU performance
================================================================================

The cycle model demonstrates that:

1. NAIVE TRANSPOSE (worst):
   - Good: coalesced reads from input
   - Bad: strided writes to output (non-coalesced)
   - Result: Poor memory bandwidth utilization

2. SHARED MEMORY TRANSPOSE (better):
   - Read tile into shared memory (coalesced)
   - Write transposed tile from shared memory (coalesced)
   - Result: Much better memory throughput

3. DIAGONAL TRANSPOSE (optimal):
   - Shared memory + diagonal reordering
   - Avoids partition camping on some architectures
   - Result: Best performance on multi-memory-controller GPUs

Performance improvement from naive to optimized demonstrates
the importance of memory access patterns in GPU computing.
EOF
