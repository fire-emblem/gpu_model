#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$BUILD_DIR/real-hip-kernel-regression"
mkdir -p "$OUT_DIR"

gpu_model_require_cmd hipcc
gpu_model_ensure_targets "$BUILD_DIR" gpu_model_tests gpu_model_hip_runtime_abi

TEST_BIN="$BUILD_DIR/tests/gpu_model_tests"

ATOMIC_FILTER='AmdgpuCodeObjectDecoderTest.DecodesRawInstructionsFromHipAtomicCountExecutable:'
ATOMIC_FILTER+='HipRuntimeAbiTest.LaunchesHipAtomicCountExecutableThroughRegisteredHostFunction:'
ATOMIC_FILTER+='HipRuntimeTest.LaunchesHipAtomicCountExecutableInRawGcnPath:'
ATOMIC_FILTER+='HipccParallelExecutionTest.EncodedAtomicReductionMatchesBetweenStMtAndCycleAndReportsClosedStats'

echo "[real-hip] shared-heavy ring"
"$ROOT/scripts/run_shared_heavy_regression.sh" 2>&1 | tee "$OUT_DIR/shared_heavy.txt"

echo "[real-hip] atomic ring"
"$TEST_BIN" --gtest_filter="$ATOMIC_FILTER" | tee "$OUT_DIR/atomic.txt"

echo "[real-hip] example 04 atomic reduction"
"$ROOT/examples/04-atomic-reduction/run.sh" 2>&1 | tee "$OUT_DIR/example_04_atomic_reduction.txt"

grep -q "\\[shared-heavy\\] ok" "$OUT_DIR/shared_heavy.txt"
grep -q "PASSED" "$OUT_DIR/atomic.txt"
grep -q "atomic_reduction value=257 expected=257" "$OUT_DIR/example_04_atomic_reduction.txt"

cat <<'EOF' | tee "$OUT_DIR/summary.txt"
[real-hip] ok
- shared-heavy ring passed
- atomic focused ring passed
- example 04 atomic reduction passed
EOF
