#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$BUILD_DIR/shared-heavy-regression"
mkdir -p "$OUT_DIR"

gpu_model_require_cmd hipcc
gpu_model_ensure_targets "$BUILD_DIR" gpu_model_tests gpu_model_hip_runtime_abi

TEST_BIN="$BUILD_DIR/tests/gpu_model_tests"

FOCUSED_FILTER='AmdgpuCodeObjectDecoderTest.DecodesHipSharedReverseExecutable:'
FOCUSED_FILTER+='AmdgpuCodeObjectDecoderTest.DecodesHipDynamicSharedExecutableWithoutUnknownInstructions:'
FOCUSED_FILTER+='AmdgpuCodeObjectDecoderTest.DecodesHipBlockReduceExecutableWithoutUnknownInstructions:'
FOCUSED_FILTER+='AmdgpuCodeObjectDecoderTest.DecodesHipSoftmaxExecutableWithoutUnknownInstructions:'
FOCUSED_FILTER+='HipRuntimeAbiTest.LaunchesHipSharedReverseExecutableThroughRegisteredHostFunction:'
FOCUSED_FILTER+='HipRuntimeAbiTest.LaunchesHipDynamicSharedExecutableThroughRegisteredHostFunction:'
FOCUSED_FILTER+='HipRuntimeAbiTest.LaunchesHipBlockReduceExecutableThroughRegisteredHostFunction:'
FOCUSED_FILTER+='HipRuntimeAbiTest.LaunchesHipSoftmaxExecutableThroughRegisteredHostFunction:'
FOCUSED_FILTER+='HipRuntimeTest.LaunchesHipDynamicSharedExecutableInRawGcnPath:'
FOCUSED_FILTER+='HipRuntimeTest.LaunchesHipBlockReduceExecutableInRawGcnPath:'
FOCUSED_FILTER+='HipRuntimeTest.LaunchesHipSoftmaxExecutableInRawGcnPath:'
FOCUSED_FILTER+='HipRuntimeTest.LaunchesHipSharedReverseExecutableAndValidatesOutput:'
FOCUSED_FILTER+='HipccParallelExecutionTest.EncodedSharedReverseKernelMatchesBetweenStMtAndCycleAndReportsClosedStats:'
FOCUSED_FILTER+='HipccParallelExecutionTest.EncodedSoftmaxKernelMatchesBetweenStMtAndCycleAndReportsClosedStats'

CTS_FILTER='HipRuntimeTest.HipCtsFullCaseCountIsOneHundred:'
CTS_FILTER+='HipRuntimeTest.CtsCaseCountIsOneHundred:'
CTS_FILTER+='HipRuntimeTest.FeatureCtsCaseCountIsOneHundred:'
CTS_FILTER+='HipRuntimeCTS/*:HipRuntimeAbiCTS/*:HipRuntimeFeatureCTS/*:HipRuntimeAbiFeatureCTS/*'

echo "[shared-heavy] focused gtests"
"$TEST_BIN" --gtest_filter="$FOCUSED_FILTER" | tee "$OUT_DIR/focused.txt"

echo "[shared-heavy] cts gtests"
"$TEST_BIN" --gtest_filter="$CTS_FILTER" | tee "$OUT_DIR/cts.txt"

echo "[shared-heavy] example 03 shared reverse"
"$ROOT/examples/03-shared-reverse/run.sh" 2>&1 | tee "$OUT_DIR/example_03_shared_reverse.txt"

echo "[shared-heavy] example 05 softmax reduction"
"$ROOT/examples/05-softmax-reduction/run.sh" 2>&1 | tee "$OUT_DIR/example_05_softmax_reduction.txt"

echo "[shared-heavy] example 09 dynamic shared sum"
"$ROOT/examples/09-dynamic-shared-sum/run.sh" 2>&1 | tee "$OUT_DIR/example_09_dynamic_shared_sum.txt"

echo "[shared-heavy] example 10 block reduce sum"
"$ROOT/examples/10-block-reduce-sum/run.sh" 2>&1 | tee "$OUT_DIR/example_10_block_reduce_sum.txt"

grep -q "PASSED" "$OUT_DIR/focused.txt"
grep -q "PASSED" "$OUT_DIR/cts.txt"
grep -q "shared_reverse mismatches=0" "$OUT_DIR/example_03_shared_reverse.txt"
grep -q "softmax_reduction mismatches=0" "$OUT_DIR/example_05_softmax_reduction.txt"
grep -q "dynamic_shared_sum mismatches=0" "$OUT_DIR/example_09_dynamic_shared_sum.txt"
grep -q "block_reduce_sum mismatches=0" "$OUT_DIR/example_10_block_reduce_sum.txt"

cat <<'EOF' | tee "$OUT_DIR/summary.txt"
[shared-heavy] ok
- focused decode/hip-runtime-abi/runtime/parallel ring passed
- hip cts + feature cts ring passed
- examples 03/05/09/10 passed
EOF
