#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$BUILD_DIR/abi-regression"
mkdir -p "$OUT_DIR"

gpu_model_require_cmd hipcc
gpu_model_ensure_targets "$BUILD_DIR" gpu_model_tests gpu_model_hip_runtime_abi

TEST_BIN="$BUILD_DIR/tests/gpu_model_tests"

FOCUSED_FILTER='AmdgpuCodeObjectDecoderTest.DecodesHipByValueAggregateExecutable:'
FOCUSED_FILTER+='AmdgpuCodeObjectDecoderTest.DecodesHipThreeDimensionalHiddenArgsExecutable:'
FOCUSED_FILTER+='AmdgpuCodeObjectDecoderTest.DecodesHipThreeDimensionalBuiltinIdsExecutable:'
FOCUSED_FILTER+='HipRuntimeAbiTest.LaunchesHipByValueAggregateExecutableThroughRegisteredHostFunction:'
FOCUSED_FILTER+='HipRuntimeAbiTest.LaunchesHipThreeDimensionalHiddenArgsExecutableThroughRegisteredHostFunction:'
FOCUSED_FILTER+='HipRuntimeAbiTest.LaunchesHipThreeDimensionalBuiltinIdsExecutableThroughRegisteredHostFunction:'
FOCUSED_FILTER+='HipRuntimeTest.LaunchesHipMixedArgsAggregateExecutableInRawGcnPath:'
FOCUSED_FILTER+='HipRuntimeTest.LaunchesHipThreeDimensionalHiddenArgsExecutableInRawGcnPath:'
FOCUSED_FILTER+='HipRuntimeTest.LaunchesHipThreeDimensionalBuiltinIdsExecutableInRawGcnPath:'
FOCUSED_FILTER+='HipRuntimeTest.LaunchesLlvmMcAggregateByValueObjectInRawGcnPath:'
FOCUSED_FILTER+='HipRuntimeTest.LaunchesLlvmMcThreeDimensionalHiddenArgsObjectInRawGcnPath:'
FOCUSED_FILTER+='HipRuntimeTest.LaunchesLlvmMcFallbackAbiObjectInRawGcnPath'

echo "[abi] focused gtests"
"$TEST_BIN" --gtest_filter="$FOCUSED_FILTER" | tee "$OUT_DIR/focused.txt"

grep -q "PASSED" "$OUT_DIR/focused.txt"

cat <<'EOF' | tee "$OUT_DIR/summary.txt"
[abi] ok
- aggregate by-value regressions passed
- 3d hidden-args / builtin-id regressions passed
- llvm-mc abi fixture regressions passed
EOF
