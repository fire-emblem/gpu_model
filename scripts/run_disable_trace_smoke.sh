#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${GPU_MODEL_DISABLE_TRACE_BUILD_DIR:-$ROOT/build-ninja}"
JOBS="${JOBS:-8}"

FILTER="${GPU_MODEL_DISABLE_TRACE_GTEST_FILTER:-ExecutionStatsTest.*:ExecutedFlowProgramCycleStatsTest.*:FmaLoopFunctionalTest.RunsLoopedFmaKernelAndValidatesOutput:SharedBarrierFunctionalTest.ReversesValuesWithinEachBlock:SharedBarrierFunctionalTest.MatchesResultsAcrossSingleThreadedAndMultiThreadedModes}"

echo "[disable-trace-smoke] build gpu_model_tests"
cmake --build "$BUILD_DIR" --target gpu_model_tests -j "$JOBS"

echo "[disable-trace-smoke] run with GPU_MODEL_DISABLE_TRACE=1"
echo "[disable-trace-smoke] gtest_filter=$FILTER"
GPU_MODEL_DISABLE_TRACE=1 "$BUILD_DIR/tests/gpu_model_tests" --gtest_filter="$FILTER"

