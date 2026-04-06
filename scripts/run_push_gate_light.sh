#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEBUG_BUILD_DIR="${GPU_MODEL_GATE_DEBUG_BUILD_DIR:-$ROOT/build-asan}"
RELEASE_BUILD_DIR="${GPU_MODEL_GATE_RELEASE_BUILD_DIR:-$ROOT/build-gate-release}"
GATE_LOG_DIR="${GPU_MODEL_GATE_LOG_DIR:-$ROOT/results/push-gate-light}"
JOBS="${JOBS:-8}"
SMOKE_FILTER="${GPU_MODEL_GATE_LIGHT_GTEST_FILTER:-RuntimeNamingTest.*:RuntimeProgramCompatibilityAliasTest.*:HipRuntimeAbiTest.RunsHipHostExecutableThroughLdPreloadHipRuntimeAbi:HipRuntimeTest.LaunchKernelCanReadMaterializedDataPool:ModelRuntimeCoreTest.SimulatesMallocMemcpyLaunchAndSynchronizeFlow:TraceTest.NativePerfettoProtoContainsHierarchicalTracksAndEvents}"

mkdir -p "$GATE_LOG_DIR"

if command -v ninja >/dev/null 2>&1; then
  GENERATOR=(-G Ninja)
else
  GENERATOR=()
fi

configure_build() {
  local build_dir="$1"
  shift
  cmake -S "$ROOT" -B "$build_dir" "${GENERATOR[@]}" "$@"
}

run_release_smoke() {
  echo "[push-gate-light] configure release build"
  configure_build "$RELEASE_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DGPU_MODEL_ENABLE_ASAN=OFF \
    2>&1 | tee "$GATE_LOG_DIR/release.configure.log"
  echo "[push-gate-light] build release smoke targets"
  cmake --build "$RELEASE_BUILD_DIR" --target gpu_model_tests gpu_model_hip_runtime_abi -j "$JOBS" \
    2>&1 | tee "$GATE_LOG_DIR/release.build.log"
  echo "[push-gate-light] run release smoke filter=$SMOKE_FILTER"
  "$RELEASE_BUILD_DIR/tests/gpu_model_tests" \
    --gtest_filter="$SMOKE_FILTER" \
    2>&1 | tee "$GATE_LOG_DIR/release.smoke.log"
}

run_debug_asan_smoke() {
  echo "[push-gate-light] configure debug+asan build"
  configure_build "$DEBUG_BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug -DGPU_MODEL_ENABLE_ASAN=ON \
    2>&1 | tee "$GATE_LOG_DIR/debug_asan.configure.log"
  echo "[push-gate-light] build debug+asan smoke targets"
  cmake --build "$DEBUG_BUILD_DIR" --target gpu_model_tests gpu_model_hip_runtime_abi -j "$JOBS" \
    2>&1 | tee "$GATE_LOG_DIR/debug_asan.build.log"
  echo "[push-gate-light] run debug+asan smoke filter=$SMOKE_FILTER"
  "$DEBUG_BUILD_DIR/tests/gpu_model_tests" \
    --gtest_filter="$SMOKE_FILTER" \
    2>&1 | tee "$GATE_LOG_DIR/debug_asan.smoke.log"
}

run_debug_asan_smoke &
debug_pid=$!

run_release_smoke &
release_pid=$!

wait "$debug_pid"
wait "$release_pid"

cat <<EOF
[push-gate-light] ok
- debug+asan smoke passed
- release smoke passed
- full gate remains available via scripts/run_push_gate.sh
EOF
