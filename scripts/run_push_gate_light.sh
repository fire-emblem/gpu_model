#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEBUG_BUILD_DIR="${GPU_MODEL_GATE_DEBUG_BUILD_DIR:-$ROOT/build-asan}"
RELEASE_BUILD_DIR="${GPU_MODEL_GATE_RELEASE_BUILD_DIR:-$ROOT/build-gate-release}"
GATE_LOG_DIR="${GPU_MODEL_GATE_LOG_DIR:-$ROOT/results/push-gate-light}"
JOBS="${JOBS:-8}"
SMOKE_FILTER="${GPU_MODEL_GATE_LIGHT_GTEST_FILTER:-RuntimeNamingTest.*:RuntimeProgramCompatibilityAliasTest.*:HipLdPreloadTest.RunsHipHostExecutableThroughLdPreloadHipLdPreload:HipRuntimeTest.LaunchKernelCanReadMaterializedDataPool:ModelRuntimeCoreTest.SimulatesMallocMemcpyLaunchAndSynchronizeFlow:TraceTest.NativePerfettoProtoContainsHierarchicalTracksAndEvents}"

mkdir -p "$GATE_LOG_DIR"

if command -v ninja >/dev/null 2>&1; then
  GENERATOR=(-G Ninja)
else
  GENERATOR=()
fi

CMAKE_COMPILER_ARGS=()
if [[ -z "${CXX:-}" ]]; then
  for candidate in \
    "$ROOT/tools/gcc/bin/g++" \
    "$HOME/bin/g++" \
    /usr/bin/g++-13 \
    /usr/bin/g++-12 \
    /usr/bin/g++-11 \
    /usr/bin/clang++-17; do
    if [[ -x "$candidate" ]]; then
      CMAKE_COMPILER_ARGS+=("-DCMAKE_CXX_COMPILER=$candidate")
      break
    fi
  done
fi
if [[ -z "${CC:-}" ]]; then
  for candidate in \
    "$ROOT/tools/gcc/bin/gcc" \
    "$HOME/bin/gcc" \
    /usr/bin/gcc-13 \
    /usr/bin/gcc-12 \
    /usr/bin/gcc-11 \
    /usr/bin/clang-17; do
    if [[ -x "$candidate" ]]; then
      CMAKE_COMPILER_ARGS+=("-DCMAKE_C_COMPILER=$candidate")
      break
    fi
  done
fi

configure_build() {
  local build_dir="$1"
  shift
  cmake -S "$ROOT" -B "$build_dir" "${GENERATOR[@]}" "${CMAKE_COMPILER_ARGS[@]}" "$@"
}

# Build targets, skipping gpu_model_hip_ld_preload if ROCm headers are absent.
build_targets() {
  local build_dir="$1"
  shift
  local targets=(gpu_model_tests)
  if cmake --build "$build_dir" --target help 2>/dev/null | grep -q "gpu_model_hip_ld_preload"; then
    targets+=(gpu_model_hip_ld_preload)
  fi
  cmake --build "$build_dir" --target "${targets[@]}" -j "$JOBS" "$@"
}

run_release_smoke() {
  echo "[push-gate-light] configure release build"
  configure_build "$RELEASE_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DGPU_MODEL_ENABLE_ASAN=OFF \
    2>&1 | tee "$GATE_LOG_DIR/release.configure.log"
  echo "[push-gate-light] build release smoke targets"
  build_targets "$RELEASE_BUILD_DIR" \
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
  build_targets "$DEBUG_BUILD_DIR" \
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
