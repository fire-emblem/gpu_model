#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEBUG_BUILD_DIR="${GPU_MODEL_GATE_DEBUG_BUILD_DIR:-$ROOT/build-asan}"
RELEASE_BUILD_DIR="${GPU_MODEL_GATE_RELEASE_BUILD_DIR:-$ROOT/build-gate-release}"
EXAMPLES_BUILD_DIR="${GPU_MODEL_GATE_EXAMPLES_BUILD_DIR:-$ROOT/build-gate-examples}"
JOBS="${JOBS:-8}"

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

run_release_tests() {
  echo "[push-gate] configure release build"
  configure_build "$RELEASE_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DGPU_MODEL_ENABLE_ASAN=OFF
  echo "[push-gate] build release targets"
  cmake --build "$RELEASE_BUILD_DIR" --target gpu_model_tests gpu_model_hip_interposer gpu_model_perfetto_waitcnt_slots_demo -j "$JOBS"
  echo "[push-gate] run release gpu_model_tests"
  "$RELEASE_BUILD_DIR/tests/gpu_model_tests"
}

run_debug_asan_tests() {
  echo "[push-gate] configure debug+asan build"
  configure_build "$DEBUG_BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug -DGPU_MODEL_ENABLE_ASAN=ON
  echo "[push-gate] build debug+asan targets"
  cmake --build "$DEBUG_BUILD_DIR" --target gpu_model_tests gpu_model_hip_interposer gpu_model_perfetto_waitcnt_slots_demo -j "$JOBS"
  echo "[push-gate] run debug+asan gpu_model_tests"
  "$DEBUG_BUILD_DIR/tests/gpu_model_tests"
}

run_all_examples() {
  echo "[push-gate] configure examples release build"
  configure_build "$EXAMPLES_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DGPU_MODEL_ENABLE_ASAN=OFF
  echo "[push-gate] build examples release targets"
  cmake --build "$EXAMPLES_BUILD_DIR" --target gpu_model_tests gpu_model_hip_interposer gpu_model_perfetto_waitcnt_slots_demo -j "$JOBS"
  echo "[push-gate] run all examples on examples release build"
  local examples=(
    "01-vecadd-basic"
    "02-fma-loop"
    "03-shared-reverse"
    "04-atomic-reduction"
    "05-softmax-reduction"
    "06-mma-gemm"
    "07-vecadd-cycle-splitting"
    "08-conditional-multibarrier"
    "09-dynamic-shared-sum"
    "10-block-reduce-sum"
    "11-perfetto-waitcnt-slots"
  )
  for name in "${examples[@]}"; do
    echo "[push-gate] example $name"
    GPU_MODEL_BUILD_DIR="$EXAMPLES_BUILD_DIR" GPU_MODEL_USE_HIPCC_CACHE=0 \
      "$ROOT/examples/$name/run.sh"
  done
}

run_debug_asan_tests &
debug_pid=$!

run_release_tests &
release_pid=$!

run_all_examples &
examples_pid=$!

wait "$debug_pid"
wait "$release_pid"
wait "$examples_pid"

cat <<EOF
[push-gate] ok
- debug+asan tests passed
- release tests passed
- all examples passed
EOF
