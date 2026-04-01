#!/usr/bin/env bash

gpu_model_detect_build_dir() {
  local root="$1"
  if [[ -n "${GPU_MODEL_BUILD_DIR:-}" ]]; then
    echo "$GPU_MODEL_BUILD_DIR"
    return
  fi
  if [[ -d "$root/build-ninja" ]]; then
    echo "$root/build-ninja"
    return
  fi
  echo "$root/build"
}

gpu_model_require_cmd() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing tool: $tool" >&2
    exit 1
  fi
}

gpu_model_ensure_targets() {
  local build_dir="$1"
  shift
  cmake --build "$build_dir" --target "$@" -j "${JOBS:-8}"
}
