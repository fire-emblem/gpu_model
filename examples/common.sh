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

gpu_model_detect_rocm_lib_dir() {
  if [[ -d /opt/rocm/lib ]]; then
    echo "/opt/rocm/lib"
    return
  fi
  if [[ -d /opt/rocm/lib64 ]]; then
    echo "/opt/rocm/lib64"
    return
  fi
}

gpu_model_mode_dir() {
  local base_dir="$1"
  local mode="$2"
  local case_name="${3:-}"
  if [[ -n "$case_name" ]]; then
    echo "$base_dir/$mode/$case_name"
  else
    echo "$base_dir/$mode"
  fi
}

gpu_model_run_interposed_mode() {
  local so_path="$1"
  local exe_path="$2"
  local mode_dir="$3"
  local mode="$4"

  rm -rf "$mode_dir"
  mkdir -p "$mode_dir"

  local exec_mode="functional"
  local functional_mode="st"
  local worker_threads=""
  case "$mode" in
    st)
      functional_mode="st"
      ;;
    mt)
      functional_mode="mt"
      worker_threads="${GPU_MODEL_MT_WORKERS:-4}"
      ;;
    cycle)
      exec_mode="cycle"
      functional_mode="${GPU_MODEL_CYCLE_FUNCTIONAL_MODE:-st}"
      ;;
    *)
      echo "unsupported mode: $mode" >&2
      return 1
      ;;
  esac

  local rocm_lib
  rocm_lib="$(gpu_model_detect_rocm_lib_dir)"

  local -a env_args=(
    "GPU_MODEL_EXECUTION_MODE=$exec_mode"
    "GPU_MODEL_FUNCTIONAL_MODE=$functional_mode"
    "GPU_MODEL_TRACE_DIR=$mode_dir"
    "GPU_MODEL_HIP_INTERPOSER_DEBUG=1"
    "LD_PRELOAD=$so_path"
  )
  if [[ -n "$worker_threads" ]]; then
    env_args+=("GPU_MODEL_FUNCTIONAL_WORKERS=$worker_threads")
  fi
  if [[ -n "$rocm_lib" ]]; then
    env_args+=("LD_LIBRARY_PATH=$rocm_lib:${LD_LIBRARY_PATH:-}")
  fi

  env "${env_args[@]}" "$exe_path" 2>&1 | tee "$mode_dir/stdout.txt"
}

gpu_model_assert_trace_artifacts() {
  local mode_dir="$1"
  [[ -f "$mode_dir/trace.txt" ]]
  [[ -f "$mode_dir/trace.jsonl" ]]
  [[ -f "$mode_dir/timeline.perfetto.json" ]]
  [[ -f "$mode_dir/launch_summary.txt" ]]
  grep -q "kind=Launch" "$mode_dir/trace.txt"
  grep -q '"kind":"Launch"' "$mode_dir/trace.jsonl"
  grep -q '"traceEvents"' "$mode_dir/timeline.perfetto.json"
  grep -q 'execution_mode=' "$mode_dir/launch_summary.txt"
}

gpu_model_assert_mode_success() {
  local mode_dir="$1"
  local pattern="$2"
  grep -q "$pattern" "$mode_dir/stdout.txt"
  gpu_model_assert_trace_artifacts "$mode_dir"
}

gpu_model_summary_field() {
  local summary_path="$1"
  local key="$2"
  awk -v key="$key" '
    {
      for (i = 1; i <= NF; ++i) {
        split($i, kv, "=")
        if (kv[1] == key) {
          value = kv[2]
        }
      }
    }
    END {
      if (value != "") {
        print value
      }
    }
  ' "$summary_path"
}
