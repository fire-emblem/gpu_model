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

gpu_model_detect_results_dir() {
  local root="$1"
  local case_dir="$2"
  echo "$case_dir/results"
}

gpu_model_require_cmd() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing tool: $tool" >&2
    exit 1
  fi
}

gpu_model_compile_hip_source() {
  local root="$1"
  shift
  local -a compile_args=("$@")
  local has_explicit_arch=0
  local default_arch="${GPU_MODEL_HIP_OFFLOAD_ARCH:-gfx90a}"

  for arg in "${compile_args[@]}"; do
    case "$arg" in
      --offload-arch=*|--offload-arch|--amdgpu-target=*|--amdgpu-target)
        has_explicit_arch=1
        break
        ;;
    esac
  done

  if [[ "$has_explicit_arch" -eq 0 ]]; then
    compile_args=(--offload-arch="$default_arch" "${compile_args[@]}")
  fi

  if [[ "${GPU_MODEL_USE_HIPCC_CACHE:-1}" != "0" ]]; then
    "$root/tools/hipcc_cache.sh" "${compile_args[@]}"
  else
    hipcc "${compile_args[@]}"
  fi
}

gpu_model_ensure_targets() {
  local build_dir="$1"
  shift
  local lock_dir="$build_dir/.gpu_model_build.lock"
  mkdir -p "$lock_dir"
  local lock_file="$lock_dir/build.lock"
  exec 9>"$lock_file"
  flock 9
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

gpu_model_ld_preload_value() {
  local so_path="$1"
  local asan_path=""
  asan_path="$(ldd "$so_path" 2>/dev/null | awk '/libasan/ && /=>/ {print $3; exit}')"
  if [[ -n "$asan_path" && -f "$asan_path" ]]; then
    echo "$asan_path:$so_path"
  else
    echo "$so_path"
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
  local preload_value
  preload_value="$(gpu_model_ld_preload_value "$so_path")"

  local -a env_args=(
    "GPU_MODEL_EXECUTION_MODE=$exec_mode"
    "GPU_MODEL_FUNCTIONAL_MODE=$functional_mode"
    "GPU_MODEL_DISABLE_TRACE=0"
    "GPU_MODEL_TRACE_DIR=$mode_dir"
    "GPU_MODEL_LOG_MODULES=hip_ld_preload"
    "GPU_MODEL_LOG_LEVEL=info"
    "LD_PRELOAD=$preload_value"
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
  # Verify structured trace output
  grep -q "GPU_MODEL TRACE" "$mode_dir/trace.txt"
  grep -q "\[RUN\]" "$mode_dir/trace.txt"
  grep -q "\[KERNEL\]" "$mode_dir/trace.txt"
  grep -q "\[EVENTS\]" "$mode_dir/trace.txt"
  grep -q "\[SUMMARY\]" "$mode_dir/trace.txt"
  grep -q '"type":"run_snapshot"' "$mode_dir/trace.jsonl"
  grep -q '"type":"summary_snapshot"' "$mode_dir/trace.jsonl"
  grep -q '"traceEvents"' "$mode_dir/timeline.perfetto.json"
  # Verify launch_index is present (merged from launch_summary.txt)
  grep -q 'launch_index=' "$mode_dir/trace.txt"
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

gpu_model_trace_summary_field() {
  local trace_path="$1"
  local key="$2"
  awk -v key="$key" '
    $0 == "[SUMMARY]" {
      in_summary = 1
      next
    }
    /^\[/ {
      if (in_summary) {
        exit
      }
    }
    in_summary {
      split($0, kv, "=")
      if (kv[1] == key) {
        print kv[2]
        exit
      }
    }
  ' "$trace_path"
}

gpu_model_cycle_metric() {
  local mode_dir="$1"
  local metric="$2"
  local launch_summary="$mode_dir/launch_summary.txt"
  local trace_path="$mode_dir/trace.txt"
  local value=""

  case "$metric" in
    total_cycles)
      if [[ -f "$launch_summary" ]]; then
        value="$(gpu_model_summary_field "$launch_summary" total_cycles)"
      fi
      if [[ -z "$value" && -f "$trace_path" ]]; then
        value="$(gpu_model_trace_summary_field "$trace_path" gpu_tot_sim_cycle)"
      fi
      ;;
    active_cycles)
      if [[ -f "$launch_summary" ]]; then
        value="$(gpu_model_summary_field "$launch_summary" active_cycles)"
      fi
      if [[ -z "$value" ]]; then
        value="$(gpu_model_cycle_metric "$mode_dir" total_cycles)"
      fi
      ;;
    ipc)
      if [[ -f "$launch_summary" ]]; then
        value="$(gpu_model_summary_field "$launch_summary" ipc)"
      fi
      if [[ -z "$value" && -f "$trace_path" ]]; then
        value="$(gpu_model_trace_summary_field "$trace_path" gpu_tot_ipc)"
      fi
      ;;
    *)
      echo "unsupported cycle metric: $metric" >&2
      return 1
      ;;
  esac

  echo "$value"
}
