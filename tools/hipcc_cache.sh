#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${GPU_MODEL_HIPCC_CACHE_DIR:-/tmp/gpu_model_hipcc_cache}"
mkdir -p "$CACHE_DIR"

# Resolve hipcc: prefer full ROCm SDK over conda hipcc (conda lacks device libs)
resolve_hipcc() {
  if command -v hipcc >/dev/null 2>&1; then
    printf '%s' "hipcc"
    return 0
  fi
  # Check full ROCm SDK installation first (has device libs)
  local rocm_candidates=(
    "${HOME}/tools/rocm/rocm/opt/rocm-6.2.0"
    "/opt/rocm"
  )
  for rocm in "${rocm_candidates[@]}"; do
    if [[ -x "${rocm}/bin/hipcc" ]]; then
      printf '%s' "${rocm}/bin/hipcc"
      return 0
    fi
  done
  # Check project-local conda hipcc as fallback
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -x "${script_dir}/hipcc/bin/hipcc" ]]; then
    printf '%s' "${script_dir}/hipcc/bin/hipcc"
    return 0
  fi
  return 1
}

HIPCC_BIN="$(resolve_hipcc)" || {
  echo "missing tool: hipcc (not in PATH, not in tools/hipcc, no ROCm SDK found)" >&2
  exit 1
}

# Ensure HIP_DEVICE_LIB_PATH is set if not already, so hipcc can find ROCm device libs
if [[ -z "${HIP_DEVICE_LIB_PATH:-}" ]]; then
  # Try ROCm 6.2 SDK first, then conda hipcc bundled libs
  for candidate in \
    "${HOME}/tools/rocm/rocm/opt/rocm-6.2.0/amdgcn/bitcode" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hipcc/lib/amdgcn/bitcode" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hipcc/rocm/lib/amdgcn/bitcode" \
    "/opt/rocm/amdgcn/bitcode"; do
    if [[ -d "${candidate}" && -f "${candidate}/ockl.bc" ]]; then
      export HIP_DEVICE_LIB_PATH="${candidate}"
      break
    fi
  done
fi

# Set ROCM_PATH if using ROCm SDK hipcc, so it can find its own device libs
if [[ -z "${ROCM_PATH:-}" ]]; then
  # Infer from resolved hipcc binary path
  case "${HIPCC_BIN}" in
    */rocm-*/bin/hipcc)
      ROCM_PATH="$(cd "$(dirname "${HIPCC_BIN}")/.." && pwd)"
      export ROCM_PATH
      ;;
  esac
fi

# Set HIP_CLANG_PATH if using ROCm SDK hipcc, so it can find clang++
if [[ -z "${HIP_CLANG_PATH:-}" ]]; then
  case "${HIPCC_BIN}" in
    */rocm-*/bin/hipcc)
      clang_dir="$(cd "$(dirname "${HIPCC_BIN}")/../lib/llvm/bin" && pwd)"
      if [[ -x "${clang_dir}/clang++" ]]; then
        export HIP_CLANG_PATH="${clang_dir}"
      fi
      ;;
  esac
fi

resolve_tool_dir() {
  local candidate
  for candidate in \
    "${GPU_MODEL_TOOLCHAIN_DIR:-}" \
    "${ROCM_PATH:-}/llvm/bin" \
    "${ROCM_PATH:-}/lib/llvm/bin" \
    "${ROCM_HOME:-}/llvm/bin" \
    "${ROCM_HOME:-}/lib/llvm/bin" \
    "${HIP_PATH:-}/llvm/bin" \
    "${HIP_PATH:-}/lib/llvm/bin" \
    "$(cd "$(dirname "${HIPCC_BIN}")/.." && pwd)/llvm/bin" \
    "$(cd "$(dirname "${HIPCC_BIN}")/.." && pwd)/lib/llvm/bin" \
    "/opt/rocm/llvm/bin" \
    "/opt/rocm/lib/llvm/bin" \
    "/opt/rocm-6.0.2/llvm/bin" \
    "/opt/rocm-6.0.2/lib/llvm/bin" \
    "/opt/rocm-6.2.0/llvm/bin" \
    "/opt/rocm-6.2.0/lib/llvm/bin"; do
    if [[ -n "$candidate" && -x "$candidate/clang-offload-bundler" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! command -v clang-offload-bundler >/dev/null 2>&1; then
  if tool_dir="$(resolve_tool_dir)"; then
    export PATH="$tool_dir:$PATH"
  else
    echo "missing tool: clang-offload-bundler" >&2
    exit 1
  fi
fi

if ! command -v sha256sum >/dev/null 2>&1; then
  echo "missing tool: sha256sum" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "usage: hipcc_cache.sh <hipcc args...>" >&2
  exit 1
fi

out_path=""
declare -a hipcc_args=("$@")
declare -a input_files=()

expect_out=0
for arg in "${hipcc_args[@]}"; do
  if [[ $expect_out -eq 1 ]]; then
    out_path="$arg"
    expect_out=0
    continue
  fi
  case "$arg" in
    -o)
      expect_out=1
      ;;
    -*)
      ;;
    *)
      if [[ -f "$arg" ]]; then
        input_files+=("$(realpath "$arg")")
      fi
      ;;
  esac
done

if [[ -z "$out_path" ]]; then
  echo "hipcc_cache.sh currently requires an explicit -o <output>" >&2
  exit 1
fi

if [[ ${#input_files[@]} -eq 0 ]]; then
  echo "hipcc_cache.sh could not identify any input source files" >&2
  exit 1
fi

hipcc_version="$("${HIPCC_BIN}" --version 2>/dev/null || true)"

key_material_file="$(mktemp)"
trap 'rm -f "$key_material_file"' EXIT
{
  printf 'hipcc_version\n%s\n' "$hipcc_version"
  printf 'args\0'
  printf '%s\0' "${hipcc_args[@]}"
  printf 'inputs\n'
  for input in "${input_files[@]}"; do
    printf '%s\n' "$input"
    sha256sum "$input"
  done
} >"$key_material_file"

cache_key="$(sha256sum "$key_material_file" | awk '{print $1}')"
cache_entry_dir="$CACHE_DIR/$cache_key"
cache_output_name="$(basename "$out_path")"
cache_output_path="$cache_entry_dir/$cache_output_name"
lock_path="$cache_entry_dir.lock"

mkdir -p "$CACHE_DIR"
exec 9>"$lock_path"
flock 9

mkdir -p "$cache_entry_dir"

if [[ -f "$cache_output_path" ]]; then
  mkdir -p "$(dirname "$out_path")"
  cp -f "$cache_output_path" "$out_path"
  exit 0
fi

tmp_output_path="$(mktemp "$cache_entry_dir/${cache_output_name}.tmp.XXXXXX")"
rm -f "$tmp_output_path"

declare -a compile_args=()
expect_out=0
for arg in "${hipcc_args[@]}"; do
  if [[ $expect_out -eq 1 ]]; then
    compile_args+=("$tmp_output_path")
    expect_out=0
    continue
  fi
  compile_args+=("$arg")
  if [[ "$arg" == "-o" ]]; then
    expect_out=1
  fi
done

"${HIPCC_BIN}" "${compile_args[@]}"

mv "$tmp_output_path" "$cache_output_path"
mkdir -p "$(dirname "$out_path")"
cp -f "$cache_output_path" "$out_path"
