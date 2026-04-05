#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${GPU_MODEL_HIPCC_CACHE_DIR:-/tmp/gpu_model_hipcc_cache}"
mkdir -p "$CACHE_DIR"

if ! command -v hipcc >/dev/null 2>&1; then
  echo "missing tool: hipcc" >&2
  exit 1
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

hipcc_version="$(hipcc --version 2>/dev/null || true)"

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

hipcc "${compile_args[@]}"

mv "$tmp_output_path" "$cache_output_path"
mkdir -p "$(dirname "$out_path")"
cp -f "$cache_output_path" "$out_path"
