#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CASE_DIR/../.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$(gpu_model_detect_results_dir "$ROOT" "$CASE_DIR")"

split_large_trace_txt() {
  local mode_dir="$1"
  local trace_path="$mode_dir/trace.txt"
  local part_dir="$mode_dir/trace_parts"
  local line_limit=1800
  local preview_lines=120

  [[ -f "$trace_path" ]] || return 0

  local line_count
  line_count="$(wc -l < "$trace_path")"
  if (( line_count <= 2000 )); then
    rm -rf "$part_dir"
    return 0
  fi

  rm -rf "$part_dir"
  mkdir -p "$part_dir"
  split -d -a 3 -l "$line_limit" --additional-suffix=.txt \
    "$trace_path" "$part_dir/trace_part_"

  local part_count
  part_count="$(find "$part_dir" -maxdepth 1 -type f -name 'trace_part_*.txt' | wc -l)"
  local preview_path
  preview_path="$(mktemp)"
  sed -n "1,${preview_lines}p" "$trace_path" > "$preview_path"
  {
    echo "# trace.txt preview"
    echo "# full text was split because the raw trace exceeded 2000 lines"
    echo "# original_lines=$line_count split_parts=$part_count part_lines=$line_limit"
    echo "# read full text from trace_parts/trace_part_000.txt ..."
    echo
    cat "$preview_path"
  } > "$trace_path"
  rm -f "$preview_path"
}

gpu_model_ensure_targets "$BUILD_DIR" gpu_model_perfetto_waitcnt_slots_demo
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
"$BUILD_DIR/gpu_model_perfetto_waitcnt_slots_demo" "$OUT_DIR" | tee "$OUT_DIR/summary.txt"

[[ -f "$OUT_DIR/guide.txt" ]]

for mode in st mt cycle; do
  for case_name in timeline_gap same_peu_slots switch_away_heavy; do
    mode_dir="$OUT_DIR/$mode/$case_name"
    [[ -f "$mode_dir/stdout.txt" ]]
    gpu_model_assert_trace_artifacts "$mode_dir"
    grep -q "ok=1" "$mode_dir/stdout.txt"
    split_large_trace_txt "$mode_dir"
  done
done

{
  echo
  echo "Recommended quick-start:"
  echo "1. Open results/guide.txt"
  echo "2. Open results/cycle/timeline_gap/timeline.perfetto.json for obvious blank bubbles"
  echo "3. Open results/cycle/same_peu_slots/timeline.perfetto.json for resident slots + wait/arrive"
  echo "4. Open results/st/same_peu_slots/timeline.perfetto.json for logical_unbounded slots in ST"
  echo "5. Open results/mt/same_peu_slots/timeline.perfetto.json for logical_unbounded slots in MT"
  echo "6. Open results/cycle/switch_away_heavy/timeline.perfetto.json for dense wave_switch_away rotation"
  echo
  echo "Key files:"
  echo "- results/guide.txt"
  echo "- results/cycle/timeline_gap/timeline.perfetto.json"
  echo "- results/cycle/same_peu_slots/timeline.perfetto.json"
  echo "- results/st/same_peu_slots/timeline.perfetto.json"
  echo "- results/mt/same_peu_slots/timeline.perfetto.json"
  echo "- results/cycle/switch_away_heavy/timeline.perfetto.json"
} >> "$OUT_DIR/summary.txt"

grep -q "Recommended quick-start:" "$OUT_DIR/summary.txt"
grep -q "results/cycle/switch_away_heavy/timeline.perfetto.json" "$OUT_DIR/summary.txt"
grep -q "results/st/same_peu_slots/timeline.perfetto.json" "$OUT_DIR/summary.txt"
grep -q "results/mt/same_peu_slots/timeline.perfetto.json" "$OUT_DIR/summary.txt"

echo "STATUS: generated perfetto waitcnt slot demos under $OUT_DIR"
