#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CASE_DIR/../.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$CASE_DIR/results"

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
  done
done

{
  echo
  echo "Recommended quick-start:"
  echo "1. Open results/guide.txt"
  echo "2. Open results/cycle/timeline_gap/timeline.perfetto.pb for obvious blank bubbles"
  echo "3. Open results/cycle/same_peu_slots/timeline.perfetto.pb for resident slots + wait/arrive"
  echo "4. Open results/st/same_peu_slots/timeline.perfetto.pb for logical_unbounded slots in ST"
  echo "5. Open results/mt/same_peu_slots/timeline.perfetto.pb for logical_unbounded slots in MT"
  echo "6. Open results/cycle/switch_away_heavy/timeline.perfetto.pb for dense wave_switch_away rotation"
  echo
  echo "Key files:"
  echo "- results/guide.txt"
  echo "- results/cycle/timeline_gap/timeline.perfetto.pb"
  echo "- results/cycle/same_peu_slots/timeline.perfetto.pb"
  echo "- results/st/same_peu_slots/timeline.perfetto.pb"
  echo "- results/mt/same_peu_slots/timeline.perfetto.pb"
  echo "- results/cycle/switch_away_heavy/timeline.perfetto.pb"
} >> "$OUT_DIR/summary.txt"

grep -q "Recommended quick-start:" "$OUT_DIR/summary.txt"
grep -q "results/cycle/switch_away_heavy/timeline.perfetto.pb" "$OUT_DIR/summary.txt"
grep -q "results/st/same_peu_slots/timeline.perfetto.pb" "$OUT_DIR/summary.txt"
grep -q "results/mt/same_peu_slots/timeline.perfetto.pb" "$OUT_DIR/summary.txt"

echo "STATUS: generated perfetto waitcnt slot demos under $OUT_DIR"
