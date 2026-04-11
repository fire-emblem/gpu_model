#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${GPU_MODEL_QUALITY_RESULTS_DIR:-$ROOT/results/quality}"
CACHE_DIR="${GPU_MODEL_QUALITY_CACHE_DIR:-$ROOT/.cache/quality-tools}"
LIZARD_VENV="${GPU_MODEL_QUALITY_LIZARD_VENV:-$CACHE_DIR/lizard-venv}"
LIZARD_BIN="$LIZARD_VENV/bin/lizard"
QUALITY_BUILD_DIR="${GPU_MODEL_QUALITY_BUILD_DIR:-$ROOT/build-quality}"
JSCPD_VERSION="${GPU_MODEL_JSCPD_VERSION:-4.0.9}"
JSCPD_MIN_LINES="${GPU_MODEL_JSCPD_MIN_LINES:-10}"
JSCPD_MIN_TOKENS="${GPU_MODEL_JSCPD_MIN_TOKENS:-80}"
JSCPD_THRESHOLD="${GPU_MODEL_JSCPD_THRESHOLD:-100}"
LIZARD_CCN_LIMIT="${GPU_MODEL_LIZARD_CCN_LIMIT:-15}"
LIZARD_LENGTH_LIMIT="${GPU_MODEL_LIZARD_LENGTH_LIMIT:-200}"
LIZARD_PARAM_LIMIT="${GPU_MODEL_LIZARD_PARAM_LIMIT:-8}"
QUALITY_STRICT="${GPU_MODEL_QUALITY_STRICT:-0}"
JSCPD_IGNORE="**/third_party/**,**/build*/**,**/.cache/**,**/.omx/**,**/docs/**,**/src/spec/**,**/results/**,**/generated_encoded_gcn_full_opcode_table.cpp,**/generated_encoded_gcn_inst_db.cpp"

log() {
  echo "[quality] $*"
}

fail() {
  echo "[quality] error: $*" >&2
  exit 1
}

require_command() {
  local name="$1"
  command -v "$name" >/dev/null 2>&1 || fail "missing required command: $name"
}

require_quality_dependencies() {
  require_command cmake
  if ! command -v npx >/dev/null 2>&1; then
    fail "missing required command: npx; install Node.js/npm first"
  fi
  if ! command -v cppcheck >/dev/null 2>&1; then
    fail "missing required command: cppcheck; run ./scripts/install_quality_tools.sh first"
  fi
}

detect_generator() {
  if command -v ninja >/dev/null 2>&1; then
    echo "Ninja"
    return
  fi
  echo ""
}

ensure_lizard() {
  if [ -x "$LIZARD_BIN" ]; then
    return
  fi

  log "bootstrap local lizard runtime"
  "$ROOT/scripts/install_quality_tools.sh"
}

run_jscpd() {
  log "run duplication detection via jscpd"
  npx --yes "jscpd@${JSCPD_VERSION}" \
    "$ROOT/src" "$ROOT/tests" "$ROOT/examples" \
    --gitignore \
    --ignore "$JSCPD_IGNORE" \
    --format cpp \
    --min-lines "$JSCPD_MIN_LINES" \
    --min-tokens "$JSCPD_MIN_TOKENS" \
    --reporters console,json \
    --output "$RESULTS_DIR/jscpd" \
    --threshold "$JSCPD_THRESHOLD" \
    2>&1 | tee "$RESULTS_DIR/jscpd.console.log"
}

run_lizard() {
  log "run cyclomatic complexity scan via lizard"
  "$LIZARD_BIN" \
    -l cpp \
    -C "$LIZARD_CCN_LIMIT" \
    -L "$LIZARD_LENGTH_LIMIT" \
    -a "$LIZARD_PARAM_LIMIT" \
    -i -1 \
    -x "$ROOT/examples/*/results/*" \
    -x "$ROOT/src/instruction/encoded/internal/generated_encoded_gcn_full_opcode_table.cpp" \
    -x "$ROOT/src/instruction/encoded/internal/generated_encoded_gcn_inst_db.cpp" \
    "$ROOT/src" "$ROOT/tests" "$ROOT/examples" \
    2>&1 | tee "$RESULTS_DIR/lizard.console.log"
}

configure_cppcheck_project() {
  local generator_name
  generator_name="$(detect_generator)"
  local generator_args=()
  if [ -n "$generator_name" ]; then
    generator_args=(-G "$generator_name")
  fi

  log "configure compile_commands.json for cppcheck"
  cmake -S "$ROOT" -B "$QUALITY_BUILD_DIR" \
    "${generator_args[@]}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DGPU_MODEL_ENABLE_ASAN=OFF \
    2>&1 | tee "$RESULTS_DIR/cppcheck.configure.log"
}

run_cppcheck() {
  local cppcheck_args=(
    --project="$QUALITY_BUILD_DIR/compile_commands.json"
    --enable=warning,style,performance,portability,information
    --std=c++20
    --inline-suppr
    --suppress=missingIncludeSystem
    --suppress=missingInclude
    --suppress=unmatchedSuppression
    -i"$ROOT/third_party"
    -i"$ROOT/tests"
    -i"$ROOT/src/spec"
    -i"$ROOT/src/instruction/encoded/internal/generated_encoded_gcn_full_opcode_table.cpp"
    -i"$ROOT/src/instruction/encoded/internal/generated_encoded_gcn_inst_db.cpp"
  )

  if [ "$QUALITY_STRICT" = "1" ]; then
    cppcheck_args+=(--error-exitcode=2)
  fi

  log "run static analysis via cppcheck"
  cppcheck "${cppcheck_args[@]}" 2>&1 | tee "$RESULTS_DIR/cppcheck.log"
}

main() {
  require_quality_dependencies
  mkdir -p "$RESULTS_DIR" "$CACHE_DIR"
  ensure_lizard
  run_jscpd
  run_lizard
  configure_cppcheck_project
  run_cppcheck

  cat <<EOF | tee "$RESULTS_DIR/summary.txt"
[quality] ok
- duplication report: $RESULTS_DIR/jscpd
- duplication console log: $RESULTS_DIR/jscpd.console.log
- complexity console log: $RESULTS_DIR/lizard.console.log
- cppcheck configure log: $RESULTS_DIR/cppcheck.configure.log
- cppcheck diagnostics: $RESULTS_DIR/cppcheck.log
- strict mode: $QUALITY_STRICT
EOF
}

main "$@"
