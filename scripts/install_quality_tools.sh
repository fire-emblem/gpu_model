#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${GPU_MODEL_QUALITY_CACHE_DIR:-$ROOT/.cache/quality-tools}"
LIZARD_VENV="${GPU_MODEL_QUALITY_LIZARD_VENV:-$CACHE_DIR/lizard-venv}"
LIZARD_BIN="$LIZARD_VENV/bin/lizard"
JSCPD_VERSION="${GPU_MODEL_JSCPD_VERSION:-4.0.9}"

log() {
  echo "[quality-install] $*"
}

fail() {
  echo "[quality-install] error: $*" >&2
  exit 1
}

require_command() {
  local name="$1"
  command -v "$name" >/dev/null 2>&1 || fail "missing required command: $name"
}

ensure_lizard() {
  if [ -x "$LIZARD_BIN" ]; then
    log "reuse lizard: $LIZARD_BIN"
    return
  fi

  require_command python3
  mkdir -p "$CACHE_DIR"

  log "create local venv: $LIZARD_VENV"
  python3 -m venv "$LIZARD_VENV"

  log "install lizard into local venv"
  "$LIZARD_VENV/bin/pip" install lizard
}

ensure_cppcheck() {
  if command -v cppcheck >/dev/null 2>&1; then
    log "reuse cppcheck: $(command -v cppcheck)"
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    fail "cppcheck is missing and apt-get is unavailable; please install cppcheck manually"
  fi

  if [ "$(id -u)" -eq 0 ]; then
    log "install cppcheck via apt-get"
    apt-get install -y cppcheck
    return
  fi

  if command -v sudo >/dev/null 2>&1; then
    log "install cppcheck via sudo apt-get"
    sudo apt-get install -y cppcheck
    return
  fi

  fail "cppcheck is missing and current user cannot install it automatically"
}

ensure_jscpd_runtime() {
  require_command npx
  log "verify jscpd runtime via npx"
  npx --yes "jscpd@${JSCPD_VERSION}" --version
}

main() {
  ensure_lizard
  ensure_cppcheck
  ensure_jscpd_runtime

  cat <<EOF
[quality-install] ok
- lizard: $LIZARD_BIN
- cppcheck: $(command -v cppcheck)
- jscpd: npx jscpd@${JSCPD_VERSION}
EOF
}

main "$@"
