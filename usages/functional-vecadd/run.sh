#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/usages/functional-vecadd/results"
mkdir -p "$OUT_DIR"

if [[ ! -x "$ROOT/build/vecadd_main" ]]; then
  echo "missing executable: $ROOT/build/vecadd_main" >&2
  exit 1
fi

"$ROOT/build/vecadd_main" | tee "$OUT_DIR/stdout.txt"
