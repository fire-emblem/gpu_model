#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

git config core.hooksPath "$ROOT/.githooks"
chmod +x "$ROOT/.githooks/pre-push"
chmod +x "$ROOT/scripts/run_push_gate.sh"

echo "Installed git hooks:"
echo "- core.hooksPath=$ROOT/.githooks"
