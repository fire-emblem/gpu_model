#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/usages/llvm-amdgpu-asm/results"
mkdir -p "$OUT_DIR"

if ! command -v llc >/dev/null 2>&1; then
  echo "missing llc in PATH" >&2
  exit 1
fi

cat > "$OUT_DIR/min_amdgpu.ll" <<'EOF'
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @empty_kernel() {
entry:
  ret void
}
EOF

"$ROOT/scripts/emit_amdgpu_asm.sh" \
  "$OUT_DIR/min_amdgpu.ll" \
  "$OUT_DIR/min_amdgpu.s" \
  gfx900

{
  echo "llvm_toolchain=$(llc --version | head -n 1)"
  echo "output=$OUT_DIR/min_amdgpu.s"
  sed -n '1,80p' "$OUT_DIR/min_amdgpu.s"
} | tee "$OUT_DIR/stdout.txt"
