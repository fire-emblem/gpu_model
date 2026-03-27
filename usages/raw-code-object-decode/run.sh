#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/usages/raw-code-object-decode/results"
mkdir -p "$OUT_DIR"

if [[ ! -x "$ROOT/build/code_object_dump_main" ]]; then
  echo "missing executable: $ROOT/build/code_object_dump_main" >&2
  exit 1
fi

if ! command -v hipcc >/dev/null 2>&1; then
  echo "missing tool: hipcc" >&2
  exit 1
fi

SRC="$OUT_DIR/hip_vecadd.cpp"
EXE="$OUT_DIR/hip_vecadd.out"

cat > "$SRC" <<'EOF'
#include <hip/hip_runtime.h>

extern "C" __global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  return 0;
}
EOF

hipcc "$SRC" -o "$EXE"
"$ROOT/build/code_object_dump_main" "$EXE" vecadd | tee "$OUT_DIR/stdout.txt"
