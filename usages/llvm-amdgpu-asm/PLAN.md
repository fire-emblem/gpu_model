# Plan

1. Verify `llc` exists in `PATH`.
2. Materialize a minimal LLVM IR kernel.
3. Emit AMDGPU assembly with `scripts/emit_amdgpu_asm.sh`.
4. Inspect the generated `.s` output and metadata.
