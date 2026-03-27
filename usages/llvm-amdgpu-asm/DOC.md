# Doc

This usage bundle emits AMDGPU assembly from a minimal LLVM IR kernel using the
system-installed LLVM backend.

Command executed:

```bash
./scripts/emit_amdgpu_asm.sh \
  usages/llvm-amdgpu-asm/results/min_amdgpu.ll \
  usages/llvm-amdgpu-asm/results/min_amdgpu.s \
  gfx900
```

Expected artifacts:

- `results/min_amdgpu.ll`
- `results/min_amdgpu.s`
- `results/stdout.txt`

Expected behavior:

- `llc` accepts the LLVM IR
- the output assembly includes `s_endpgm`
- the output contains `.amdhsa_kernel` metadata
