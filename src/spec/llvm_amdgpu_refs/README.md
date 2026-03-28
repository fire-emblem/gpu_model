# LLVM AMDGPU HTML References

Downloaded official LLVM documentation references used by the project:

- `AMDGPUAsmGFX9.html`
  - GFX9 instruction names, syntax, operand forms
- `AMDGPUUsage.html`
  - ABI, metadata, code object, kernel state, calling convention

These files are kept locally so the project can:

- cross-check generated opcode enums and names
- validate operand syntax against LLVM docs
- review ABI / metadata rules without depending on network access
