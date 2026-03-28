# Spec References

This directory stores long-lived engineering references that directly drive code generation, loader design, ABI initialization, decode tables, and semantic-handler coverage.

Current references:

- [gcn-isa-binary-and-semantics-reference.md](/data/gpu_model/src/spec/gcn-isa-binary-and-semantics-reference.md)
  - GCN/CDNA ISA binary organization, format classes, operand model, semantic-family layering, and generation strategy for decode/disasm/exec tables
- [gcn-isa-db-format.md](/data/gpu_model/src/spec/gcn-isa-db-format.md)
  - recommended machine-readable source format for opcode/encoding/field definitions and generated C++ decode tables
- [gcn-optype-opcode-table.md](/data/gpu_model/src/spec/gcn-optype-opcode-table.md)
  - generated markdown table of `optype` encodings, per-format opcode enums, and opcode names currently present in the system
- [llvm-amdgpu-model-contract-reference.md](/data/gpu_model/src/spec/llvm-amdgpu-model-contract-reference.md)
  - LLVM AMDGPU backend / AMDHSA code object contract relevant to loader, metadata parsing, module load, kernarg packing, and wave launch ABI initialization
- [llvm_amdgpu_refs/README.md](/data/gpu_model/src/spec/llvm_amdgpu_refs/README.md)
  - locally downloaded official LLVM AMDGPU HTML references
- [llvm_amdgpu_backend_td/README.md](/data/gpu_model/src/spec/llvm_amdgpu_backend_td/README.md)
  - locally downloaded LLVM AMDGPU backend `.td` sources used for cross-checking opcode and encoding definitions

Usage rule:

- if decode/disasm changes, update the GCN ISA reference first
- if loader/runtime/ABI changes, update the LLVM AMDGPU contract reference first
- code changes that materially change either contract should update the matching reference in the same commit
