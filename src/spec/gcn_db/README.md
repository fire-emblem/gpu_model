# GCN DB Source Layout

This directory is reserved for machine-readable GCN ISA source data.

Current source-of-truth format:

- YAML files
- split by:
  - format class
  - instruction definitions

Current files:

- `profiles.yaml`
- `operand_kinds.yaml`
- `semantic_families.yaml`
- `format_classes.yaml`
- `instructions.yaml`

Current flow:

1. maintain YAML here
2. generate C++ static tables with `scripts/gen_gcn_isa_db.py`
3. use generated tables for:
   - binary decode
   - disassembly
   - definition lookup
   - future code generation

Planned follow-on split:

- `opcodes/*.yaml`
