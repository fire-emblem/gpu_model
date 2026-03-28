# GCN DB Source Layout

This directory is reserved for machine-readable GCN ISA source data.

Planned source-of-truth format:

- YAML files
- split by:
  - architecture profile
  - format class
  - operand kinds
  - semantic families
  - opcode groups

Planned flow:

1. maintain YAML here
2. generate C++ static tables
3. use generated tables for:
   - binary decode
   - disassembly
   - semantic-family lookup
   - future code generation

This directory is intentionally only scaffolded in this step.
