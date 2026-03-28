# Plan

1. Build `gpu_model_tests` and `code_object_dump_main`.
2. Run focused gtests that validate:
   - `text bytes -> raw instruction array -> decoded instruction array`
   - code object decode populates instantiated instruction objects
3. Materialize a minimal HIP `vecadd` executable with `hipcc`.
4. Run `code_object_dump_main` on the resulting `.out`.
5. Validate that dump output contains:
   - `op_type=...`
   - `class=...`
   - `class=s_load_dword` for the first scalar memory object in the vecadd kernel.
