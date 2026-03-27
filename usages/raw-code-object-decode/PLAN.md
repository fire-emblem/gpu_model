# Plan

1. Build `code_object_dump_main`.
2. Materialize a minimal HIP `vecadd` executable with `hipcc`.
3. Run `code_object_dump_main` on the resulting `.out`.
4. Capture the decoded instruction list.
