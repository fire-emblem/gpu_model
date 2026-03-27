# Plan

1. Verify the HIP and LLVM helper tools exist in `PATH`.
2. Materialize a minimal HIP source with an empty kernel.
3. Build both a host object and a host executable with `hipcc`.
4. Dump `.hip_fatbin` from the executable and unbundle the AMDGPU code object.
5. Inspect the extracted code object with `readelf` and `llvm-objdump`.
6. Run the gtest launch path that loads the executable artifact and executes the empty kernel.
