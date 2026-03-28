# Plan

1. Build `libgpu_model_hip_interposer.so`.
2. Materialize a HIP host+device `fma_loop` program with five kernel args.
3. Compile it with `hipcc` into a real `.out` executable.
4. Run the executable with `LD_PRELOAD=libgpu_model_hip_interposer.so`.
5. Verify that:
   - host `main()` executes normally
   - HIP API calls are intercepted
   - raw GCN decode recognizes loop/SCC/FMA instructions
   - dynamic kernarg packing handles `3*ptr + 2*int`
   - host-side result validation passes
