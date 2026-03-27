# Plan

1. Build `libgpu_model_hip_interposer.so`.
2. Materialize a minimal HIP host+device `vecadd` program.
3. Compile it with `hipcc` into a real `.out` executable.
4. Run the executable with `LD_PRELOAD=libgpu_model_hip_interposer.so`.
5. Verify that:
   - host `main()` executes normally
   - HIP API calls are intercepted
   - kernel launch is redirected to the model
   - host-side result validation passes
