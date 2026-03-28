# Demo VecAdd

This demo is the end-to-end command-line flow:

1. compile `vecadd.hip` with `hipcc`
2. produce a real HIP host executable `.out`
3. inject `libgpu_model_hip_interposer.so`
4. execute host `main()` normally
5. redirect HIP runtime calls into the model
6. print sample outputs and validate all results
