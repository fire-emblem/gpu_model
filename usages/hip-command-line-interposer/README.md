# HIP Command-Line Interposer

Run a real HIP executable on the host CPU while redirecting HIP runtime calls to
the GPU model through `LD_PRELOAD`, so host `main()` continues to run natively
while device kernels execute inside the model.
