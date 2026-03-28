# HIP FMA Loop Interposer

Run a real HIP host+device executable with a loop-carried `fma_loop` kernel
while redirecting HIP runtime calls to the GPU model through `LD_PRELOAD`.
