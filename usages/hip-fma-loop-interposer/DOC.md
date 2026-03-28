# Doc

This usage bundle extends the command-line HIP interposer path beyond `vecadd`
into a looped floating-point kernel that requires additional raw GCN coverage:

- scalar compare on `scc`
- scalar conditional and unconditional branches
- loop-carried `v_fma_f32`
- dynamic kernarg packing based on code-object arg layout

Expected artifacts:

- `results/hip_fma_loop_host.cpp`
- `results/hip_fma_loop_host.out`
- `results/stdout.txt`
