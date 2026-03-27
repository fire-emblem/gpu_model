# Doc

This usage bundle demonstrates the current HIP integration boundary:

- `hipcc` can produce host ELF artifacts that embed device code in `.hip_fatbin`
- the simulator can extract an `amdgcn-amd-amdhsa` bundle from that artifact
- the extracted device code object can be passed through the existing loader path

Current limitation:

- execution still depends on the simulator's supported AMD-style ISA subset
- the bundled empty-kernel example works because it lowers to a trivially supported instruction stream
- arbitrary HIP kernels such as compiled `vecadd` are not yet guaranteed to execute

Expected artifacts:

- `results/hip_empty_kernel.cpp`
- `results/hip_empty_kernel.o`
- `results/hip_empty_kernel.out`
- `results/hip_empty_kernel.hip_fatbin`
- `results/hip_empty_kernel.gfx.co`
- `results/bundles.txt`
- `results/device_readelf_header.txt`
- `results/device_objdump.txt`
- `results/stdout.txt`
