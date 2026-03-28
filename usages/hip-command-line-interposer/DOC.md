# Doc

This usage bundle exercises the intended long-term execution shape:

- the HIP executable runs as a normal host process
- host CPU code, including `main()`, runs natively
- intercepted HIP runtime APIs are redirected into the model
- control returns to host code after kernel launch completion

Current intercepted API subset:

- `__hipRegisterFatBinary`
- `__hipRegisterFunction`
- `__hipPushCallConfiguration`
- `__hipPopCallConfiguration`
- `hipMalloc`
- `hipMallocManaged`
- `hipFree`
- `hipMemcpy`
- `hipMemcpyAsync`
- `hipMemset`
- `hipMemsetD8`
- `hipMemsetD32`
- `hipGetDeviceCount`
- `hipGetDevice`
- `hipSetDevice`
- `hipGetLastError`
- `hipPeekAtLastError`
- `hipStreamCreate`
- `hipStreamDestroy`
- `hipStreamSynchronize`
- `hipDeviceSynchronize`
- `hipLaunchKernel`

Current limitation:

- device execution still depends on the current GCN text ingestion path
- final production direction is still:
  - ELF code section bytes
  - binary decode
  - raw instruction execution

Expected artifacts:

- `results/hip_vecadd_host.cpp`
- `results/hip_vecadd_host.out`
- `results/stdout.txt`
