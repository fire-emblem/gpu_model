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

- command-line `.out` 主线已经可以走 descriptor + metadata + raw decode/exec 执行多个真实 HIP kernel
- 剩余限制主要在：
  - 全 GCN ISA 覆盖仍未完成
  - graphics/image/export/interp family 仍不是主覆盖方向
  - runtime API 仍未扩展到“任意 HIP 程序”所需的完整子集

Expected artifacts:

- `results/hip_vecadd_host.cpp`
- `results/hip_vecadd_host.out`
- `results/stdout.txt`
