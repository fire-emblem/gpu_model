#include <hip/hip_runtime.h>

#include <cstdio>

extern "C" __global__ void atomic_count(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(out, 1);
  }
}

int main() {
  constexpr int n = 257;
  int zero = 0;
  int value = -1;
  int* dout = nullptr;
  if (hipMalloc(&dout, sizeof(int)) != hipSuccess) return 10;
  if (hipMemcpy(dout, &zero, sizeof(int), hipMemcpyHostToDevice) != hipSuccess) return 11;

  hipLaunchKernelGGL(atomic_count, dim3(3), dim3(128), 0, 0, dout, n);
  if (hipDeviceSynchronize() != hipSuccess) return 12;
  if (hipMemcpy(&value, dout, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) return 13;

  std::printf("atomic_reduction value=%d expected=%d\n", value, n);
  return value == n ? 0 : 20;
}
