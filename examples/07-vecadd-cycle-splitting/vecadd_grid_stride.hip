#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

extern "C" __global__ void vecadd_grid_stride(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  constexpr int n = 1024;
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (int i = 0; i < n; ++i) {
    a[i] = 0.5f * i;
    b[i] = 0.25f * (100 + i);
  }

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  if (hipMalloc(&da, n * sizeof(float)) != hipSuccess) return 10;
  if (hipMalloc(&db, n * sizeof(float)) != hipSuccess) return 11;
  if (hipMalloc(&dc, n * sizeof(float)) != hipSuccess) return 12;
  if (hipMemcpy(da, a.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 13;
  if (hipMemcpy(db, b.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 14;

  hipLaunchKernelGGL(vecadd_grid_stride, dim3(4), dim3(64), 0, 0, da, db, dc, n);
  if (hipDeviceSynchronize() != hipSuccess) return 15;
  if (hipMemcpy(c.data(), dc, n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 16;

  for (int i = 0; i < n; ++i) {
    if (c[i] != a[i] + b[i]) return 20;
  }
  std::puts("vecadd_grid_stride validation ok");
  return 0;
}
