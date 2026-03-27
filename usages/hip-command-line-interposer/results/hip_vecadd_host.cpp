#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

extern "C" __global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  constexpr int n = 257;
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
  if (hipMemcpy(dc, c.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 15;
  hipLaunchKernelGGL(vecadd, dim3(3), dim3(128), 0, 0, da, db, dc, n);
  if (hipDeviceSynchronize() != hipSuccess) return 16;
  if (hipMemcpy(c.data(), dc, n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 17;

  for (int i = 0; i < n; ++i) {
    const float expected = a[i] + b[i];
    if (c[i] != expected) {
      std::printf("mismatch %d got=%f expected=%f\n", i, c[i], expected);
      return 20;
    }
  }

  std::puts("vecadd host path ok");
  return 0;
}
