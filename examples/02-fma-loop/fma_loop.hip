#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

extern "C" __global__ void fma_loop(const float* a, const float* b, float* c, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float x = a[i];
  float y = b[i];
  float acc = 0.0f;
  for (int k = 0; k < iters; ++k) {
    acc = acc * x + y;
  }
  c[i] = acc;
}

int main() {
  constexpr int n = 257;
  constexpr int iters = 7;
  std::vector<float> a(n), b(n), c(n, -1.0f), expect(n);
  for (int i = 0; i < n; ++i) {
    a[i] = 1.0f + 0.001f * i;
    b[i] = 2.0f + 0.002f * i;
    float acc = 0.0f;
    for (int k = 0; k < iters; ++k) {
      acc = acc * a[i] + b[i];
    }
    expect[i] = acc;
  }

  float* da = nullptr;
  float* db = nullptr;
  float* dc = nullptr;
  if (hipMalloc(&da, n * sizeof(float)) != hipSuccess) return 10;
  if (hipMalloc(&db, n * sizeof(float)) != hipSuccess) return 11;
  if (hipMalloc(&dc, n * sizeof(float)) != hipSuccess) return 12;
  if (hipMemcpy(da, a.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 13;
  if (hipMemcpy(db, b.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 14;
  if (hipMemcpy(dc, c.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 15;

  hipLaunchKernelGGL(fma_loop, dim3(3), dim3(128), 0, 0, da, db, dc, n, iters);
  if (hipDeviceSynchronize() != hipSuccess) return 16;
  if (hipMemcpy(c.data(), dc, n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 17;

  for (int i = 0; i < n; ++i) {
    if (std::fabs(c[i] - expect[i]) > 1e-5f) {
      std::printf("mismatch %d got=%f expected=%f\n", i, c[i], expect[i]);
      return 20;
    }
  }

  std::puts("fma_loop host path ok");
  return 0;
}
