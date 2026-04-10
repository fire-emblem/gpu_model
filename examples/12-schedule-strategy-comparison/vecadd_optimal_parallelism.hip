#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

// Strategy 3: 16 blocks, 256 threads - optimal parallelism
// Each thread processes exactly 1 element (1:1 mapping)
extern "C" __global__ void vecadd_optimal_parallelism(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  constexpr int n = 4096;
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

  // 16 blocks x 256 threads = 4096 threads total, 1:1 mapping
  hipLaunchKernelGGL(vecadd_optimal_parallelism, dim3(16), dim3(256), 0, 0, da, db, dc, n);
  if (hipDeviceSynchronize() != hipSuccess) return 15;
  if (hipMemcpy(c.data(), dc, n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 16;

  for (int i = 0; i < n; ++i) {
    if (c[i] != a[i] + b[i]) return 20;
  }
  std::puts("vecadd_optimal_parallelism validation ok");
  return 0;
}
