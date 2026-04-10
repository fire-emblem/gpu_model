#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

// Strategy 1: 1 block, 64 threads - minimal parallelism
// Uses grid-stride loop to process all elements
extern "C" __global__ void vecadd_low_parallelism(const float* a, const float* b, float* c, int n) {
  int stride = blockDim.x * gridDim.x;  // 64
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
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

  // 1 block x 64 threads = only 64 threads total
  hipLaunchKernelGGL(vecadd_low_parallelism, dim3(1), dim3(64), 0, 0, da, db, dc, n);
  if (hipDeviceSynchronize() != hipSuccess) return 15;
  if (hipMemcpy(c.data(), dc, n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 16;

  for (int i = 0; i < n; ++i) {
    if (c[i] != a[i] + b[i]) return 20;
  }
  std::puts("vecadd_low_parallelism validation ok");
  return 0;
}
