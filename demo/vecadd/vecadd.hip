#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

extern "C" __global__ void vecadd(const float* a, const float* b, float* c, int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  constexpr int n = 257;
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (int i = 0; i < n; ++i) {
    a[i] = 0.5f * static_cast<float>(i);
    b[i] = 0.25f * static_cast<float>(100 + i);
  }

  float* da = nullptr;
  float* db = nullptr;
  float* dc = nullptr;
  if (hipMalloc(&da, n * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc(da) failed\n");
    return 10;
  }
  if (hipMalloc(&db, n * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc(db) failed\n");
    return 11;
  }
  if (hipMalloc(&dc, n * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc(dc) failed\n");
    return 12;
  }
  if (hipMemcpy(da, a.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
    std::fprintf(stderr, "hipMemcpy H2D(a) failed\n");
    return 13;
  }
  if (hipMemcpy(db, b.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
    std::fprintf(stderr, "hipMemcpy H2D(b) failed\n");
    return 14;
  }
  if (hipMemcpy(dc, c.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
    std::fprintf(stderr, "hipMemcpy H2D(c) failed\n");
    return 15;
  }

  hipLaunchKernelGGL(vecadd, dim3(3), dim3(128), 0, 0, da, db, dc, n);
  if (hipDeviceSynchronize() != hipSuccess) {
    std::fprintf(stderr, "hipDeviceSynchronize failed\n");
    return 16;
  }
  if (hipMemcpy(c.data(), dc, n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
    std::fprintf(stderr, "hipMemcpy D2H(c) failed\n");
    return 17;
  }

  std::puts("vecadd sample outputs:");
  for (int i = 0; i < 8; ++i) {
    const float expected = a[i] + b[i];
    std::printf("  c[%d] = %.6f expected = %.6f\n", i, c[i], expected);
  }

  int mismatches = 0;
  for (int i = 0; i < n; ++i) {
    const float expected = a[i] + b[i];
    if (std::fabs(c[i] - expected) > 1e-6f) {
      if (mismatches < 8) {
        std::printf("mismatch i=%d got=%.6f expected=%.6f\n", i, c[i], expected);
      }
      ++mismatches;
    }
  }

  std::printf("vecadd validation: mismatches=%d n=%d\n", mismatches, n);
  if (mismatches != 0) {
    return 20;
  }
  std::puts("vecadd validation ok");
  return 0;
}
