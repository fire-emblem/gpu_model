#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

// Naive Algorithm: Direct transpose
// Each thread reads column-wise (coalesced) and writes row-wise (strided)
// Good for read, bad for write - causes memory bank conflicts
constexpr int TILE = 16;

extern "C" __global__ void transpose_naive(const float* in, float* out, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Transpose: out[x][y] = in[y][x]
    // For row-major: out[x * height + y] = in[y * width + x]
    out[x * height + y] = in[y * width + x];
  }
}

int main() {
  constexpr int width = 256;
  constexpr int height = 256;
  constexpr int n = width * height;

  std::vector<float> in(n), out(n, -1.0f);
  for (int i = 0; i < n; ++i) {
    in[i] = static_cast<float>(i);
  }

  float *d_in = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_in, n * sizeof(float)) != hipSuccess) return 10;
  if (hipMalloc(&d_out, n * sizeof(float)) != hipSuccess) return 11;
  if (hipMemcpy(d_in, in.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 12;

  dim3 block(TILE, TILE);
  dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);
  hipLaunchKernelGGL(transpose_naive, grid, block, 0, 0, d_in, d_out, width, height);
  if (hipDeviceSynchronize() != hipSuccess) return 15;
  if (hipMemcpy(out.data(), d_out, n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 16;

  // Validate
  int mismatches = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float expected = in[y * width + x];
      if (out[x * height + y] != expected) {
        if (mismatches < 5) {
          printf("mismatch at (%d,%d): got %.0f expected %.0f\n", x, y, out[x * height + y], expected);
        }
        ++mismatches;
      }
    }
  }

  if (mismatches != 0) {
    printf("transpose_naive: %d mismatches\n", mismatches);
    return 20;
  }
  puts("transpose_naive validation ok");
  return 0;
}
