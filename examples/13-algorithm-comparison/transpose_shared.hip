#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

// Optimized Algorithm: Shared memory transpose using dynamic shared memory
constexpr int TILE = 16;

extern "C" __global__ void transpose_shared(const float* in, float* out, int width, int height) {
  // Use dynamic shared memory
  extern __shared__ float tile[];

  // Load from input block (bx, by) - coalesced reads
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;

  if (x < width && y < height) {
    tile[threadIdx.y * TILE + threadIdx.x] = in[y * width + x];
  }
  __syncthreads();

  // Write to output block (by, bx) - coalesced writes
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;

  if (x < height && y < width) {
    out[y * width + x] = tile[threadIdx.x * TILE + threadIdx.y];
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
  hipLaunchKernelGGL(transpose_shared, grid, block, TILE * TILE * sizeof(float), 0, d_in, d_out, width, height);
  if (hipDeviceSynchronize() != hipSuccess) return 15;
  if (hipMemcpy(out.data(), d_out, n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 16;

  // Validate: out[y][x] should equal in[x][y]
  int mismatches = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float expected = in[x * height + y];
      if (out[y * width + x] != expected) {
        if (mismatches < 5) {
          printf("mismatch at (%d,%d): got %.0f expected %.0f\n", x, y, out[y * width + x], expected);
        }
        ++mismatches;
      }
    }
  }

  if (mismatches != 0) {
    printf("transpose_shared: %d mismatches\n", mismatches);
    return 20;
  }
  puts("transpose_shared validation ok");
  return 0;
}
