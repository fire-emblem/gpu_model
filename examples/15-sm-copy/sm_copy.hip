#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

constexpr int TILE = 16;

extern "C" __global__ void sm_copy(const float* in, float* out, int width, int height) {
  __shared__ float tile[TILE][TILE + 1];
  
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;

  if (x < width && y < height) {
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  }
  __syncthreads();

  if (x < width && y < height) {
    out[y * width + x] = tile[threadIdx.y][threadIdx.x];
  }
}

int main() {
  constexpr int width = 32;
  constexpr int height = 32;
  constexpr int n = width * height;

  std::vector<float> in(n), out(n, -1.0f);
  for (int i = 0; i < n; ++i) in[i] = static_cast<float>(i);

  float *d_in = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_in, n * sizeof(float)) != hipSuccess) return 10;
  if (hipMalloc(&d_out, n * sizeof(float)) != hipSuccess) return 11;
  if (hipMemcpy(d_in, in.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 12;

  dim3 block(TILE, TILE);
  dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);
  hipLaunchKernelGGL(sm_copy, grid, block, TILE * (TILE + 1) * sizeof(float), 0, d_in, d_out, width, height);
  if (hipDeviceSynchronize() != hipSuccess) return 15;
  if (hipMemcpy(out.data(), d_out, n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 16;

  int mismatches = 0;
  for (int i = 0; i < n; ++i) {
    if (out[i] != in[i]) {
      if (mismatches < 5) printf("mismatch %d: got %.0f expected %.0f\n", i, out[i], in[i]);
      ++mismatches;
    }
  }
  printf("sm_copy: %d mismatches\n", mismatches);
  return mismatches == 0 ? 0 : 20;
}
