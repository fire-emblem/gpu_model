#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

extern "C" __global__ void block_reduce_sum(const float* in, float* out, int n) {
  __shared__ float scratch[256];
  const int tid = threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int idx = blockIdx.x * blockDim.x + tid;

  float sum = 0.0f;
  for (int i = idx; i < n; i += stride) {
    sum += in[i];
  }
  scratch[tid] = sum;
  __syncthreads();

  for (int step = blockDim.x / 2; step > 0; step >>= 1) {
    if (tid < step) {
      scratch[tid] += scratch[tid + step];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[blockIdx.x] = scratch[0];
  }
}

int main() {
  constexpr int n = 1024;
  constexpr int grid_dim = 4;
  constexpr int block_dim = 256;

  std::vector<float> in(n, 1.0f);
  std::vector<float> out(grid_dim, -1.0f);
  std::vector<float> expect(grid_dim, static_cast<float>(n / grid_dim));

  float* din = nullptr;
  float* dout = nullptr;
  if (hipMalloc(&din, n * sizeof(float)) != hipSuccess) return 10;
  if (hipMalloc(&dout, grid_dim * sizeof(float)) != hipSuccess) return 11;
  if (hipMemcpy(din, in.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 12;
  if (hipMemcpy(dout, out.data(), grid_dim * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 13;

  hipLaunchKernelGGL(block_reduce_sum, dim3(grid_dim), dim3(block_dim), 0, 0, din, dout, n);
  if (hipDeviceSynchronize() != hipSuccess) return 14;
  if (hipMemcpy(out.data(), dout, grid_dim * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 15;

  int mismatches = 0;
  for (int i = 0; i < grid_dim; ++i) {
    if (fabsf(out[i] - expect[i]) > 1.0e-4f) {
      if (mismatches < 8) {
        std::printf("mismatch %d got=%f expected=%f\n", i, out[i], expect[i]);
      }
      ++mismatches;
    }
  }

  std::printf("block_reduce_sum mismatches=%d blocks=%d block_dim=%d n=%d\n",
              mismatches,
              grid_dim,
              block_dim,
              n);
  return mismatches == 0 ? 0 : 20;
}
