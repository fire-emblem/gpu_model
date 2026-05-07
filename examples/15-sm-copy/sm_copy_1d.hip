#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

extern "C" __global__ void sm_copy_1d(const float* in, float* out, int n) {
  __shared__ float tile[64];
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    tile[threadIdx.x] = in[idx];
  }
  __syncthreads();

  if (idx < n) {
    out[idx] = tile[threadIdx.x];
  }
}

int main() {
  constexpr int n = 128;

  std::vector<float> in(n), out(n, -1.0f);
  for (int i = 0; i < n; ++i) in[i] = static_cast<float>(i);

  float *d_in = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_in, n * sizeof(float)) != hipSuccess) return 10;
  if (hipMalloc(&d_out, n * sizeof(float)) != hipSuccess) return 11;
  if (hipMemcpy(d_in, in.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 12;

  hipLaunchKernelGGL(sm_copy_1d, dim3(2), dim3(64), 64 * sizeof(float), 0, d_in, d_out, n);
  if (hipDeviceSynchronize() != hipSuccess) return 15;
  if (hipMemcpy(out.data(), d_out, n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 16;

  int mismatches = 0;
  for (int i = 0; i < n; ++i) {
    if (out[i] != in[i]) {
      if (mismatches < 5) printf("mismatch %d: got %.0f expected %.0f\n", i, out[i], in[i]);
      ++mismatches;
    }
  }
  printf("sm_copy_1d: %d mismatches\n", mismatches);
  return mismatches == 0 ? 0 : 20;
}
