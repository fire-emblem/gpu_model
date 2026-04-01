#include <hip/hip_runtime.h>
#include <math.h>

#include <cstdio>
#include <vector>

extern "C" __global__ void softmax_row(const float* in, float* out, int n) {
  __shared__ float scratch[64];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  float x = idx < n ? in[idx] : -1.0e20f;
  scratch[tid] = x;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    __syncthreads();
  }
  float m = scratch[0];
  float e = idx < n ? expf(x - m) : 0.0f;
  scratch[tid] = e;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  if (idx < n) out[idx] = e / scratch[0];
}

int main() {
  constexpr int n = 64;
  std::vector<float> in(n, 1.0f), out(n, 0.0f);
  constexpr float expect = 1.0f / 64.0f;

  float* din = nullptr;
  float* dout = nullptr;
  if (hipMalloc(&din, n * sizeof(float)) != hipSuccess) return 10;
  if (hipMalloc(&dout, n * sizeof(float)) != hipSuccess) return 11;
  if (hipMemcpy(din, in.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 12;
  if (hipMemcpy(dout, out.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 13;

  hipLaunchKernelGGL(softmax_row, dim3(1), dim3(64), 0, 0, din, dout, n);
  if (hipDeviceSynchronize() != hipSuccess) return 14;
  if (hipMemcpy(out.data(), dout, n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 15;

  int mismatches = 0;
  for (int i = 0; i < n; ++i) {
    if (fabsf(out[i] - expect) > 1.0e-4f) {
      if (mismatches < 8) {
        std::printf("mismatch %d got=%f expected=%f\n", i, out[i], expect);
      }
      ++mismatches;
    }
  }
  std::printf("softmax_reduction mismatches=%d n=%d\n", mismatches, n);
  return mismatches == 0 ? 0 : 20;
}
