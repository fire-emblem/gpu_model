#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

extern "C" __global__ void shared_reverse(const int* in, int* out, int n) {
  __shared__ int scratch[64];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  if (idx < n) {
    scratch[tid] = in[idx];
  }
  __syncthreads();
  if (idx < n) {
    out[idx] = scratch[blockDim.x - 1 - tid];
  }
}

int main() {
  constexpr int n = 128;
  std::vector<int> in(n), out(n, -1), expect(n, -1);
  for (int i = 0; i < n; ++i) {
    in[i] = i + 1;
  }
  for (int block = 0; block < 2; ++block) {
    const int base = block * 64;
    for (int lane = 0; lane < 64; ++lane) {
      expect[base + lane] = in[base + (63 - lane)];
    }
  }

  int* din = nullptr;
  int* dout = nullptr;
  if (hipMalloc(&din, n * sizeof(int)) != hipSuccess) return 10;
  if (hipMalloc(&dout, n * sizeof(int)) != hipSuccess) return 11;
  if (hipMemcpy(din, in.data(), n * sizeof(int), hipMemcpyHostToDevice) != hipSuccess) return 12;
  if (hipMemcpy(dout, out.data(), n * sizeof(int), hipMemcpyHostToDevice) != hipSuccess) return 13;

  hipLaunchKernelGGL(shared_reverse, dim3(2), dim3(64), 0, 0, din, dout, n);
  if (hipDeviceSynchronize() != hipSuccess) return 14;
  if (hipMemcpy(out.data(), dout, n * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) return 15;

  int mismatches = 0;
  for (int i = 0; i < n; ++i) {
    if (out[i] != expect[i]) {
      if (mismatches < 8) {
        std::printf("mismatch %d got=%d expected=%d\n", i, out[i], expect[i]);
      }
      ++mismatches;
    }
  }
  std::printf("shared_reverse mismatches=%d n=%d\n", mismatches, n);
  return mismatches == 0 ? 0 : 20;
}
