#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

extern "C" __global__ void dynamic_shared_sum(int* out) {
  extern __shared__ int scratch[];
  const int tid = threadIdx.x;
  scratch[tid] = tid + 1;
  __syncthreads();
  if (tid == 0) {
    int acc = 0;
    for (int i = 0; i < blockDim.x; ++i) {
      acc += scratch[i];
    }
    out[blockIdx.x] = acc;
  }
}

int main() {
  constexpr int blocks = 4;
  constexpr int block_dim = 64;
  const int expected_value = block_dim * (block_dim + 1) / 2;

  std::vector<int> out(blocks, 0), expect(blocks, expected_value);

  int* dout = nullptr;
  if (hipMalloc(&dout, blocks * static_cast<int>(sizeof(int))) != hipSuccess) return 10;
  if (hipMemcpy(dout,
                out.data(),
                blocks * static_cast<int>(sizeof(int)),
                hipMemcpyHostToDevice) != hipSuccess) {
    return 11;
  }

  hipLaunchKernelGGL(dynamic_shared_sum,
                     dim3(blocks),
                     dim3(block_dim),
                     block_dim * sizeof(int),
                     0,
                     dout);
  if (hipDeviceSynchronize() != hipSuccess) return 12;
  if (hipMemcpy(out.data(),
                dout,
                blocks * static_cast<int>(sizeof(int)),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return 13;
  }

  int mismatches = 0;
  for (int i = 0; i < blocks; ++i) {
    if (out[i] != expect[i]) {
      if (mismatches < 8) {
        std::printf("mismatch %d got=%d expected=%d\n", i, out[i], expect[i]);
      }
      ++mismatches;
    }
  }

  std::printf("dynamic_shared_sum mismatches=%d blocks=%d block_dim=%d expected=%d\n",
              mismatches,
              blocks,
              block_dim,
              expected_value);
  return mismatches == 0 ? 0 : 20;
}
