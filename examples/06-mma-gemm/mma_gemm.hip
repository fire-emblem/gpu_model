#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>

typedef float v4f __attribute__((ext_vector_type(4)));

extern "C" __global__ void mma_gemm_probe(float* out) {
  v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
  acc = __builtin_amdgcn_mfma_f32_16x16x4f32(1.0f, 1.0f, acc, 0, 0, 0);
  if (threadIdx.x == 0) {
    out[0] = acc[0] ;
  }
}

int main() {
  float init = 0.0f;
  float out = 0.0f;
  float* dout = nullptr;
  if (hipMalloc(&dout, sizeof(float)) != hipSuccess) return 10;
  if (hipMemcpy(dout, &init, sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 11;

  hipLaunchKernelGGL(mma_gemm_probe, dim3(1), dim3(64), 0, 0, dout);
  if (hipDeviceSynchronize() != hipSuccess) return 12;
  if (hipMemcpy(&out, dout, sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 13;

  std::printf("mma_gemm out=%f expected=%f\n", out, 4.0f);
  return std::fabs(out - 4.0f) < 1.0e-5f ? 0 : 20;
}
