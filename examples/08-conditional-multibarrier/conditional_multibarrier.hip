#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr uint32_t kBlockDim = 128;
constexpr uint32_t kHalfBlock = kBlockDim / 2;
constexpr uint32_t kQuarterBlock = kBlockDim / 4;
constexpr uint32_t kThreeQuarterBlock = 3 * kQuarterBlock;

bool CheckHip(hipError_t status, const char* api) {
  if (status == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s\n", api, hipGetErrorString(status));
  return false;
}

}  // namespace

extern "C" __global__ void conditional_multibarrier(int* out) {
  __shared__ int tile[kBlockDim];
  const int tid = static_cast<int>(threadIdx.x);
  const int block = static_cast<int>(blockIdx.x);
  const int base = block * blockDim.x;
  const int value = base + tid;

  tile[tid] = value + 3;
  __syncthreads();

  if (tid < static_cast<int>(kHalfBlock)) {
    tile[tid] += tile[kBlockDim - 1 - tid];
  } else {
    tile[tid] -= tile[kBlockDim - 1 - tid];
  }
  __syncthreads();

  int mixed = tile[tid];
  if (tid < static_cast<int>(kQuarterBlock)) {
    mixed += 11;
  } else if (tid < static_cast<int>(kThreeQuarterBlock)) {
    mixed -= 7;
  } else {
    mixed += 5;
  }

  if (tid < static_cast<int>(kHalfBlock)) {
    mixed += tile[(tid + 17) & (kBlockDim - 1)];
  } else {
    mixed -= tile[(tid + 23) & (kBlockDim - 1)];
  }

  tile[tid] = mixed;
  __syncthreads();

  out[base + tid] = tile[tid];
}

static std::vector<int32_t> BuildExpected(uint32_t grid_dim) {
  std::vector<int32_t> expected(grid_dim * kBlockDim, 0);

  for (uint32_t block = 0; block < grid_dim; ++block) {
    const uint32_t base = block * kBlockDim;
    std::vector<int32_t> tile(kBlockDim, 0);
    for (uint32_t tid = 0; tid < kBlockDim; ++tid) {
      tile[tid] = static_cast<int32_t>(base + tid) + 3;
    }

    std::vector<int32_t> stage1(kBlockDim, 0);
    for (uint32_t tid = 0; tid < kBlockDim; ++tid) {
      if (tid < kHalfBlock) {
        stage1[tid] = tile[tid] + tile[kBlockDim - 1 - tid];
      } else {
        stage1[tid] = tile[tid] - tile[kBlockDim - 1 - tid];
      }
    }

    std::vector<int32_t> stage2(kBlockDim, 0);
    for (uint32_t tid = 0; tid < kBlockDim; ++tid) {
      int32_t mixed = stage1[tid];
      if (tid < kQuarterBlock) {
        mixed += 11;
      } else if (tid < kThreeQuarterBlock) {
        mixed -= 7;
      } else {
        mixed += 5;
      }

      if (tid < kHalfBlock) {
        mixed += stage1[(tid + 17) & (kBlockDim - 1)];
      } else {
        mixed -= stage1[(tid + 23) & (kBlockDim - 1)];
      }
      stage2[tid] = mixed;
    }

    for (uint32_t tid = 0; tid < kBlockDim; ++tid) {
      expected[base + tid] = stage2[tid];
    }
  }

  return expected;
}

int main() {
  constexpr uint32_t kGridDim = 8;
  constexpr uint32_t kElementCount = kGridDim * kBlockDim;

  int* device_out = nullptr;
  std::vector<int32_t> host_out(kElementCount, -1);
  const std::vector<int32_t> expected = BuildExpected(kGridDim);

  if (!CheckHip(hipMalloc(&device_out, kElementCount * sizeof(int32_t)), "hipMalloc")) return 10;
  if (!CheckHip(hipMemset(device_out, 0, kElementCount * sizeof(int32_t)), "hipMemset")) return 11;

  hipLaunchKernelGGL(
      conditional_multibarrier, dim3(kGridDim), dim3(kBlockDim), 0, 0, device_out);

  const auto launch_status = hipGetLastError();
  if (launch_status != hipSuccess) {
    CheckHip(hipFree(device_out), "hipFree");
    std::fprintf(stderr, "hipLaunchKernelGGL failed: %s\n", hipGetErrorString(launch_status));
    return 12;
  }
  if (!CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize")) {
    CheckHip(hipFree(device_out), "hipFree");
    return 13;
  }
  if (!CheckHip(hipMemcpy(host_out.data(),
                          device_out,
                          kElementCount * sizeof(int32_t),
                          hipMemcpyDeviceToHost),
                "hipMemcpy")) {
    CheckHip(hipFree(device_out), "hipFree");
    return 14;
  }

  uint32_t mismatches = 0;
  for (uint32_t i = 0; i < kElementCount; ++i) {
    if (host_out[i] != expected[i]) {
      if (mismatches < 8) {
        const uint32_t block = i / kBlockDim;
        const uint32_t tid = i % kBlockDim;
        std::printf("mismatch idx=%u block=%u tid=%u got=%d expected=%d\n",
                    i,
                    block,
                    tid,
                    host_out[i],
                    expected[i]);
      }
      ++mismatches;
    }
  }

  std::printf("conditional_multibarrier mismatches=%u n=%u\n", mismatches, kElementCount);
  if (!CheckHip(hipFree(device_out), "hipFree")) return 15;
  return mismatches == 0 ? 0 : 20;
}
