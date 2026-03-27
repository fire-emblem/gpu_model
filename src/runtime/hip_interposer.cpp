#include <hip/hip_runtime_api.h>

#include <cstdarg>
#include <cstdlib>
#include <cstdio>

#include "gpu_model/runtime/hip_interposer_state.h"

using gpu_model::HipInterposerState;
using gpu_model::KernelArgPack;
using gpu_model::LaunchConfig;

namespace {

bool DebugEnabled() {
  return std::getenv("GPU_MODEL_HIP_INTERPOSER_DEBUG") != nullptr;
}

void DebugLog(const char* fmt, ...) {
  if (!DebugEnabled()) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  std::fputs("[gpu_model_hip_interposer] ", stderr);
  std::vfprintf(stderr, fmt, args);
  std::fputc('\n', stderr);
  va_end(args);
}

}  // namespace

extern "C" {

void** __hipRegisterFatBinary(const void*) {
  static int token = 0;
  DebugLog("__hipRegisterFatBinary");
  return reinterpret_cast<void**>(&token);
}

void __hipUnregisterFatBinary(void**) {}

void __hipRegisterFunction(void**, const void* hostFunction, char*, const char* deviceName, int,
                           void*, void*, dim3*, dim3*, int*) {
  HipInterposerState::Instance().RegisterFunction(
      hostFunction, deviceName != nullptr ? deviceName : "");
  DebugLog("__hipRegisterFunction host=%p device=%s", hostFunction,
           deviceName != nullptr ? deviceName : "<null>");
}

hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMemBytes,
                                      hipStream_t) {
  HipInterposerState::Instance().PushLaunchConfiguration(
      LaunchConfig{
          .grid_dim_x = gridDim.x,
          .grid_dim_y = gridDim.y,
          .block_dim_x = blockDim.x,
          .block_dim_y = blockDim.y,
      },
      sharedMemBytes);
  DebugLog("__hipPushCallConfiguration grid=(%u,%u,%u) block=(%u,%u,%u) shared=%zu", gridDim.x,
           gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMemBytes);
  return hipSuccess;
}

hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMemBytes,
                                     hipStream_t* stream) {
  const auto config = HipInterposerState::Instance().PopLaunchConfiguration();
  if (!config.has_value()) {
    return hipErrorInvalidValue;
  }
  if (gridDim != nullptr) {
    *gridDim = dim3(config->grid_dim_x, config->grid_dim_y, 1);
  }
  if (blockDim != nullptr) {
    *blockDim = dim3(config->block_dim_x, config->block_dim_y, 1);
  }
  if (sharedMemBytes != nullptr) {
    *sharedMemBytes = config->shared_memory_bytes;
  }
  if (stream != nullptr) {
    *stream = nullptr;
  }
  DebugLog("__hipPopCallConfiguration");
  return hipSuccess;
}

hipError_t hipMalloc(void** devPtr, size_t size) {
  if (devPtr == nullptr) {
    return hipErrorInvalidValue;
  }
  *devPtr = HipInterposerState::Instance().AllocateDevice(size);
  DebugLog("hipMalloc size=%zu -> %p", size, *devPtr);
  return hipSuccess;
}

hipError_t hipFree(void* ptr) {
  return HipInterposerState::Instance().FreeDevice(ptr) ? hipSuccess : hipErrorInvalidValue;
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  auto& state = HipInterposerState::Instance();
  DebugLog("hipMemcpy kind=%d bytes=%zu dst=%p src=%p", static_cast<int>(kind), sizeBytes, dst, src);
  switch (kind) {
    case hipMemcpyHostToDevice:
      state.MemcpyHostToDevice(dst, src, sizeBytes);
      return hipSuccess;
    case hipMemcpyDeviceToHost:
      state.MemcpyDeviceToHost(dst, src, sizeBytes);
      return hipSuccess;
    case hipMemcpyDeviceToDevice:
      state.MemcpyDeviceToDevice(dst, src, sizeBytes);
      return hipSuccess;
    default:
      return hipErrorInvalidValue;
  }
}

hipError_t hipDeviceSynchronize() {
  HipInterposerState::Instance().hooks().DeviceSynchronize();
  DebugLog("hipDeviceSynchronize");
  return hipSuccess;
}

hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks, void** args,
                           size_t sharedMemBytes, hipStream_t) {
  DebugLog("hipLaunchKernel host=%p grid=(%u,%u,%u) block=(%u,%u,%u) shared=%zu", function_address,
           numBlocks.x, numBlocks.y, numBlocks.z, dimBlocks.x, dimBlocks.y, dimBlocks.z,
           sharedMemBytes);
  LaunchConfig config{
      .grid_dim_x = numBlocks.x,
      .grid_dim_y = numBlocks.y,
      .block_dim_x = dimBlocks.x,
      .block_dim_y = dimBlocks.y,
      .shared_memory_bytes = static_cast<uint32_t>(sharedMemBytes),
  };
  const auto result = HipInterposerState::Instance().LaunchExecutableKernel(
      HipInterposerState::CurrentExecutablePath(), function_address, config, args);
  DebugLog("hipLaunchKernel result ok=%d err=%s", result.ok ? 1 : 0,
           result.error_message.c_str());
  return result.ok ? hipSuccess : hipErrorLaunchFailure;
}

}  // extern "C"
