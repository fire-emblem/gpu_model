#include <hip/hip_runtime_api.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "gpu_model/runtime/hip_interposer_state.h"

using gpu_model::HipInterposerState;
using gpu_model::KernelArgPack;
using gpu_model::LaunchConfig;

namespace {

thread_local hipError_t g_last_error = hipSuccess;
thread_local int g_current_device = 0;
uintptr_t g_next_stream_id = 1;

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

hipError_t Remember(hipError_t error) {
  g_last_error = error;
  return error;
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
  return Remember(hipSuccess);
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
  return Remember(hipSuccess);
}

hipError_t hipMalloc(void** devPtr, size_t size) {
  if (devPtr == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  *devPtr = HipInterposerState::Instance().AllocateDevice(size);
  DebugLog("hipMalloc size=%zu -> %p", size, *devPtr);
  return Remember(hipSuccess);
}

hipError_t hipMallocManaged(void** devPtr, size_t size, unsigned int flags) {
  if (devPtr == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  *devPtr = HipInterposerState::Instance().AllocateManaged(size);
  DebugLog("hipMallocManaged size=%zu flags=%u -> %p", size, flags, *devPtr);
  return Remember(hipSuccess);
}

hipError_t hipFree(void* ptr) {
  return Remember(HipInterposerState::Instance().FreeDevice(ptr) ? hipSuccess : hipErrorInvalidValue);
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  auto& state = HipInterposerState::Instance();
  DebugLog("hipMemcpy kind=%d bytes=%zu dst=%p src=%p", static_cast<int>(kind), sizeBytes, dst, src);
  switch (kind) {
    case hipMemcpyHostToDevice:
      state.MemcpyHostToDevice(dst, src, sizeBytes);
      return Remember(hipSuccess);
    case hipMemcpyDeviceToHost:
      state.MemcpyDeviceToHost(dst, src, sizeBytes);
      return Remember(hipSuccess);
    case hipMemcpyDeviceToDevice:
      state.MemcpyDeviceToDevice(dst, src, sizeBytes);
      return Remember(hipSuccess);
    default:
      return Remember(hipErrorInvalidValue);
  }
}

hipError_t hipMemcpyAsync(void* dst,
                          const void* src,
                          size_t sizeBytes,
                          hipMemcpyKind kind,
                          hipStream_t) {
  return hipMemcpy(dst, src, sizeBytes, kind);
}

hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  auto& state = HipInterposerState::Instance();
  if (!state.IsDevicePointer(dst)) {
    return Remember(hipErrorInvalidValue);
  }
  state.MemsetDevice(dst, static_cast<uint8_t>(value), sizeBytes);
  DebugLog("hipMemset dst=%p value=%d bytes=%zu", dst, value, sizeBytes);
  return Remember(hipSuccess);
}

hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count) {
  auto& state = HipInterposerState::Instance();
  void* ptr = reinterpret_cast<void*>(dest);
  if (!state.IsDevicePointer(ptr)) {
    return Remember(hipErrorInvalidValue);
  }
  state.MemsetDevice(ptr, value, count);
  DebugLog("hipMemsetD8 dst=%p value=%u count=%zu", ptr, static_cast<unsigned>(value), count);
  return Remember(hipSuccess);
}

hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count) {
  auto& state = HipInterposerState::Instance();
  void* ptr = reinterpret_cast<void*>(dest);
  if (!state.IsDevicePointer(ptr)) {
    return Remember(hipErrorInvalidValue);
  }
  state.MemsetDeviceD32(ptr, static_cast<uint32_t>(value), count);
  DebugLog("hipMemsetD32 dst=%p value=%d count=%zu", ptr, value, count);
  return Remember(hipSuccess);
}

hipError_t hipDeviceSynchronize() {
  HipInterposerState::Instance().hooks().DeviceSynchronize();
  HipInterposerState::Instance().SyncManagedDeviceToHost();
  DebugLog("hipDeviceSynchronize");
  return Remember(hipSuccess);
}

hipError_t hipGetDeviceCount(int* count) {
  if (count == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  *count = 1;
  return Remember(hipSuccess);
}

hipError_t hipGetDevice(int* deviceId) {
  if (deviceId == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  *deviceId = g_current_device;
  return Remember(hipSuccess);
}

hipError_t hipSetDevice(int deviceId) {
  if (deviceId != 0) {
    return Remember(hipErrorInvalidDevice);
  }
  g_current_device = deviceId;
  return Remember(hipSuccess);
}

hipError_t hipGetLastError() {
  const hipError_t error = g_last_error;
  g_last_error = hipSuccess;
  return error;
}

hipError_t hipPeekAtLastError() {
  return g_last_error;
}

hipError_t hipStreamCreate(hipStream_t* stream) {
  if (stream == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  *stream = reinterpret_cast<hipStream_t>(g_next_stream_id++);
  return Remember(hipSuccess);
}

hipError_t hipStreamDestroy(hipStream_t) {
  return Remember(hipSuccess);
}

hipError_t hipStreamSynchronize(hipStream_t) {
  return Remember(hipSuccess);
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
  return Remember(result.ok ? hipSuccess : hipErrorLaunchFailure);
}

}  // extern "C"
