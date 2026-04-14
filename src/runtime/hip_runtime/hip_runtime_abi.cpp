#include <hip/hip_runtime_api.h>

#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include <loguru.hpp>

#include "debug/trace/artifact_recorder.h"
#include "runtime/hip_runtime/hip_runtime.h"
#include "runtime/model_runtime/runtime_session.h"
#include "utils/logging/log_macros.h"

using gpu_model::LaunchConfig;
using gpu_model::TraceArtifactRecorder;

namespace {

gpu_model::HipRuntime& HipApi();

bool DebugEnabled() {
  return gpu_model::logging::ShouldLog("hip_runtime_abi", loguru::Verbosity_INFO);
}

[[maybe_unused]] const char* ToFunctionalModeName(gpu_model::FunctionalExecutionMode mode) {
  switch (mode) {
    case gpu_model::FunctionalExecutionMode::SingleThreaded:
      return "st";
    case gpu_model::FunctionalExecutionMode::MultiThreaded:
      return "mt";
  }
  return "unknown";
}

gpu_model::ExecutionMode ResolveExecutionModeFromEnv() {
  const char* env = std::getenv("GPU_MODEL_EXECUTION_MODE");
  if (env == nullptr || env[0] == '\0') {
    return gpu_model::ExecutionMode::Functional;
  }

  const std::string_view mode(env);
  if (mode == "cycle") {
    return gpu_model::ExecutionMode::Cycle;
  }
  return gpu_model::ExecutionMode::Functional;
}

TraceArtifactRecorder* ResolveTraceArtifactRecorder() {
  return HipApi().ResolveTraceArtifactRecorderFromEnv();
}

gpu_model::HipRuntime& HipApi() {
  static gpu_model::HipRuntime runtime;
  return runtime;
}

void DebugLog(const char* fmt, ...) {
  if (!DebugEnabled()) {
    return;
  }
  std::array<char, 4096> buffer{};
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer.data(), buffer.size(), fmt, args);
  va_end(args);
  GPU_MODEL_LOG_INFO("hip_runtime_abi", "%s", buffer.data());
}

hipError_t Remember(hipError_t error) {
  HipApi().SetLastError(static_cast<int>(error));
  return error;
}

bool IsValidStream(hipStream_t stream) {
  std::optional<uintptr_t> stream_id;
  if (stream != nullptr) {
    stream_id = reinterpret_cast<uintptr_t>(stream);
  }
  return HipApi().IsValidStream(stream_id);
}

gpu_model::RuntimeSubmissionContext CurrentSubmissionContext() {
  gpu_model::RuntimeSubmissionContext submission_context;
  submission_context.device_id = HipApi().GetDevice();
  if (const auto stream_id = HipApi().active_stream_id();
      stream_id.has_value()) {
    submission_context.stream_id = *stream_id;
  }
  return submission_context;
}

gpu_model::RuntimeSubmissionContext SubmissionContextForStream(hipStream_t stream) {
  auto submission_context = CurrentSubmissionContext();
  if (stream != nullptr) {
    submission_context.stream_id = reinterpret_cast<uintptr_t>(stream);
  }
  return submission_context;
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
  HipApi().RegisterFunction(hostFunction, deviceName != nullptr ? deviceName : "");
  DebugLog("__hipRegisterFunction host=%p device=%s", hostFunction,
           deviceName != nullptr ? deviceName : "<null>");
}

hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMemBytes,
                                      hipStream_t stream) {
  if (!IsValidStream(stream)) {
    return Remember(hipErrorInvalidHandle);
  }
  LaunchConfig config{
      .grid_dim_x = gridDim.x,
      .grid_dim_y = gridDim.y,
      .grid_dim_z = gridDim.z,
      .block_dim_x = blockDim.x,
      .block_dim_y = blockDim.y,
      .block_dim_z = blockDim.z,
      .shared_memory_bytes = static_cast<uint32_t>(sharedMemBytes),
  };
  HipApi().PushLaunchConfiguration(config, sharedMemBytes);
  DebugLog("__hipPushCallConfiguration grid=(%u,%u,%u) block=(%u,%u,%u) shared=%zu", gridDim.x,
           gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMemBytes);
  return Remember(hipSuccess);
}

hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMemBytes,
                                     hipStream_t* stream) {
  const auto config = HipApi().PopLaunchConfiguration();
  if (!config.has_value()) {
    return hipErrorInvalidValue;
  }
  if (gridDim != nullptr) {
    *gridDim = dim3(config->grid_dim_x, config->grid_dim_y, config->grid_dim_z);
  }
  if (blockDim != nullptr) {
    *blockDim = dim3(config->block_dim_x, config->block_dim_y, config->block_dim_z);
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
  *devPtr = HipApi().AllocateDevice(size);
  DebugLog("hipMalloc size=%zu -> %p", size, *devPtr);
  return Remember(hipSuccess);
}

hipError_t hipMallocManaged(void** devPtr, size_t size, unsigned int flags) {
  if (devPtr == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  *devPtr = HipApi().AllocateManaged(size);
  DebugLog("hipMallocManaged size=%zu flags=%u -> %p", size, flags, *devPtr);
  return Remember(hipSuccess);
}

hipError_t hipFree(void* ptr) {
  return Remember(HipApi().FreeDevice(ptr) ? hipSuccess : hipErrorInvalidValue);
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  DebugLog("hipMemcpy kind=%d bytes=%zu dst=%p src=%p", static_cast<int>(kind), sizeBytes, dst, src);
  try {
    switch (kind) {
      case hipMemcpyHostToDevice:
        HipApi().MemcpyHostToDevice(dst, src, sizeBytes);
        return Remember(hipSuccess);
      case hipMemcpyDeviceToHost:
        HipApi().MemcpyDeviceToHost(dst, src, sizeBytes);
        return Remember(hipSuccess);
      case hipMemcpyDeviceToDevice:
        HipApi().MemcpyDeviceToDevice(dst, src, sizeBytes);
        return Remember(hipSuccess);
      default:
        return Remember(hipErrorInvalidValue);
    }
  } catch (const std::invalid_argument&) {
    return Remember(hipErrorInvalidValue);
  }
}

hipError_t hipMemcpyAsync(void* dst,
                          const void* src,
                          size_t sizeBytes,
                          hipMemcpyKind kind,
                          hipStream_t stream) {
  if (!IsValidStream(stream)) {
    return Remember(hipErrorInvalidHandle);
  }
  return hipMemcpy(dst, src, sizeBytes, kind);
}

hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  if (!HipApi().IsDevicePointer(dst)) {
    return Remember(hipErrorInvalidValue);
  }
  HipApi().MemsetDevice(dst, static_cast<uint8_t>(value), sizeBytes);
  DebugLog("hipMemset dst=%p value=%d bytes=%zu", dst, value, sizeBytes);
  return Remember(hipSuccess);
}

hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  if (!IsValidStream(stream)) {
    return Remember(hipErrorInvalidHandle);
  }
  return hipMemset(dst, value, sizeBytes);
}

hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count) {
  void* ptr = reinterpret_cast<void*>(dest);
  if (!HipApi().IsDevicePointer(ptr)) {
    return Remember(hipErrorInvalidValue);
  }
  HipApi().MemsetDevice(ptr, value, count);
  DebugLog("hipMemsetD8 dst=%p value=%u count=%zu", ptr, static_cast<unsigned>(value), count);
  return Remember(hipSuccess);
}

hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count) {
  void* ptr = reinterpret_cast<void*>(dest);
  if (!HipApi().IsDevicePointer(ptr)) {
    return Remember(hipErrorInvalidValue);
  }
  HipApi().MemsetDeviceD16(ptr, static_cast<uint16_t>(value), count);
  DebugLog("hipMemsetD16 dst=%p value=%u count=%zu", ptr, static_cast<unsigned>(value), count);
  return Remember(hipSuccess);
}

hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count) {
  void* ptr = reinterpret_cast<void*>(dest);
  if (!HipApi().IsDevicePointer(ptr)) {
    return Remember(hipErrorInvalidValue);
  }
  HipApi().MemsetDeviceD32(ptr, static_cast<uint32_t>(value), count);
  DebugLog("hipMemsetD32 dst=%p value=%d count=%zu", ptr, value, count);
  return Remember(hipSuccess);
}

hipError_t hipDeviceSynchronize() {
  HipApi().DeviceSynchronize();
  HipApi().SyncManagedDeviceToHost();
  DebugLog("hipDeviceSynchronize");
  return Remember(hipSuccess);
}

hipError_t hipGetDeviceCount(int* count) {
  if (count == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  *count = HipApi().GetDeviceCount();
  return Remember(hipSuccess);
}

hipError_t hipGetDevice(int* deviceId) {
  if (deviceId == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  *deviceId = HipApi().GetDevice();
  return Remember(hipSuccess);
}

hipError_t hipSetDevice(int deviceId) {
  if (!HipApi().SetDevice(deviceId)) {
    return Remember(hipErrorInvalidDevice);
  }
  return Remember(hipSuccess);
}

hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600* prop, int deviceId) {
  if (prop == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  if (deviceId < 0 || deviceId >= HipApi().GetDeviceCount()) {
    return Remember(hipErrorInvalidDevice);
  }
  const auto props = HipApi().GetDeviceProperties(deviceId);
  std::memset(prop, 0, sizeof(*prop));
  std::snprintf(prop->name, sizeof(prop->name), "%s", props.name.c_str());
  prop->totalGlobalMem = props.total_global_mem;
  prop->sharedMemPerBlock = props.shared_mem_per_block;
  prop->regsPerBlock = props.regs_per_block;
  prop->warpSize = props.warp_size;
  prop->maxThreadsPerBlock = props.max_threads_per_block;
  prop->maxThreadsDim[0] = props.max_threads_dim[0];
  prop->maxThreadsDim[1] = props.max_threads_dim[1];
  prop->maxThreadsDim[2] = props.max_threads_dim[2];
  prop->maxGridSize[0] = props.max_grid_size[0];
  prop->maxGridSize[1] = props.max_grid_size[1];
  prop->maxGridSize[2] = props.max_grid_size[2];
  prop->clockRate = props.clock_rate_khz;
  prop->totalConstMem = props.total_const_mem;
  prop->major = props.compute_capability_major;
  prop->minor = props.compute_capability_minor;
  prop->multiProcessorCount = props.multi_processor_count;
  prop->integrated = props.integrated;
  prop->canMapHostMemory = props.can_map_host_memory;
  prop->concurrentKernels = props.concurrent_kernels;
  prop->asyncEngineCount = props.async_engine_count;
  prop->unifiedAddressing = props.unified_addressing;
  prop->memoryClockRate = props.memory_clock_rate_khz;
  prop->memoryBusWidth = props.memory_bus_width_bits;
  prop->l2CacheSize = props.l2_cache_size;
  prop->maxThreadsPerMultiProcessor = props.max_threads_per_multiprocessor;
  prop->sharedMemPerMultiprocessor = props.shared_mem_per_multiprocessor;
  prop->regsPerMultiprocessor = props.regs_per_multiprocessor;
  prop->managedMemory = props.managed_memory;
  prop->concurrentManagedAccess = props.concurrent_managed_access;
  prop->cooperativeLaunch = props.cooperative_launch;
  prop->hostRegisterSupported = props.host_register_supported;
  std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "%s", "mac500");
  prop->maxSharedMemoryPerMultiProcessor = props.max_shared_mem_per_multiprocessor;
  prop->pciBusID = props.pci_bus_id;
  prop->pciDeviceID = props.pci_device_id;
  prop->pciDomainID = props.pci_domain_id;
  return Remember(hipSuccess);
}

hipError_t hipDeviceGetAttribute(int* value, hipDeviceAttribute_t attr, int deviceId) {
  if (value == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  if (deviceId < 0 || deviceId >= HipApi().GetDeviceCount()) {
    return Remember(hipErrorInvalidDevice);
  }
  using A = gpu_model::RuntimeDeviceAttribute;
  std::optional<int> resolved;
  switch (attr) {
    case hipDeviceAttributeWarpSize:
      resolved = HipApi().GetDeviceAttribute(A::WarpSize, deviceId);
      break;
    case hipDeviceAttributeMaxThreadsPerBlock:
      resolved = HipApi().GetDeviceAttribute(A::MaxThreadsPerBlock, deviceId);
      break;
    case hipDeviceAttributeMaxBlockDimX:
      resolved = HipApi().GetDeviceAttribute(A::MaxBlockDimX, deviceId);
      break;
    case hipDeviceAttributeMaxBlockDimY:
      resolved = HipApi().GetDeviceAttribute(A::MaxBlockDimY, deviceId);
      break;
    case hipDeviceAttributeMaxBlockDimZ:
      resolved = HipApi().GetDeviceAttribute(A::MaxBlockDimZ, deviceId);
      break;
    case hipDeviceAttributeMaxGridDimX:
      resolved = HipApi().GetDeviceAttribute(A::MaxGridDimX, deviceId);
      break;
    case hipDeviceAttributeMaxGridDimY:
      resolved = HipApi().GetDeviceAttribute(A::MaxGridDimY, deviceId);
      break;
    case hipDeviceAttributeMaxGridDimZ:
      resolved = HipApi().GetDeviceAttribute(A::MaxGridDimZ, deviceId);
      break;
    case hipDeviceAttributeMultiprocessorCount:
      resolved = HipApi().GetDeviceAttribute(A::MultiprocessorCount, deviceId);
      break;
    case hipDeviceAttributeMaxThreadsPerMultiProcessor:
      resolved = HipApi().GetDeviceAttribute(A::MaxThreadsPerMultiprocessor, deviceId);
      break;
    case hipDeviceAttributeMaxSharedMemoryPerBlock:
      resolved = HipApi().GetDeviceAttribute(A::SharedMemPerBlock, deviceId);
      break;
    case hipDeviceAttributeSharedMemPerMultiprocessor:
      resolved = HipApi().GetDeviceAttribute(A::SharedMemPerMultiprocessor, deviceId);
      break;
    case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:
      resolved = HipApi().GetDeviceAttribute(A::MaxSharedMemPerMultiprocessor, deviceId);
      break;
    case hipDeviceAttributeMaxRegistersPerBlock:
      resolved = HipApi().GetDeviceAttribute(A::RegistersPerBlock, deviceId);
      break;
    case hipDeviceAttributeMaxRegistersPerMultiprocessor:
      resolved = HipApi().GetDeviceAttribute(A::RegistersPerMultiprocessor, deviceId);
      break;
    case hipDeviceAttributeTotalConstantMemory:
      resolved = HipApi().GetDeviceAttribute(A::TotalConstantMemory, deviceId);
      break;
    case hipDeviceAttributeL2CacheSize:
      resolved = HipApi().GetDeviceAttribute(A::L2CacheSize, deviceId);
      break;
    case hipDeviceAttributeClockRate:
      resolved = HipApi().GetDeviceAttribute(A::ClockRateKHz, deviceId);
      break;
    case hipDeviceAttributeMemoryClockRate:
      resolved = HipApi().GetDeviceAttribute(A::MemoryClockRateKHz, deviceId);
      break;
    case hipDeviceAttributeMemoryBusWidth:
      resolved = HipApi().GetDeviceAttribute(A::MemoryBusWidthBits, deviceId);
      break;
    case hipDeviceAttributeIntegrated:
      resolved = HipApi().GetDeviceAttribute(A::Integrated, deviceId);
      break;
    case hipDeviceAttributeConcurrentKernels:
      resolved = HipApi().GetDeviceAttribute(A::ConcurrentKernels, deviceId);
      break;
    case hipDeviceAttributeCooperativeLaunch:
      resolved = HipApi().GetDeviceAttribute(A::CooperativeLaunch, deviceId);
      break;
    case hipDeviceAttributeCanMapHostMemory:
      resolved = HipApi().GetDeviceAttribute(A::CanMapHostMemory, deviceId);
      break;
    case hipDeviceAttributeManagedMemory:
      resolved = HipApi().GetDeviceAttribute(A::ManagedMemory, deviceId);
      break;
    case hipDeviceAttributeConcurrentManagedAccess:
      resolved = HipApi().GetDeviceAttribute(A::ConcurrentManagedAccess, deviceId);
      break;
    case hipDeviceAttributeHostRegisterSupported:
      resolved = HipApi().GetDeviceAttribute(A::HostRegisterSupported, deviceId);
      break;
    case hipDeviceAttributeUnifiedAddressing:
      resolved = HipApi().GetDeviceAttribute(A::UnifiedAddressing, deviceId);
      break;
    case hipDeviceAttributeComputeCapabilityMajor:
      resolved = HipApi().GetDeviceAttribute(A::ComputeCapabilityMajor, deviceId);
      break;
    case hipDeviceAttributeComputeCapabilityMinor:
      resolved = HipApi().GetDeviceAttribute(A::ComputeCapabilityMinor, deviceId);
      break;
    default:
      return Remember(hipErrorInvalidValue);
  }
  if (!resolved.has_value()) {
    return Remember(hipErrorInvalidValue);
  }
  *value = *resolved;
  return Remember(hipSuccess);
}

hipError_t hipGetLastError() {
  return static_cast<hipError_t>(HipApi().ConsumeLastError());
}

hipError_t hipPeekAtLastError() {
  return static_cast<hipError_t>(HipApi().PeekLastError());
}

hipError_t hipStreamCreate(hipStream_t* stream) {
  if (stream == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  const auto stream_id = HipApi().CreateStream();
  if (!stream_id.has_value()) {
    return Remember(hipErrorInvalidValue);
  }
  *stream = reinterpret_cast<hipStream_t>(*stream_id);
  return Remember(hipSuccess);
}

hipError_t hipStreamDestroy(hipStream_t stream) {
  if (stream == nullptr ||
      !HipApi().DestroyStream(reinterpret_cast<uintptr_t>(stream))) {
    return Remember(hipErrorInvalidHandle);
  }
  return Remember(hipSuccess);
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  if (!IsValidStream(stream)) {
    return Remember(hipErrorInvalidHandle);
  }
  HipApi().StreamSynchronizeCompatibility(SubmissionContextForStream(stream));
  return Remember(hipSuccess);
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int) {
  if (!IsValidStream(stream)) {
    return Remember(hipErrorInvalidHandle);
  }
  if (!HipApi().HasEvent(reinterpret_cast<uintptr_t>(event))) {
    return Remember(hipErrorInvalidHandle);
  }
  return Remember(hipSuccess);
}

hipError_t hipEventCreate(hipEvent_t* event) {
  if (event == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  *event = reinterpret_cast<hipEvent_t>(HipApi().CreateEvent());
  return Remember(hipSuccess);
}

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned) {
  return hipEventCreate(event);
}

hipError_t hipEventDestroy(hipEvent_t event) {
  if (!HipApi().DestroyEvent(reinterpret_cast<uintptr_t>(event))) {
    return Remember(hipErrorInvalidHandle);
  }
  return Remember(hipSuccess);
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  if (!IsValidStream(stream)) {
    return Remember(hipErrorInvalidHandle);
  }
  std::optional<uintptr_t> stream_id;
  if (stream != nullptr) {
    stream_id = reinterpret_cast<uintptr_t>(stream);
  }
  if (!HipApi().RecordEvent(reinterpret_cast<uintptr_t>(event), stream_id)) {
    return Remember(hipErrorInvalidHandle);
  }
  return Remember(hipSuccess);
}

hipError_t hipEventRecordWithFlags(hipEvent_t event, hipStream_t stream, unsigned) {
  return hipEventRecord(event, stream);
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  if (!HipApi().HasEvent(reinterpret_cast<uintptr_t>(event))) {
    return Remember(hipErrorInvalidHandle);
  }
  return Remember(hipSuccess);
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
  if (ms == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  if (!HipApi().HasEvent(reinterpret_cast<uintptr_t>(start)) ||
      !HipApi().HasEvent(reinterpret_cast<uintptr_t>(stop))) {
    return Remember(hipErrorInvalidHandle);
  }
  *ms = 0.0f;
  return Remember(hipSuccess);
}

hipError_t hipLaunchKernel(const void* function_address,
                           dim3 numBlocks,
                           dim3 dimBlocks,
                           void** args,
                           size_t sharedMemBytes,
                           hipStream_t stream) {
  if (!IsValidStream(stream)) {
    return Remember(hipErrorInvalidHandle);
  }
  DebugLog("hipLaunchKernel host=%p grid=(%u,%u,%u) block=(%u,%u,%u) shared=%zu", function_address,
           numBlocks.x, numBlocks.y, numBlocks.z, dimBlocks.x, dimBlocks.y, dimBlocks.z,
           sharedMemBytes);
  LaunchConfig config{
      .grid_dim_x = numBlocks.x,
      .grid_dim_y = numBlocks.y,
      .grid_dim_z = numBlocks.z,
      .block_dim_x = dimBlocks.x,
      .block_dim_y = dimBlocks.y,
      .block_dim_z = dimBlocks.z,
      .shared_memory_bytes = static_cast<uint32_t>(sharedMemBytes),
  };
  const auto execution_mode = ResolveExecutionModeFromEnv();
  auto* trace = ResolveTraceArtifactRecorder();
  const auto result = HipApi().LaunchExecutableKernel(
      gpu_model::HipRuntime::CurrentExecutablePath(),
      function_address,
      config,
      args,
      execution_mode,
      "mac500",
      trace,
      CurrentSubmissionContext());
  if (trace != nullptr) {
    trace->FlushTimeline();
  }
  DebugLog("hipLaunchKernel result ok=%d err=%s", result.ok ? 1 : 0,
           result.error_message.c_str());
  return Remember(result.ok ? hipSuccess : hipErrorLaunchFailure);
}

}  // extern "C"
