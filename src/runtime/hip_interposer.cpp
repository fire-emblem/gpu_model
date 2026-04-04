#include <hip/hip_runtime_api.h>

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
#include <string>
#include <string_view>
#include <unordered_map>

#include "gpu_model/debug/trace/artifact_recorder.h"
#include "gpu_model/runtime/hip_interposer_state.h"

using gpu_model::HipInterposerState;
using gpu_model::KernelArgPack;
using gpu_model::LaunchConfig;
using gpu_model::TraceArtifactRecorder;

namespace {

thread_local hipError_t g_last_error = hipSuccess;
thread_local int g_current_device = 0;
thread_local std::optional<uintptr_t> g_active_stream_id;
uintptr_t g_next_event_id = 1;

struct EventState {
  bool recorded = false;
  hipStream_t stream = nullptr;
};

std::unordered_map<uintptr_t, EventState> g_events;
std::unique_ptr<TraceArtifactRecorder> g_trace_artifacts;
std::string g_trace_artifacts_dir;
uint64_t g_launch_index = 0;

bool DebugEnabled() {
  return std::getenv("GPU_MODEL_HIP_INTERPOSER_DEBUG") != nullptr;
}

const char* ToFunctionalModeName(gpu_model::FunctionalExecutionMode mode) {
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
  const char* env = std::getenv("GPU_MODEL_TRACE_DIR");
  if (env == nullptr || env[0] == '\0') {
    g_trace_artifacts.reset();
    g_trace_artifacts_dir.clear();
    return nullptr;
  }

  if (!g_trace_artifacts || g_trace_artifacts_dir != env) {
    g_trace_artifacts_dir = env;
    g_trace_artifacts = std::make_unique<TraceArtifactRecorder>(g_trace_artifacts_dir);
  }
  return g_trace_artifacts.get();
}

void AppendLaunchSummary(const std::filesystem::path& output_dir,
                         const std::string& kernel_name,
                         gpu_model::ExecutionMode execution_mode,
                         gpu_model::FunctionalExecutionMode functional_mode,
                         const gpu_model::LaunchResult& result) {
  std::ofstream out(output_dir / "launch_summary.txt", std::ios::app);
  if (!out) {
    throw std::runtime_error("failed to open launch summary file");
  }

  out << "launch_index=" << g_launch_index++
      << " kernel=" << kernel_name
      << " execution_mode="
      << (execution_mode == gpu_model::ExecutionMode::Cycle ? "cycle" : "functional")
      << " functional_mode=" << ToFunctionalModeName(functional_mode)
      << " ok=" << (result.ok ? 1 : 0)
      << " submit_cycle=" << result.submit_cycle
      << " begin_cycle=" << result.begin_cycle
      << " end_cycle=" << result.end_cycle
      << " total_cycles=" << result.total_cycles
      << " program_total_cycles=";
  if (result.program_cycle_stats.has_value()) {
    out << result.program_cycle_stats->total_cycles;
  } else {
    out << "na";
  }
  out << '\n';
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

bool IsValidStream(hipStream_t stream) {
  if (stream == nullptr) {
    return true;
  }
  return g_active_stream_id.has_value() &&
         reinterpret_cast<uintptr_t>(stream) == *g_active_stream_id;
}

gpu_model::RuntimeSubmissionContext CurrentSubmissionContext() {
  gpu_model::RuntimeSubmissionContext submission_context;
  submission_context.device_id = g_current_device;
  if (g_active_stream_id.has_value()) {
    submission_context.stream_id = *g_active_stream_id;
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
  HipInterposerState::Instance().RegisterFunction(
      hostFunction, deviceName != nullptr ? deviceName : "");
  DebugLog("__hipRegisterFunction host=%p device=%s", hostFunction,
           deviceName != nullptr ? deviceName : "<null>");
}

hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMemBytes,
                                      hipStream_t stream) {
  if (!IsValidStream(stream)) {
    return Remember(hipErrorInvalidHandle);
  }
  HipInterposerState::Instance().PushLaunchConfiguration(
      LaunchConfig{
          .grid_dim_x = gridDim.x,
          .grid_dim_y = gridDim.y,
          .grid_dim_z = gridDim.z,
          .block_dim_x = blockDim.x,
          .block_dim_y = blockDim.y,
          .block_dim_z = blockDim.z,
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
                          hipStream_t stream) {
  if (!IsValidStream(stream)) {
    return Remember(hipErrorInvalidHandle);
  }
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

hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  if (!IsValidStream(stream)) {
    return Remember(hipErrorInvalidHandle);
  }
  return hipMemset(dst, value, sizeBytes);
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
  *count = HipInterposerState::Instance().model_runtime().GetDeviceCount();
  return Remember(hipSuccess);
}

hipError_t hipGetDevice(int* deviceId) {
  if (deviceId == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  *deviceId = HipInterposerState::Instance().model_runtime().GetDevice();
  return Remember(hipSuccess);
}

hipError_t hipSetDevice(int deviceId) {
  if (!HipInterposerState::Instance().model_runtime().SetDevice(deviceId)) {
    return Remember(hipErrorInvalidDevice);
  }
  g_current_device = deviceId;
  return Remember(hipSuccess);
}

hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600* prop, int deviceId) {
  if (prop == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  auto& hooks = HipInterposerState::Instance().hooks();
  if (deviceId < 0 || deviceId >= hooks.GetDeviceCount()) {
    return Remember(hipErrorInvalidDevice);
  }
  const auto props = hooks.GetDeviceProperties(deviceId);
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
  std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "%s", "c500");
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
  auto& hooks = HipInterposerState::Instance().hooks();
  if (deviceId < 0 || deviceId >= hooks.GetDeviceCount()) {
    return Remember(hipErrorInvalidDevice);
  }
  using A = gpu_model::RuntimeDeviceAttribute;
  std::optional<int> resolved;
  switch (attr) {
    case hipDeviceAttributeWarpSize:
      resolved = hooks.GetDeviceAttribute(A::WarpSize, deviceId);
      break;
    case hipDeviceAttributeMaxThreadsPerBlock:
      resolved = hooks.GetDeviceAttribute(A::MaxThreadsPerBlock, deviceId);
      break;
    case hipDeviceAttributeMaxBlockDimX:
      resolved = hooks.GetDeviceAttribute(A::MaxBlockDimX, deviceId);
      break;
    case hipDeviceAttributeMaxBlockDimY:
      resolved = hooks.GetDeviceAttribute(A::MaxBlockDimY, deviceId);
      break;
    case hipDeviceAttributeMaxBlockDimZ:
      resolved = hooks.GetDeviceAttribute(A::MaxBlockDimZ, deviceId);
      break;
    case hipDeviceAttributeMaxGridDimX:
      resolved = hooks.GetDeviceAttribute(A::MaxGridDimX, deviceId);
      break;
    case hipDeviceAttributeMaxGridDimY:
      resolved = hooks.GetDeviceAttribute(A::MaxGridDimY, deviceId);
      break;
    case hipDeviceAttributeMaxGridDimZ:
      resolved = hooks.GetDeviceAttribute(A::MaxGridDimZ, deviceId);
      break;
    case hipDeviceAttributeMultiprocessorCount:
      resolved = hooks.GetDeviceAttribute(A::MultiprocessorCount, deviceId);
      break;
    case hipDeviceAttributeMaxThreadsPerMultiProcessor:
      resolved = hooks.GetDeviceAttribute(A::MaxThreadsPerMultiprocessor, deviceId);
      break;
    case hipDeviceAttributeMaxSharedMemoryPerBlock:
      resolved = hooks.GetDeviceAttribute(A::SharedMemPerBlock, deviceId);
      break;
    case hipDeviceAttributeSharedMemPerMultiprocessor:
      resolved = hooks.GetDeviceAttribute(A::SharedMemPerMultiprocessor, deviceId);
      break;
    case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:
      resolved = hooks.GetDeviceAttribute(A::MaxSharedMemPerMultiprocessor, deviceId);
      break;
    case hipDeviceAttributeMaxRegistersPerBlock:
      resolved = hooks.GetDeviceAttribute(A::RegistersPerBlock, deviceId);
      break;
    case hipDeviceAttributeMaxRegistersPerMultiprocessor:
      resolved = hooks.GetDeviceAttribute(A::RegistersPerMultiprocessor, deviceId);
      break;
    case hipDeviceAttributeTotalConstantMemory:
      resolved = hooks.GetDeviceAttribute(A::TotalConstantMemory, deviceId);
      break;
    case hipDeviceAttributeL2CacheSize:
      resolved = hooks.GetDeviceAttribute(A::L2CacheSize, deviceId);
      break;
    case hipDeviceAttributeClockRate:
      resolved = hooks.GetDeviceAttribute(A::ClockRateKHz, deviceId);
      break;
    case hipDeviceAttributeMemoryClockRate:
      resolved = hooks.GetDeviceAttribute(A::MemoryClockRateKHz, deviceId);
      break;
    case hipDeviceAttributeMemoryBusWidth:
      resolved = hooks.GetDeviceAttribute(A::MemoryBusWidthBits, deviceId);
      break;
    case hipDeviceAttributeIntegrated:
      resolved = hooks.GetDeviceAttribute(A::Integrated, deviceId);
      break;
    case hipDeviceAttributeConcurrentKernels:
      resolved = hooks.GetDeviceAttribute(A::ConcurrentKernels, deviceId);
      break;
    case hipDeviceAttributeCooperativeLaunch:
      resolved = hooks.GetDeviceAttribute(A::CooperativeLaunch, deviceId);
      break;
    case hipDeviceAttributeCanMapHostMemory:
      resolved = hooks.GetDeviceAttribute(A::CanMapHostMemory, deviceId);
      break;
    case hipDeviceAttributeManagedMemory:
      resolved = hooks.GetDeviceAttribute(A::ManagedMemory, deviceId);
      break;
    case hipDeviceAttributeConcurrentManagedAccess:
      resolved = hooks.GetDeviceAttribute(A::ConcurrentManagedAccess, deviceId);
      break;
    case hipDeviceAttributeHostRegisterSupported:
      resolved = hooks.GetDeviceAttribute(A::HostRegisterSupported, deviceId);
      break;
    case hipDeviceAttributeUnifiedAddressing:
      resolved = hooks.GetDeviceAttribute(A::UnifiedAddressing, deviceId);
      break;
    case hipDeviceAttributeComputeCapabilityMajor:
      resolved = hooks.GetDeviceAttribute(A::ComputeCapabilityMajor, deviceId);
      break;
    case hipDeviceAttributeComputeCapabilityMinor:
      resolved = hooks.GetDeviceAttribute(A::ComputeCapabilityMinor, deviceId);
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
  if (g_active_stream_id.has_value()) {
    return Remember(hipErrorInvalidValue);
  }
  static constexpr uintptr_t kSingleStreamId =
      static_cast<uintptr_t>(std::numeric_limits<uint32_t>::max());
  g_active_stream_id = kSingleStreamId;
  *stream = reinterpret_cast<hipStream_t>(*g_active_stream_id);
  return Remember(hipSuccess);
}

hipError_t hipStreamDestroy(hipStream_t stream) {
  if (!g_active_stream_id.has_value() ||
      stream == nullptr ||
      reinterpret_cast<uintptr_t>(stream) != *g_active_stream_id) {
    return Remember(hipErrorInvalidHandle);
  }
  g_active_stream_id.reset();
  return Remember(hipSuccess);
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  if (!IsValidStream(stream)) {
    return Remember(hipErrorInvalidHandle);
  }
  HipInterposerState::Instance().hooks().StreamSynchronize(SubmissionContextForStream(stream));
  return Remember(hipSuccess);
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int) {
  if (!IsValidStream(stream)) {
    return Remember(hipErrorInvalidHandle);
  }
  const auto it = g_events.find(reinterpret_cast<uintptr_t>(event));
  if (it == g_events.end()) {
    return Remember(hipErrorInvalidHandle);
  }
  return Remember(hipSuccess);
}

hipError_t hipEventCreate(hipEvent_t* event) {
  if (event == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  const uintptr_t id = g_next_event_id++;
  g_events[id] = EventState{};
  *event = reinterpret_cast<hipEvent_t>(id);
  return Remember(hipSuccess);
}

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned) {
  return hipEventCreate(event);
}

hipError_t hipEventDestroy(hipEvent_t event) {
  const auto it = g_events.find(reinterpret_cast<uintptr_t>(event));
  if (it == g_events.end()) {
    return Remember(hipErrorInvalidHandle);
  }
  g_events.erase(it);
  return Remember(hipSuccess);
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  if (!IsValidStream(stream)) {
    return Remember(hipErrorInvalidHandle);
  }
  const auto it = g_events.find(reinterpret_cast<uintptr_t>(event));
  if (it == g_events.end()) {
    return Remember(hipErrorInvalidHandle);
  }
  it->second.recorded = true;
  it->second.stream = stream;
  return Remember(hipSuccess);
}

hipError_t hipEventRecordWithFlags(hipEvent_t event, hipStream_t stream, unsigned) {
  return hipEventRecord(event, stream);
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  const auto it = g_events.find(reinterpret_cast<uintptr_t>(event));
  if (it == g_events.end()) {
    return Remember(hipErrorInvalidHandle);
  }
  return Remember(hipSuccess);
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
  if (ms == nullptr) {
    return Remember(hipErrorInvalidValue);
  }
  const auto start_it = g_events.find(reinterpret_cast<uintptr_t>(start));
  const auto stop_it = g_events.find(reinterpret_cast<uintptr_t>(stop));
  if (start_it == g_events.end() || stop_it == g_events.end()) {
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
  auto& state = HipInterposerState::Instance();
  const auto kernel_name = state.ResolveKernelName(function_address);
  const auto result = state.LaunchExecutableKernel(HipInterposerState::CurrentExecutablePath(),
                                                   function_address,
                                                   config,
                                                   args,
                                                   execution_mode,
                                                   "c500",
                                                   trace,
                                                   CurrentSubmissionContext());
  if (trace != nullptr) {
    trace->FlushTimeline();
    AppendLaunchSummary(trace->output_dir(),
                        kernel_name.value_or("<unregistered>"),
                        execution_mode,
                        state.model_runtime().runtime().functional_execution_config().mode,
                        result);
  }
  DebugLog("hipLaunchKernel result ok=%d err=%s", result.ok ? 1 : 0,
           result.error_message.c_str());
  return Remember(result.ok ? hipSuccess : hipErrorLaunchFailure);
}

}  // extern "C"
