#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>

#include "program/loader/device_image_loader.h"
#include "program/program_object/program_object.h"
#include "runtime/config/launch_request.h"
#include "runtime/model_runtime/compat/launch/runtime_submission_context.h"
#include "utils/config/execution_mode.h"

namespace gpu_model {

class MemorySystem;
class TraceSink;

struct PreparedExecutableLaunch {
  ProgramObject image;
  DeviceLoadResult device_load;
  LaunchRequest request;
};

ProgramObject LoadRegisteredExecutableImage(
    const std::filesystem::path& executable_path,
    const void* host_function,
    const std::function<std::optional<std::string>(const void*)>& resolve_kernel_symbol);

DeviceLoadPlan BuildRegisteredExecutableLoadPlan(
    const std::filesystem::path& executable_path,
    const void* host_function,
    const std::function<std::optional<std::string>(const void*)>& resolve_kernel_symbol);

PreparedExecutableLaunch PrepareRegisteredExecutableLaunch(
    const std::filesystem::path& executable_path,
    const void* host_function,
    LaunchConfig config,
    void** args,
    ExecutionMode mode,
    const std::string& arch_name,
    TraceSink* trace,
    RuntimeSubmissionContext submission_context,
    uint64_t launch_index,
    FunctionalExecutionMode functional_mode,
    MemorySystem& memory,
    const std::function<std::optional<std::string>(const void*)>& resolve_kernel_symbol,
    const std::function<KernelArgPack(const MetadataBlob&, void**)>& pack_abi_args);

}  // namespace gpu_model
