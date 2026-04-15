#include "runtime/model_runtime/compat/runtime_executable_launch_helper.h"

#include <stdexcept>

#include "state/memory/memory_system.h"
#include "program/loader/device_image_loader.h"
#include "program/loader/device_segment_image.h"
#include "program/program_object/object_reader.h"

namespace gpu_model {

namespace {

std::string ResolveRegisteredKernelName(
    const void* host_function,
    const std::function<std::optional<std::string>(const void*)>& resolve_kernel_symbol) {
  const auto kernel_name = resolve_kernel_symbol(host_function);
  if (!kernel_name.has_value()) {
    throw std::invalid_argument("unregistered HIP host function");
  }
  return *kernel_name;
}

}  // namespace

ProgramObject LoadRegisteredExecutableImage(
    const std::filesystem::path& executable_path,
    const void* host_function,
    const std::function<std::optional<std::string>(const void*)>& resolve_kernel_symbol) {
  return ObjectReader{}.LoadProgramObject(
      executable_path, ResolveRegisteredKernelName(host_function, resolve_kernel_symbol));
}

DeviceLoadPlan BuildRegisteredExecutableLoadPlan(
    const std::filesystem::path& executable_path,
    const void* host_function,
    const std::function<std::optional<std::string>(const void*)>& resolve_kernel_symbol) {
  return BuildDeviceLoadPlan(
      LoadRegisteredExecutableImage(executable_path, host_function, resolve_kernel_symbol));
}

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
    const std::function<KernelArgPack(const MetadataBlob&, void**)>& pack_abi_args) {
  PreparedExecutableLaunch prepared;
  prepared.image =
      LoadRegisteredExecutableImage(executable_path, host_function, resolve_kernel_symbol);
  prepared.device_load = DeviceImageLoader{}.Materialize(BuildDeviceLoadPlan(prepared.image), memory);
  prepared.request.arch_name = arch_name;
  prepared.request.program_object = &prepared.image;
  prepared.request.device_load = &prepared.device_load;
  prepared.request.submission_context = submission_context;
  prepared.request.config = std::move(config);
  prepared.request.args = pack_abi_args(prepared.image.metadata(), args);
  prepared.request.mode = mode;
  prepared.request.trace = trace;
  prepared.request.launch_index = launch_index;
  if (mode == ExecutionMode::Functional) {
    prepared.request.functional_mode =
        functional_mode == FunctionalExecutionMode::SingleThreaded ? "st" : "mt";
  }
  return prepared;
}

}  // namespace gpu_model
