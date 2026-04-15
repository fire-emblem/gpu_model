#include "runtime/model_runtime/model_runtime_launch_helper.h"

#include <utility>

#include "program/loader/device_image_loader.h"
#include "program/loader/device_segment_image.h"
#include "runtime/exec_engine/exec_engine.h"

namespace gpu_model {

DeviceLoadResult MaterializeProgramObjectLoadResult(ExecEngine& runtime, const ProgramObject& image) {
  return DeviceImageLoader{}.Materialize(BuildDeviceLoadPlan(image), runtime.memory());
}

LaunchRequest BuildProgramObjectLaunchRequest(const ProgramObject& image,
                                              const DeviceLoadResult* device_load,
                                              LaunchConfig config,
                                              KernelArgPack args,
                                              ExecutionMode mode,
                                              std::string arch_name,
                                              TraceSink* trace,
                                              RuntimeSubmissionContext submission_context) {
  LaunchRequest request;
  request.arch_name = std::move(arch_name);
  request.program_object = &image;
  request.device_load = device_load;
  request.submission_context = submission_context;
  request.config = config;
  request.args = std::move(args);
  request.mode = mode;
  request.trace = trace;
  return request;
}

LaunchRequest BuildKernelLaunchRequest(const ExecutableKernel& kernel,
                                       LaunchConfig config,
                                       KernelArgPack args,
                                       ExecutionMode mode,
                                       const std::string& arch_name,
                                       TraceSink* trace,
                                       RuntimeSubmissionContext submission_context) {
  LaunchRequest request;
  request.arch_name = arch_name;
  request.kernel = &kernel;
  request.submission_context = submission_context;
  request.config = config;
  request.args = std::move(args);
  request.mode = mode;
  request.trace = trace;
  return request;
}

}  // namespace gpu_model
