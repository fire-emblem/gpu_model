#pragma once

#include <string>

#include "program/loader/device_image_loader.h"
#include "runtime/config/launch_request.h"

namespace gpu_model {

class ExecEngine;
class ExecutableKernel;
class ProgramObject;
class TraceSink;

DeviceLoadResult MaterializeProgramObjectLoadResult(ExecEngine& runtime, const ProgramObject& image);

LaunchRequest BuildProgramObjectLaunchRequest(const ProgramObject& image,
                                              const DeviceLoadResult* device_load,
                                              LaunchConfig config,
                                              KernelArgPack args,
                                              ExecutionMode mode,
                                              std::string arch_name,
                                              TraceSink* trace,
                                              RuntimeSubmissionContext submission_context);

LaunchRequest BuildKernelLaunchRequest(const ExecutableKernel& kernel,
                                       LaunchConfig config,
                                       KernelArgPack args,
                                       ExecutionMode mode,
                                       const std::string& arch_name,
                                       TraceSink* trace,
                                       RuntimeSubmissionContext submission_context);

}  // namespace gpu_model
