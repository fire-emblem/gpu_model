#pragma once

#include <optional>
#include <string>

#include "gpu_arch/device/gpu_arch_spec.h"
#include "program/executable/executable_kernel.h"
#include "program/program_object/program_object.h"
#include "utils/config/launch_request.h"

namespace gpu_model {

struct ValidatedLaunchRequest {
  std::string arch_name;
  const GpuArchSpec* spec = nullptr;
  const ProgramObject* program_object = nullptr;
  const ExecutableKernel* kernel = nullptr;
  std::optional<ExecutableKernel> parsed_kernel;
  bool use_program_object_payload = false;
  LaunchConfig adjusted_config;

  const ExecutableKernel& kernel_ref() const {
    return parsed_kernel.has_value() ? *parsed_kernel : *kernel;
  }

  const MetadataBlob& launch_metadata_source() const {
    if (use_program_object_payload) {
      return program_object->metadata();
    }
    return kernel_ref().metadata();
  }

  std::string kernel_name() const {
    return use_program_object_payload ? program_object->kernel_name() : kernel_ref().name();
  }
};

std::optional<ValidatedLaunchRequest> ValidateAndPrepareLaunch(const LaunchRequest& request,
                                                              std::string& error_message);

}  // namespace gpu_model
