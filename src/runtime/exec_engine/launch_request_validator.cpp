#include "runtime/exec_engine/launch_request_validator.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>

#include "gpu_arch/chip_config/arch_registry.h"
#include "instruction/isa/kernel_metadata.h"
#include "program/loader/asm_parser.h"
#include "program/loader/device_image_loader.h"

namespace gpu_model {

std::optional<ValidatedLaunchRequest> ValidateAndPrepareLaunch(const LaunchRequest& request,
                                                              std::string& error_message) {
  ValidatedLaunchRequest validated;
  validated.program_object = request.program_object;

  if (request.arch_spec_override.has_value()) {
    validated.spec_storage =
        std::make_shared<GpuArchSpec>(*request.arch_spec_override);
    validated.spec = validated.spec_storage.get();
    validated.arch_name = request.arch_name.empty() ? validated.spec->name : request.arch_name;
  } else {
    validated.arch_name = request.arch_name;
    if (validated.arch_name.empty() && request.program_object != nullptr) {
      const auto it = request.program_object->metadata().values.find("arch");
      if (it != request.program_object->metadata().values.end()) {
        validated.arch_name = it->second;
      }
    }
    if (validated.arch_name.empty()) {
      validated.arch_name = "mac500";
    }

    const auto spec = ArchRegistry::Get(validated.arch_name);
    if (!spec) {
      error_message = "unknown architecture: " + validated.arch_name;
      return std::nullopt;
    }
    validated.spec_storage = spec;
    validated.spec = validated.spec_storage.get();
  }

  validated.kernel = request.kernel;
  validated.use_program_object_payload =
      request.program_object != nullptr && request.program_object->has_encoded_payload();
  if (!validated.use_program_object_payload && validated.kernel == nullptr &&
      request.program_object != nullptr) {
    validated.parsed_kernel = AsmParser{}.Parse(*request.program_object);
  }

  if (validated.kernel == nullptr && !validated.parsed_kernel.has_value() &&
      !validated.use_program_object_payload) {
    error_message = "launch request missing kernel or program object";
    return std::nullopt;
  }

  if (request.config.grid_dim_x == 0 || request.config.grid_dim_y == 0 ||
      request.config.grid_dim_z == 0 || request.config.block_dim_x == 0 ||
      request.config.block_dim_y == 0 || request.config.block_dim_z == 0) {
    error_message = "grid and block dimensions must be non-zero";
    return std::nullopt;
  }

  validated.adjusted_config = request.config;

  try {
    const auto launch_metadata = ParseKernelLaunchMetadata(validated.launch_metadata_source());
    if (launch_metadata.arch.has_value() && *launch_metadata.arch != validated.spec->name) {
      error_message = "kernel metadata arch does not match selected architecture";
      return std::nullopt;
    }

    const std::string kernel_name = validated.kernel_name();
    if (launch_metadata.entry.has_value() && *launch_metadata.entry != kernel_name) {
      error_message = "kernel metadata entry does not match kernel name";
      return std::nullopt;
    }
    if (!launch_metadata.module_kernels.empty()) {
      const bool found = std::find(launch_metadata.module_kernels.begin(),
                                   launch_metadata.module_kernels.end(),
                                   kernel_name) != launch_metadata.module_kernels.end();
      if (!found) {
        error_message = "kernel name is not present in module_kernels metadata";
        return std::nullopt;
      }
    }
    if (launch_metadata.arg_count.has_value() &&
        request.args.size() != *launch_metadata.arg_count) {
      error_message = "kernel argument count does not match metadata";
      return std::nullopt;
    }

    const uint32_t statically_loaded_shared_bytes =
        request.device_load != nullptr ? request.device_load->required_shared_bytes : 0u;
    const uint32_t available_shared_bytes =
        std::max({request.config.shared_memory_bytes,
                  launch_metadata.group_segment_fixed_size.value_or(0u),
                  statically_loaded_shared_bytes});
    if (launch_metadata.required_shared_bytes.has_value() &&
        available_shared_bytes < *launch_metadata.required_shared_bytes) {
      error_message = "shared memory launch size is smaller than metadata requirement";
      return std::nullopt;
    }
    validated.adjusted_config.shared_memory_bytes = available_shared_bytes;
    if (available_shared_bytes > validated.spec->shared_mem_per_block) {
      error_message = "shared memory launch size exceeds block limit";
      return std::nullopt;
    }
    if (available_shared_bytes > validated.spec->max_shared_mem_per_multiprocessor) {
      error_message = "shared memory launch size exceeds multiprocessor limit";
      return std::nullopt;
    }

    if (launch_metadata.block_dim_multiple.has_value() &&
        request.config.block_dim_x % *launch_metadata.block_dim_multiple != 0) {
      error_message = "block_dim_x does not satisfy metadata multiple requirement";
      return std::nullopt;
    }
    if (launch_metadata.max_block_dim.has_value() &&
        request.config.block_dim_x > *launch_metadata.max_block_dim) {
      error_message = "block_dim_x exceeds metadata maximum";
      return std::nullopt;
    }
  } catch (const std::exception& ex) {
    error_message = std::string("failed to parse kernel metadata: ") + ex.what();
    return std::nullopt;
  }

  return validated;
}

}  // namespace gpu_model
