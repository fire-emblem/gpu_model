#include "gpu_model/program/execution_route.h"

#include <filesystem>
#include <stdexcept>

#include "gpu_model/isa/target_isa.h"
#include "gpu_model/loader/amdgpu_code_object_decoder.h"
#include "gpu_model/program/encoded_program_object.h"
#include "gpu_model/program/program_object.h"

namespace gpu_model {

namespace {

std::optional<std::filesystem::path> ArtifactPathForProgramImage(const ProgramObject& image) {
  const auto artifact_path = image.metadata().values.find("artifact_path");
  if (artifact_path == image.metadata().values.end()) {
    return std::nullopt;
  }
  return std::filesystem::path(artifact_path->second);
}

ProgramObject CreateLoweredModeledProgramImage(const ProgramObject& image) {
  MetadataBlob metadata = image.metadata();
  SetTargetIsa(metadata, TargetIsa::GcnAsm);
  return ProgramObject(image.kernel_name(),
                       image.assembly_text(),
                       std::move(metadata),
                       image.const_segment(),
                       image.raw_data_segment());
}

}  // namespace

PreparedExecutionRoute PrepareExecutionRoute(const ProgramObject& image,
                                             ExecutionRoute requested_route) {
  PreparedExecutionRoute prepared;
  const bool is_raw_program = ResolveTargetIsa(image.metadata()) == TargetIsa::GcnRawAsm;

  if (requested_route == ExecutionRoute::AutoSelect) {
    prepared.resolved_route =
        is_raw_program ? ExecutionRoute::EncodedRaw
                       : ExecutionRoute::LoweredModeled;
  } else {
    prepared.resolved_route = requested_route;
  }

  if (prepared.resolved_route == ExecutionRoute::EncodedRaw) {
    const auto artifact_path = ArtifactPathForProgramImage(image);
    if (!artifact_path.has_value()) {
      throw std::invalid_argument("raw code object execution requires artifact_path metadata");
    }
    prepared.owned_raw_code_object =
        std::make_shared<EncodedProgramObject>(
            AmdgpuCodeObjectDecoder{}.Decode(*artifact_path, image.kernel_name()));
    prepared.raw_code_object = prepared.owned_raw_code_object.get();
    return prepared;
  }

  if (prepared.resolved_route == ExecutionRoute::LoweredModeled && is_raw_program) {
    prepared.owned_program_image =
        std::make_shared<ProgramObject>(CreateLoweredModeledProgramImage(image));
    prepared.execution_image = prepared.owned_program_image.get();
    return prepared;
  }

  prepared.execution_image = &image;
  return prepared;
}

}  // namespace gpu_model
