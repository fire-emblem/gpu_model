#include "gpu_model/runtime/program_execution.h"

#include <filesystem>
#include <stdexcept>

#include "gpu_model/isa/target_isa.h"

namespace gpu_model {

namespace {

std::optional<std::filesystem::path> ArtifactPathForProgramImage(const ProgramImage& image) {
  const auto artifact_path = image.metadata().values.find("artifact_path");
  if (artifact_path == image.metadata().values.end()) {
    return std::nullopt;
  }
  return std::filesystem::path(artifact_path->second);
}

ProgramImage CreateLoweredModeledProgramImage(const ProgramImage& image) {
  MetadataBlob metadata = image.metadata();
  SetTargetIsa(metadata, TargetIsa::GcnAsm);
  return ProgramImage(image.kernel_name(),
                      image.assembly_text(),
                      std::move(metadata),
                      image.const_segment(),
                      image.raw_data_segment());
}

}  // namespace

PreparedProgramExecution PrepareProgramExecution(const ProgramImage& image,
                                                 ProgramExecutionRoute requested_route) {
  PreparedProgramExecution prepared;
  const bool is_raw_program = ResolveTargetIsa(image.metadata()) == TargetIsa::GcnRawAsm;

  if (requested_route == ProgramExecutionRoute::AutoSelect) {
    prepared.resolved_route =
        is_raw_program ? ProgramExecutionRoute::EncodedRaw
                       : ProgramExecutionRoute::LoweredModeled;
  } else {
    prepared.resolved_route = requested_route;
  }

  if (prepared.resolved_route == ProgramExecutionRoute::EncodedRaw) {
    const auto artifact_path = ArtifactPathForProgramImage(image);
    if (!artifact_path.has_value()) {
      throw std::invalid_argument("raw code object execution requires artifact_path metadata");
    }
    prepared.owned_raw_code_object.emplace(
        AmdgpuCodeObjectDecoder{}.Decode(*artifact_path, image.kernel_name()));
    prepared.raw_code_object = &*prepared.owned_raw_code_object;
    return prepared;
  }

  if (prepared.resolved_route == ProgramExecutionRoute::LoweredModeled && is_raw_program) {
    prepared.owned_program_image.emplace(CreateLoweredModeledProgramImage(image));
    prepared.execution_image = &*prepared.owned_program_image;
    return prepared;
  }

  prepared.execution_image = &image;
  return prepared;
}

}  // namespace gpu_model
