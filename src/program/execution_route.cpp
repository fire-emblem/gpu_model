#include "gpu_model/program/execution_route.h"

#include <filesystem>
#include <stdexcept>

#include "gpu_model/isa/target_isa.h"
#include "gpu_model/program/encoded_program_object.h"
#include "gpu_model/program/object_reader.h"
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

}  // namespace

PreparedExecutionRoute PrepareExecutionRoute(const ProgramObject& image) {
  PreparedExecutionRoute prepared;
  const bool is_raw_program = ResolveTargetIsa(image.metadata()) == TargetIsa::GcnRawAsm;

  if (is_raw_program) {
    const auto artifact_path = ArtifactPathForProgramImage(image);
    if (!artifact_path.has_value()) {
      throw std::invalid_argument("raw code object execution requires artifact_path metadata");
    }
    prepared.owned_raw_code_object =
        std::make_shared<EncodedProgramObject>(
            ObjectReader{}.LoadEncodedObject(*artifact_path, image.kernel_name()));
    prepared.raw_code_object = prepared.owned_raw_code_object.get();
    return prepared;
  }

  prepared.execution_image = &image;
  return prepared;
}

}  // namespace gpu_model
