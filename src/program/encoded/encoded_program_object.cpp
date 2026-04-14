#include "program/program_object/object_reader.h"

#include "program/loader/code_object_builder.h"

namespace gpu_model {

ProgramObject ObjectReader::LoadProgramObject(const std::filesystem::path& path,
                                              std::optional<std::string> kernel_name) const {
  return LoadEncodedProgramObject(path, std::move(kernel_name));
}

}  // namespace gpu_model
