#pragma once

#include <filesystem>
#include <string>

#include "instruction/isa/kernel_metadata.h"
#include "program/loader/temp_dir_manager.h"
#include "program/program_object/program_object.h"

namespace gpu_model {

struct MaterializedCodeObject {
  std::filesystem::path path;
  MetadataBlob metadata;
};

MaterializedCodeObject MaterializeDeviceCodeObject(const std::filesystem::path& path,
                                                   const ScopedTempDir& temp_dir);

MetadataBlob BuildMetadataFromNotes(const std::filesystem::path& note_source_path,
                                    const std::string& kernel_name,
                                    MetadataBlob metadata = {});

}  // namespace gpu_model
