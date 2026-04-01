#pragma once

#include <string_view>

#include "gpu_model/isa/metadata.h"

namespace gpu_model {

enum class TargetIsa {
  CanonicalAsm,
  GcnRawAsm,
};

std::string_view ToString(TargetIsa isa);
TargetIsa ParseTargetIsa(std::string_view text);
TargetIsa ResolveTargetIsa(const MetadataBlob& metadata);
void SetTargetIsa(MetadataBlob& metadata, TargetIsa isa);

}  // namespace gpu_model
