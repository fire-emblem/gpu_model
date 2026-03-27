#include "gpu_model/isa/target_isa.h"

#include <stdexcept>
#include <string>

namespace gpu_model {

std::string_view ToString(TargetIsa isa) {
  switch (isa) {
    case TargetIsa::CanonicalAsm:
      return "canonical_asm";
    case TargetIsa::GcnAsm:
      return "gcn_asm";
  }
  return "canonical_asm";
}

TargetIsa ParseTargetIsa(std::string_view text) {
  if (text.empty() || text == "canonical_asm") {
    return TargetIsa::CanonicalAsm;
  }
  if (text == "gcn_asm") {
    return TargetIsa::GcnAsm;
  }
  throw std::invalid_argument("unknown target ISA: " + std::string(text));
}

TargetIsa ResolveTargetIsa(const MetadataBlob& metadata) {
  const auto it = metadata.values.find("target_isa");
  if (it == metadata.values.end()) {
    return TargetIsa::CanonicalAsm;
  }
  return ParseTargetIsa(it->second);
}

void SetTargetIsa(MetadataBlob& metadata, TargetIsa isa) {
  metadata.values["target_isa"] = std::string(ToString(isa));
}

}  // namespace gpu_model
