#pragma once

#include <string>
#include <unordered_map>

namespace gpu_model {

struct MetadataBlob {
  std::unordered_map<std::string, std::string> values;
};

}  // namespace gpu_model
