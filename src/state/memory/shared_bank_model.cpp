#include "state/memory/shared_bank_model.h"

#include <algorithm>
#include <vector>

namespace gpu_model {

uint64_t SharedBankModel::ConflictPenalty(const MemoryRequest& request) const {
  if (!spec_.enabled || spec_.bank_count == 0 || spec_.bank_width_bytes == 0) {
    return 0;
  }

  std::vector<uint32_t> counts(spec_.bank_count, 0);
  for (const auto& lane : request.lanes) {
    if (!lane.active) {
      continue;
    }
    const uint32_t bank =
        static_cast<uint32_t>((lane.addr / spec_.bank_width_bytes) % spec_.bank_count);
    ++counts[bank];
  }

  const auto max_it = std::max_element(counts.begin(), counts.end());
  if (max_it == counts.end() || *max_it == 0) {
    return 0;
  }
  return static_cast<uint64_t>(*max_it - 1);
}

}  // namespace gpu_model
