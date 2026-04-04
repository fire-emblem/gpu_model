#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "gpu_model/debug/cycle_timeline.h"

namespace gpu_model {

struct SlotKey {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;

  bool operator<(const SlotKey& other) const;
};

struct RowDescriptor {
  uint32_t pid = 0;
  uint32_t tid = 0;
  int32_t process_sort_index = 0;
  int32_t thread_sort_index = 0;
  std::string process_name;
  std::string thread_name;

  bool operator<(const RowDescriptor& other) const;
};

std::string SlotLabel(const SlotKey& key);
std::string PeuLabel(const SlotKey& key);
std::string ApLabel(const SlotKey& key);
std::string DpcLabel(const SlotKey& key);
std::string ProcessName(const SlotKey& key, CycleTimelineGroupBy group_by);
RowDescriptor DescribeRow(const SlotKey& key,
                          CycleTimelineGroupBy group_by,
                          std::optional<uint32_t> block_id = std::nullopt);

}  // namespace gpu_model
