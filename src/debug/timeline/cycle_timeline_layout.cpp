#include "cycle_timeline_internal.h"

#include <iomanip>
#include <sstream>
#include <tuple>

namespace gpu_model {

namespace {

std::string BlockLabel(uint32_t block_id) {
  return "B" + std::to_string(block_id);
}

std::string FormatPaddedLabel(std::string_view prefix, uint32_t value, int width) {
  std::ostringstream out;
  out << prefix << std::setw(width) << std::setfill('0') << value;
  return out.str();
}

std::string ThreadLabel(const SlotKey& key,
                        CycleTimelineGroupBy group_by,
                        std::optional<uint32_t> block_id = std::nullopt) {
  switch (group_by) {
    case CycleTimelineGroupBy::Wave:
      return SlotLabel(key);
    case CycleTimelineGroupBy::Block:
      return BlockLabel(block_id.value_or(0));
    case CycleTimelineGroupBy::Peu:
      return PeuLabel(key);
    case CycleTimelineGroupBy::Ap:
      return ApLabel(key);
    case CycleTimelineGroupBy::Dpc:
      return DpcLabel(key);
  }
  return SlotLabel(key);
}

uint32_t TracePid(const SlotKey& key, CycleTimelineGroupBy group_by) {
  switch (group_by) {
    case CycleTimelineGroupBy::Wave:
      return 1u + (key.dpc_id << 12) + (key.ap_id << 4) + key.peu_id;
    case CycleTimelineGroupBy::Block:
      return 1u;
    case CycleTimelineGroupBy::Peu:
      return 1u + (key.dpc_id << 8) + key.ap_id;
    case CycleTimelineGroupBy::Ap:
      return 1u + key.dpc_id;
    case CycleTimelineGroupBy::Dpc:
      return 1u;
  }
  return 1u;
}

uint32_t TraceTid(const SlotKey& key,
                  CycleTimelineGroupBy group_by,
                  std::optional<uint32_t> block_id = std::nullopt) {
  switch (group_by) {
    case CycleTimelineGroupBy::Wave:
      return key.slot_id;
    case CycleTimelineGroupBy::Block:
      return block_id.value_or(0);
    case CycleTimelineGroupBy::Peu:
      return key.peu_id;
    case CycleTimelineGroupBy::Ap:
      return key.ap_id;
    case CycleTimelineGroupBy::Dpc:
      return key.dpc_id;
  }
  return key.slot_id;
}

int32_t ProcessSortIndex(const SlotKey& key, CycleTimelineGroupBy group_by) {
  switch (group_by) {
    case CycleTimelineGroupBy::Wave:
      return static_cast<int32_t>((key.dpc_id << 8) + (key.ap_id << 4) + key.peu_id);
    case CycleTimelineGroupBy::Block:
      return 0;
    case CycleTimelineGroupBy::Peu:
      return static_cast<int32_t>((key.dpc_id << 8) + key.ap_id);
    case CycleTimelineGroupBy::Ap:
      return static_cast<int32_t>(key.dpc_id);
    case CycleTimelineGroupBy::Dpc:
      return 0;
  }
  return 0;
}

int32_t ThreadSortIndex(const SlotKey& key,
                        CycleTimelineGroupBy group_by,
                        std::optional<uint32_t> block_id = std::nullopt) {
  switch (group_by) {
    case CycleTimelineGroupBy::Wave:
      return static_cast<int32_t>(key.slot_id);
    case CycleTimelineGroupBy::Block:
      return static_cast<int32_t>(block_id.value_or(0));
    case CycleTimelineGroupBy::Peu:
      return static_cast<int32_t>(key.peu_id);
    case CycleTimelineGroupBy::Ap:
      return static_cast<int32_t>(key.ap_id);
    case CycleTimelineGroupBy::Dpc:
      return static_cast<int32_t>(key.dpc_id);
  }
  return 0;
}

}  // namespace

bool SlotKey::operator<(const SlotKey& other) const {
  return std::tie(dpc_id, ap_id, peu_id, slot_id) <
         std::tie(other.dpc_id, other.ap_id, other.peu_id, other.slot_id);
}

bool RowDescriptor::operator<(const RowDescriptor& other) const {
  return std::tie(process_sort_index, pid, thread_sort_index, tid, process_name, thread_name) <
         std::tie(other.process_sort_index, other.pid, other.thread_sort_index, other.tid,
                  other.process_name, other.thread_name);
}

std::string SlotLabel(const SlotKey& key) {
  return FormatPaddedLabel("WAVE_SLOT_", key.slot_id, 2);
}

std::string PeuLabel(const SlotKey& key) {
  return FormatPaddedLabel("PEU_", key.peu_id, 2);
}

std::string ApLabel(const SlotKey& key) {
  return FormatPaddedLabel("AP_", key.ap_id, 2);
}

std::string DpcLabel(const SlotKey& key) {
  return FormatPaddedLabel("DPC_", key.dpc_id, 2);
}

std::string ProcessName(const SlotKey& key, CycleTimelineGroupBy group_by) {
  switch (group_by) {
    case CycleTimelineGroupBy::Wave:
      return DpcLabel(key) + "/" + ApLabel(key) + "/" + PeuLabel(key);
    case CycleTimelineGroupBy::Block:
      return "Blocks";
    case CycleTimelineGroupBy::Peu:
      return DpcLabel(key) + "/" + ApLabel(key);
    case CycleTimelineGroupBy::Ap:
      return DpcLabel(key);
    case CycleTimelineGroupBy::Dpc:
      return "Device";
  }
  return "Device";
}

RowDescriptor DescribeRow(const SlotKey& key,
                          CycleTimelineGroupBy group_by,
                          std::optional<uint32_t> block_id) {
  return RowDescriptor{.pid = TracePid(key, group_by),
                       .tid = TraceTid(key, group_by, block_id),
                       .process_sort_index = ProcessSortIndex(key, group_by),
                       .thread_sort_index = ThreadSortIndex(key, group_by, block_id),
                       .process_name = ProcessName(key, group_by),
                       .thread_name = ThreadLabel(key, group_by, block_id)};
}

}  // namespace gpu_model
