#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>

namespace gpu_model {

struct RuntimeAbiEvent {
  bool recorded = false;
  std::optional<uintptr_t> stream_id;
};

class RuntimeStreamEventState {
 public:
  void Reset();

  std::optional<uintptr_t> active_stream_id() const;
  bool IsValidStream(std::optional<uintptr_t> stream_id) const;
  std::optional<uintptr_t> CreateStream();
  bool DestroyStream(uintptr_t stream_id);

  uintptr_t CreateEvent();
  bool HasEvent(uintptr_t event_id) const;
  bool DestroyEvent(uintptr_t event_id);
  bool RecordEvent(uintptr_t event_id, std::optional<uintptr_t> stream_id);

 private:
  thread_local static std::optional<uintptr_t> active_stream_id_;
  std::unordered_map<uintptr_t, RuntimeAbiEvent> abi_events_;
  uintptr_t next_event_id_ = 1;
};

}  // namespace gpu_model
