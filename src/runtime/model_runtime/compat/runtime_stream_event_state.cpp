#include "runtime/model_runtime/compat/runtime_stream_event_state.h"

namespace gpu_model {

thread_local std::optional<uintptr_t> RuntimeStreamEventState::active_stream_id_;

void RuntimeStreamEventState::Reset() {
  abi_events_.clear();
  next_event_id_ = 1;
}

std::optional<uintptr_t> RuntimeStreamEventState::active_stream_id() const {
  return active_stream_id_;
}

bool RuntimeStreamEventState::IsValidStream(std::optional<uintptr_t> stream_id) const {
  if (!stream_id.has_value()) {
    return true;
  }
  return active_stream_id_.has_value() && *stream_id == *active_stream_id_;
}

std::optional<uintptr_t> RuntimeStreamEventState::CreateStream() {
  if (active_stream_id_.has_value()) {
    return std::nullopt;
  }
  active_stream_id_ = static_cast<uintptr_t>(std::numeric_limits<uint32_t>::max());
  return active_stream_id_;
}

bool RuntimeStreamEventState::DestroyStream(uintptr_t stream_id) {
  if (!active_stream_id_.has_value() || *active_stream_id_ != stream_id) {
    return false;
  }
  active_stream_id_.reset();
  return true;
}

uintptr_t RuntimeStreamEventState::CreateEvent() {
  const uintptr_t event_id = next_event_id_++;
  abi_events_.emplace(event_id, RuntimeAbiEvent{});
  return event_id;
}

bool RuntimeStreamEventState::HasEvent(uintptr_t event_id) const {
  return abi_events_.find(event_id) != abi_events_.end();
}

bool RuntimeStreamEventState::DestroyEvent(uintptr_t event_id) {
  return abi_events_.erase(event_id) != 0;
}

bool RuntimeStreamEventState::RecordEvent(uintptr_t event_id, std::optional<uintptr_t> stream_id) {
  const auto it = abi_events_.find(event_id);
  if (it == abi_events_.end()) {
    return false;
  }
  it->second.recorded = true;
  it->second.stream_id = stream_id;
  return true;
}

}  // namespace gpu_model
