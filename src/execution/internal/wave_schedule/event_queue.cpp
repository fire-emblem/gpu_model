#include "execution/internal/wave_schedule/event_queue.h"

namespace gpu_model {

void EventQueue::Schedule(TimedEvent event) {
  event.sequence = next_sequence_++;
  events_.push(std::move(event));
}

void EventQueue::RunReady(uint64_t cycle) {
  while (!events_.empty() && events_.top().cycle <= cycle) {
    auto event = std::move(const_cast<TimedEvent&>(events_.top()));
    events_.pop();
    event.action();
  }
}

std::optional<uint64_t> EventQueue::NextCycle() const {
  if (events_.empty()) {
    return std::nullopt;
  }
  return events_.top().cycle;
}

bool EventQueue::EarlierEvent::operator()(const TimedEvent& lhs, const TimedEvent& rhs) const {
  if (lhs.cycle != rhs.cycle) {
    return lhs.cycle > rhs.cycle;
  }
  return lhs.sequence > rhs.sequence;
}

}  // namespace gpu_model
