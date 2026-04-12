#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <vector>

namespace gpu_model {

struct TimedEvent {
  uint64_t cycle = 0;
  uint64_t sequence = 0;
  std::function<void()> action;
};

class EventQueue {
 public:
  void Schedule(TimedEvent event);
  void RunReady(uint64_t cycle);
  bool empty() const { return events_.empty(); }
  std::optional<uint64_t> NextCycle() const;

 private:
  struct EarlierEvent {
    bool operator()(const TimedEvent& lhs, const TimedEvent& rhs) const;
  };

  std::priority_queue<TimedEvent, std::vector<TimedEvent>, EarlierEvent> events_;
  uint64_t next_sequence_ = 0;
};

}  // namespace gpu_model
