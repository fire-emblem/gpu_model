#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gpu_model/debug/trace_event.h"

namespace gpu_model {

class TraceSink {
 public:
  virtual ~TraceSink() = default;
  virtual void OnEvent(const TraceEvent& event) = 0;
};

class NullTraceSink final : public TraceSink {
 public:
  void OnEvent(const TraceEvent& event) override;
};

class CollectingTraceSink final : public TraceSink {
 public:
  void OnEvent(const TraceEvent& event) override;
  const std::vector<TraceEvent>& events() const { return events_; }

 private:
  std::vector<TraceEvent> events_;
};

class FileTraceSink final : public TraceSink {
 public:
  explicit FileTraceSink(const std::filesystem::path& path);
  void OnEvent(const TraceEvent& event) override;

 private:
  std::ofstream out_;
};

class JsonTraceSink final : public TraceSink {
 public:
  explicit JsonTraceSink(const std::filesystem::path& path);
  void OnEvent(const TraceEvent& event) override;

 private:
  std::ofstream out_;
};

}  // namespace gpu_model
