#pragma once

#include "debug/recorder/recorder.h"
#include "debug/trace/sink.h"

namespace gpu_model {

class RecorderTraceSink final : public TraceSink {
 public:
  explicit RecorderTraceSink(Recorder& recorder) : recorder_(recorder) {}

  void OnEvent(const TraceEvent& event) override;

 private:
  Recorder& recorder_;
};

}  // namespace gpu_model
