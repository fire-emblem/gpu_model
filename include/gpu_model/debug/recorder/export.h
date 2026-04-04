#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "gpu_model/debug/recorder/recorder.h"

namespace gpu_model {

class RecorderSerializer {
 public:
  virtual ~RecorderSerializer() = default;

  virtual std::filesystem::path DefaultArtifactPath() const = 0;
  virtual std::string Serialize(const Recorder& recorder) const = 0;
};

class TextRecorderSerializer final : public RecorderSerializer {
 public:
  std::filesystem::path DefaultArtifactPath() const override;
  std::string Serialize(const Recorder& recorder) const override;
};

class JsonRecorderSerializer final : public RecorderSerializer {
 public:
  std::filesystem::path DefaultArtifactPath() const override;
  std::string Serialize(const Recorder& recorder) const override;
};

std::unique_ptr<RecorderSerializer> MakeTextRecorderSerializer();
std::unique_ptr<RecorderSerializer> MakeJsonRecorderSerializer();

std::string RenderRecorderTextTrace(const Recorder& recorder);
std::string RenderRecorderJsonTrace(const Recorder& recorder);

}  // namespace gpu_model
