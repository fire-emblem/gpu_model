#include "runtime/model_runtime/compat/runtime_process_path.h"

#include <array>
#include <stdexcept>

#include <unistd.h>

namespace gpu_model {

std::filesystem::path ResolveCurrentExecutablePath() {
  std::array<char, 4096> buffer{};
  const ssize_t length = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length < 0) {
    throw std::runtime_error("failed to resolve /proc/self/exe");
  }
  buffer[static_cast<size_t>(length)] = '\0';
  return std::filesystem::path(buffer.data());
}

}  // namespace gpu_model
