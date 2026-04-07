#include "gpu_model/logging/runtime_log_service.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace gpu_model {
namespace {

struct LoggingState {
  std::once_flag init_once;
  std::atomic<bool> initialized = false;
  std::vector<std::string> enabled_modules;
  std::string file_path;
  std::string program_name;
};

LoggingState& State() {
  static LoggingState state;
  return state;
}

int ParseVerbosityFromEnv() {
  const char* raw = std::getenv("GPU_MODEL_LOG_LEVEL");
  if (raw == nullptr || raw[0] == '\0') {
    return loguru::Verbosity_WARNING;
  }
  std::string level(raw);
  std::transform(level.begin(), level.end(), level.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (level == "error") return loguru::Verbosity_ERROR;
  if (level == "warning" || level == "warn") return loguru::Verbosity_WARNING;
  if (level == "info") return loguru::Verbosity_INFO;
  if (level == "debug") return 1;
  if (level == "trace") return 2;
  return loguru::Verbosity_WARNING;
}

int ParseFileVerbosityFromEnv() {
  const char* raw = std::getenv("GPU_MODEL_LOG_FILE_LEVEL");
  if (raw == nullptr || raw[0] == '\0') {
    return loguru::Verbosity_INFO;
  }
  std::string level(raw);
  std::transform(level.begin(), level.end(), level.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (level == "error") return loguru::Verbosity_ERROR;
  if (level == "warning" || level == "warn") return loguru::Verbosity_WARNING;
  if (level == "info") return loguru::Verbosity_INFO;
  if (level == "debug") return 1;
  if (level == "trace") return 2;
  return loguru::Verbosity_INFO;
}

std::string ProgramName() {
  std::array<char, 4096> exe_path{};
  const ssize_t length = ::readlink("/proc/self/exe", exe_path.data(), exe_path.size() - 1);
  if (length > 0) {
    exe_path[static_cast<size_t>(length)] = '\0';
    return std::filesystem::path(exe_path.data()).filename().string();
  }
  const char* raw = std::getenv("GPU_MODEL_LOG_PROGRAM");
  if (raw != nullptr && raw[0] != '\0') {
    return raw;
  }
  return "gpu_model";
}

bool ShouldDisableLoguru() {
  const char* raw = std::getenv("GPU_MODEL_DISABLE_LOGURU");
  if (raw != nullptr && raw[0] != '\0' && std::string_view(raw) != "0") {
    return true;
  }
  return false;
}

bool IsChattyModule(std::string_view module) {
  return module == "encoded_exec" || module == "encoded_mt";
}

std::string LogFilePath() {
  const char* raw = std::getenv("GPU_MODEL_LOG_FILE");
  if (raw != nullptr && raw[0] != '\0') {
    return raw;
  }
  const std::filesystem::path program_path(ProgramName());
  const std::string stem =
      program_path.stem().empty() ? std::string("gpu_model") : program_path.stem().string();
  return "logs/" + stem + "." + std::to_string(::getpid()) + ".log";
}

void LoadModulesFromEnv() {
  auto& enabled_modules = State().enabled_modules;
  enabled_modules.clear();

  const char* raw = std::getenv("GPU_MODEL_LOG_MODULES");
  if (raw == nullptr || raw[0] == '\0') {
    return;
  }

  std::string modules(raw);
  size_t start = 0;
  while (start < modules.size()) {
    const size_t comma = modules.find(',', start);
    const std::string_view token =
        comma == std::string::npos ? std::string_view(modules).substr(start)
                                   : std::string_view(modules).substr(start, comma - start);
    size_t first = 0;
    while (first < token.size() && std::isspace(static_cast<unsigned char>(token[first]))) {
      ++first;
    }
    size_t last = token.size();
    while (last > first && std::isspace(static_cast<unsigned char>(token[last - 1]))) {
      --last;
    }
    if (last > first) {
      std::string value(token.substr(first, last - first));
      std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
      });
      enabled_modules.push_back(std::move(value));
    }
    if (comma == std::string::npos) break;
    start = comma + 1;
  }

}

}  // namespace

RuntimeLogService& GetRuntimeLogService() {
  static RuntimeLogService service;
  return service;
}

void RuntimeLogService::EnsureInitialized() {
  auto& state = State();
  std::call_once(state.init_once, [&state] {
    if (ShouldDisableLoguru()) {
      state.initialized.store(false, std::memory_order_release);
      return;
    }
    state.program_name = ProgramName();
    int argc = 1;
    char* argv[] = {state.program_name.data(), nullptr};
    loguru::init(argc, argv);
    loguru::g_stderr_verbosity = ParseVerbosityFromEnv();
    loguru::g_preamble_date = true;
    loguru::g_preamble_time = true;
    loguru::g_preamble_uptime = false;
    loguru::g_preamble_thread = true;
    loguru::g_preamble_file = true;
    loguru::g_preamble_verbose = true;
    LoadModulesFromEnv();
    state.file_path = LogFilePath();
    const auto parent = std::filesystem::path(state.file_path).parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    loguru::add_file(state.file_path.c_str(), loguru::Append, ParseFileVerbosityFromEnv());
    state.initialized.store(true, std::memory_order_release);
  });
}

bool RuntimeLogService::IsInitialized() const {
  return State().initialized.load(std::memory_order_acquire);
}

bool RuntimeLogService::ShouldLog(std::string_view module, int verbosity) {
  EnsureInitialized();
  if (verbosity <= loguru::Verbosity_WARNING) {
    return true;
  }
  const auto& enabled_modules = State().enabled_modules;
  if (enabled_modules.empty()) {
    if (IsChattyModule(module)) {
      return false;
    }
    return IsInitialized() && verbosity <= loguru::g_stderr_verbosity;
  }
  std::string lowered(module);
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return std::find(enabled_modules.begin(), enabled_modules.end(), lowered) !=
         enabled_modules.end();
}

void RuntimeLogService::LogMessage(int verbosity, std::string_view module, const char* fmt, ...) {
  if (!ShouldLog(module, verbosity)) {
    return;
  }
  std::array<char, 4096> buffer{};
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer.data(), buffer.size(), fmt, args);
  va_end(args);
  loguru::log(verbosity, __FILE__, __LINE__, "[%.*s] %s",
              static_cast<int>(module.size()), module.data(), buffer.data());
}

void RuntimeLogService::LogMessageForced(int verbosity, std::string_view module, const char* fmt, ...) {
  EnsureInitialized();
  std::array<char, 4096> buffer{};
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer.data(), buffer.size(), fmt, args);
  va_end(args);
  std::fprintf(stderr, "[%.*s] %s\n",
               static_cast<int>(module.size()), module.data(), buffer.data());
  if (!IsInitialized()) {
    return;
  }
  loguru::log(verbosity, __FILE__, __LINE__, "[%.*s] %s",
              static_cast<int>(module.size()), module.data(), buffer.data());
}

namespace logging {

void LogMessage(int verbosity, std::string_view module, const char* fmt, ...) {
  std::array<char, 4096> buffer{};
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer.data(), buffer.size(), fmt, args);
  va_end(args);
  GetRuntimeLogService().LogMessage(verbosity, module, "%s", buffer.data());
}

void LogMessageForced(int verbosity, std::string_view module, const char* fmt, ...) {
  std::array<char, 4096> buffer{};
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer.data(), buffer.size(), fmt, args);
  va_end(args);
  GetRuntimeLogService().LogMessageForced(verbosity, module, "%s", buffer.data());
}

}  // namespace logging

}  // namespace gpu_model
