#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace gpu_model {
namespace {

std::filesystem::path SelfExecutablePath() {
  std::array<char, 4096> buffer{};
  const ssize_t length = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length <= 0) {
    throw std::runtime_error("failed to resolve /proc/self/exe");
  }
  buffer[static_cast<size_t>(length)] = '\0';
  return std::filesystem::path(buffer.data());
}

std::set<std::string> LogFileSet(const std::filesystem::path& log_dir) {
  std::set<std::string> files;
  if (!std::filesystem::exists(log_dir)) {
    return files;
  }
  for (const auto& entry : std::filesystem::directory_iterator(log_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (entry.path().extension() != ".log") {
      continue;
    }
    files.insert(entry.path().filename().string());
  }
  return files;
}

std::vector<std::string> NewLogFiles(const std::filesystem::path& log_dir,
                                     const std::set<std::string>& before) {
  std::vector<std::string> files;
  for (const auto& file : LogFileSet(log_dir)) {
    if (!before.contains(file)) {
      files.push_back(file);
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    return {};
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

TEST(RuntimeLoggingTest, DefaultLogFileUsesProgramStemAndPid) {
  const auto exe_path = SelfExecutablePath();
  const auto log_dir = std::filesystem::current_path() / "logs";
  const auto before = LogFileSet(log_dir);

  const std::string command =
      "cd /data/gpu_model && "
      "env -u GPU_MODEL_DISABLE_LOGURU "
      "GPU_MODEL_LOG_LEVEL=info " + exe_path.string() +
      " --gtest_filter=RuntimeNamingTest.NewRuntimeTypesAreConcreteAndUsable >/dev/null 2>&1";
  ASSERT_EQ(std::system(command.c_str()), 0);

  const auto added = NewLogFiles(log_dir, before);
  ASSERT_FALSE(added.empty());
  const auto it = std::find_if(added.begin(), added.end(), [](const std::string& name) {
    return name.rfind("gpu_model_tests.", 0) == 0;
  });
  ASSERT_NE(it, added.end());
  EXPECT_NE(it->find(".log"), std::string::npos);
}

TEST(RuntimeLoggingTest, HipRuntimeAbiDebugStillInitializesLoguru) {
  const auto exe_path = SelfExecutablePath();
  const auto log_dir = std::filesystem::current_path() / "logs";
  const auto before = LogFileSet(log_dir);

  const std::string command =
      "cd /data/gpu_model && "
      "env -u GPU_MODEL_DISABLE_LOGURU "
      "GPU_MODEL_HIP_RUNTIME_ABI_DEBUG=1 " + exe_path.string() +
      " --gtest_filter=RuntimeNamingTest.NewRuntimeTypesAreConcreteAndUsable >/dev/null 2>&1";
  ASSERT_EQ(std::system(command.c_str()), 0);

  const auto added = NewLogFiles(log_dir, before);
  ASSERT_FALSE(added.empty());
  const auto it = std::find_if(added.begin(), added.end(), [](const std::string& name) {
    return name.rfind("gpu_model_tests.", 0) == 0;
  });
  ASSERT_NE(it, added.end());
  const std::string text = ReadTextFile(log_dir / *it);
  EXPECT_NE(text.find("Logging to"), std::string::npos);
}

}  // namespace
}  // namespace gpu_model
