#include "debug/info/debug_info_io.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_model {

namespace {

std::string EscapeJson(const std::string& text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (const char ch : text) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      default:
        escaped.push_back(ch);
        break;
    }
  }
  return escaped;
}

std::vector<uint64_t> SortedPcs(const KernelDebugInfo& info) {
  std::vector<uint64_t> pcs;
  pcs.reserve(info.pc_to_debug_loc.size());
  for (const auto& [pc, loc] : info.pc_to_debug_loc) {
    (void)loc;
    pcs.push_back(pc);
  }
  std::sort(pcs.begin(), pcs.end());
  return pcs;
}

}  // namespace

void DebugInfoIO::WriteText(const std::filesystem::path& path, const KernelDebugInfo& info) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open debug info text file");
  }

  out << "kernel=" << info.kernel_name << '\n';
  for (const uint64_t pc : SortedPcs(info)) {
    const auto& loc = info.pc_to_debug_loc.at(pc);
    out << "pc=" << pc << " file=" << loc.file << " line=" << loc.line;
    if (!loc.label.empty()) {
      out << " label=" << loc.label;
    }
    out << '\n';
  }
}

void DebugInfoIO::WriteJson(const std::filesystem::path& path, const KernelDebugInfo& info) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open debug info json file");
  }

  out << "{\"kernel_name\":\"" << EscapeJson(info.kernel_name) << "\",\"entries\":[";
  bool first = true;
  for (const uint64_t pc : SortedPcs(info)) {
    const auto& loc = info.pc_to_debug_loc.at(pc);
    if (!first) {
      out << ',';
    }
    first = false;
    out << "{\"pc\":" << pc << ",\"file\":\"" << EscapeJson(loc.file) << "\",\"line\":"
        << loc.line << ",\"label\":\"" << EscapeJson(loc.label) << "\"}";
  }
  out << "]}";
}

}  // namespace gpu_model
