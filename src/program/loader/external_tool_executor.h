#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace gpu_model {

/// ExternalToolExecutor — 外部工具调用封装
///
/// 提供对外部命令（readelf, llvm-mc 等）的统一调用接口。
/// 将 popen/system 调用封装到独立模块，便于测试和替换。
class ExternalToolExecutor {
 public:
  /// 运行命令并返回 stdout 输出
  static std::string RunCommand(const std::string& command);

  /// 检查命令是否可用
  static bool HasCommand(std::string_view command_name);

  /// 解析外部工具路径，优先使用项目/ROCm 工具链
  static std::string ResolveTool(std::string_view tool_name);

  /// 读取 ELF header 信息
  static std::string ReadElfHeader(const std::filesystem::path& path);

  /// 读取 ELF section table
  static std::string ReadElfSectionTable(const std::filesystem::path& path);

  /// 读取 ELF symbol table
  static std::string ReadElfSymbolTable(const std::filesystem::path& path);

  /// 读取 ELF notes
  static std::string ReadElfNotes(const std::filesystem::path& path);

  /// 读取 AMDGPU file headers
  static std::string ReadAmdgpuFileHeaders(const std::filesystem::path& path);

  /// 导出 ELF section 到文件
  static void DumpElfSection(const std::filesystem::path& path,
                             const std::string& section_name,
                             const std::filesystem::path& output_path);

  /// 列出 offload bundles
  static std::string ListOffloadBundles(const std::filesystem::path& fatbin_path);

  /// 解包 offload target
  static void UnbundleOffloadTarget(const std::filesystem::path& fatbin_path,
                                    const std::string& target,
                                    const std::filesystem::path& output_path);

  /// 使用 llvm-mc 反汇编 hex byte stream
  static std::string DisassembleHexByteStreamWithLlvmMc(
      const std::filesystem::path& input_path);

  /// 检查是否是 AMDGPU ELF
  static bool IsAmdgpuElf(const std::filesystem::path& path);

  /// 检查是否包含 HIP fatbin
  static bool HasHipFatbin(const std::filesystem::path& path);

  /// 检查 llvm-mc 是否可用
  static bool HasLlvmMc();
};

}  // namespace gpu_model
