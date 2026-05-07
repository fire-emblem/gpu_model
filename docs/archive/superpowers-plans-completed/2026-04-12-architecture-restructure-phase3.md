# Architecture Restructure Phase 3 — Program/Loader Pipeline Split

## Status: ✅ 已完成

## Context

Phase 1-2 已完成（utils/, gpu_arch/ 基础设施层落地）。Phase 4-5 已完成（instruction/semantics 拆分 + execution 精简）。

Phase 3 目标：让 artifact ingestion 更可测试、更可替换。

## Current State

`encoded_program_object.cpp` (856 行) 混合了：
- 外部工具调用 (RunCommand, popen)
- 临时目录管理 (ScopedTempDir)
- 解析函数 (ParseSectionInfo, ParseSymbols, ParseKernelMetadataNotes)
- 组装函数 (MaterializeDeviceCodeObject, BuildMetadataFromNotes)

## Completion Criteria

1. 外部工具调用、临时目录管理、artifact 解析、ProgramObject 组装已经分层
2. ObjectReader 收敛为 façade
3. 至少存在 extractor / parser / builder 级别的 focused tests

## Approach

### Task 1: Extract ExternalToolExecutor [P0]

将外部工具调用提取到独立模块：

**文件：**
- Create: `src/gpu_model/loader/external_tool_executor.h`
- Create: `src/loader/external_tool_executor.cpp`
- Modify: `src/program/encoded_program_object.cpp` — 使用新模块

**内容：**
```cpp
// external_tool_executor.h
class ExternalToolExecutor {
 public:
  static std::string RunCommand(const std::string& command);
  static bool HasCommand(std::string_view command_name);
  static std::string ReadElfHeader(const std::filesystem::path& path);
  static std::string ReadElfSectionTable(const std::filesystem::path& path);
  static std::string ReadElfSymbolTable(const std::filesystem::path& path);
  static std::string ReadElfNotes(const std::filesystem::path& path);
  static std::string ReadAmdgpuFileHeaders(const std::filesystem::path& path);
  static void DumpElfSection(const std::filesystem::path& path,
                             const std::string& section_name,
                             const std::filesystem::path& output_path);
};
```

### Task 2: Extract TempDirManager [P1]

将临时目录管理提取到独立模块：

**文件：**
- Create: `src/gpu_model/loader/temp_dir_manager.h`
- Create: `src/loader/temp_dir_manager.cpp`

### Task 3: Extract ArtifactParser [P0]

将解析函数提取到独立模块：

**文件：**
- Create: `src/gpu_model/loader/artifact_parser.h`
- Create: `src/loader/artifact_parser.cpp`

**内容：**
```cpp
// artifact_parser.h
struct SectionInfo { ... };
struct SymbolInfo { ... };
struct NoteKernelMetadata { ... };

class ArtifactParser {
 public:
  static SectionInfo ParseSectionInfo(const std::string& sections, std::string_view section_name);
  static std::vector<SymbolInfo> ParseSymbols(const std::string& symbols);
  static std::vector<NoteKernelMetadata> ParseKernelMetadataNotes(const std::string& notes);
  static AmdgpuKernelDescriptor ParseKernelDescriptor(std::span<const std::byte> bytes);
};
```

### Task 4: Make ObjectReader a Façade [P0]

将 ObjectReader 改为真正的 façade，委托给 extractor/parser/builder：

**文件：**
- Modify: `src/gpu_model/program/object_reader.h`
- Modify: `src/program/encoded_program_object.cpp` — 重构为 ObjectReaderImpl

### Task 5: Add Focused Tests [P1]

为 extractor/parser/builder 添加 focused tests：

**文件：**
- Create: `tests/loader/external_tool_executor_test.cpp`
- Create: `tests/loader/artifact_parser_test.cpp`

---

## Commit / Push Policy

- Each task: independent commit + push + push gate
- No mixing tasks in one commit

## Done Criteria

- encoded_program_object.cpp 精简到 < 300 行（只保留组装逻辑）
- ObjectReader 成为真正的 façade
- extractor/parser/builder 有独立测试
- 所有现有测试通过
