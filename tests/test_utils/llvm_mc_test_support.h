#pragma once

#include <filesystem>
#include <string>

#include "instruction/decode/encoded/instruction_object.h"
#include "program/program_object/program_object.h"
#include "program/loader/amdgpu_target_config.h"

namespace gpu_model::test_utils {

bool HasLlvmMcAmdgpuToolchain();

struct AssembledModule {
  std::filesystem::path temp_dir;
  std::filesystem::path asm_path;
  std::filesystem::path obj_path;
  ProgramObject image;
};

struct AssembledInstructionStream {
  std::filesystem::path temp_dir;
  std::filesystem::path asm_path;
  std::filesystem::path obj_path;
  std::filesystem::path text_path;
  ParsedInstructionArray parsed;
};

AssembledModule AssembleAndDecodeLlvmMcModule(const std::string& stem,
                                              const std::string& kernel_name,
                                              const std::string& assembly_text,
                                              std::string_view mcpu = kProjectAmdgpuMcpu);

AssembledInstructionStream AssembleInstructionStream(const std::string& stem,
                                                     const std::string& assembly_text,
                                                     uint64_t start_pc = 0,
                                                     std::string_view mcpu = kProjectAmdgpuMcpu);

std::string WrapAmdgpuKernelAssembly(const std::string& kernel_name,
                                     const std::string& kernel_body_asm,
                                     uint32_t next_free_sgpr,
                                     uint32_t next_free_vgpr,
                                     uint32_t kernarg_segment_size = 0,
                                     uint32_t kernarg_segment_align = 4);

}  // namespace gpu_model::test_utils
