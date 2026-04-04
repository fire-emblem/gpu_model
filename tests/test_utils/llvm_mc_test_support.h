#pragma once

#include <filesystem>
#include <string>

#include "gpu_model/instruction/encoded/instruction_object.h"
#include "gpu_model/program/encoded_program_object.h"

namespace gpu_model::test_utils {

bool HasLlvmMcAmdgpuToolchain();

struct AssembledModule {
  std::filesystem::path temp_dir;
  std::filesystem::path asm_path;
  std::filesystem::path obj_path;
  EncodedProgramObject image;
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
                                              const std::string& mcpu = "gfx900");

AssembledInstructionStream AssembleInstructionStream(const std::string& stem,
                                                     const std::string& assembly_text,
                                                     uint64_t start_pc = 0,
                                                     const std::string& mcpu = "gfx900");

std::string WrapAmdgpuKernelAssembly(const std::string& kernel_name,
                                     const std::string& kernel_body_asm,
                                     uint32_t next_free_sgpr,
                                     uint32_t next_free_vgpr,
                                     uint32_t kernarg_segment_size = 0,
                                     uint32_t kernarg_segment_align = 4);

}  // namespace gpu_model::test_utils
