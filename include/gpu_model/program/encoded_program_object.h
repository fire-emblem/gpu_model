#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/decode/raw_gcn_instruction.h"
#include "gpu_model/instruction/encoded/decoded_instruction.h"
#include "gpu_model/instruction/encoded/instruction_object.h"
#include "gpu_model/isa/metadata.h"

namespace gpu_model {

struct AmdgpuKernelDescriptor {
  uint32_t group_segment_fixed_size = 0;
  uint32_t private_segment_fixed_size = 0;
  uint32_t kernarg_size = 0;
  int64_t kernel_code_entry_byte_offset = 0;
  uint32_t compute_pgm_rsrc3 = 0;
  uint32_t compute_pgm_rsrc1 = 0;
  uint32_t compute_pgm_rsrc2 = 0;
  uint32_t setup_word = 0;
  uint16_t accum_offset = 0;
  uint16_t agpr_count = 0;

  bool enable_private_segment = false;
  uint8_t user_sgpr_count = 0;
  bool enable_sgpr_workgroup_id_x = false;
  bool enable_sgpr_workgroup_id_y = false;
  bool enable_sgpr_workgroup_id_z = false;
  bool enable_sgpr_workgroup_info = false;
  uint8_t enable_vgpr_workitem_id = 0;

  bool enable_sgpr_private_segment_buffer = false;
  bool enable_sgpr_dispatch_ptr = false;
  bool enable_sgpr_queue_ptr = false;
  bool enable_sgpr_kernarg_segment_ptr = false;
  bool enable_sgpr_dispatch_id = false;
  bool enable_sgpr_flat_scratch_init = false;
  bool enable_sgpr_private_segment_size = false;
  bool enable_wavefront_size32 = false;
  bool uses_dynamic_stack = false;
  uint8_t kernarg_preload_spec_length = 0;
  uint16_t kernarg_preload_spec_offset = 0;
};

struct EncodedProgramObject {
  std::string kernel_name;
  MetadataBlob metadata;
  AmdgpuKernelDescriptor kernel_descriptor;
  std::vector<std::byte> code_bytes;
  std::vector<RawGcnInstruction> instructions;
  std::vector<DecodedInstruction> decoded_instructions;
  std::vector<InstructionObjectPtr> instruction_objects;
};

}  // namespace gpu_model
