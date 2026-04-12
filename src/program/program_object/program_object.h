#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "instruction/decode/encoded/decoded_instruction.h"
#include "instruction/decode/encoded/encoded_gcn_instruction.h"
#include "instruction/decode/encoded/instruction_object.h"
#include "instruction/isa/metadata.h"

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

struct ConstSegment {
  std::vector<std::byte> bytes;
};

struct DataSegment {
  std::vector<std::byte> bytes;
};

class ProgramObject {
 public:
  ProgramObject() = default;
  ProgramObject(std::string kernel_name,
                std::string assembly_text,
                MetadataBlob metadata = {},
                ConstSegment const_segment = {},
                DataSegment data_segment = {})
      : kernel_name_(std::move(kernel_name)),
        assembly_text_(std::move(assembly_text)),
        metadata_(std::move(metadata)),
        const_segment_(std::move(const_segment)),
        data_segment_(std::move(data_segment)) {}

  const std::string& kernel_name() const { return kernel_name_; }
  std::string& kernel_name() { return kernel_name_; }
  const std::string& assembly_text() const { return assembly_text_; }
  std::string& assembly_text() { return assembly_text_; }
  const MetadataBlob& metadata() const { return metadata_; }
  MetadataBlob& metadata() { return metadata_; }
  const ConstSegment& const_segment() const { return const_segment_; }
  ConstSegment& const_segment() { return const_segment_; }
  const DataSegment& data_segment() const { return data_segment_; }
  DataSegment& data_segment() { return data_segment_; }
  const AmdgpuKernelDescriptor& kernel_descriptor() const { return kernel_descriptor_; }
  AmdgpuKernelDescriptor& kernel_descriptor() { return kernel_descriptor_; }
  const std::vector<std::byte>& code_bytes() const { return code_bytes_; }
  std::vector<std::byte>& code_bytes() { return code_bytes_; }
  const std::vector<EncodedGcnInstruction>& instructions() const { return instructions_; }
  std::vector<EncodedGcnInstruction>& instructions() { return instructions_; }
  const std::vector<DecodedInstruction>& decoded_instructions() const {
    return decoded_instructions_;
  }
  std::vector<DecodedInstruction>& decoded_instructions() { return decoded_instructions_; }
  const std::vector<InstructionObjectPtr>& instruction_objects() const {
    return instruction_objects_;
  }
  std::vector<InstructionObjectPtr>& instruction_objects() { return instruction_objects_; }
  bool has_encoded_payload() const {
    return !code_bytes_.empty() || !decoded_instructions_.empty() || !instruction_objects_.empty();
  }

  void set_kernel_name(std::string kernel_name) { kernel_name_ = std::move(kernel_name); }
  void set_assembly_text(std::string assembly_text) { assembly_text_ = std::move(assembly_text); }
  void set_metadata(MetadataBlob metadata) { metadata_ = std::move(metadata); }
  void set_const_segment(ConstSegment const_segment) { const_segment_ = std::move(const_segment); }
  void set_data_segment(DataSegment data_segment) { data_segment_ = std::move(data_segment); }
  void set_kernel_descriptor(AmdgpuKernelDescriptor kernel_descriptor) {
    kernel_descriptor_ = std::move(kernel_descriptor);
  }
  void set_code_bytes(std::vector<std::byte> code_bytes) { code_bytes_ = std::move(code_bytes); }
  void set_instructions(std::vector<EncodedGcnInstruction> instructions) {
    instructions_ = std::move(instructions);
  }
  void set_decoded_instructions(std::vector<DecodedInstruction> decoded_instructions) {
    decoded_instructions_ = std::move(decoded_instructions);
  }
  void set_instruction_objects(std::vector<InstructionObjectPtr> instruction_objects) {
    instruction_objects_ = std::move(instruction_objects);
  }

 private:
  std::string kernel_name_;
  std::string assembly_text_;
  MetadataBlob metadata_;
  ConstSegment const_segment_;
  DataSegment data_segment_;
  AmdgpuKernelDescriptor kernel_descriptor_;
  std::vector<std::byte> code_bytes_;
  std::vector<EncodedGcnInstruction> instructions_;
  std::vector<DecodedInstruction> decoded_instructions_;
  std::vector<InstructionObjectPtr> instruction_objects_;
};

}  // namespace gpu_model
