#pragma once

#include <cstdint>
#include <vector>

#include "gpu_model/decode/gcn_inst_encoding_def.h"
#include "gpu_model/decode/gcn_inst_format.h"

namespace gpu_model {

struct GcnGeneratedFieldRef {
  const char* name;
  uint8_t word_index;
  uint8_t lsb;
  uint8_t width;
  bool sign_extend;
  const char* meaning;
};

struct GcnGeneratedFormatDef {
  const char* id;
  GcnInstFormatClass format_class;
  uint8_t size_bytes;
  GcnGeneratedFieldRef opcode_field;
  uint16_t field_begin;
  uint16_t field_count;
};

struct GcnGeneratedInstDef {
  uint32_t id;
  const char* profile;
  GcnInstFormatClass format_class;
  uint32_t opcode;
  uint8_t size_bytes;
  const char* mnemonic;
  const char* semantic_family;
  const char* issue_family;
};

const std::vector<GcnGeneratedFieldRef>& GeneratedGcnFieldRefs();
const std::vector<GcnGeneratedFormatDef>& GeneratedGcnFormatDefs();
const std::vector<GcnGeneratedInstDef>& GeneratedGcnInstDefs();
const std::vector<GcnInstEncodingDef>& GeneratedGcnEncodingDefs();

}  // namespace gpu_model
