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

struct GcnGeneratedProfileDef {
  const char* id;
  const char* display_name;
  uint32_t wave_size;
  bool has_accvgpr;
  const char* waitcnt_layout;
};

struct GcnGeneratedOperandKindDef {
  const char* id;
  const char* category;
  const char* description;
};

struct GcnGeneratedSemanticFamilyDef {
  const char* id;
  const char* description;
};

enum GcnGeneratedInstFlags : uint64_t {
  kGcnInstFlagNone = 0,
  kGcnInstFlagIsBranch = 1ull << 0,
  kGcnInstFlagIsMemory = 1ull << 1,
  kGcnInstFlagIsAtomic = 1ull << 2,
  kGcnInstFlagIsBarrier = 1ull << 3,
  kGcnInstFlagWritesExec = 1ull << 4,
  kGcnInstFlagWritesVcc = 1ull << 5,
  kGcnInstFlagWritesScc = 1ull << 6,
  kGcnInstFlagIsWaitcnt = 1ull << 7,
  kGcnInstFlagIsMatrix = 1ull << 8,
};

struct GcnGeneratedImplicitRegRef {
  const char* name;
  bool is_write;
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
  uint64_t flags;
  uint16_t implicit_begin;
  uint16_t implicit_count;
};

const std::vector<GcnGeneratedProfileDef>& GeneratedGcnProfileDefs();
const std::vector<GcnGeneratedOperandKindDef>& GeneratedGcnOperandKindDefs();
const std::vector<GcnGeneratedSemanticFamilyDef>& GeneratedGcnSemanticFamilyDefs();
const std::vector<GcnGeneratedImplicitRegRef>& GeneratedGcnImplicitRegRefs();
const std::vector<GcnGeneratedFieldRef>& GeneratedGcnFieldRefs();
const std::vector<GcnGeneratedFormatDef>& GeneratedGcnFormatDefs();
const std::vector<GcnGeneratedInstDef>& GeneratedGcnInstDefs();
const std::vector<GcnInstEncodingDef>& GeneratedGcnEncodingDefs();

}  // namespace gpu_model
