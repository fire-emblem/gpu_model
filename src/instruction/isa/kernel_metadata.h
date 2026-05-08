#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "instruction/isa/metadata.h"

namespace gpu_model {

enum class KernelArgValueKind {
  Unknown,
  GlobalBuffer,
  ByValue,
};

enum class KernelHiddenArgKind {
  Unknown,
  BlockCountX,
  BlockCountY,
  BlockCountZ,
  GroupSizeX,
  GroupSizeY,
  GroupSizeZ,
  RemainderX,
  RemainderY,
  RemainderZ,
  GlobalOffsetX,
  GlobalOffsetY,
  GlobalOffsetZ,
  GridDims,
  DynamicLdsSize,
  PrivateBase,
  SharedBase,
  QueuePtr,
  HostcallBuffer,
  MultigridSyncArg,
  HeapV1,
  DefaultQueue,
  CompletionAction,
  None,
};

struct KernelArgLayoutEntry {
  KernelArgValueKind kind = KernelArgValueKind::Unknown;
  std::string kind_name;
  std::optional<uint32_t> offset;
  uint32_t size = 0;
};

struct KernelHiddenArgLayoutEntry {
  KernelHiddenArgKind kind = KernelHiddenArgKind::Unknown;
  std::string kind_name;
  uint32_t offset = 0;
  uint32_t size = 0;
};

/// NoteKernelArgLayoutEntry — kernel argument layout from AMDGPU notes
struct NoteKernelArgLayoutEntry {
  KernelArgValueKind arg_kind = KernelArgValueKind::Unknown;
  KernelHiddenArgKind hidden_kind = KernelHiddenArgKind::Unknown;
  std::string kind_name;
  uint32_t offset = 0;
  uint32_t size = 0;
};

/// NoteKernelMetadata — kernel metadata from AMDGPU notes
struct NoteKernelMetadata {
  std::string name;
  std::vector<NoteKernelArgLayoutEntry> args;
  std::vector<NoteKernelArgLayoutEntry> hidden_args;
  uint32_t group_segment_fixed_size = 0;
  uint32_t kernarg_segment_size = 0;
  uint32_t private_segment_fixed_size = 0;
  uint32_t sgpr_count = 0;
  uint32_t vgpr_count = 0;
  uint32_t agpr_count = 0;
  uint32_t wavefront_size = 0;
  bool uniform_work_group_size = false;
  std::string symbol;
};

struct KernelLaunchMetadata {
  std::optional<std::string> arch;
  std::optional<std::string> entry;
  std::vector<std::string> module_kernels;
  std::optional<uint32_t> arg_count;
  std::optional<uint32_t> required_shared_bytes;
  std::optional<uint32_t> group_segment_fixed_size;
  std::optional<uint32_t> block_dim_multiple;
  std::optional<uint32_t> max_block_dim;
  std::optional<uint32_t> kernarg_segment_size;
  std::vector<KernelArgLayoutEntry> arg_layout;
  std::vector<KernelHiddenArgLayoutEntry> hidden_arg_layout;
};

KernelLaunchMetadata ParseKernelLaunchMetadata(const MetadataBlob& metadata);
uint32_t EstimateVisibleKernargBytes(const KernelLaunchMetadata& metadata);
uint32_t RequiredKernargTemplateBytes(const KernelLaunchMetadata& metadata);
KernelArgValueKind ParseKernelArgValueKind(std::string_view text);
KernelHiddenArgKind ParseKernelHiddenArgKind(std::string_view text);
std::string_view ToString(KernelArgValueKind kind);
std::string_view ToString(KernelHiddenArgKind kind);

}  // namespace gpu_model
