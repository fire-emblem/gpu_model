#include "gpu_model/instruction/semantics/internal/handler_support.h"

namespace gpu_model {
namespace semantics {

using handler_support::BaseHandler;
using handler_support::FlatAtomicOperands;
using handler_support::HandlerRegistry;
using handler_support::RequireCanonicalOpcode;
using handler_support::RequireVectorRange;
using handler_support::ResolveFlatAtomicAddress;
using handler_support::ResolveFlatAtomicOperands;
using handler_support::ResolveScalarLike;
using handler_support::ResolveSharedAtomicOperands;
using handler_support::ResolveVectorLane;
using handler_support::StoreFlatAtomicReturnValue;
using handler_support::StoreSharedAtomicReturnValue;
using handler_support::ThrowUnsupportedInstruction;
// These are in gpu_model namespace, not handler_support:
// MaskFromU64, RequireScalarIndex, RequireScalarRange, RequireVectorIndex,
// ResolveScalarPair, LaneCount, LoadU32, StoreU32 are available via transitive includes

class FlatMemoryHandler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    MemoryRequest request;
    request.space = MemorySpace::Global;
    if (instruction.mnemonic == "global_load_dword") {
      request.kind = AccessKind::Load;
      const int64_t offset = instruction.operands.size() >= 3 && instruction.operands.back().info.has_immediate
                                 ? instruction.operands.back().info.immediate
                                 : 0;
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      request.dst = RegRef{.file = RegisterFile::Vector, .index = vdst};
      if (instruction.operands.at(1).kind == DecodedInstructionOperandKind::VectorRegRange) {
        // Flat-style: address is in a vector register pair
        const auto [addr, _] = RequireVectorRange(instruction.operands.at(1));
        for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
          if (!context.wave.exec.test(lane)) {
            continue;
          }
          const uint64_t lo = static_cast<uint32_t>(context.wave.vgpr.Read(addr, lane));
          const uint64_t hi = static_cast<uint32_t>(context.wave.vgpr.Read(addr + 1, lane));
          const uint64_t address = static_cast<uint64_t>(static_cast<int64_t>((hi << 32u) | lo) + offset);
          const uint32_t value = context.memory.LoadGlobalValue<uint32_t>(address);
          request.exec_snapshot.set(lane);
          request.lanes[lane] = LaneAccess{
              .active = true,
              .addr = address,
              .bytes = 4,
              .value = value,
              .has_read_value = true,
              .read_value = value,
          };
          context.wave.vgpr.Write(vdst, lane, value);
        }
      } else {
        // Scalar-base: address = saddr + vaddr + offset
        // Decoded operand order: [vdst, saddr, vaddr, offset]
        const auto [saddr, _] = RequireScalarRange(instruction.operands.at(1));
        const uint32_t vaddr = RequireVectorIndex(instruction.operands.at(2));
        const uint64_t base = static_cast<uint64_t>(context.wave.sgpr.Read(saddr)) |
                              (static_cast<uint64_t>(context.wave.sgpr.Read(saddr + 1)) << 32u);
        for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
          if (!context.wave.exec.test(lane)) {
            continue;
          }
          const int32_t voffset = static_cast<int32_t>(context.wave.vgpr.Read(vaddr, lane));
          const uint64_t address = static_cast<uint64_t>(static_cast<int64_t>(base) + voffset + offset);
          const uint32_t value = context.memory.LoadGlobalValue<uint32_t>(address);
          request.exec_snapshot.set(lane);
          request.lanes[lane] = LaneAccess{
              .active = true,
              .addr = address,
              .bytes = 4,
              .value = value,
              .has_read_value = true,
              .read_value = value,
          };
          context.wave.vgpr.Write(vdst, lane, value);
        }
      }
      ++context.stats.global_loads;
    } else if (instruction.mnemonic == "global_store_dword") {
      request.kind = AccessKind::Store;
      const int64_t offset = instruction.operands.size() >= 4 && instruction.operands.back().info.has_immediate
                                 ? instruction.operands.back().info.immediate
                                 : 0;
      if (instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange) {
        const auto [addr, _] = RequireVectorRange(instruction.operands.at(0));
        const uint32_t data = RequireVectorIndex(instruction.operands.at(1));
        for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
          if (!context.wave.exec.test(lane)) {
            continue;
          }
          const uint64_t lo = static_cast<uint32_t>(context.wave.vgpr.Read(addr, lane));
          const uint64_t hi = static_cast<uint32_t>(context.wave.vgpr.Read(addr + 1, lane));
          const uint64_t address = static_cast<uint64_t>(static_cast<int64_t>((hi << 32u) | lo) + offset);
          request.exec_snapshot.set(lane);
          request.lanes[lane] = LaneAccess{
              .active = true,
              .addr = address,
              .bytes = 4,
              .value = static_cast<uint32_t>(context.wave.vgpr.Read(data, lane)),
              .has_write_value = true,
              .write_value = static_cast<uint32_t>(context.wave.vgpr.Read(data, lane)),
          };
          context.memory.StoreGlobalValue<uint32_t>(
              address, static_cast<uint32_t>(context.wave.vgpr.Read(data, lane)));
        }
      } else {
        // AMDGPU asm syntax: global_store_dword vaddr, saddr, vdata [offset]
        // Decoded operand order: [vaddr, saddr, vdata, offset]
        // For SADDR variant: vaddr=vreg, saddr=sreg_pair, vdata=vreg
        // For flat variant: vaddr=vreg_pair (addr), vdata=vreg
        const uint32_t vaddr = RequireVectorIndex(instruction.operands.at(0));
        const auto [saddr, _] = RequireScalarRange(instruction.operands.at(1));
        const uint32_t data = RequireVectorIndex(instruction.operands.at(2));
        const uint64_t base = static_cast<uint64_t>(context.wave.sgpr.Read(saddr)) |
                              (static_cast<uint64_t>(context.wave.sgpr.Read(saddr + 1)) << 32u);
        for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
          if (!context.wave.exec.test(lane)) {
            continue;
          }
          const int32_t voffset = static_cast<int32_t>(context.wave.vgpr.Read(vaddr, lane));
          const uint64_t address = static_cast<uint64_t>(static_cast<int64_t>(base) + voffset + offset);
          const uint32_t store_val = static_cast<uint32_t>(context.wave.vgpr.Read(data, lane));
          request.exec_snapshot.set(lane);
          request.lanes[lane] = LaneAccess{
              .active = true,
              .addr = address,
              .bytes = 4,
              .value = store_val,
              .has_write_value = true,
              .write_value = store_val,
          };
          context.memory.StoreGlobalValue<uint32_t>(address, store_val);
        }
      }
      ++context.stats.global_stores;
    } else if (instruction.mnemonic == "global_atomic_add") {
      request.kind = AccessKind::Atomic;
      const FlatAtomicOperands operands = ResolveFlatAtomicOperands(instruction);
      const uint32_t data = RequireVectorIndex(*operands.data);
      if (operands.return_dest != nullptr) {
        request.dst = RegRef{.file = RegisterFile::Vector,
                             .index = RequireVectorIndex(*operands.return_dest)};
      }
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint64_t address = ResolveFlatAtomicAddress(operands, context, lane);
        const uint32_t old_value = context.memory.LoadGlobalValue<uint32_t>(address);
        const uint32_t add_value = static_cast<uint32_t>(context.wave.vgpr.Read(data, lane));
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = address,
            .bytes = 4,
            .value = add_value,
            .has_read_value = true,
            .read_value = old_value,
            .has_write_value = true,
            .write_value = old_value + add_value,
        };
        context.memory.StoreGlobalValue<uint32_t>(address, old_value + add_value);
        StoreFlatAtomicReturnValue(operands, context, lane, old_value);
      }
      ++context.stats.global_loads;
      ++context.stats.global_stores;
    } else if (instruction.mnemonic == "global_atomic_smin") {
      request.kind = AccessKind::Atomic;
      const FlatAtomicOperands operands = ResolveFlatAtomicOperands(instruction);
      const uint32_t data = RequireVectorIndex(*operands.data);
      if (operands.return_dest != nullptr) {
        request.dst = RegRef{.file = RegisterFile::Vector,
                             .index = RequireVectorIndex(*operands.return_dest)};
      }
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint64_t address = ResolveFlatAtomicAddress(operands, context, lane);
        const int32_t old_value = context.memory.LoadGlobalValue<int32_t>(address);
        const int32_t new_value = static_cast<int32_t>(context.wave.vgpr.Read(data, lane));
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = address,
            .bytes = 4,
            .value = static_cast<uint32_t>(new_value),
            .has_read_value = true,
            .read_value = static_cast<uint32_t>(old_value),
            .has_write_value = true,
            .write_value = static_cast<uint32_t>(std::min(old_value, new_value)),
        };
        context.memory.StoreGlobalValue<int32_t>(address, std::min(old_value, new_value));
        StoreFlatAtomicReturnValue(operands, context, lane, static_cast<uint32_t>(old_value));
      }
      ++context.stats.global_loads;
      ++context.stats.global_stores;
    } else if (instruction.mnemonic == "global_atomic_smax") {
      request.kind = AccessKind::Atomic;
      const FlatAtomicOperands operands = ResolveFlatAtomicOperands(instruction);
      const uint32_t data = RequireVectorIndex(*operands.data);
      if (operands.return_dest != nullptr) {
        request.dst = RegRef{.file = RegisterFile::Vector,
                             .index = RequireVectorIndex(*operands.return_dest)};
      }
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint64_t address = ResolveFlatAtomicAddress(operands, context, lane);
        const int32_t old_value = context.memory.LoadGlobalValue<int32_t>(address);
        const int32_t new_value = static_cast<int32_t>(context.wave.vgpr.Read(data, lane));
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = address,
            .bytes = 4,
            .value = static_cast<uint32_t>(new_value),
            .has_read_value = true,
            .read_value = static_cast<uint32_t>(old_value),
            .has_write_value = true,
            .write_value = static_cast<uint32_t>(std::max(old_value, new_value)),
        };
        context.memory.StoreGlobalValue<int32_t>(address, std::max(old_value, new_value));
        StoreFlatAtomicReturnValue(operands, context, lane, static_cast<uint32_t>(old_value));
      }
      ++context.stats.global_loads;
      ++context.stats.global_stores;
    } else if (instruction.mnemonic == "global_atomic_swap") {
      request.kind = AccessKind::Atomic;
      const FlatAtomicOperands operands = ResolveFlatAtomicOperands(instruction);
      const uint32_t data = RequireVectorIndex(*operands.data);
      if (operands.return_dest != nullptr) {
        request.dst = RegRef{.file = RegisterFile::Vector,
                             .index = RequireVectorIndex(*operands.return_dest)};
      }
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint64_t address = ResolveFlatAtomicAddress(operands, context, lane);
        const uint32_t old_value = context.memory.LoadGlobalValue<uint32_t>(address);
        const uint32_t new_value = static_cast<uint32_t>(context.wave.vgpr.Read(data, lane));
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = address,
            .bytes = 4,
            .value = new_value,
            .has_read_value = true,
            .read_value = old_value,
            .has_write_value = true,
            .write_value = new_value,
        };
        context.memory.StoreGlobalValue<uint32_t>(address, new_value);
        StoreFlatAtomicReturnValue(operands, context, lane, old_value);
      }
      ++context.stats.global_loads;
      ++context.stats.global_stores;
    } else {
      ThrowUnsupportedInstruction("unsupported flat memory opcode: ", instruction);
    }
    if (context.captured_memory_request != nullptr) {
      *context.captured_memory_request = request;
    }
  }
};

class BufferMemoryHandler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    MemoryRequest request;
    request.space = MemorySpace::Global;
    request.kind = AccessKind::Atomic;

    // MUBUF atomic instructions use buffer descriptor (4 scalar registers) + vector offset
    // Format: buffer_atomic_* vdst, data, s[base:base+3], voffset
    // For simplicity, we extract the base address from the buffer descriptor

    if (instruction.mnemonic == "buffer_atomic_smax") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t data = RequireVectorIndex(instruction.operands.at(1));
      // Buffer descriptor is in scalar registers (4 consecutive registers)
      // For now, we use a simplified model where the base address is in s[base:base+1]
      const auto [sbase, _] = RequireScalarRange(instruction.operands.at(2));
      const uint64_t base = static_cast<uint64_t>(context.wave.sgpr.Read(sbase)) |
                            (static_cast<uint64_t>(context.wave.sgpr.Read(sbase + 1)) << 32u);
      request.dst = RegRef{.file = RegisterFile::Vector, .index = vdst};
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const int32_t old_value = context.memory.LoadGlobalValue<int32_t>(base);
        const int32_t new_value = static_cast<int32_t>(context.wave.vgpr.Read(data, lane));
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = base,
            .bytes = 4,
            .value = static_cast<uint32_t>(new_value),
            .has_read_value = true,
            .read_value = static_cast<uint32_t>(old_value),
            .has_write_value = true,
            .write_value = static_cast<uint32_t>(std::max(old_value, new_value)),
        };
        context.memory.StoreGlobalValue<int32_t>(base, std::max(old_value, new_value));
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(old_value));
      }
      ++context.stats.global_loads;
      ++context.stats.global_stores;
    } else if (instruction.mnemonic == "buffer_atomic_smin") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t data = RequireVectorIndex(instruction.operands.at(1));
      const auto [sbase, _] = RequireScalarRange(instruction.operands.at(2));
      const uint64_t base = static_cast<uint64_t>(context.wave.sgpr.Read(sbase)) |
                            (static_cast<uint64_t>(context.wave.sgpr.Read(sbase + 1)) << 32u);
      request.dst = RegRef{.file = RegisterFile::Vector, .index = vdst};
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const int32_t old_value = context.memory.LoadGlobalValue<int32_t>(base);
        const int32_t new_value = static_cast<int32_t>(context.wave.vgpr.Read(data, lane));
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = base,
            .bytes = 4,
            .value = static_cast<uint32_t>(new_value),
            .has_read_value = true,
            .read_value = static_cast<uint32_t>(old_value),
            .has_write_value = true,
            .write_value = static_cast<uint32_t>(std::min(old_value, new_value)),
        };
        context.memory.StoreGlobalValue<int32_t>(base, std::min(old_value, new_value));
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(old_value));
      }
      ++context.stats.global_loads;
      ++context.stats.global_stores;
    } else if (instruction.mnemonic == "buffer_atomic_swap") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t data = RequireVectorIndex(instruction.operands.at(1));
      const auto [sbase, _] = RequireScalarRange(instruction.operands.at(2));
      const uint64_t base = static_cast<uint64_t>(context.wave.sgpr.Read(sbase)) |
                            (static_cast<uint64_t>(context.wave.sgpr.Read(sbase + 1)) << 32u);
      request.dst = RegRef{.file = RegisterFile::Vector, .index = vdst};
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t old_value = context.memory.LoadGlobalValue<uint32_t>(base);
        const uint32_t new_value = static_cast<uint32_t>(context.wave.vgpr.Read(data, lane));
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = base,
            .bytes = 4,
            .value = new_value,
            .has_read_value = true,
            .read_value = old_value,
            .has_write_value = true,
            .write_value = new_value,
        };
        context.memory.StoreGlobalValue<uint32_t>(base, new_value);
        context.wave.vgpr.Write(vdst, lane, old_value);
      }
      ++context.stats.global_loads;
      ++context.stats.global_stores;
    } else {
      ThrowUnsupportedInstruction("unsupported buffer memory opcode: ", instruction);
    }
    if (context.captured_memory_request != nullptr) {
      *context.captured_memory_request = request;
    }
  }
};

class SharedMemoryHandler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    MemoryRequest request;
    request.space = MemorySpace::Shared;
    if (instruction.mnemonic == "ds_write_b32") {
      request.kind = AccessKind::Store;
      const uint32_t addr_vgpr = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t data_vgpr = RequireVectorIndex(instruction.operands.at(1));
      const uint32_t offset = instruction.operands.size() >= 3 && instruction.operands.back().info.has_immediate
                                  ? static_cast<uint32_t>(instruction.operands.back().info.immediate)
                                  : 0u;
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        uint32_t byte_offset = static_cast<uint32_t>(context.wave.vgpr.Read(addr_vgpr, lane));
        if (!context.wave.exec.test(lane) || context.block.shared_memory.empty()) {
          continue;
        }
        byte_offset += offset;
        byte_offset %= static_cast<uint32_t>(context.block.shared_memory.size());
        if (static_cast<size_t>(byte_offset) + sizeof(uint32_t) > context.block.shared_memory.size()) {
          continue;
        }
        const uint32_t value =
            static_cast<uint32_t>(context.wave.vgpr.Read(data_vgpr, lane));
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = byte_offset,
            .bytes = 4,
            .value = value,
            .has_write_value = true,
            .write_value = value,
        };
        StoreU32(context.block.shared_memory, byte_offset, value);
      }
      ++context.stats.shared_stores;
      if (context.captured_memory_request != nullptr) {
        *context.captured_memory_request = request;
      }
      return;
    }
    if (instruction.mnemonic == "ds_read_b32") {
      request.kind = AccessKind::Load;
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t addr_vgpr = RequireVectorIndex(instruction.operands.at(1));
      const uint32_t offset = instruction.operands.size() >= 3 && instruction.operands.back().info.has_immediate
                                  ? static_cast<uint32_t>(instruction.operands.back().info.immediate)
                                  : 0u;
      request.dst = RegRef{.file = RegisterFile::Vector, .index = vdst};
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        uint32_t byte_offset = static_cast<uint32_t>(context.wave.vgpr.Read(addr_vgpr, lane));
        if (!context.wave.exec.test(lane) || context.block.shared_memory.empty()) {
          continue;
        }
        byte_offset += offset;
        byte_offset %= static_cast<uint32_t>(context.block.shared_memory.size());
        if (static_cast<size_t>(byte_offset) + sizeof(uint32_t) > context.block.shared_memory.size()) {
          continue;
        }
        request.exec_snapshot.set(lane);
        const uint32_t value = LoadU32(context.block.shared_memory, byte_offset);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = byte_offset,
            .bytes = 4,
            .value = value,
            .has_read_value = true,
            .read_value = value,
        };
        context.wave.vgpr.Write(vdst, lane, value);
      }
      ++context.stats.shared_loads;
      if (context.captured_memory_request != nullptr) {
        *context.captured_memory_request = request;
      }
      return;
    }
    if (instruction.mnemonic == "ds_read2_b32") {
      const auto [vdst, _] = RequireVectorRange(instruction.operands.at(0));
      const uint32_t addr_vgpr = RequireVectorIndex(instruction.operands.at(1));
      const uint32_t offset0 =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const uint32_t offset1 =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(3), context));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t base_byte_offset =
            static_cast<uint32_t>(context.wave.vgpr.Read(addr_vgpr, lane));
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t byte_offset0 = base_byte_offset + offset0 * sizeof(uint32_t);
        const uint32_t byte_offset1 = base_byte_offset + offset1 * sizeof(uint32_t);
        if (static_cast<size_t>(byte_offset0) + sizeof(uint32_t) > context.block.shared_memory.size() ||
            static_cast<size_t>(byte_offset1) + sizeof(uint32_t) > context.block.shared_memory.size()) {
          continue;
        }
        context.wave.vgpr.Write(vdst, lane, LoadU32(context.block.shared_memory, byte_offset0));
        context.wave.vgpr.Write(vdst + 1, lane, LoadU32(context.block.shared_memory, byte_offset1));
      }
      ++context.stats.shared_loads;
      return;
    }
    // DS atomic operations
    if (instruction.mnemonic == "ds_add_u32") {
      request.kind = AccessKind::Atomic;
      const auto operands = ResolveSharedAtomicOperands(instruction);
      const uint32_t addr_vgpr = RequireVectorIndex(*operands.address);
      const uint32_t data_vgpr = RequireVectorIndex(*operands.data);
      if (operands.return_dest != nullptr) {
        request.dst = RegRef{
            .file = RegisterFile::Vector,
            .index = RequireVectorIndex(*operands.return_dest),
        };
      }
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        uint32_t byte_offset = static_cast<uint32_t>(context.wave.vgpr.Read(addr_vgpr, lane));
        if (!context.wave.exec.test(lane) || context.block.shared_memory.empty()) {
          continue;
        }
        byte_offset += operands.offset;
        byte_offset %= static_cast<uint32_t>(context.block.shared_memory.size());
        if (static_cast<size_t>(byte_offset) + sizeof(uint32_t) > context.block.shared_memory.size()) {
          continue;
        }
        const uint32_t old_value = LoadU32(context.block.shared_memory, byte_offset);
        const uint32_t add_value = static_cast<uint32_t>(context.wave.vgpr.Read(data_vgpr, lane));
        StoreU32(context.block.shared_memory, byte_offset, old_value + add_value);
        StoreSharedAtomicReturnValue(operands, context, lane, old_value);
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = byte_offset,
            .bytes = 4,
            .value = add_value,
            .has_read_value = true,
            .read_value = old_value,
            .has_write_value = true,
            .write_value = old_value + add_value,
        };
      }
      ++context.stats.shared_loads;
      ++context.stats.shared_stores;
      if (context.captured_memory_request != nullptr) {
        *context.captured_memory_request = request;
      }
      return;
    }
    if (instruction.mnemonic == "ds_min_i32") {
      request.kind = AccessKind::Atomic;
      const auto operands = ResolveSharedAtomicOperands(instruction);
      const uint32_t addr_vgpr = RequireVectorIndex(*operands.address);
      const uint32_t data_vgpr = RequireVectorIndex(*operands.data);
      if (operands.return_dest != nullptr) {
        request.dst = RegRef{
            .file = RegisterFile::Vector,
            .index = RequireVectorIndex(*operands.return_dest),
        };
      }
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        uint32_t byte_offset = static_cast<uint32_t>(context.wave.vgpr.Read(addr_vgpr, lane));
        if (!context.wave.exec.test(lane) || context.block.shared_memory.empty()) {
          continue;
        }
        byte_offset += operands.offset;
        byte_offset %= static_cast<uint32_t>(context.block.shared_memory.size());
        if (static_cast<size_t>(byte_offset) + sizeof(uint32_t) > context.block.shared_memory.size()) {
          continue;
        }
        const int32_t old_value = static_cast<int32_t>(LoadU32(context.block.shared_memory, byte_offset));
        const int32_t new_value = static_cast<int32_t>(context.wave.vgpr.Read(data_vgpr, lane));
        StoreU32(context.block.shared_memory, byte_offset, static_cast<uint32_t>(std::min(old_value, new_value)));
        StoreSharedAtomicReturnValue(operands, context, lane, static_cast<uint32_t>(old_value));
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = byte_offset,
            .bytes = 4,
            .value = static_cast<uint32_t>(new_value),
            .has_read_value = true,
            .read_value = static_cast<uint32_t>(old_value),
            .has_write_value = true,
            .write_value = static_cast<uint32_t>(std::min(old_value, new_value)),
        };
      }
      ++context.stats.shared_loads;
      ++context.stats.shared_stores;
      if (context.captured_memory_request != nullptr) {
        *context.captured_memory_request = request;
      }
      return;
    }
    if (instruction.mnemonic == "ds_max_i32") {
      request.kind = AccessKind::Atomic;
      const auto operands = ResolveSharedAtomicOperands(instruction);
      const uint32_t addr_vgpr = RequireVectorIndex(*operands.address);
      const uint32_t data_vgpr = RequireVectorIndex(*operands.data);
      if (operands.return_dest != nullptr) {
        request.dst = RegRef{
            .file = RegisterFile::Vector,
            .index = RequireVectorIndex(*operands.return_dest),
        };
      }
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        uint32_t byte_offset = static_cast<uint32_t>(context.wave.vgpr.Read(addr_vgpr, lane));
        if (!context.wave.exec.test(lane) || context.block.shared_memory.empty()) {
          continue;
        }
        byte_offset += operands.offset;
        byte_offset %= static_cast<uint32_t>(context.block.shared_memory.size());
        if (static_cast<size_t>(byte_offset) + sizeof(uint32_t) > context.block.shared_memory.size()) {
          continue;
        }
        const int32_t old_value = static_cast<int32_t>(LoadU32(context.block.shared_memory, byte_offset));
        const int32_t new_value = static_cast<int32_t>(context.wave.vgpr.Read(data_vgpr, lane));
        StoreU32(context.block.shared_memory, byte_offset, static_cast<uint32_t>(std::max(old_value, new_value)));
        StoreSharedAtomicReturnValue(operands, context, lane, static_cast<uint32_t>(old_value));
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = byte_offset,
            .bytes = 4,
            .value = static_cast<uint32_t>(new_value),
            .has_read_value = true,
            .read_value = static_cast<uint32_t>(old_value),
            .has_write_value = true,
            .write_value = static_cast<uint32_t>(std::max(old_value, new_value)),
        };
      }
      ++context.stats.shared_loads;
      ++context.stats.shared_stores;
      if (context.captured_memory_request != nullptr) {
        *context.captured_memory_request = request;
      }
      return;
    }
    if (instruction.mnemonic == "ds_swap_b32") {
      request.kind = AccessKind::Atomic;
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t addr_vgpr = RequireVectorIndex(instruction.operands.at(1));
      const uint32_t data_vgpr = RequireVectorIndex(instruction.operands.at(2));
      request.dst = RegRef{.file = RegisterFile::Vector, .index = vdst};
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        uint32_t byte_offset = static_cast<uint32_t>(context.wave.vgpr.Read(addr_vgpr, lane));
        if (!context.wave.exec.test(lane) || context.block.shared_memory.empty()) {
          continue;
        }
        byte_offset %= static_cast<uint32_t>(context.block.shared_memory.size());
        if (static_cast<size_t>(byte_offset) + sizeof(uint32_t) > context.block.shared_memory.size()) {
          continue;
        }
        const uint32_t old_value = LoadU32(context.block.shared_memory, byte_offset);
        const uint32_t new_value = static_cast<uint32_t>(context.wave.vgpr.Read(data_vgpr, lane));
        StoreU32(context.block.shared_memory, byte_offset, new_value);
        context.wave.vgpr.Write(vdst, lane, old_value);
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = byte_offset,
            .bytes = 4,
            .value = new_value,
            .has_read_value = true,
            .read_value = old_value,
            .has_write_value = true,
            .write_value = new_value,
        };
      }
      ++context.stats.shared_loads;
      ++context.stats.shared_stores;
      if (context.captured_memory_request != nullptr) {
        *context.captured_memory_request = request;
      }
      return;
    }
    if (instruction.mnemonic == "ds_wrxchg_rtn_b32") {
      request.kind = AccessKind::Atomic;
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t addr_vgpr = RequireVectorIndex(instruction.operands.at(1));
      const uint32_t data_vgpr = RequireVectorIndex(instruction.operands.at(2));
      const uint32_t offset = instruction.operands.size() >= 4 &&
                                      instruction.operands.back().info.has_immediate
                                  ? static_cast<uint32_t>(instruction.operands.back().info.immediate)
                                  : 0u;
      request.dst = RegRef{.file = RegisterFile::Vector, .index = vdst};
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        uint32_t byte_offset = static_cast<uint32_t>(context.wave.vgpr.Read(addr_vgpr, lane));
        if (!context.wave.exec.test(lane) || context.block.shared_memory.empty()) {
          continue;
        }
        byte_offset += offset;
        byte_offset %= static_cast<uint32_t>(context.block.shared_memory.size());
        if (static_cast<size_t>(byte_offset) + sizeof(uint32_t) > context.block.shared_memory.size()) {
          continue;
        }
        const uint32_t old_value = LoadU32(context.block.shared_memory, byte_offset);
        const uint32_t new_value = static_cast<uint32_t>(context.wave.vgpr.Read(data_vgpr, lane));
        StoreU32(context.block.shared_memory, byte_offset, new_value);
        context.wave.vgpr.Write(vdst, lane, old_value);
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = byte_offset,
            .bytes = 4,
            .value = new_value,
            .has_read_value = true,
            .read_value = old_value,
            .has_write_value = true,
            .write_value = new_value,
        };
      }
      ++context.stats.shared_loads;
      ++context.stats.shared_stores;
      if (context.captured_memory_request != nullptr) {
        *context.captured_memory_request = request;
      }
      return;
    }
    ThrowUnsupportedInstruction("unsupported shared memory opcode: ", instruction);
  }
};

// Static instances
static const FlatMemoryHandler kFlatMemoryHandler;
static const BufferMemoryHandler kBufferMemoryHandler;
static const SharedMemoryHandler kSharedMemoryHandler;

// Accessors
const IEncodedSemanticHandler& GetFlatMemoryHandler() { return kFlatMemoryHandler; }
const IEncodedSemanticHandler& GetBufferMemoryHandler() { return kBufferMemoryHandler; }
const IEncodedSemanticHandler& GetSharedMemoryHandler() { return kSharedMemoryHandler; }

// Self-registration for mnemonic-based lookup
struct MemoryHandlerRegistrar {
  MemoryHandlerRegistrar() {
    auto& registry = HandlerRegistry::MutableInstance();
    registry.Register("ds_read2_b32", &kSharedMemoryHandler);
    registry.Register("ds_read_b32", &kSharedMemoryHandler);
    registry.Register("ds_write_b32", &kSharedMemoryHandler);
    registry.Register("ds_wrxchg_rtn_b32", &kSharedMemoryHandler);
    registry.Register("buffer_atomic_smax", &kBufferMemoryHandler);
    registry.Register("buffer_atomic_smin", &kBufferMemoryHandler);
    registry.Register("buffer_atomic_swap", &kBufferMemoryHandler);
  }
};
static MemoryHandlerRegistrar s_memory_registrar;

}  // namespace semantics
}  // namespace gpu_model
