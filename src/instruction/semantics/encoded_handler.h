#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

#include "instruction/decode/encoded/decoded_instruction.h"
#include "instruction/decode/encoded/instruction_object.h"
#include "gpu_arch/memory/memory_request.h"
#include "state/memory/memory_system.h"
#include "state/wave/wave_runtime_state.h"
#include "state/execution_stats.h"

namespace gpu_model {

// Forward declarations
class TraceSink;
struct EncodedWaveContext;

/// EncodedBlockContext — block 执行上下文
///
/// 包含 block 级别的共享状态，不依赖 runtime 层。
struct EncodedBlockContext {
  std::vector<std::byte>& shared_memory;
  uint64_t& barrier_generation;
  uint32_t& barrier_arrivals;
  uint32_t wave_count = 0;
};

/// Instruction execution callback types for logging/tracing
using InstructionExecuteCallback = std::function<void(const DecodedInstruction&,
                                                       const EncodedWaveContext&,
                                                       const char* phase)>;

/// EncodedWaveContext — 编码指令执行上下文
///
/// 包含 wave 执行所需的所有状态引用。
/// 这是一个运行时上下文，但它的定义属于 state 层，
/// 因为它只是状态的组合，不包含执行逻辑。
struct EncodedWaveContext : InstructionExecutionContext {
  WaveContext& wave;
  uint64_t& vcc;
  const std::vector<std::byte>& kernarg;
  uint64_t kernarg_base = 0;
  MemorySystem& memory;
  ExecutionStats& stats;
  EncodedBlockContext& block;
  std::optional<MemoryRequest>* captured_memory_request = nullptr;
  TraceSink* trace_sink = nullptr;
  uint64_t trace_cycle = 0;
  uint32_t trace_slot_id = 0;
  InstructionExecuteCallback on_execute = nullptr;

  EncodedWaveContext(WaveContext& wave_ref,
                     uint64_t& vcc_ref,
                     const std::vector<std::byte>& kernarg_ref,
                     uint64_t kernarg_base_value,
                     MemorySystem& memory_ref,
                     ExecutionStats& stats_ref,
                     EncodedBlockContext& block_ref,
                     std::optional<MemoryRequest>* captured_memory_request_ref = nullptr,
                     TraceSink* trace_sink_ref = nullptr,
                     uint64_t trace_cycle_value = 0,
                     uint32_t trace_slot_id_value = 0,
                     InstructionExecuteCallback on_execute_fn = nullptr)
      : wave(wave_ref),
        vcc(vcc_ref),
        kernarg(kernarg_ref),
        kernarg_base(kernarg_base_value),
        memory(memory_ref),
        stats(stats_ref),
        block(block_ref),
        captured_memory_request(captured_memory_request_ref),
        trace_sink(trace_sink_ref),
        trace_cycle(trace_cycle_value),
        trace_slot_id(trace_slot_id_value),
        on_execute(std::move(on_execute_fn)) {}
};

/// IEncodedSemanticHandler — 编码指令语义处理器接口
///
/// 继承自 InstructionSemanticHandler，提供编码指令的执行接口。
/// 这个接口属于 instruction/semantics 层，因为它是指令语义的一部分。
class IEncodedSemanticHandler : public InstructionSemanticHandler {
 public:
  virtual ~IEncodedSemanticHandler() = default;
  virtual void Execute(const DecodedInstruction& instruction,
                       EncodedWaveContext& context) const = 0;
  void Execute(const DecodedInstruction& instruction,
               InstructionExecutionContext& context) const final {
    Execute(instruction, static_cast<EncodedWaveContext&>(context));
  }
};

/// EncodedSemanticHandlerRegistry — 指令处理器注册表
///
/// 提供 mnemonic -> handler 的查找服务。
class EncodedSemanticHandlerRegistry {
 public:
  static const IEncodedSemanticHandler& Get(const DecodedInstruction& instruction);
  static const IEncodedSemanticHandler& Get(std::string_view mnemonic);
};

}  // namespace gpu_model
