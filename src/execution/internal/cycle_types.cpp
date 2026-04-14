#include "execution/internal/cycle_types.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#include "execution/cycle/cycle_exec_engine.h"  // CycleTimingConfig (full definition)
#include "instruction/isa/opcode_info.h"  // GetOpcodeExecutionInfo
#include "gpu_arch/chip_config/gpu_arch_spec.h"  // GpuArchSpec
#include "gpu_arch/issue_config/issue_config.h"  // ArchitecturalIssuePolicyFromLimits

namespace gpu_model {
namespace cycle_internal {

uint64_t QuantizeIssueDuration(uint64_t cycles) {
  const uint64_t clamped = std::max<uint64_t>(kIssueTimelineQuantumCycles, cycles);
  const uint64_t remainder = clamped % kIssueTimelineQuantumCycles;
  if (remainder == 0) {
    return clamped;
  }
  return clamped + (kIssueTimelineQuantumCycles - remainder);
}

std::optional<ExecutedStepClass> ClassifyCycleInstruction(const Instruction& instruction,
                                                          const OpPlan& plan) {
  // Sync instructions: barrier, waitcnt
  if (plan.sync_barrier || plan.sync_wave_barrier || plan.wait_cnt) {
    return ExecutedStepClass::Sync;
  }

  // Vector memory instructions: global, shared, private
  if (plan.memory.has_value()) {
    return ExecutedStepClass::VectorMem;
  }

  // Classify by semantic family (hardware execution unit)
  switch (GetOpcodeExecutionInfo(instruction.opcode).family) {
    case SemanticFamily::ScalarAlu:
    case SemanticFamily::ScalarCompare:
      return ExecutedStepClass::ScalarAlu;
    case SemanticFamily::VectorAluInt:
    case SemanticFamily::VectorAluFloat:
    case SemanticFamily::VectorCompare:
      return ExecutedStepClass::VectorAlu;
    case SemanticFamily::ScalarMemory:
      return ExecutedStepClass::ScalarMem;
    case SemanticFamily::VectorMemory:
    case SemanticFamily::LocalDataShare:
      return ExecutedStepClass::VectorMem;
    case SemanticFamily::Branch:
      return ExecutedStepClass::Branch;
    case SemanticFamily::Builtin:
    case SemanticFamily::Mask:
    case SemanticFamily::Sync:
    case SemanticFamily::Special:
      return ExecutedStepClass::Other;
  }
  return ExecutedStepClass::Other;
}

uint64_t CostForCycleStep(const OpPlan& plan,
                          ExecutedStepClass step_class,
                          const ProgramCycleStatsConfig& config) {
  switch (step_class) {
    case ExecutedStepClass::ScalarAlu:
    case ExecutedStepClass::VectorAlu:
    case ExecutedStepClass::Branch:
    case ExecutedStepClass::Sync:
      return plan.issue_cycles;
    case ExecutedStepClass::Tensor:
      return config.tensor_cycles;
    case ExecutedStepClass::ScalarMem:
      return config.scalar_mem_cycles;
    case ExecutedStepClass::VectorMem:
      // VectorMem includes global, shared, private memory
      // Use global_mem_cycles as default (dominant case)
      return config.global_mem_cycles;
    case ExecutedStepClass::Other:
      return plan.issue_cycles == 0 ? config.default_issue_cycles : plan.issue_cycles;
  }
  return config.default_issue_cycles;
}

void AccumulateProgramCycleStep(ProgramCycleStats& stats,
                                ExecutedStepClass step_class,
                                uint64_t cost_cycles,
                                uint64_t work_weight) {
  const uint64_t weighted_cycles = cost_cycles * work_weight;
  stats.total_issued_work_cycles += weighted_cycles;
  switch (step_class) {
    case ExecutedStepClass::ScalarAlu:
      stats.scalar_alu_cycles += weighted_cycles;
      stats.scalar_alu_insts += 1;
      return;
    case ExecutedStepClass::ScalarMem:
      stats.scalar_mem_cycles += weighted_cycles;
      stats.scalar_mem_insts += 1;
      return;
    case ExecutedStepClass::VectorAlu:
      stats.vector_alu_cycles += weighted_cycles;
      stats.vector_alu_insts += 1;
      return;
    case ExecutedStepClass::VectorMem:
      stats.global_mem_cycles += weighted_cycles;
      stats.vector_mem_insts += 1;
      return;
    case ExecutedStepClass::Branch:
      stats.branch_insts += 1;
      return;
    case ExecutedStepClass::Sync:
      stats.barrier_cycles += weighted_cycles;
      stats.sync_insts += 1;
      return;
    case ExecutedStepClass::Tensor:
      stats.tensor_cycles += weighted_cycles;
      stats.tensor_insts += 1;
      return;
    case ExecutedStepClass::Other:
      stats.other_insts += 1;
      return;
  }
}

bool IssueLimitsUnset(const ArchitecturalIssueLimits& limits) {
  return limits.branch == 0 && limits.scalar_alu_or_memory == 0 && limits.vector_alu == 0 &&
         limits.vector_memory == 0 && limits.local_data_share == 0 &&
         limits.global_data_share_or_export == 0 && limits.special == 0;
}

ArchitecturalIssuePolicy ResolveIssuePolicy(const CycleTimingConfig& timing_config,
                                            const GpuArchSpec& spec) {
  if (timing_config.issue_policy.has_value()) {
    return *timing_config.issue_policy;
  }
  if (IssueLimitsUnset(timing_config.issue_limits)) {
    return CycleIssuePolicyForSpec(spec);
  }
  return ArchitecturalIssuePolicyFromLimits(timing_config.issue_limits);
}

uint64_t ModeledAsyncCompletionDelay(uint32_t issue_cycles, uint32_t default_issue_cycles) {
  if (issue_cycles <= default_issue_cycles) {
    return 0;
  }
  return static_cast<uint64_t>(issue_cycles - default_issue_cycles);
}

}  // namespace cycle_internal
}  // namespace gpu_model
