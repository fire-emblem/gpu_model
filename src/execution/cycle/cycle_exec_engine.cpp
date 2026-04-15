#include "execution/cycle/cycle_exec_engine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#include "debug/trace/event.h"
#include "debug/trace/event_factory.h"
#include "debug/trace/instruction_trace.h"
#include "debug/trace/wave_launch_trace.h"
#include "execution/internal/plan/semantics.h"
#include "execution/internal/sync_ops/barrier_resource_pool.h"
#include "execution/internal/cost_model/cycle_issue_policy.h"
#include "execution/internal/cost_model/cycle_types.h"
#include "execution/internal/issue_logic/cycle_issue_schedule.h"
#include "execution/internal/wave_schedule/cycle_wave_schedule.h"
#include "execution/internal/wave_schedule/event_queue.h"
#include "gpu_arch/issue_config/issue_config.h"
#include "execution/internal/issue_logic/issue_scheduler.h"
#include "instruction/isa/opcode_info.h"
#include "execution/internal/commit_logic/async_scoreboard.h"
#include "execution/internal/commit_logic/memory_ops.h"
#include "execution/internal/commit_logic/plan_apply.h"
#include "execution/internal/sync_ops/sync_ops.h"
#include "state/wave/barrier_state.h"
#include "execution/internal/block_schedule/wave_context_builder.h"
#include "execution/internal/issue_logic/issue_eligibility.h"
#include "instruction/isa/opcode.h"
#include "program/loader/device_image_loader.h"
#include "state/memory/cache_model.h"
#include "state/memory/shared_bank_model.h"
#include "runtime/model_runtime/program_cycle_tracker.h"
#include "runtime/model_runtime/program_cycle_stats.h"

namespace gpu_model {

CycleExecEngine::CycleExecEngine(CycleTimingConfig timing_config)
    : timing_config_(timing_config), semantics_(std::make_unique<Semantics>()) {}

CycleExecEngine::~CycleExecEngine() = default;

using namespace cycle_internal;

namespace {

uint64_t LoadLaneValue(const MemorySystem& memory, MemoryPoolKind pool, const LaneAccess& lane) {
  return memory_ops::LoadPoolLaneValue(memory, pool, lane);
}

uint64_t LoadLaneValue(const MemorySystem& memory, const LaneAccess& lane) {
  return memory_ops::LoadGlobalLaneValue(memory, lane);
}

uint64_t ConstantPoolBase(const ExecutionContext& context) {
  if (context.device_load == nullptr) {
    return 0;
  }
  for (const auto& segment : context.device_load->segments) {
    if (segment.segment.kind == DeviceSegmentKind::ConstantData) {
      return segment.allocation.range.base;
    }
  }
  return 0;
}

bool ResumeWaitcntWaveIfReady(const ExecutableKernel& kernel, WaveContext& wave) {
  if (wave.run_state != WaveRunState::Waiting) {
    return false;
  }
  if (!kernel.ContainsPc(wave.pc)) {
    return false;
  }
  const Instruction& instruction = kernel.InstructionAtPc(wave.pc);
  if (instruction.opcode != Opcode::SWaitCnt ||
      !ResumeMemoryWaitStateIfSatisfied(WaitCntThresholdsForInstruction(instruction), wave)) {
    return false;
  }
  const auto next_pc = kernel.NextPc(wave.pc);
  if (!next_pc.has_value()) {
    throw std::out_of_range("next instruction pc not found");
  }
  wave.pc = *next_pc;
  wave.valid_entry = true;
  if (wave.status != WaveStatus::Exited && !wave.waiting_at_barrier) {
    wave.status = WaveStatus::Active;
  }
  return true;
}

AsyncArriveResult MakeCycleArriveResult(const ExecutableKernel& kernel,
                                        const WaveContext& wave,
                                        MemoryWaitDomain arrive_domain) {
  if (wave.run_state != WaveRunState::Waiting) {
    return {};
  }
  if (!kernel.ContainsPc(wave.pc)) {
    return {};
  }
  const Instruction& instruction = kernel.InstructionAtPc(wave.pc);
  if (instruction.opcode != Opcode::SWaitCnt) {
    return {};
  }
  return MakeAsyncArriveResult(wave, arrive_domain, WaitCntThresholdsForInstruction(instruction));
}

void StoreLaneValue(MemorySystem& memory, const LaneAccess& lane) {
  memory_ops::StoreGlobalLaneValue(memory, lane);
}

uint64_t LoadLaneValue(const std::vector<std::byte>& memory, const LaneAccess& lane) {
  return memory_ops::LoadByteLaneValue(memory, lane);
}

void StoreLaneValue(std::vector<std::byte>& memory, const LaneAccess& lane) {
  memory_ops::StoreByteLaneValue(memory, lane);
}

uint64_t LoadLaneValue(const std::array<std::vector<std::byte>, kWaveSize>& memory,
                       uint32_t lane_id,
                       const LaneAccess& lane) {
  return memory_ops::LoadPrivateLaneValue(memory, lane_id, lane);
}

void StoreLaneValue(std::array<std::vector<std::byte>, kWaveSize>& memory,
                    uint32_t lane_id,
                    const LaneAccess& lane) {
  memory_ops::StorePrivateLaneValue(memory, lane_id, lane);
}

}  // namespace

uint64_t CycleExecEngine::Run(ExecutionContext& context) {
  const uint64_t run_begin_cycle = context.cycle;
  ProgramCycleStatsConfig cycle_stats_config;
  cycle_stats_config.default_issue_cycles = context.spec.default_issue_cycles;
  program_cycle_stats_ = ProgramCycleStats{};

  auto blocks = MaterializeBlocks(context.placement, context.launch_config);

  // Count waves launched
  if (program_cycle_stats_.has_value()) {
    for (const auto& block : blocks) {
      program_cycle_stats_->waves_launched += static_cast<uint32_t>(block.waves.size());
    }
  }

  const uint32_t resident_wave_slots_per_peu = ResidentWaveSlotCapacityPerPeu(context.spec);
  auto slots = BuildPeuSlots(blocks, resident_wave_slots_per_peu);
  EventQueue events;
  std::map<L1Key, CacheModel> l1_caches;
  CacheModel l2_cache(timing_config_.cache_model);
  SharedBankModel shared_bank_model(timing_config_.shared_bank_model);
  std::map<uint32_t, ApResidentState> ap_states;

  for (auto& slot : slots) {
    slot.busy_until = 0;
    slot.selection_ready_cycle = 0;
    slot.last_bundle_commit_cycle = 0;
    l1_caches.emplace(L1Key{.dpc_id = slot.dpc_id, .ap_id = slot.ap_id},
                      CacheModel(timing_config_.cache_model));
  }

  for (auto& block : blocks) {
    auto& ap_state = ap_states[block.global_ap_id];
    ap_state.global_ap_id = block.global_ap_id;
    ap_state.barrier_slot_capacity = context.spec.cycle_resources.barrier_slots_per_ap;
    ap_state.pending_blocks.push_back(&block);
  }
  for (auto& [global_ap_id, ap_state] : ap_states) {
    (void)global_ap_id;
    AdmitResidentBlocks(ap_state,
                        context.cycle,
                        slots,
                        resident_wave_slots_per_peu,
                        context.spec.max_issuable_waves,
                        timing_config_.launch_timing.wave_generation_cycles,
                        timing_config_.launch_timing.wave_dispatch_cycles,
                        timing_config_.launch_timing.wave_launch_cycles,
                        events,
                        context.trace);
  }

  uint64_t cycle = context.cycle;
  while (true) {
    context.cycle = cycle;
    events.RunReady(cycle);
    for (auto& slot : slots) {
      FillDispatchWindow(slot,
                         cycle,
                         context.spec.max_issuable_waves,
                         timing_config_.launch_timing.wave_launch_cycles,
                         events,
                         context.trace);
    }
    events.RunReady(cycle);

    if (AllWavesExited(blocks) && events.empty()) {
      if (program_cycle_stats_.has_value()) {
        program_cycle_stats_->total_cycles = cycle - run_begin_cycle;
      }
      return cycle;
    }

    // Track active/idle cycles
    if (program_cycle_stats_.has_value()) {
      bool has_active_wave = false;
      for (const auto& block : blocks) {
        for (const auto& scheduled_wave : block.waves) {
          if (scheduled_wave.wave.status != WaveStatus::Exited) {
            has_active_wave = true;
            break;
          }
        }
        if (has_active_wave) break;
      }
      if (has_active_wave) {
        program_cycle_stats_->active_cycles += 1;
      } else {
        program_cycle_stats_->idle_cycles += 1;
      }
    }

    bool issued_any = false;
    for (auto& slot : slots) {
      // Selection gate: PEU can select when selection_ready_cycle is satisfied
      if (slot.selection_ready_cycle > cycle) {
        continue;
      }

      std::vector<ResidentIssueSlot*> ordered_resident_slots;
      const auto candidates =
          BuildResidentIssueCandidates(slot, context.kernel, cycle, ap_states, ordered_resident_slots);
      const auto bundle = IssueScheduler::SelectIssueBundle(
          candidates,
          slot.issue_round_robin_index,
          timing_config_.eligible_wave_selection_policy,
          ResolveIssuePolicy(timing_config_, context.spec));
      slot.issue_round_robin_index = bundle.next_round_robin_index;

      if (bundle.selected_candidate_indices.empty()) {
        if (const auto blocked =
                PickFirstBlockedResidentWave(slot, context.kernel, cycle, ap_states)) {
          std::optional<WaitCntThresholds> waitcnt_thresholds;
          const WaveContext& blocked_wave = blocked->first->wave;
          if (context.kernel.ContainsPc(blocked_wave.pc)) {
            const Instruction& blocked_instruction = context.kernel.InstructionAtPc(blocked_wave.pc);
            if (blocked_instruction.opcode == Opcode::SWaitCnt) {
              waitcnt_thresholds = WaitCntThresholdsForInstruction(blocked_instruction);
            }
          }
          context.trace.OnEvent(MakeTraceBlockedStallEvent(
              MakeTraceWaveView(*blocked->first, TraceSlotId(*blocked->first)),
              cycle,
              blocked->second,
              TraceSlotModelKind::ResidentFixed,
              std::numeric_limits<uint64_t>::max(),
              MakeOptionalTraceWaitcntState(blocked_wave, waitcnt_thresholds)));
        }
        continue;
      }

      if (const auto ready_unselected =
              PickFirstReadyUnselectedResidentWave(candidates, bundle, ordered_resident_slots)) {
        context.trace.OnEvent(MakeTraceBlockedStallEvent(
            MakeTraceWaveView(*ready_unselected->first, TraceSlotId(*ready_unselected->first)),
            cycle,
            ready_unselected->second,
            TraceSlotModelKind::ResidentFixed));
      }

      uint64_t bundle_commit_cycle = cycle;
      uint64_t bundle_last_wave_tag = slot.last_wave_tag;
      for (const size_t selected_candidate_index : bundle.selected_candidate_indices) {
        ResidentIssueSlot& resident_slot = *ordered_resident_slots.at(selected_candidate_index);
        ScheduledWave* candidate = resident_slot.resident_wave;
        if (candidate == nullptr || !context.kernel.ContainsPc(candidate->wave.pc)) {
          throw std::out_of_range("wave pc out of range");
        }

        WaveContext& wave = candidate->wave;
        const Instruction instruction = context.kernel.InstructionAtPc(wave.pc);
        const uint32_t slot_id = TraceSlotId(*candidate);
        const uint64_t wave_tag = WaveTag(wave);

        // Calculate actual_issue_cycle considering switch penalty and wave timing
        const bool wave_switched = slot.last_wave_tag != std::numeric_limits<uint64_t>::max() &&
                                    slot.last_wave_tag != wave_tag;
        const uint64_t switch_penalty = wave_switched 
            ? timing_config_.launch_timing.warp_switch_cycles 
            : 0;
        
        // Update switch_ready_cycle if wave switched
        if (wave_switched && switch_penalty > 0) {
          slot.switch_ready_cycle = cycle + switch_penalty;
        }
        
        // Calculate actual issue cycle
        const uint64_t actual_issue_cycle = std::max({cycle, candidate->next_issue_cycle, slot.switch_ready_cycle});
        
        // Emit switch events if penalty applies
        if (wave_switched && switch_penalty > 0 && slot.last_wave_trace.has_value()) {
          context.trace.OnEvent(MakeTraceWaveSwitchAwayEvent(
              *slot.last_wave_trace, cycle, TraceSlotModelKind::ResidentFixed, slot.last_wave_pc));
          context.trace.OnEvent(MakeTraceWaveSwitchStallEvent(
              *slot.last_wave_trace, cycle, TraceSlotModelKind::ResidentFixed, slot.last_wave_pc));
        }
        
                context.trace.OnEvent(MakeTraceIssueSelectEvent(
            MakeTraceWaveView(*candidate, slot_id), cycle, TraceSlotModelKind::ResidentFixed));
        if (context.stats != nullptr) {
          ++context.stats->wave_steps;
          ++context.stats->instructions_issued;
        }
  const OpPlan plan = semantics_->BuildPlan(instruction, wave, context);
        if (program_cycle_stats_.has_value()) {
          // Track total instructions
          program_cycle_stats_->instructions_executed += 1;

          if (const auto step_class = ClassifyCycleInstruction(instruction, plan); step_class.has_value()) {
            AccumulateProgramCycleStep(*program_cycle_stats_,
                                       *step_class,
                                       CostForCycleStep(plan, *step_class, cycle_stats_config),
                                       wave.exec.count());
          }
        }
        context.trace.OnEvent(MakeTraceWaveStepEvent(MakeTraceWaveView(*candidate, slot_id),
                                                     actual_issue_cycle,
                                                     TraceSlotModelKind::ResidentFixed,
                                                     FormatWaveStepMessage(instruction, wave),
                                                     std::numeric_limits<uint64_t>::max(),
                                                     QuantizeIssueDuration(plan.issue_cycles)));
        const uint64_t commit_cycle = actual_issue_cycle + plan.issue_cycles;
        bundle_commit_cycle = std::max(bundle_commit_cycle, commit_cycle);
        bundle_last_wave_tag = wave_tag;
        slot.last_wave_trace = MakeTraceWaveView(*candidate, slot_id);
        slot.last_wave_pc = wave.pc;
        // Update issue timing state
        candidate->last_issue_cycle = actual_issue_cycle;
        candidate->next_issue_cycle = commit_cycle;
        candidate->eligible_since_valid = false;
        wave.status = WaveStatus::Stalled;
        wave.valid_entry = false;
        uint64_t flow_id = 0;
        if (plan.memory.has_value()) {
          IncrementPendingMemoryOps(wave, MemoryDomainForOpcode(instruction.opcode));
        }
        if (instruction.opcode == Opcode::BBranch || instruction.opcode == Opcode::BIfSmask ||
            instruction.opcode == Opcode::BIfNoexec) {
          wave.branch_pending = true;
        }

        if (plan.memory.has_value()) {
          if (context.stats != nullptr) {
            ++context.stats->memory_ops;
            const auto& request = *plan.memory;
            if (request.space == MemorySpace::Global) {
              if (request.kind == AccessKind::Load) {
                ++context.stats->global_loads;
              } else if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
                ++context.stats->global_stores;
              }
            } else if (request.space == MemorySpace::Shared) {
              if (request.kind == AccessKind::Load) {
                ++context.stats->shared_loads;
              } else if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
                ++context.stats->shared_stores;
              }
            } else if (request.space == MemorySpace::Private) {
              if (request.kind == AccessKind::Load) {
                ++context.stats->private_loads;
              } else if (request.kind == AccessKind::Store) {
                ++context.stats->private_stores;
              }
            } else if (request.space == MemorySpace::Constant && request.kind == AccessKind::Load) {
              ++context.stats->constant_loads;
            }
          }
          // Track memory ops in program_cycle_stats
          if (program_cycle_stats_.has_value()) {
            const auto& request = *plan.memory;
            if (request.space == MemorySpace::Global) {
              if (request.kind == AccessKind::Load) {
                program_cycle_stats_->global_loads += 1;
              } else if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
                program_cycle_stats_->global_stores += 1;
              }
            } else if (request.space == MemorySpace::Shared) {
              if (request.kind == AccessKind::Load) {
                program_cycle_stats_->shared_loads += 1;
              } else if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
                program_cycle_stats_->shared_stores += 1;
              }
            } else if (request.space == MemorySpace::Private) {
              if (request.kind == AccessKind::Load) {
                program_cycle_stats_->private_loads += 1;
              } else if (request.kind == AccessKind::Store) {
                program_cycle_stats_->private_stores += 1;
              }
            } else if (request.space == MemorySpace::Constant && request.kind == AccessKind::Load) {
              program_cycle_stats_->scalar_loads += 1;
            }
          }
          flow_id = context.AllocateTraceFlowId();
          TraceEvent issue_event = MakeTraceWaveEvent(
              MakeTraceWaveView(*candidate, slot_id),
              TraceEventKind::MemoryAccess,
              cycle,
              TraceSlotModelKind::ResidentFixed,
              plan.memory->kind == AccessKind::Load ? "load_issue" : "store_issue");
          issue_event.flow_id = flow_id;
          issue_event.flow_phase = TraceFlowPhase::Start;
          context.trace.OnEvent(std::move(issue_event));
        }

        events.Schedule(TimedEvent{
            .cycle = commit_cycle,
            .action =
                [&, candidate, instruction, plan, commit_cycle, slot_id, flow_id]() {
                context.cycle = commit_cycle;

                ApplyExecutionPlanRegisterWrites(plan, candidate->wave);
                if (const auto mask_text = MaybeFormatExecutionMaskUpdate(plan, candidate->wave);
                    mask_text.has_value()) {
                  context.trace.OnEvent(MakeTraceWaveEvent(MakeTraceWaveView(*candidate, slot_id),
                                                           TraceEventKind::ExecMaskUpdate,
                                                           commit_cycle,
                                                           TraceSlotModelKind::ResidentFixed,
                                                           *mask_text));
                }

                context.trace.OnEvent(MakeTraceCommitEvent(
                    MakeTraceWaveView(*candidate, slot_id),
                    commit_cycle,
                    TraceSlotModelKind::ResidentFixed));

                if (plan.memory.has_value()) {
                  const MemoryRequest request = *plan.memory;
                  if (request.space == MemorySpace::Global) {
                    auto& l1_cache =
                        l1_caches.at(L1Key{.dpc_id = candidate->block->dpc_id, .ap_id = candidate->block->ap_id});
                    const std::vector<uint64_t> addrs = ActiveAddresses(request);
                    const CacheProbeResult l1_probe = l1_cache.Probe(addrs);
                    const CacheProbeResult l2_probe = l2_cache.Probe(addrs);
                    uint64_t arrive_latency = std::min(l1_probe.latency, l2_probe.latency);
                    // gem5: store uses 2x bus latency
                    if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
                      arrive_latency *= cycle_stats_config.store_latency_multiplier;
                    }
                    if (context.stats != nullptr) {
                      if (l1_probe.l1_hits > 0) {
                        context.stats->l1_hits += l1_probe.l1_hits;
                      } else if (l2_probe.l2_hits > 0) {
                        context.stats->l2_hits += l2_probe.l2_hits;
                      } else {
                        context.stats->cache_misses += std::max<uint64_t>(1, l2_probe.misses);
                      }
                    }
                    const uint64_t arrive_cycle = commit_cycle + arrive_latency;
                    const uint64_t completion_flow_id = flow_id;
                    events.Schedule(TimedEvent{
                        .cycle = arrive_cycle,
                        .action =
                            [&, candidate, request, addrs, arrive_cycle, slot_id, completion_flow_id]() {
                              context.cycle = arrive_cycle;
                              if (request.kind == AccessKind::Load) {
                                if (!request.dst.has_value()) {
                                  throw std::invalid_argument("load request missing destination");
                                }
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    const uint64_t value =
                                        LoadLaneValue(context.memory, request.lanes[lane]);
                                    candidate->wave.vgpr.Write(request.dst->index, lane, value);
                                  }
                                }
                                l2_cache.Promote(addrs);
                                l1_caches
                                    .at(L1Key{.dpc_id = candidate->block->dpc_id, .ap_id = candidate->block->ap_id})
                                    .Promote(addrs);
                              } else if (request.kind == AccessKind::Store) {
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    StoreLaneValue(context.memory, request.lanes[lane]);
                                  }
                                }
                              } else if (request.kind == AccessKind::Atomic) {
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (!request.lanes[lane].active) {
                                    continue;
                                  }
                                  const int32_t prior = static_cast<int32_t>(
                                      LoadLaneValue(context.memory, request.lanes[lane]));
                                  int32_t updated;
                                  switch (request.atomic_op) {
                                    case AtomicOp::Add:
                                      updated = prior + static_cast<int32_t>(request.lanes[lane].value);
                                      break;
                                    case AtomicOp::Max:
                                      updated = std::max(prior, static_cast<int32_t>(request.lanes[lane].value));
                                      break;
                                    case AtomicOp::Min:
                                      updated = std::min(prior, static_cast<int32_t>(request.lanes[lane].value));
                                      break;
                                    case AtomicOp::Exch:
                                      updated = static_cast<int32_t>(request.lanes[lane].value);
                                      break;
                                    default:
                                      throw std::invalid_argument("unsupported atomic op");
                                  }
                                  LaneAccess writeback = request.lanes[lane];
                                  writeback.value = static_cast<uint64_t>(static_cast<int64_t>(updated));
                                  StoreLaneValue(context.memory, writeback);
                                }
                              }

                              TraceEvent arrive_event = MakeTraceMemoryArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  arrive_cycle,
                                  request.kind == AccessKind::Load ? TraceMemoryArriveKind::Load
                                                                   : TraceMemoryArriveKind::Store,
                                  TraceSlotModelKind::ResidentFixed);
                              arrive_event.flow_id = completion_flow_id;
                              arrive_event.flow_phase = TraceFlowPhase::Finish;
                              DecrementPendingMemoryOps(candidate->wave, MemoryWaitDomain::Global);
                              const AsyncArriveResult arrive_result = MakeCycleArriveResult(
                                  context.kernel, candidate->wave, MemoryWaitDomain::Global);
                              arrive_event.waitcnt_state = arrive_result.waitcnt_state;
                              arrive_event.arrive_progress = arrive_result.arrive_progress;
                              context.trace.OnEvent(std::move(arrive_event));
                              context.trace.OnEvent(MakeTraceWaveArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  arrive_cycle,
                                  request.kind == AccessKind::Load ? TraceMemoryArriveKind::Load
                                                                   : TraceMemoryArriveKind::Store,
                                  TraceSlotModelKind::ResidentFixed,
                                  arrive_result.arrive_progress,
                                  std::numeric_limits<uint64_t>::max(),
                                  arrive_result.waitcnt_state));
                              if (!ResumeWaitcntWaveIfReady(context.kernel, candidate->wave)) {
                                candidate->wave.valid_entry = true;
                                if (candidate->wave.status != WaveStatus::Exited &&
                                    !candidate->wave.waiting_at_barrier) {
                                  candidate->wave.status = WaveStatus::Active;
                                }
                              } else {
                                candidate->eligible_since_cycle = arrive_cycle;
                                candidate->eligible_since_valid = true;
                                context.trace.OnEvent(MakeTraceWaveResumeEvent(
                                    MakeTraceWaveView(*candidate, slot_id),
                                    arrive_cycle,
                                    TraceSlotModelKind::ResidentFixed));
                              }
                            },
                    });
                  } else if (request.space == MemorySpace::Shared) {
                    const uint64_t penalty = shared_bank_model.ConflictPenalty(request);
                    uint64_t async_delay = ModeledAsyncCompletionDelay(
                        plan.issue_cycles, context.spec.default_issue_cycles);
                    // gem5: store uses 2x bus latency
                    if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
                      async_delay *= cycle_stats_config.store_latency_multiplier;
                    }
                    if (context.stats != nullptr) {
                      context.stats->shared_bank_conflict_penalty_cycles += penalty;
                    }
                    const uint64_t ready_cycle = commit_cycle + penalty + async_delay;
                    const uint64_t completion_flow_id = flow_id;
                    events.Schedule(TimedEvent{
                        .cycle = ready_cycle,
                        .action =
                            [&, candidate, request, ready_cycle, plan, slot_id, completion_flow_id]() {
                              context.cycle = ready_cycle;
                              if (request.kind == AccessKind::Load) {
                                if (!request.dst.has_value()) {
                                  throw std::invalid_argument("load request missing destination");
                                }
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    const uint64_t value =
                                        LoadLaneValue(candidate->block->shared_memory, request.lanes[lane]);
                                    candidate->wave.vgpr.Write(request.dst->index, lane, value);
                                  }
                                }
                              } else if (request.kind == AccessKind::Store) {
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    StoreLaneValue(candidate->block->shared_memory, request.lanes[lane]);
                                  }
                                }
                              } else if (request.kind == AccessKind::Atomic) {
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (!request.lanes[lane].active) {
                                    continue;
                                  }
                                  const int32_t prior = static_cast<int32_t>(
                                      LoadLaneValue(candidate->block->shared_memory, request.lanes[lane]));
                                  int32_t updated;
                                  switch (request.atomic_op) {
                                    case AtomicOp::Add:
                                      updated = prior + static_cast<int32_t>(request.lanes[lane].value);
                                      break;
                                    case AtomicOp::Max:
                                      updated = std::max(prior, static_cast<int32_t>(request.lanes[lane].value));
                                      break;
                                    case AtomicOp::Min:
                                      updated = std::min(prior, static_cast<int32_t>(request.lanes[lane].value));
                                      break;
                                    case AtomicOp::Exch:
                                      updated = static_cast<int32_t>(request.lanes[lane].value);
                                      break;
                                    default:
                                      throw std::invalid_argument("unsupported atomic op");
                                  }
                                  LaneAccess writeback = request.lanes[lane];
                                  writeback.value = static_cast<uint64_t>(static_cast<int64_t>(updated));
                                  StoreLaneValue(candidate->block->shared_memory, writeback);
                                }
                              }

                              TraceEvent arrive_event = MakeTraceMemoryArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  ready_cycle,
                                  request.kind == AccessKind::Load ? TraceMemoryArriveKind::Shared
                                                                   : TraceMemoryArriveKind::Store,
                                  TraceSlotModelKind::ResidentFixed);
                              arrive_event.flow_id = completion_flow_id;
                              arrive_event.flow_phase = TraceFlowPhase::Finish;
                              DecrementPendingMemoryOps(candidate->wave, MemoryWaitDomain::Shared);
                              const AsyncArriveResult arrive_result = MakeCycleArriveResult(
                                  context.kernel, candidate->wave, MemoryWaitDomain::Shared);
                              arrive_event.waitcnt_state = arrive_result.waitcnt_state;
                              arrive_event.arrive_progress = arrive_result.arrive_progress;
                              context.trace.OnEvent(std::move(arrive_event));
                              context.trace.OnEvent(MakeTraceWaveArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  ready_cycle,
                                  TraceMemoryArriveKind::Shared,
                                  TraceSlotModelKind::ResidentFixed,
                                  arrive_result.arrive_progress,
                                  std::numeric_limits<uint64_t>::max(),
                                  arrive_result.waitcnt_state));
                              if (!ResumeWaitcntWaveIfReady(context.kernel, candidate->wave)) {
                                candidate->wave.status = WaveStatus::Active;
                              } else {
                                candidate->eligible_since_cycle = ready_cycle;
                                candidate->eligible_since_valid = true;
                                context.trace.OnEvent(MakeTraceWaveResumeEvent(
                                    MakeTraceWaveView(*candidate, slot_id),
                                    ready_cycle,
                                    TraceSlotModelKind::ResidentFixed));
                              }
                            },
                    });
                  } else if (request.space == MemorySpace::Private) {
                    uint64_t async_delay = ModeledAsyncCompletionDelay(plan.issue_cycles, context.spec.default_issue_cycles);
                    // gem5: store uses 2x bus latency
                    if (request.kind == AccessKind::Store) {
                      async_delay *= cycle_stats_config.store_latency_multiplier;
                    }
                    const uint64_t arrive_cycle = commit_cycle + async_delay;
                    const uint64_t completion_flow_id = flow_id;
                    events.Schedule(TimedEvent{
                        .cycle = arrive_cycle,
                        .action =
                            [&, candidate, request, arrive_cycle, slot_id, completion_flow_id]() {
                              context.cycle = arrive_cycle;
                              if (request.kind == AccessKind::Load) {
                                if (!request.dst.has_value()) {
                                  throw std::invalid_argument("load request missing destination");
                                }
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    const uint64_t value = LoadLaneValue(candidate->wave.private_memory,
                                                                         lane,
                                                                         request.lanes[lane]);
                                    candidate->wave.vgpr.Write(request.dst->index, lane, value);
                                  }
                                }
                              } else if (request.kind == AccessKind::Store) {
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    StoreLaneValue(candidate->wave.private_memory,
                                                   lane,
                                                   request.lanes[lane]);
                                  }
                                }
                              }
                              TraceEvent arrive_event = MakeTraceMemoryArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  arrive_cycle,
                                  request.kind == AccessKind::Load ? TraceMemoryArriveKind::Private
                                                                   : TraceMemoryArriveKind::Store,
                                  TraceSlotModelKind::ResidentFixed);
                              arrive_event.flow_id = completion_flow_id;
                              arrive_event.flow_phase = TraceFlowPhase::Finish;
                              DecrementPendingMemoryOps(candidate->wave, MemoryWaitDomain::Private);
                              const AsyncArriveResult arrive_result = MakeCycleArriveResult(
                                  context.kernel, candidate->wave, MemoryWaitDomain::Private);
                              arrive_event.waitcnt_state = arrive_result.waitcnt_state;
                              arrive_event.arrive_progress = arrive_result.arrive_progress;
                              context.trace.OnEvent(std::move(arrive_event));
                              context.trace.OnEvent(MakeTraceWaveArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  arrive_cycle,
                                  TraceMemoryArriveKind::Private,
                                  TraceSlotModelKind::ResidentFixed,
                                  arrive_result.arrive_progress,
                                  std::numeric_limits<uint64_t>::max(),
                                  arrive_result.waitcnt_state));
                              if (!ResumeWaitcntWaveIfReady(context.kernel, candidate->wave)) {
                                candidate->wave.valid_entry = true;
                              } else {
                                candidate->eligible_since_cycle = arrive_cycle;
                                candidate->eligible_since_valid = true;
                                context.trace.OnEvent(MakeTraceWaveResumeEvent(
                                    MakeTraceWaveView(*candidate, slot_id),
                                    arrive_cycle,
                                    TraceSlotModelKind::ResidentFixed));
                              }
                            },
                    });
                  } else if (request.space == MemorySpace::Constant) {
                    const uint64_t arrive_cycle =
                        commit_cycle +
                        ModeledAsyncCompletionDelay(plan.issue_cycles, context.spec.default_issue_cycles);
                    const uint64_t completion_flow_id = flow_id;
                    events.Schedule(TimedEvent{
                        .cycle = arrive_cycle,
                        .action =
                            [&, candidate, request, arrive_cycle, slot_id, completion_flow_id]() {
                              context.cycle = arrive_cycle;
                              if (!request.dst.has_value()) {
                                throw std::invalid_argument("load request missing destination");
                              }
                              if (request.dst->file == RegisterFile::Scalar) {
                                uint64_t loaded_value = 0;
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (!request.lanes[lane].active) {
                                    continue;
                                  }
                                  const LaneAccess pool_lane{
                                      .active = request.lanes[lane].active,
                                      .addr = ConstantPoolBase(context) + request.lanes[lane].addr,
                                      .bytes = request.lanes[lane].bytes,
                                      .value = request.lanes[lane].value,
                                  };
                                  if (context.memory.HasRange(MemoryPoolKind::Constant,
                                                              pool_lane.addr,
                                                              request.lanes[lane].bytes)) {
                                    loaded_value =
                                        LoadLaneValue(context.memory, MemoryPoolKind::Constant, pool_lane);
                                  } else {
                                    loaded_value =
                                        LoadLaneValue(context.kernel.const_segment().bytes,
                                                      request.lanes[lane]);
                                  }
                                  break;
                                }
                                candidate->wave.sgpr.Write(request.dst->index, loaded_value);
                              } else {
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    const LaneAccess pool_lane{
                                        .active = request.lanes[lane].active,
                                        .addr = ConstantPoolBase(context) + request.lanes[lane].addr,
                                        .bytes = request.lanes[lane].bytes,
                                        .value = request.lanes[lane].value,
                                    };
                                    const uint64_t value =
                                        context.memory.HasRange(MemoryPoolKind::Constant,
                                                                pool_lane.addr,
                                                                request.lanes[lane].bytes)
                                            ? LoadLaneValue(context.memory,
                                                            MemoryPoolKind::Constant,
                                                            pool_lane)
                                            : LoadLaneValue(context.kernel.const_segment().bytes,
                                                            request.lanes[lane]);
                                    candidate->wave.vgpr.Write(request.dst->index, lane, value);
                                  }
                                }
                              }
                              TraceEvent arrive_event = MakeTraceMemoryArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  arrive_cycle,
                                  TraceMemoryArriveKind::ScalarBuffer,
                                  TraceSlotModelKind::ResidentFixed);
                              arrive_event.flow_id = completion_flow_id;
                              arrive_event.flow_phase = TraceFlowPhase::Finish;
                              DecrementPendingMemoryOps(candidate->wave,
                                                        MemoryWaitDomain::ScalarBuffer);
                              const AsyncArriveResult arrive_result = MakeCycleArriveResult(
                                  context.kernel, candidate->wave, MemoryWaitDomain::ScalarBuffer);
                              arrive_event.waitcnt_state = arrive_result.waitcnt_state;
                              arrive_event.arrive_progress = arrive_result.arrive_progress;
                              context.trace.OnEvent(std::move(arrive_event));
                              context.trace.OnEvent(MakeTraceWaveArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  arrive_cycle,
                                  TraceMemoryArriveKind::ScalarBuffer,
                                  TraceSlotModelKind::ResidentFixed,
                                  arrive_result.arrive_progress,
                                  std::numeric_limits<uint64_t>::max(),
                                  arrive_result.waitcnt_state));
                              if (!ResumeWaitcntWaveIfReady(context.kernel, candidate->wave)) {
                                candidate->wave.valid_entry = true;
                              } else {
                                candidate->eligible_since_cycle = arrive_cycle;
                                candidate->eligible_since_valid = true;
                                context.trace.OnEvent(MakeTraceWaveResumeEvent(
                                    MakeTraceWaveView(*candidate, slot_id),
                                    arrive_cycle,
                                    TraceSlotModelKind::ResidentFixed));
                              }
                            },
                    });
                  } else {
                    throw std::invalid_argument("unsupported memory space in cycle executor");
                  }
                }

                if (plan.sync_wave_barrier) {
                  if (context.stats != nullptr) {
                    ++context.stats->barriers;
                  }
                  context.trace.OnEvent(MakeTraceBarrierWaveEvent(
                      MakeTraceWaveView(*candidate, slot_id),
                      commit_cycle,
                      TraceSlotModelKind::ResidentFixed));
                  ApplyExecutionPlanControlFlow(context.kernel, plan, candidate->wave, true, true);
                  candidate->wave.status = WaveStatus::Active;
                  return;
                }

                if (plan.sync_barrier) {
                  if (context.stats != nullptr) {
                    ++context.stats->barriers;
                  }
                  auto& ap_state = ap_states[candidate->block->global_ap_id];
                  const bool acquired = TryAcquireBarrierSlot(ap_state.barrier_slot_capacity,
                                                              ap_state.barrier_slots_in_use,
                                                              candidate->block->barrier_slot_acquired);
                  if (!acquired) {
                    candidate->wave.status = WaveStatus::Active;
                    candidate->wave.valid_entry = true;
                    return;
                  }
                  MarkWaveAtBarrier(candidate->wave,
                                    candidate->block->barrier_generation,
                                    candidate->block->barrier_arrivals,
                                    true);
                  PeuSlot& peu_slot = slots.at(candidate->peu_slot_index);
                  DeactivateResidentSlot(ResidentSlotForWave(peu_slot, *candidate));
                  context.trace.OnEvent(MakeTraceBarrierArriveEvent(
                      MakeTraceWaveView(*candidate, slot_id),
                      commit_cycle,
                      TraceSlotModelKind::ResidentFixed));
                  context.trace.OnEvent(MakeTraceWaveWaitEvent(
                      MakeTraceWaveView(*candidate, slot_id),
                      commit_cycle,
                      TraceSlotModelKind::ResidentFixed));
                  context.trace.OnEvent(MakeTraceWaveSwitchAwayEvent(
                      MakeTraceWaveView(*candidate, slot_id),
                      commit_cycle,
                      TraceSlotModelKind::ResidentFixed,
                      wave.pc));
                  context.trace.OnEvent(MakeTraceWaveSwitchStallEvent(
                      MakeTraceWaveView(*candidate, slot_id),
                      commit_cycle,
                      TraceSlotModelKind::ResidentFixed,
                      wave.pc));
                  RefillActiveWindow(peu_slot,
                                     commit_cycle,
                                     context.spec.max_issuable_waves,
                                     timing_config_.launch_timing.wave_launch_cycles,
                                     events,
                                     context.trace,
                                     true);
                  std::vector<ScheduledWave*> barrier_generation_waves;
                  barrier_generation_waves.reserve(candidate->block->waves.size());
                  for (auto& waiting_wave : candidate->block->waves) {
                    if (waiting_wave.wave.waiting_at_barrier &&
                        waiting_wave.wave.barrier_generation ==
                            candidate->block->barrier_generation) {
                      barrier_generation_waves.push_back(&waiting_wave);
                    }
                  }
                  std::vector<WaveContext*> waiting_waves;
                  waiting_waves.reserve(candidate->block->waves.size());
                  for (auto& waiting_wave : candidate->block->waves) {
                    waiting_waves.push_back(&waiting_wave.wave);
                  }
                  if (sync_ops::ReleaseBarrierIfReady(waiting_waves,
                                                      context.kernel,
                                                      candidate->block->barrier_generation,
                                                      candidate->block->barrier_arrivals,
                                                      true)) {
                    ReleaseBarrierSlot(ap_state.barrier_slots_in_use,
                                       candidate->block->barrier_slot_acquired);
                    context.trace.OnEvent(MakeTraceBarrierReleaseEvent(
                        candidate->block->dpc_id,
                        candidate->block->ap_id,
                        candidate->block->block_id,
                        commit_cycle));
                    std::vector<size_t> refill_slots;
                    refill_slots.reserve(barrier_generation_waves.size());
                    for (ScheduledWave* released_wave : barrier_generation_waves) {
                      if (released_wave == nullptr || released_wave->wave.waiting_at_barrier) {
                        continue;
                      }
                      released_wave->eligible_since_cycle = commit_cycle;
                      released_wave->eligible_since_valid = true;
                      context.trace.OnEvent(MakeTraceWaveResumeEvent(
                          MakeTraceWaveView(*released_wave, TraceSlotId(*released_wave)),
                          commit_cycle,
                          TraceSlotModelKind::ResidentFixed));
                      PeuSlot& released_slot = slots.at(released_wave->peu_slot_index);
                      QueueResidentWaveForRefill(released_slot, *released_wave);
                      if (std::find(refill_slots.begin(),
                                    refill_slots.end(),
                                    released_wave->peu_slot_index) == refill_slots.end()) {
                        refill_slots.push_back(released_wave->peu_slot_index);
                      }
                    }
                    for (size_t slot_index : refill_slots) {
                      RefillActiveWindow(slots.at(slot_index),
                                         commit_cycle,
                                         context.spec.max_issuable_waves,
                                         timing_config_.launch_timing.wave_launch_cycles,
                                         events,
                                         context.trace,
                                         false);
                    }
                  }
                  return;
                }

                if (plan.exit_wave) {
                  if (context.stats != nullptr) {
                    ++context.stats->wave_exits;
                  }
                  // Track wave completion in program_cycle_stats
                  if (program_cycle_stats_.has_value()) {
                    program_cycle_stats_->waves_completed += 1;
                  }
                  ApplyExecutionPlanControlFlow(context.kernel, plan, candidate->wave, false, false);
                  PeuSlot& peu_slot = slots.at(candidate->peu_slot_index);
                  if (peu_slot.last_wave_tag == WaveTag(candidate->wave)) {
                    peu_slot.last_wave_tag = std::numeric_limits<uint64_t>::max();
                    peu_slot.last_wave_trace.reset();
                    peu_slot.last_wave_pc = 0;
                  }
                  RemoveResidentWave(peu_slot, *candidate);
                  context.trace.OnEvent(MakeTraceWaveExitEvent(
                      MakeTraceWaveView(*candidate, slot_id),
                      commit_cycle,
                      TraceSlotModelKind::ResidentFixed));
                  if (candidate->block->active && !candidate->block->completed &&
                      AllWavesExited(*candidate->block)) {
                    candidate->block->active = false;
                    candidate->block->completed = true;
                    const uint32_t global_ap_id = candidate->block->global_ap_id;
                    auto ap_state_it = ap_states.find(global_ap_id);
                    if (ap_state_it != ap_states.end()) {
                      const bool removed =
                          RetireResidentBlock(ap_state_it->second, candidate->block);
                      if (removed) {
                        context.trace.OnEvent(MakeTraceBlockRetireEvent(
                            candidate->block->dpc_id,
                            candidate->block->ap_id,
                            candidate->block->block_id,
                            commit_cycle,
                            "ap=" + std::to_string(candidate->block->ap_id)));
                      }
                      if (removed && CanScheduleDelayedReadmit(ap_state_it->second)) {
                        ++ap_state_it->second.scheduled_readmit_count;
                      events.Schedule(TimedEvent{
                            .cycle = commit_cycle + timing_config_.launch_timing.block_launch_cycles,
                            .action =
                                [&, global_ap_id, commit_cycle]() {
                                  auto state_it = ap_states.find(global_ap_id);
                                  if (state_it == ap_states.end()) {
                                    return;
                                  }
                                  if (state_it->second.scheduled_readmit_count > 0) {
                                    --state_it->second.scheduled_readmit_count;
                                  }
                                  AdmitOneResidentBlock(
                                      state_it->second,
                                      commit_cycle + timing_config_.launch_timing.block_launch_cycles,
                                      slots,
                                      resident_wave_slots_per_peu,
                                      context.spec.max_issuable_waves,
                                      timing_config_.launch_timing.wave_generation_cycles,
                                      timing_config_.launch_timing.wave_dispatch_cycles,
                                      timing_config_.launch_timing.wave_launch_cycles,
                                      events,
                                      context.trace);
                                },
                        });
                      }
                    }
                  }
                  RefillActiveWindow(peu_slot,
                                     commit_cycle,
                                     context.spec.max_issuable_waves,
                                     timing_config_.launch_timing.wave_launch_cycles,
                                     events,
                                     context.trace,
                                     true);
                  return;
                }

                if (instruction.opcode == Opcode::SWaitCnt) {
                  const WaitCntThresholds thresholds = WaitCntThresholdsForInstruction(instruction);
                  if (EnterMemoryWaitState(thresholds, candidate->wave)) {
                    candidate->wave.valid_entry = true;
                    candidate->wave.status = WaveStatus::Active;
                    context.trace.OnEvent(MakeTraceWaveWaitEvent(
                        MakeTraceWaveView(*candidate, slot_id),
                        commit_cycle,
                        TraceSlotModelKind::ResidentFixed,
                        TraceStallReasonForWaitReason(candidate->wave.wait_reason),
                        std::numeric_limits<uint64_t>::max(),
                        MakeTraceWaitcntState(candidate->wave, thresholds)));
                    context.trace.OnEvent(MakeTraceWaitStallEvent(
                        MakeTraceWaveView(*candidate, slot_id),
                        commit_cycle,
                        TraceStallReasonForWaitReason(candidate->wave.wait_reason),
                        TraceSlotModelKind::ResidentFixed,
                        std::numeric_limits<uint64_t>::max(),
                        MakeTraceWaitcntState(candidate->wave, thresholds)));
                    context.trace.OnEvent(MakeTraceWaveSwitchAwayEvent(
                        MakeTraceWaveView(*candidate, slot_id),
                        commit_cycle,
                        TraceSlotModelKind::ResidentFixed,
                        candidate->wave.pc));
                    context.trace.OnEvent(MakeTraceWaveSwitchStallEvent(
                        MakeTraceWaveView(*candidate, slot_id),
                        commit_cycle,
                        TraceSlotModelKind::ResidentFixed,
                        candidate->wave.pc));
                    return;
                  }
                }

                candidate->wave.status = WaveStatus::Active;
                ApplyExecutionPlanControlFlow(context.kernel, plan, candidate->wave, true, true);
                },
        });
        issued_any = true;
      }
      // Update timing state:
      // - selection_ready_cycle: when PEU can next select
      // - last_bundle_commit_cycle: when current bundle finishes committing
      // busy_until remains a mirrored alias until external callers finish moving to the split fields.
      slot.selection_ready_cycle = bundle_commit_cycle;
      slot.last_bundle_commit_cycle = bundle_commit_cycle;
      slot.busy_until = bundle_commit_cycle;
      slot.last_wave_tag = bundle_last_wave_tag;
    }

    if (!issued_any && events.empty()) {
      throw std::runtime_error("cycle execution stalled without pending events");
    }

    ++cycle;
  }
}

}  // namespace gpu_model
