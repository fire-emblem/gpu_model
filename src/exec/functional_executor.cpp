#include "gpu_model/exec/functional_executor.h"

#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "gpu_model/debug/trace_event.h"
#include "gpu_model/isa/opcode.h"

namespace gpu_model {

namespace {

struct ExecutableBlock {
  uint32_t block_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  std::vector<WaveState> waves;
};

std::vector<ExecutableBlock> MaterializeBlocks(const PlacementMap& placement) {
  std::vector<ExecutableBlock> blocks;
  blocks.reserve(placement.blocks.size());

  for (const auto& block_placement : placement.blocks) {
    ExecutableBlock block{
        .block_id = block_placement.block_id,
        .dpc_id = block_placement.dpc_id,
        .ap_id = block_placement.ap_id,
        .waves = {},
    };
    block.waves.reserve(block_placement.waves.size());
    for (const auto& wave_placement : block_placement.waves) {
      WaveState wave;
      wave.block_id = block_placement.block_id;
      wave.wave_id = wave_placement.wave_id;
      wave.peu_id = wave_placement.peu_id;
      wave.ap_id = block_placement.ap_id;
      wave.thread_count = wave_placement.lane_count;
      wave.ResetInitialExec();
      block.waves.push_back(wave);
    }
    blocks.push_back(std::move(block));
  }

  return blocks;
}

uint64_t LoadLaneValue(const MemorySystem& memory, const LaneAccess& lane) {
  switch (lane.bytes) {
    case 4: {
      const int32_t value = memory.LoadGlobalValue<int32_t>(lane.addr);
      return static_cast<uint64_t>(static_cast<int64_t>(value));
    }
    case 8:
      return memory.LoadGlobalValue<uint64_t>(lane.addr);
    default:
      throw std::invalid_argument("unsupported load width");
  }
}

void StoreLaneValue(MemorySystem& memory, const LaneAccess& lane) {
  switch (lane.bytes) {
    case 4:
      memory.StoreGlobalValue<int32_t>(lane.addr, static_cast<int32_t>(lane.value));
      return;
    case 8:
      memory.StoreGlobalValue<uint64_t>(lane.addr, lane.value);
      return;
    default:
      throw std::invalid_argument("unsupported store width");
  }
}

}  // namespace

uint64_t FunctionalExecutor::Run(ExecutionContext& context) {
  auto blocks = MaterializeBlocks(context.placement);

  auto has_active_wave = [&blocks]() {
    for (const auto& block : blocks) {
      for (const auto& wave : block.waves) {
        if (wave.status == WaveStatus::Active) {
          return true;
        }
      }
    }
    return false;
  };

  while (has_active_wave()) {
    bool made_progress = false;

    for (auto& block : blocks) {
      for (auto& wave : block.waves) {
        if (wave.status != WaveStatus::Active) {
          continue;
        }
        if (wave.pc >= context.kernel.instructions().size()) {
          throw std::out_of_range("wave pc out of range");
        }

        const Instruction& instruction = context.kernel.instructions().at(wave.pc);
        context.trace.OnEvent(TraceEvent{
            .kind = TraceEventKind::WaveStep,
            .cycle = context.cycle,
            .block_id = wave.block_id,
            .wave_id = wave.wave_id,
            .pc = wave.pc,
            .message = std::string(ToString(instruction.opcode)),
        });

        const OpPlan plan = semantics_.BuildPlan(instruction, wave, context);

        for (const auto& write : plan.scalar_writes) {
          wave.sgpr.Write(write.reg_index, write.value);
        }
        for (const auto& write : plan.vector_writes) {
          for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
            if (write.mask.test(lane)) {
              wave.vgpr.Write(write.reg_index, lane, write.values[lane]);
            }
          }
        }
        if (plan.cmask_write.has_value()) {
          wave.cmask = *plan.cmask_write;
        }
        if (plan.smask_write.has_value()) {
          wave.smask = *plan.smask_write;
        }
        if (plan.exec_write.has_value()) {
          wave.exec = *plan.exec_write;
          std::ostringstream mask_text;
          mask_text << wave.exec;
          context.trace.OnEvent(TraceEvent{
              .kind = TraceEventKind::ExecMaskUpdate,
              .cycle = context.cycle,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = mask_text.str(),
          });
        }
        if (plan.memory.has_value()) {
          const auto& request = *plan.memory;
          context.trace.OnEvent(TraceEvent{
              .kind = TraceEventKind::MemoryAccess,
              .cycle = context.cycle,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = request.kind == AccessKind::Load ? "load" : "store",
          });

          if (request.kind == AccessKind::Load) {
            if (!request.dst.has_value()) {
              throw std::invalid_argument("load request missing destination");
            }
            std::array<uint64_t, 64> loaded_values{};
            for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
              if (request.lanes[lane].active) {
                loaded_values[lane] = LoadLaneValue(context.memory, request.lanes[lane]);
              }
            }
            for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
              if (request.exec_snapshot.test(lane)) {
                wave.vgpr.Write(request.dst->index, lane, loaded_values[lane]);
              }
            }
          } else if (request.kind == AccessKind::Store) {
            for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
              if (request.lanes[lane].active) {
                StoreLaneValue(context.memory, request.lanes[lane]);
              }
            }
          } else {
            throw std::invalid_argument("unsupported memory access kind in functional executor");
          }
        }

        if (plan.exit_wave) {
          wave.status = WaveStatus::Exited;
          context.trace.OnEvent(TraceEvent{
              .kind = TraceEventKind::WaveExit,
              .cycle = context.cycle,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = "exit",
          });
        } else if (plan.branch_target.has_value()) {
          wave.pc = *plan.branch_target;
        } else if (plan.advance_pc) {
          ++wave.pc;
        }

        made_progress = true;
      }
    }

    if (!made_progress) {
      throw std::runtime_error("functional execution stalled without progress");
    }
  }

  return 0;
}

}  // namespace gpu_model
