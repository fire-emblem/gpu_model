#pragma once

#include <cstdint>
#include <vector>

#include "program/executable/executable_kernel.h"
#include "state/wave/wave_runtime_state.h"

namespace gpu_model {

namespace sync_ops {

void MarkWaveAtBarrier(WaveContext& wave,
                       uint64_t barrier_generation,
                       uint32_t& barrier_arrivals,
                       bool set_valid_entry_on_arrive);

bool ReleaseBarrierIfReady(std::vector<WaveContext>& waves,
                           uint64_t& barrier_generation,
                           uint32_t& barrier_arrivals,
                           uint64_t pc_increment,
                           bool set_valid_entry_on_release);

bool ReleaseBarrierIfReady(std::vector<WaveContext>& waves,
                           const ExecutableKernel& kernel,
                           uint64_t& barrier_generation,
                           uint32_t& barrier_arrivals,
                           bool set_valid_entry_on_release);

bool ReleaseBarrierIfReady(const std::vector<WaveContext*>& waves,
                           uint64_t& barrier_generation,
                           uint32_t& barrier_arrivals,
                           uint64_t pc_increment,
                           bool set_valid_entry_on_release);

bool ReleaseBarrierIfReady(const std::vector<WaveContext*>& waves,
                           const ExecutableKernel& kernel,
                           uint64_t& barrier_generation,
                           uint32_t& barrier_arrivals,
                           bool set_valid_entry_on_release);

}  // namespace sync_ops

}  // namespace gpu_model
