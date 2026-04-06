#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "gpu_model/arch/gpu_arch_spec.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/state/dpc_state.h"

namespace gpu_model {

struct GpuState {
  std::shared_ptr<const GpuArchSpec> spec;
  MemorySystem memory;
  std::vector<DpcState> dpcs;

  explicit GpuState(std::shared_ptr<const GpuArchSpec> in_spec = nullptr)
      : spec(std::move(in_spec)) {
    if (!spec) {
      return;
    }

    dpcs.reserve(spec->dpc_count);
    for (uint32_t dpc_id = 0; dpc_id < spec->dpc_count; ++dpc_id) {
      DpcState dpc;
      dpc.dpc_id = dpc_id;
      dpc.aps.reserve(spec->ap_per_dpc);
      for (uint32_t ap_id = 0; ap_id < spec->ap_per_dpc; ++ap_id) {
        ApState ap;
        ap.dpc_id = dpc_id;
        ap.ap_id = ap_id;
        ap.peus.reserve(spec->peu_per_ap);
        for (uint32_t peu_id = 0; peu_id < spec->peu_per_ap; ++peu_id) {
          ap.peus.push_back(PeuState{.peu_id = peu_id});
        }
        dpc.aps.push_back(std::move(ap));
      }
      dpcs.push_back(std::move(dpc));
    }
  }
};

}  // namespace gpu_model
