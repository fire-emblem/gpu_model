#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gpu_model {

// TraceWaveStepDetail captures structured instruction execution facts.
// All fields are producer-owned; trace renderer only consumes.
struct TraceWaveStepDetail {
  // Assembly representation
  std::string asm_text;
  
  // Operand reads/writes (formatted strings from producer)
  std::vector<std::string> scalar_reads;
  std::vector<std::string> vector_reads;
  std::vector<std::string> scalar_writes;
  std::vector<std::string> vector_writes;
  
  // Memory operation summary
  std::string mem_summary;
  
  // Exec mask before/after
  std::string exec_before;
  std::string exec_after;
  
  // Timing (modeled cycles, not physical time)
  uint64_t issue_cycle = 0;
  uint64_t commit_cycle = 0;
  uint64_t duration_cycles = 0;
  
  // State delta summary
  std::string state_summary;
};

}  // namespace gpu_model
