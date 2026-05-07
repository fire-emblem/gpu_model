# gpu_model `trace.txt` Design Reference And Target Template

## Purpose

This document captures the target design for `gpu_model`'s human-readable `trace.txt`.
It is intended to serve as a stable reference for:

- trace format evolution
- trace producer / exporter implementation
- tests that validate trace readability and field coverage

This design is informed by the existing simulator-style reference log pattern, but it is
adapted to fit `gpu_model`'s current architecture constraints:

- trace only consumes facts already produced by engine / state machine
- trace does not infer business logic
- trace `cycle` is modeled time, not physical hardware time
- `wave_step` is the authoritative instruction-execution fact

## Design Goals

`trace.txt` should optimize for both:

1. Human readability
2. Stable, machine-checkable structure

The target format therefore uses:

- a readable sectioned header for static run context
- compact single-line events for non-instruction events
- multi-line expanded blocks for `wave_step`

This is intentionally not a free-form simulator log dump.

## What To Learn From Traditional Simulator Logs

Useful aspects worth keeping:

- rich initialization / configuration context
- explicit resource configuration
- kernel launch context
- wave initialization context
- instruction-level execution facts
- final summary counters

Useful summary additions worth keeping at trace tail:

- instruction category distribution ratios
- branch / control-flow counters
- unit utilization counters
- issue-efficiency style aggregate counters

Aspects that should not be copied directly:

- overly verbose prose-style scheduler logs
- leaking internal function names as trace protocol
- mixing every category into a single unstructured text stream
- treating `cycle` as real hardware time

## Scope Of `trace.txt`

`trace.txt` should cover five layers of information:

1. Initialization parameter configuration
2. Resource / model configuration
3. Kernel run parameters
4. Wave initialization parameters
5. Wave execution events, especially per-instruction execution facts

The trace tail should also include a performance-analysis summary section so a reader can
quickly inspect execution mix and structural pressure without opening a separate report.

For instruction execution, the trace should record:

- modeled execution cycle
- pc
- full instruction assembly text
- all operand reads
- all operand writes
- memory access summary when applicable
- key state changes that affect later execution

## Separation Of Concerns

The target split is:

- `trace.txt`: human-readable primary trace
- `trace.jsonl`: strict machine-readable equivalent
- stats / timeline artifacts: separate views over the same execution facts

`trace.txt` must not become a second business-logic layer.

## Event Principles

Dynamic events in `trace.txt` must reflect facts already decided by engine / state machine.

In particular:

- `arrive_resume` means ready / eligible, not guaranteed issue
- `wave_step` means actual issued / executed instruction fact
- `cycle` means modeled `global_cycle`

## Vector Register Rendering Policy

Vector register display is controlled by execution options and affects presentation only.
It does not alter execution semantics or trace business fields.

Recommended option:

```text
-trace_vector_step={N}
```

Rules:

- `N=1`: show every lane
- `N=2`: show lanes `0,2,4,6,...`
- `N=4`: show lanes `0,4,8,12,...`
- `N=k`: show lanes `0,k,2k,3k,...`

Recommended default:

```text
-trace_vector_step=4
```

Display rules:

- vector values are aligned in columns
- integer values are shown in hexadecimal only
- floating-point values are shown in hexadecimal and floating-point form
- if operand type cannot be identified reliably, fall back to hexadecimal only

Floating-point special values should preserve raw bits while rendering the FP column as:

- `nan`
- `+inf`
- `-inf`

## Read / Write Coverage Policy

For `wave_step`, the `rw` block must include all actual reads and writes involved in the
instruction semantics, not just visually explicit assembly operands.

This includes:

- explicit source operand reads
- explicit destination operand writes
- implicit register / state reads and writes
- `exec`, `vcc`, `scc`, `m0`, `pc`
- waitcnt / barrier / readiness state when semantically affected
- address-generation inputs for memory instructions
- atomic memory read/write effects

## Target `trace.txt` Template

```text
================================================================================
GPU_MODEL TRACE
================================================================================
trace_format_version: 1
project: gpu_model
runtime_version: {runtime_version}
build_tag: {build_tag}
execution_model: {functional_st|functional_mt|cycle|encoded}
trace_time_basis: modeled_cycle
trace_cycle_is_physical_time: false
trace_sink_enabled: {true|false}
trace_disable_env: GPU_MODEL_DISABLE_TRACE

[RUN]
invocation: {gpu_model_env_vars} {command_line}
run_id: {run_id}
pid: {pid}
start_time_utc: {start_time_utc}
binary_path: {binary_path}
resolved_binary_path: {resolved_binary_path}
input_file: {input_file}
config_file: {config_file}
workload_name: {workload_name}
iteration: {iteration}
result: {PASS|FAIL|UNKNOWN}

[RUNTIME_CONFIG]
project_target: {target}
force_capability: {capability}
log_level: {log_level}
trace_enabled: {true|false}
stats_enabled: {true|false}
checkpoint_enabled: {true|false}
checkpoint_interval: {checkpoint_interval}
resume_enabled: {true|false}
resume_from: {resume_from}

[MODEL_CONFIG]
num_dpc: {num_dpc}
num_ap_per_dpc: {num_ap_per_dpc}
num_peu_per_ap: {num_peu_per_ap}
num_wave_slots_per_peu: {num_wave_slots_per_peu}
max_concurrent_blocks: {max_concurrent_blocks}
max_issuable_waves: {max_issuable_waves}
issue_quantum_cycles: {issue_quantum_cycles}
wave_launch_cycles: {wave_launch_cycles}
wave_dispatch_cycles: {wave_dispatch_cycles}
global_cycle_start: 0

[RESOURCE_CONFIG]
shared_mem_size_per_core: {shared_mem_size}
register_file_size_per_core: {register_file_size}
l1_config: {l1_config}
l2_config: {l2_config}
buffer_size_request: {buffer_size_request}
buffer_size_response: {buffer_size_response}
clock_rate_core: {clock_rate_core}
clock_rate_mem: {clock_rate_mem}
addr_dec_mask_global: {addr_dec_mask_global}
addr_dec_mask_local: {addr_dec_mask_local}
addr_dec_mask_shared: {addr_dec_mask_shared}
sub_partition_id_mask: {sub_partition_id_mask}

[TRACE_DISPLAY]
vector_step: {N}
vector_expand_policy: sampled
integer_format: hex
float_format: hex+float
vector_align_columns: true

[KERNEL]
kernel_name: {kernel_name}
kernel_launch_uid: {kernel_launch_uid}
stream_id: {stream_id}
kernel_handle: {kernel_handle}
host_fun_ptr: {host_fun_ptr}
grid_dim: ({grid_x},{grid_y},{grid_z})
block_dim: ({block_x},{block_y},{block_z})
regs_per_thread: {regs}
lmem_per_thread: {lmem}
smem_per_block: {smem}
cmem_bytes: {cmem}
task_per_core: {task_per_core}
occupancy_limiter: {limiter}

[WAVE_INIT]
wave={stable_wave_id} block={block_id} loc=dpc{dpc_id}/ap{ap_id}/peu{peu_id}/slot{slot_id} slot_model={slot_model} start_pc={start_pc} exec_mask={exec_mask} vgpr_base={vgpr_base} sgpr_base={sgpr_base} wave_cycle_total=0 wave_cycle_active=0 ready_at={ready_at_global_cycle} next_issue_at={next_issue_earliest_global_cycle} waitcnt_init={waitcnt_init} barrier_init={barrier_init}

[EVENTS]
# cycle     seq   kind            wave      pc        asm/details
{event_lines...}

[SUMMARY]
kernel_status: {kernel_status}
gpu_tot_sim_cycle: {gpu_tot_sim_cycle}
gpu_tot_sim_insn: {gpu_tot_sim_insn}
gpu_tot_ipc: {gpu_tot_ipc}
gpu_tot_issued_blocks: {gpu_tot_issued_blocks}
gpu_tot_wave_exits: {gpu_tot_wave_exits}
stall_scoreboard: {stall_scoreboard}
stall_dependency: {stall_dependency}
stall_waitcnt: {stall_waitcnt}
stall_barrier: {stall_barrier}
stall_resource_busy: {stall_resource_busy}
stall_warp_switch: {stall_warp_switch}
core_activity: {core_activity}
l2_total_cache_accesses: {l2_total_cache_accesses}
l2_total_cache_misses: {l2_total_cache_misses}
icnt_total_pkts_mem_to_core: {icnt_total_pkts_mem_to_core}
icnt_total_pkts_core_to_mem: {icnt_total_pkts_core_to_mem}
simulation_elapsed_sec: {elapsed_total_sec}
simulation_rate_inst_sec: {simulation_rate_inst_sec}
simulation_rate_cycle_sec: {simulation_rate_cycle_sec}
silicon_slowdown: {slowdown_factor}

[PERF_ANALYSIS]
instruction_mix_total: {instruction_mix_total}
instruction_mix_scalar_alu: {scalar_alu_count} ({scalar_alu_ratio}%)
instruction_mix_scalar_mem: {scalar_mem_count} ({scalar_mem_ratio}%)
instruction_mix_vector_alu: {vector_alu_count} ({vector_alu_ratio}%)
instruction_mix_vector_mem: {vector_mem_count} ({vector_mem_ratio}%)
instruction_mix_branch: {branch_count} ({branch_ratio}%)
instruction_mix_sync: {sync_count} ({sync_ratio}%)
instruction_mix_tensor: {tensor_count} ({tensor_ratio}%)
instruction_mix_other: {other_count} ({other_ratio}%)

branch_total: {branch_total}
branch_taken: {branch_taken}
branch_not_taken: {branch_not_taken}
branch_divergent: {branch_divergent}
barrier_total: {barrier_total}
waitcnt_total: {waitcnt_total}

dpc_utilization_pct: {dpc_utilization_pct}
ap_utilization_pct: {ap_utilization_pct}
peu_utilization_pct: {peu_utilization_pct}
wave_slot_utilization_pct: {wave_slot_utilization_pct}
issue_slot_utilization_pct: {issue_slot_utilization_pct}
memory_pipeline_utilization_pct: {memory_pipeline_utilization_pct}
shared_pipeline_utilization_pct: {shared_pipeline_utilization_pct}
tensor_pipeline_utilization_pct: {tensor_pipeline_utilization_pct}

issue_eligible_cycles: {issue_eligible_cycles}
issue_selected_cycles: {issue_selected_cycles}
issue_empty_cycles: {issue_empty_cycles}
issue_conflict_cycles: {issue_conflict_cycles}
issue_backpressure_cycles: {issue_backpressure_cycles}

[END]
exit_detected: true
```

## Event Templates

### Compact Single-Line Events

Non-`wave_step` events should remain single-line for readability.

```text
[000001] #2   slot_bind      w700000  0x100   slot=0 model=resident_fixed reason=launch
[000004] #3   issue_select   w700000  0x100   asm="s_load_dword s4, s[0:1], 0x0" eligible=1 budget=1
[000005] #5   wave_wait      w700000  0x104   reason=waitcnt blocked=global deps="vmcnt>0"
[000008] #6   wave_arrive    w700000  0x104   kind=load progress=complete flow=91 resumed_ready=1
[000008] #7   wave_resume    w700000  0x104   reason=arrive_ready ready_at=8 next_issue_at=8
[000032] #9   wave_exit      w700000  0x180   wave_cycle_total=32 wave_cycle_active=24
```

### Expanded `wave_step`

`wave_step` is the authoritative instruction execution record and may expand into multiple
readable lines.

```text
[000012] #8   wave_step      w700000  0x104   v_add_f32 v0, v1, v2
  rw:
    R:
      scalar: exec=0xffffffffffffffff
      v1[sampled step=4]:
        lane  value(hex)   value(fp)
        0     0x3f800000   1.000000
        4     0x40800000   4.000000
        8     0x41000000   8.000000
        12    0x41400000   12.000000
      v2[sampled step=4]:
        lane  value(hex)   value(fp)
        0     0x40000000   2.000000
        4     0x40400000   3.000000
        8     0x40800000   4.000000
        12    0x40a00000   5.000000
    W:
      scalar: vcc=0x0000000000000000
      v0[sampled step=4]:
        lane  value(hex)   value(fp)
        0     0x40400000   3.000000
        4     0x40e00000   7.000000
        8     0x41400000   12.000000
        12    0x41880000   17.000000
  mem: none
  mask: exec_before=0xffffffffffffffff exec_after=0xffffffffffffffff
  timing: issue=12 commit=16 dur=4
  state: waitcnt_before=vmcnt=0 waitcnt_after=vmcnt=0 scc_before=0 scc_after=0
```

## Full Example

```text
================================================================================
GPU_MODEL TRACE
================================================================================
trace_format_version: 1
project: gpu_model
runtime_version: 0.3.2
build_tag: dev-abc123
execution_model: cycle
trace_time_basis: modeled_cycle
trace_cycle_is_physical_time: false
trace_sink_enabled: true
trace_disable_env: GPU_MODEL_DISABLE_TRACE

[RUN]
invocation: GPU_MODEL_EXECUTION_MODE=cycle GPU_MODEL_DISABLE_TRACE=0 ./build/example kernel.co
run_id: 42
pid: 18342
start_time_utc: 2026-04-10T11:20:33Z
binary_path: build/example
resolved_binary_path: build/example
input_file: kernel.co
config_file: configs/mac500.yaml
workload_name: vecadd
iteration: 1
result: UNKNOWN

[TRACE_DISPLAY]
vector_step: 4
vector_expand_policy: sampled
integer_format: hex
float_format: hex+float
vector_align_columns: true

[KERNEL]
kernel_name: vecadd_kernel
kernel_launch_uid: 7
stream_id: 0
kernel_handle: 0x1
host_fun_ptr: 0x7f00
grid_dim: (128,1,1)
block_dim: (256,1,1)
regs_per_thread: 24
lmem_per_thread: 0
smem_per_block: 0
cmem_bytes: 64
task_per_core: 8
occupancy_limiter: registers

[WAVE_INIT]
wave=700000 block=0 loc=dpc0/ap0/peu0/slot0 slot_model=resident_fixed start_pc=0x100 exec_mask=0xffffffffffffffff vgpr_base=0 sgpr_base=0 wave_cycle_total=0 wave_cycle_active=0 ready_at=0 next_issue_at=0 waitcnt_init=vmcnt=0 barrier_init=none

[EVENTS]
[000000] #1   wave_generate  w700000  0x100   block=0 slot=0
[000001] #2   slot_bind      w700000  0x100   slot=0 model=resident_fixed reason=launch
[000004] #3   issue_select   w700000  0x100   asm="s_load_dword s4, s[0:1], 0x0" eligible=1 budget=1
[000004] #4   wave_step      w700000  0x100   s_load_dword s4, s[0:1], 0x0
  rw:
    R:
      scalar: s0=0x00002000 s1=0x00000000 exec=0xffffffffffffffff
    W:
      scalar: s4=0x12345678
  mem: LOAD global addr=0x00002000 size=4 data=0x12345678
  mask: exec_before=0xffffffffffffffff exec_after=0xffffffffffffffff
  timing: issue=4 commit=8 dur=4
  state: waitcnt_before=vmcnt=1 waitcnt_after=vmcnt=0
[000012] #5   wave_step      w700000  0x104   v_add_f32 v0, v1, v2
  rw:
    R:
      scalar: exec=0xffffffffffffffff
      v1[sampled step=4]:
        lane  value(hex)   value(fp)
        0     0x3f800000   1.000000
        4     0x40800000   4.000000
        8     0x41000000   8.000000
        12    0x41400000   12.000000
      v2[sampled step=4]:
        lane  value(hex)   value(fp)
        0     0x40000000   2.000000
        4     0x40400000   3.000000
        8     0x40800000   4.000000
        12    0x40a00000   5.000000
    W:
      scalar: vcc=0x0000000000000000
      v0[sampled step=4]:
        lane  value(hex)   value(fp)
        0     0x40400000   3.000000
        4     0x40e00000   7.000000
        8     0x41400000   12.000000
        12    0x41880000   17.000000
  mem: none
  mask: exec_before=0xffffffffffffffff exec_after=0xffffffffffffffff
  timing: issue=12 commit=16 dur=4
  state: waitcnt_before=vmcnt=0 waitcnt_after=vmcnt=0 scc_before=0 scc_after=0
[000032] #6   wave_exit      w700000  0x180   wave_cycle_total=32 wave_cycle_active=24

[SUMMARY]
kernel_status: completed
gpu_tot_sim_cycle: 120
gpu_tot_sim_insn: 4096
gpu_tot_ipc: 1.23
gpu_tot_issued_blocks: 128
gpu_tot_wave_exits: 512
stall_scoreboard: 0
stall_dependency: 0
stall_waitcnt: 64
stall_barrier: 0
stall_resource_busy: 3
stall_warp_switch: 12
core_activity: 0.73
l2_total_cache_accesses: 1024
l2_total_cache_misses: 16
icnt_total_pkts_mem_to_core: 128
icnt_total_pkts_core_to_mem: 128
simulation_elapsed_sec: 0.32
simulation_rate_inst_sec: 128000
simulation_rate_cycle_sec: 375
silicon_slowdown: 1500

[PERF_ANALYSIS]
instruction_mix_total: 4096
instruction_mix_scalar_alu: 256 (6.25%)
instruction_mix_scalar_mem: 512 (12.50%)
instruction_mix_vector_alu: 2048 (50.00%)
instruction_mix_vector_mem: 896 (21.88%)
instruction_mix_branch: 192 (4.69%)
instruction_mix_sync: 160 (3.91%)
instruction_mix_tensor: 0 (0.00%)
instruction_mix_other: 32 (0.78%)

branch_total: 192
branch_taken: 144
branch_not_taken: 48
branch_divergent: 12
barrier_total: 16
waitcnt_total: 80

dpc_utilization_pct: 71.2
ap_utilization_pct: 68.4
peu_utilization_pct: 63.7
wave_slot_utilization_pct: 59.1
issue_slot_utilization_pct: 52.8
memory_pipeline_utilization_pct: 47.5
shared_pipeline_utilization_pct: 8.0
tensor_pipeline_utilization_pct: 0.0

issue_eligible_cycles: 104
issue_selected_cycles: 96
issue_empty_cycles: 24
issue_conflict_cycles: 11
issue_backpressure_cycles: 7

[END]
exit_detected: true
```
