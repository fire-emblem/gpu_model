# gpu_model `trace.txt` Recommended Default Template

## Purpose

This document defines the recommended default `trace.txt` template for `gpu_model`.

It is narrower than the broader target-template reference:

- keep the information that is usually actionable for HPC performance analysis
- keep the information needed for instruction-level semantic debugging
- remove fields that are expensive, noisy, or only useful in special investigations

The default template is intended to answer four questions quickly:

1. What ran
2. How it progressed
3. Why it stalled
4. Whether the bottleneck is compute, memory, control flow, or scheduling

## Default-Include Principles

By default, `trace.txt` should include information that satisfies at least one of:

- explains an observed execution fact
- supports immediate performance diagnosis
- helps distinguish algorithm / compiler / runtime / scheduling causes

By default, `trace.txt` should avoid information that:

- repeats static context on every event line
- leaks unstable implementation details
- produces high volume with low decision value
- belongs in `trace.jsonl` or a deeper debug artifact instead

## Recommended Default Scope

Include by default:

- run / config / kernel / wave-init context
- only the minimum static context needed to interpret the run
- compact lifecycle events
- expanded `wave_step` instruction facts
- core stall summary
- compact performance summary
- warning / anomaly summary

Do not include by default:

- full lane expansion for vectors
- full register-file dumps
- full memory dumps
- per-cycle empty/no-op logs
- exhaustive per-unit diagnostics that lack direct actionability
- fields that can be trivially derived from other already-printed defaults

## Vector Rendering Policy

Vector rendering is presentation-only and controlled by execution options.

Recommended default:

```text
-trace_vector_step=4
```

Rules:

- `1`: full lane expansion
- `2`: print one value for every 2 lanes
- `4`: print one value for every 4 lanes
- `N`: print lanes `0, N, 2N, 3N, ...`

Formatting rules:

- vector values are column-aligned
- integer values are rendered in hexadecimal
- floating-point values are rendered as raw hexadecimal plus floating-point value
- unknown types fall back to hexadecimal only

## Recommended Default Template

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

[RUN]
run_id: {run_id}
input_file: {input_file}
config_file: {config_file}
result: {PASS|FAIL|UNKNOWN}

[MODEL_CONFIG]
num_peu_per_ap: {num_peu_per_ap}
num_wave_slots_per_peu: {num_wave_slots_per_peu}
max_issuable_waves: {max_issuable_waves}
issue_quantum_cycles: {issue_quantum_cycles}
wave_launch_cycles: {wave_launch_cycles}
wave_dispatch_cycles: {wave_dispatch_cycles}

[TRACE_DISPLAY]
vector_step: {N}
integer_format: hex
float_format: hex+float
vector_align_columns: true

[KERNEL]
kernel_name: {kernel_name}
kernel_launch_uid: {kernel_launch_uid}
grid_dim: ({grid_x},{grid_y},{grid_z})
block_dim: ({block_x},{block_y},{block_z})
regs_per_thread: {regs}
smem_per_block: {smem}
occupancy_limiter: {limiter}

[WAVE_INIT]
wave={stable_wave_id} block={block_id} loc=dpc{dpc_id}/ap{ap_id}/peu{peu_id}/slot{slot_id} slot_model={slot_model} start_pc={start_pc} exec_mask={exec_mask}

[EVENTS]
# cycle     seq   kind            wave      pc        asm/details
[000000] #1   wave_generate  w{stable_wave_id}  {start_pc}   block={block_id} slot={slot_id}
[000001] #2   slot_bind      w{stable_wave_id}  {start_pc}   slot={slot_id} model={slot_model} reason={bind_reason}
[000004] #3   issue_select   w{stable_wave_id}  {pc}         eligible={eligible_count} budget={issue_budget}
[000005] #4   wave_wait      w{stable_wave_id}  {pc}         reason={stall_reason} blocked={blocked_domain} deps="{deps}"
[000008] #5   wave_arrive    w{stable_wave_id}  {pc}         kind={arrive_kind} progress={arrive_progress} flow={flow_id} resumed_ready={0|1}
[000008] #6   wave_resume    w{stable_wave_id}  {pc}         reason={resume_reason}
[000032] #7   wave_exit      w{stable_wave_id}  {pc}         wave_cycle_total={wave_cycle_total} wave_cycle_active={wave_cycle_active}

[000012] #8   wave_step      w{stable_wave_id}  {pc}         {full_asm}
  rw:
    R:
      scalar: {scalar_reads}
      {vector_read_blocks}
    W:
      scalar: {scalar_writes}
      {vector_write_blocks}
  mem: {none|LOAD|STORE|ATOMIC} {mem_detail}
  mask: exec_before={exec_before} exec_after={exec_after}
  timing: issue={issue_cycle} commit={commit_cycle} dur={duration_cycles}
  state: waitcnt_before={waitcnt_before} waitcnt_after={waitcnt_after} {other_state_changes}

[SUMMARY]
kernel_status: {kernel_status}
gpu_tot_sim_cycle: {gpu_tot_sim_cycle}
gpu_tot_sim_insn: {gpu_tot_sim_insn}
gpu_tot_ipc: {gpu_tot_ipc}
gpu_tot_wave_exits: {gpu_tot_wave_exits}

[STALL_SUMMARY]
stall_scoreboard: {stall_scoreboard}
stall_dependency: {stall_dependency}
stall_waitcnt: {stall_waitcnt}
stall_barrier: {stall_barrier}
stall_resource_busy: {stall_resource_busy}
stall_warp_switch: {stall_warp_switch}

[MEMORY_AND_RESOURCES]
shared_loads: {shared_loads}
shared_stores: {shared_stores}
shared_bank_conflict_penalty_cycles: {shared_bank_conflict_penalty_cycles}
shared_bank_conflict_pct: {shared_bank_conflict_pct}
private_loads: {private_loads}
private_stores: {private_stores}
private_mem_bytes: {private_mem_bytes}
global_mem_loads: {global_mem_loads}
global_mem_stores: {global_mem_stores}
global_mem_bytes: {global_mem_bytes}
regs_per_thread: {regs}
register_pressure_pct: {register_pressure_pct}
smem_per_block: {smem}
shared_mem_utilization_pct: {shared_mem_utilization_pct}
waves_limited_by_registers: {waves_limited_by_registers}
waves_limited_by_shared_mem: {waves_limited_by_shared_mem}

[PERF]
instruction_mix_total: {instruction_mix_total}
scalar_alu: {scalar_alu_count} ({scalar_alu_ratio}%)
scalar_mem: {scalar_mem_count} ({scalar_mem_ratio}%)
vector_alu: {vector_alu_count} ({vector_alu_ratio}%)
vector_mem: {vector_mem_count} ({vector_mem_ratio}%)
branch: {branch_count} ({branch_ratio}%)
sync: {sync_count} ({sync_ratio}%)
tensor: {tensor_count} ({tensor_ratio}%)
other: {other_count} ({other_ratio}%)
branch_total: {branch_total}
branch_divergent: {branch_divergent}
barrier_total: {barrier_total}
waitcnt_total: {waitcnt_total}
peu_utilization_pct: {peu_utilization_pct}
wave_slot_utilization_pct: {wave_slot_utilization_pct}
issue_slot_utilization_pct: {issue_slot_utilization_pct}
memory_pipeline_utilization_pct: {memory_pipeline_utilization_pct}
issue_eligible_cycles: {issue_eligible_cycles}
issue_selected_cycles: {issue_selected_cycles}
issue_conflict_cycles: {issue_conflict_cycles}
issue_backpressure_cycles: {issue_backpressure_cycles}

[WARNINGS]
# Only emit lines when threshold is crossed or an anomaly is detected.
# Recommended default warning kinds:
# - private_mem_bytes
# - shared_bank_conflict_pct
# - register_pressure_pct
# - shared_mem_utilization_pct
# - branch_divergent
WARNING {warning_kind} value={value} threshold={threshold} detail="{detail}"

[END]
exit_detected: true
```

## Why This Template Is A Better Default

This template keeps:

- enough static context to make the run reproducible
- enough event context to reconstruct scheduling and waits
- enough per-instruction evidence to debug semantics
- enough summary fields to classify bottlenecks
- enough memory/resource signals to diagnose common dense-kernel pathologies

This template deliberately excludes several fields from the broader reference or treats
them as optional:

- DPC/AP utilization
- shared/tensor pipeline utilization
- large environment dumps
- extra path / process metadata
- deep per-unit breakdowns
- fields derivable from already printed defaults

Those fields may still be useful, but they are better treated as optional or advanced-detail
fields rather than default always-on trace content.

## Recommended Optional Fields

If a deeper diagnosis mode is enabled, the following fields are good candidates to add:

- `dpc_utilization_pct`
- `ap_utilization_pct`
- `shared_pipeline_utilization_pct`
- `tensor_pipeline_utilization_pct`
- top-N slowest waves
- top-N longest memory waits
- per-category average issue cycles
- full vector lane expansion with `trace_vector_step=1`
- detailed bank-by-bank shared conflict histograms
- full spill/load-store address traces

## Example

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

[RUN]
run_id: 42
input_file: kernel.co
config_file: configs/mac500.yaml
result: PASS

[MODEL_CONFIG]
num_peu_per_ap: 4
num_wave_slots_per_peu: 8
max_issuable_waves: 4
issue_quantum_cycles: 4
wave_launch_cycles: 1
wave_dispatch_cycles: 1

[TRACE_DISPLAY]
vector_step: 4
integer_format: hex
float_format: hex+float
vector_align_columns: true

[KERNEL]
kernel_name: vecadd_kernel
kernel_launch_uid: 7
grid_dim: (128,1,1)
block_dim: (256,1,1)
regs_per_thread: 24
smem_per_block: 0
occupancy_limiter: registers

[WAVE_INIT]
wave=700000 block=0 loc=dpc0/ap0/peu0/slot0 slot_model=resident_fixed start_pc=0x100 exec_mask=0xffffffffffffffff

[EVENTS]
[000000] #1   wave_generate  w700000  0x100   block=0 slot=0
[000001] #2   slot_bind      w700000  0x100   slot=0 model=resident_fixed reason=launch
[000004] #3   issue_select   w700000  0x100   eligible=1 budget=1
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
[000005] #5   wave_wait      w700000  0x104   reason=waitcnt blocked=global deps="vmcnt>0"
[000008] #6   wave_arrive    w700000  0x104   kind=load progress=complete flow=91 resumed_ready=1
[000008] #7   wave_resume    w700000  0x104   reason=arrive_ready
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
[000032] #9   wave_exit      w700000  0x180   wave_cycle_total=32 wave_cycle_active=24

[SUMMARY]
kernel_status: completed
gpu_tot_sim_cycle: 120
gpu_tot_sim_insn: 4096
gpu_tot_ipc: 1.23
gpu_tot_wave_exits: 512

[STALL_SUMMARY]
stall_scoreboard: 0
stall_dependency: 0
stall_waitcnt: 64
stall_barrier: 0
stall_resource_busy: 3
stall_warp_switch: 12

[MEMORY_AND_RESOURCES]
shared_loads: 320
shared_stores: 192
shared_bank_conflict_penalty_cycles: 288
shared_bank_conflict_pct: 4.69
private_loads: 8
private_stores: 8
private_mem_bytes: 512
global_mem_loads: 896
global_mem_stores: 128
global_mem_bytes: 65536
regs_per_thread: 24
register_pressure_pct: 75.0
smem_per_block: 8192
shared_mem_utilization_pct: 12.5
waves_limited_by_registers: 1
waves_limited_by_shared_mem: 0

[PERF]
instruction_mix_total: 4096
scalar_alu: 256 (6.25%)
scalar_mem: 512 (12.50%)
vector_alu: 2048 (50.00%)
vector_mem: 896 (21.88%)
branch: 192 (4.69%)
sync: 160 (3.91%)
tensor: 0 (0.00%)
other: 32 (0.78%)
branch_total: 192
branch_divergent: 12
barrier_total: 16
waitcnt_total: 80
peu_utilization_pct: 63.7
wave_slot_utilization_pct: 59.1
issue_slot_utilization_pct: 52.8
memory_pipeline_utilization_pct: 47.5
issue_eligible_cycles: 104
issue_selected_cycles: 96
issue_conflict_cycles: 11
issue_backpressure_cycles: 7

[WARNINGS]
WARNING private_mem_bytes value=512 threshold=0 detail="private memory traffic detected; check for spills or unintended private accesses"

[END]
exit_detected: true
```
