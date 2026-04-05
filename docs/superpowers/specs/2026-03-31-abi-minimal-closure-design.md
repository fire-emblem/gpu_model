# ABI Minimal Closure Design

## Goal

快速补齐最常见的 kernel 启动 ABI 能力，让更多真实 `hipcc` / `.out` 程序在 `st`、`mt` 和 encoded execution 路径下稳定启动并正确读取参数。

本批只覆盖：

- `system SGPR/VGPR` 常见启动寄存器
- `hidden / implicit args`
- `kernarg by-value / aggregate / alignment / padding`

本批不覆盖：

- relocation / `bss` / 更完整 segment materialize
- 图形指令族
- cycle 精细资源建模

## Design Scope

### 1. Kernarg Packing

`kernarg_packer` 应成为 host-visible launch args 到 ABI kernarg bytes 的唯一规范入口。

这批需要补齐：

- by-value struct / aggregate 的 `size`、`align`、`padding`
- pointer + scalar + aggregate 混合参数布局
- 显式 arg offset 优先于“顺序紧排”
- hidden args 接在 visible args 之后时的整体 layout 一致性
- metadata typed layout 与 fallback layout 的一致性检查

### 2. Hidden / Implicit Args

运行时与执行层应统一 hidden arg 的来源顺序：

1. 如果 metadata 提供 typed hidden arg layout，则严格按 metadata 装载
2. 如果 metadata 未提供，则按项目 fallback 规则补齐常见 implicit args

本批优先支持：

- `global_offset_x/y/z`
- `printf_buffer`
- `default_queue`
- `completion_action`
- `multigrid_sync_arg`
- `hostcall_buffer`
- `heap_v1`
- `dynamic_lds`
- `private_base` / `shared_base`

要求：

- 未启用的 hidden arg 不应破坏后续 offset
- 未知 hidden arg 类型必须保守跳过或报清晰错误，不能静默写错布局

### 3. System SGPR / VGPR Initialization

执行层应把“descriptor + metadata + launch config + placement”统一变成 wave 启动寄存器状态。

本批优先保证：

- `kernarg_segment_ptr`
- `workgroup_id_x/y/z`
- `workgroup_info`
- `private_segment_size`
- `private_segment_buffer`
- `dispatch_ptr`
- `queue_ptr`
- `dispatch_id`
- `flat_scratch_init`
- `workitem_id x/y/z`

要求：

- `st` / `mt` 的初始化结果一致
- modeled 路径和 encoded 路径读取到的 builtin 值一致
- descriptor 明确关闭的字段不应被意外写入

## Implementation Boundaries

### Runtime / Program

保留现有职责：

- `ExecEngine`
- `ModelRuntime`
- `ObjectReader`
- `EncodedProgramObject`

这层只负责：

- 读取 metadata / descriptor
- 组织 launch request
- 提供 kernarg bytes 与 launch config

不负责：

- 直接写 wave 执行寄存器

### Execution

执行层负责：

- wave 启动寄存器装载
- hidden arg 语义消费
- ABI builtin 值落到 `WaveContext`

优先修改：

- `src/execution/encoded_exec_engine.cpp`
- `src/execution/functional_exec_engine.cpp`
- `src/execution/wave_context_builder.cpp`

### Kernarg

`kernarg_packer` 负责：

- 按 typed layout 生成 visible + hidden args bytes
- 处理 alignment / offset / padding
- 提供 fallback layout 时的最小稳定规则

优先修改：

- `include/gpu_model/runtime/kernarg_packer.h`
- `src/runtime/kernarg_packer.cpp`

## Test Matrix

本批不只修当前已有测试，而是新增一组更常见 ABI 场景。

### Unit Tests

- by-value 小 struct
- 带 padding 的 aggregate
- pointer + scalar + aggregate 混合参数
- 显式 offset layout
- hidden args typed layout
- hidden args fallback layout

### Integration Tests

- 2D / 3D hidden args kernel
- dynamic shared memory kernel
- builtin + hidden arg 混合读取
- encoded path / modeled path ABI 一致性
- `st` / `mt` ABI 一致性

### Real HIP Executable Tests

- by-value aggregate kernel
- 2D / 3D builtin-id kernel
- dynamic LDS kernel
- mixed args kernel

## Success Criteria

本批完成后，应满足：

1. `kernarg_packer` 能稳定处理最常见 by-value / aggregate / padding 场景
2. typed hidden arg layout 与 fallback hidden arg layout 都有明确测试覆盖
3. encoded execution 和 functional execution 在 wave 启动 ABI 上行为一致
4. 新增真实 `hipcc` 程序能在 `st` / `mt` 下通过
5. 不破坏现有全量 `gpu_model_tests`

## Non-Goals

本批不做：

- relocation / `bss` / 更完整 loader 修补
- image / sampler / texture
- graphics ISA 覆盖
- cycle 资源冲突细化
