# Wave Launch ABI Summary Design

## Goal

细化现有 `WaveLaunch` trace message，让它从“固定寄存器窗口摘要”提升为“可读的 ABI 语义摘要”，更直接回答：

- kernarg 指针是否装载正确
- workgroup id 是否按 `x/y/z` 维度初始化
- workitem id 是否按 `x/y/z` 维度初始化
- descriptor 驱动的关键 SGPR ABI 字段是否进入 wave 初态

同时保持：

- 仍然使用单条 `TraceEventKind::WaveLaunch`
- 仍然保持单行 message
- 不把 trace 变成完整寄存器 dump

## Scope

本轮只做 `WaveLaunch` 文本摘要细化，不扩展到其它 trace kind。

本轮范围：

- 细化 `FormatWaveLaunchTraceMessage(...)`
- 统一 `FunctionalExecEngine / CycleExecEngine / EncodedExecEngine` 三条路径的 `WaveLaunch` message 语义
- 为 raw-GCN / encoded ABI 路径增加更稳定的语义字段断言
- 保留现有拓扑/执行摘要字段

本轮明确不做：

- 不新增 trace event kind
- 不输出完整 SGPR/VGPR dump
- 不把所有 hidden arg / visible arg preload 全部按字节展开
- 不在 `WaveStep` 或 `WaveStats` 中同步改 schema
- 不尝试本轮补全更多 ABI preload 逻辑本身，本轮只提升“已存在状态”的可观测性

## Current Problem

当前 `WaveLaunch` 已经输出：

- `block_xyz`
- `dpc/ap/peu`
- `lanes/exec/cmask/smask`
- 一个固定或半固定长度的 `sgpr={...}` / `vgpr={...}` 窗口

问题在于它仍然过于“寄存器号导向”：

- `sgpr={s0=...,s1=...}` 需要读者自己知道 ABI 槽位映射
- raw-GCN fallback ABI 与 descriptor-explicit ABI 的寄存器布局不同，当前 message 不能稳定表达“这个值到底是什么语义”
- `tests/runtime/hip_runtime_test.cpp` 里已有断言直接检查 `s4/s5/s6/s7` 与 `v2[...]`，这锁定了实现细节，而不是 ABI 含义

结果是：

- trace 可读性不高
- 回归测试更像“寄存器槽位快照”而不是“ABI 合约”
- 不同 backend 想保持一致时，容易重新回到各自硬编码窗口

## Existing Data Sources

当前代码里已经有足够信息支撑“语义摘要”，关键来源包括：

### 1. `WaveContext`

已有：

- `block_idx_x/y/z`
- `thread_count`
- `exec/cmask/smask`
- `sgpr`
- `vgpr`

### 2. Encoded / raw-GCN ABI recipe

`src/execution/encoded_exec_engine.cpp` 已经显式知道：

- fallback ABI 下 `s4:s5 -> kernarg ptr`
- fallback ABI 下 `s6/s7 -> workgroup id x/y`
- fallback ABI 下 `v0/v1/v2 -> workitem id x/y/z`
- explicit descriptor ABI 下各个 `enable_sgpr_*` / `enable_vgpr_workitem_id` 对应的槽位顺序

### 3. 当前 trace helper 已经集中在一处

`include/gpu_model/debug/wave_launch_trace.h` 已经是三条执行路径共享的格式化入口，这意味着本轮可以在一个地方统一 schema，而不是在三个 executor 内分别拼接字符串。

## Design Summary

本轮继续保留 `WaveLaunch` 的单行摘要结构，但把 `sgpr={...}` / `vgpr={...}` 的内容从“原始槽位优先”改成“ABI 语义优先”。

推荐格式：

```text
block_xyz=(0x0,0x0,0x0) dpc=0x0 ap=0x0 peu=0x0 lanes=0x40 exec=0xffffffffffffffff cmask=0x0 smask=0x0 sgpr={kernarg_ptr=0x50000000,wg_id_x=0x0,wg_id_y=0x0} vgpr={workitem_id_x[0,1]={0x0,0x1},workitem_id_y[0,1]={0x0,0x0},workitem_id_z[0,1]={0x0,0x0}}
```

其中：

- 保留现有拓扑/执行字段
- `sgpr={...}` 改为语义 key
- `vgpr={...}` 改为语义 key + lane sample
- `tensor={...}` 继续保留当前格式

## Field Policy

### 保留不变的顶层字段

这些字段继续原样输出：

- `block_xyz`
- `dpc`
- `ap`
- `peu`
- `lanes`
- `exec`
- `cmask`
- `smask`

原因：

- 这些字段已经稳定
- 现有测试与 trace 阅读习惯已经依赖它们
- 本轮目标是提升 ABI 可读性，不是重写整个 `WaveLaunch` schema

### `sgpr={...}` 采用语义优先、原始回退

规则：

1. 如果当前路径知道 ABI 语义映射，则优先打印语义字段
2. 如果当前路径拿不到稳定语义映射，则退回当前原始 `sN=` 窗口格式
3. 不为未知 preload 值发明字段名

推荐语义字段：

- `kernarg_ptr`
- `dispatch_ptr`
- `queue_ptr`
- `dispatch_id`
- `flat_scratch_init`
- `private_segment_size`
- `wg_id_x`
- `wg_id_y`
- `wg_id_z`
- `workgroup_info`

对 descriptor 中难以在本轮稳定命名、或者只是“预留/占位 preload 槽位”的内容，不强行纳入语义摘要。

### `vgpr={...}` 采用 workitem 语义字段

规则：

- 如果 ABI recipe 明确 `v0/v1/v2` 的语义为 local/workitem id，则打印成：
  - `workitem_id_x[...]`
  - `workitem_id_y[...]`
  - `workitem_id_z[...]`
- lane 仍只显示少量 sample，默认保留当前 `0,1` 两个 lane
- 如果当前路径没有稳定语义映射，则退回当前 `vN[...]` 格式

## Backend Strategy

### Functional / Cycle 主线

对 modeled kernel：

- 保持单条 `WaveLaunch`
- 优先复用统一 formatter
- 如果只有通用 `WaveContext` 而没有 ABI recipe，则允许退回当前原始窗口

也就是说，本轮不要求 modeled kernel 必须凭空推导所有 ABI 字段。

### Encoded / raw-GCN 主线

对 encoded kernel：

- fallback ABI 直接按现有约定导出语义字段
- descriptor-explicit ABI 按 `AmdgpuKernelDescriptor` 的 enable 位顺序解析
- raw-GCN 路径是本轮最主要的收益面，因为它已经有最完整的 ABI recipe

## Compatibility Rule

本轮不是“删掉原始寄存器可见性”，而是“优先把已知 ABI 字段语义化”。

因此兼容规则是：

- 语义可确定：用语义 key
- 语义不可确定：保留原始 `sN=` / `vN=` 作为回退

这样能避免两个坏结果：

- 为了追求漂亮 schema 而丢失已有信息
- 为了保持兼容而继续长期锁定 `s4/s5/v2` 这类实现细节

## Testing Strategy

### 1. `TraceTest` 基础回归

继续验证：

- `WaveLaunch` 事件存在
- `lanes/exec`
- `sgpr={` / `vgpr={` 仍存在

同时把断言从“只存在寄存器组”提高到“至少一个语义字段存在”，例如：

- `kernarg_ptr=`
- `workitem_id_x[`

### 2. raw-GCN fallback ABI 回归

更新 `tests/runtime/hip_runtime_test.cpp` 中现有 fallback ABI 断言：

- 从检查 `s4/s5/s6/s7`
- 改为检查 `kernarg_ptr=` / `wg_id_x=` / `wg_id_y=`

这样测试锁定的是 ABI 语义，而不是 fallback 槽位细节。

### 3. 3D builtin-id 回归

更新现有三维 builtin-id trace 回归：

- 从检查 `v2[0,1]={0x0,0x1}`
- 改为检查 `workitem_id_z[0,1]={0x0,0x1}`

### 4. Tensor 兼容性

确保 `tensor={agpr_count=...,accum_offset=...}` 继续保留，不被新 ABI 摘要覆盖或打乱。

## Acceptance Criteria

本轮完成标准：

1. `WaveLaunch` 仍保持单条单行 message
2. `sgpr={...}` / `vgpr={...}` 在已知 ABI 路径上优先输出语义字段
3. fallback ABI 路径能稳定看到 `kernarg_ptr` 与 `wg_id_x/y`
4. builtin-id 路径能稳定看到 `workitem_id_x/y/z` 的语义样本
5. `TraceTest.*` 与受影响的 `HipRuntimeTest.*` wave-launch trace regressions 通过
6. 不影响 tensor launch trace 与现有非 ABI 相关 `WaveLaunch` 观察

## Approaches Considered

### Option A: 直接扩大原始寄存器窗口

优点：

- 实现简单
- 保持完全兼容

缺点：

- 可读性提升有限
- 测试仍然只能锁槽位，不能锁 ABI 语义

不推荐。

### Option B: 完全改成纯语义字段

优点：

- 对人最友好
- 最适合做 ABI 契约测试

缺点：

- 对语义未知的 preload 槽位不友好
- 容易丢掉现有回退信息

本轮不直接走这么激进。

### Option C: 语义优先，原始回退

优点：

- 人可读
- 对已知 ABI 字段更稳定
- 对未知字段仍保留现有可观测性
- 可以同时覆盖 modeled / cycle / encoded 三条路径

这是推荐方案。

## Next Step

在本设计获批后，再写 implementation plan。实现应优先把格式化职责继续集中在 `wave_launch_trace.h` 对应的 helper，而不是让三条 executor 各自拼 ABI 文本。
