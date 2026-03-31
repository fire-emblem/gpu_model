# Wave Launch ABI Summary Plan

## Goal Description

把 [2026-03-31-wave-launch-abi-summary-design.md](/data/gpu_model/docs/superpowers/specs/2026-03-31-wave-launch-abi-summary-design.md) 落成一份可执行实现计划，使 `WaveLaunch` trace message 从“固定寄存器窗口”提升为“ABI 语义优先、原始寄存器回退”的单行摘要。

本轮目标是提升可观测性，而不是改变 ABI preload 逻辑本身。实现必须保持：

- 仍使用 `TraceEventKind::WaveLaunch`
- 仍输出单行 message
- 保留现有 `block_xyz/dpc/ap/peu/lanes/exec/cmask/smask`
- 在已知 ABI 路径上优先输出 `kernarg_ptr`、`wg_id_x/y/z`、`workitem_id_x/y/z` 等语义字段
- 在未知语义路径上保留当前 `sN=` / `vN=` 回退

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: `WaveLaunch` 保持单条单行摘要，且现有顶层执行/拓扑字段不回归
  - Positive Tests (expected to PASS):
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.EmitsWaveLaunchEventWithInitialWaveStateSummary'` 通过，并继续看到 `lanes=0x40`、`exec=0xffffffffffffffff`、`sgpr={`、`vgpr={`
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='CycleSmokeTest.QueuedBlocksRespectApCapacityAndActivateOverflowLater'` 通过，说明 `WaveLaunch` 事件数量与 cycle launch 统计不变
  - Negative Tests (expected to FAIL):
    - 如果 `WaveLaunch` message 变成多行或移除 `lanes/exec` 顶层字段，`TraceTest.EmitsWaveLaunchEventWithInitialWaveStateSummary` 应失败
    - 如果把 `WaveLaunch` 改成新 event kind 或跳过现有发射点，`CycleSmokeTest` 中的 wave launch 计数断言应失败

- AC-2: raw-GCN fallback ABI 路径的 SGPR 摘要改为语义字段断言，而不是槽位号断言
  - Positive Tests (expected to PASS):
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.LaunchesLlvmMcFallbackAbiObjectInRawGcnPath'` 通过，并在 `WaveLaunch` 中看到 `kernarg_ptr=`、`wg_id_x=`、`wg_id_y=`
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.LaunchesHipDynamicSharedExecutableInRawGcnPath'` 通过，说明新格式没有破坏 dynamic shared raw-GCN 主线
  - Negative Tests (expected to FAIL):
    - 如果实现仍只输出 `s4/s5/s6/s7` 而不输出语义 key，更新后的 fallback ABI trace 回归应失败
    - 如果 `kernarg_ptr` 的值拼接错误或高低 32 位顺序错位，fallback ABI trace 断言应失败

- AC-3: workitem VGPR 样本改为语义字段断言，至少覆盖三维 builtin-id 路径
  - Positive Tests (expected to PASS):
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.LaunchesHipThreeDimensionalBuiltinIdsExecutableInRawGcnPath'` 通过，并在 `WaveLaunch` 中看到 `workitem_id_z[0,1]={0x0,0x1}`
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.EmitsWaveLaunchEventWithInitialWaveStateSummary'` 通过，说明通用路径仍能输出 `vgpr={` 样本
  - Negative Tests (expected to FAIL):
    - 如果实现把三维 builtin-id 仍固定暴露为 `v2[...]`，更新后的 raw-GCN builtin-id trace 回归应失败
    - 如果 lane sample 顺序从 `[0,1]` 改成其它窗口而未同步测试，builtin-id trace 回归应失败

- AC-4: tensor launch trace 与未知 ABI 路径回退行为保持兼容
  - Positive Tests (expected to PASS):
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.LaunchesHipMfmaExecutableInRawGcnPath'` 通过，并继续看到 `tensor={agpr_count=...,accum_offset=...}`
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*'` 全通过，说明 generic/modeled 路径在拿不到 ABI recipe 时仍可回退到稳定窗口
  - Negative Tests (expected to FAIL):
    - 如果新实现覆盖或删除 `tensor={...}`，`HipRuntimeTest.LaunchesHipMfmaExecutableInRawGcnPath` 的 tensor launch trace 断言应失败
    - 如果在无 ABI recipe 路径上强行输出空语义字段而移除 `sgpr={...}` / `vgpr={...}` 原始窗口，`TraceTest.*` 应失败

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

实现新增一个轻量的 ABI 摘要描述层，由共享 formatter 使用；encoded/raw-GCN 路径传入语义映射，functional/cycle 路径在拿不到稳定 ABI recipe 时自动回退到当前原始窗口。相关 runtime/trace 回归同步切到语义字段断言，并补一条最小 helper-level 覆盖或等价 focused trace 回归来锁定 fallback 与 tensor 兼容。

### Lower Bound (Minimum Acceptable Scope)

只改共享 `WaveLaunch` formatter 和 encoded/raw-GCN 调用点，使 fallback ABI 与三维 builtin-id 测试从原始 `sN/vN` 槽位断言切到语义字段断言；functional/cycle 保持原窗口回退；`TraceTest.*` 和受影响 `HipRuntimeTest.*` 通过。

### Allowed Choices

- Can use:
  - 在 [wave_launch_trace.h](/data/gpu_model/include/gpu_model/debug/wave_launch_trace.h) 增加轻量配置结构或辅助参数
  - 在 [encoded_exec_engine.cpp](/data/gpu_model/src/execution/encoded_exec_engine.cpp) 内构建 ABI 语义摘要输入
  - 更新 [trace_test.cpp](/data/gpu_model/tests/runtime/trace_test.cpp) 与 [hip_runtime_test.cpp](/data/gpu_model/tests/runtime/hip_runtime_test.cpp) 的 `WaveLaunch` 断言
  - 保留对未知 ABI 字段的原始 `sN=` / `vN=` 回退
- Cannot use:
  - 新增 `TraceEventKind`
  - 把 `WaveLaunch` 改成多行或完整寄存器 dump
  - 在三个 executor 内分别复制大段 ABI 文本拼接逻辑
  - 在没有稳定来源的情况下臆造语义字段名

> **Note on Deterministic Designs**: 本设计的关键选择已经基本固定，尤其是“单行 message”“语义优先、原始回退”“不新增 event kind”。因此上下边界接近，允许的选择主要集中在 formatter API 形状和测试切面，而不是功能方向。

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

一个可行做法是把当前 formatter 分成两层：

1. 顶层固定字段层
   - 始终输出 `block_xyz/dpc/ap/peu/lanes/exec/cmask/smask`
2. ABI 摘要层
   - 有语义映射时输出 `sgpr={kernarg_ptr=...,wg_id_x=...}` 与 `vgpr={workitem_id_x[...]...}`
   - 无语义映射时继续输出现有 `sgpr={s0=...}` / `vgpr={v0[...]...}` 窗口

可以考虑引入一个轻量描述对象，例如：

```cpp
struct WaveLaunchAbiSummary {
  std::vector<std::pair<std::string, uint64_t>> sgpr_fields;
  std::vector<NamedLaneSample> vgpr_fields;
};
```

然后：

- functional/cycle 调用 `FormatWaveLaunchTraceMessage(wave)` 或传空 summary
- encoded/raw-GCN 调用 `FormatWaveLaunchTraceMessage(wave, summary, scalar_regs, vector_regs, lanes_per_vector)`

这样能把 ABI 逻辑集中在一处，同时保留现有窗口参数用于回退。

### Relevant References

- [wave_launch_trace.h](/data/gpu_model/include/gpu_model/debug/wave_launch_trace.h) - 当前共享 `WaveLaunch` formatter 入口
- [encoded_exec_engine.cpp](/data/gpu_model/src/execution/encoded_exec_engine.cpp) - 已有 fallback ABI 和 descriptor-explicit ABI 槽位顺序
- [functional_exec_engine.cpp](/data/gpu_model/src/execution/functional_exec_engine.cpp) - functional 路径 `WaveLaunch` 调用点
- [cycle_exec_engine.cpp](/data/gpu_model/src/execution/cycle_exec_engine.cpp) - cycle 路径 `WaveLaunch` 调用点
- [trace_test.cpp](/data/gpu_model/tests/runtime/trace_test.cpp) - 通用 `WaveLaunch` trace 回归
- [hip_runtime_test.cpp](/data/gpu_model/tests/runtime/hip_runtime_test.cpp) - raw-GCN fallback ABI、3D builtin-id、tensor trace 断言
- [kernel_metadata.h](/data/gpu_model/include/gpu_model/isa/kernel_metadata.h) - hidden arg / ABI 相关 typed metadata 定义

## Dependencies and Sequence

### Milestones
1. Milestone 1: 固化 `WaveLaunch` formatter 的 ABI 摘要接口
   - Phase A: 盘点当前 formatter、encoded ABI recipe、受影响断言
   - Phase B: 设计“语义字段输入 + 原始窗口回退”的 helper API
2. Milestone 2: 接入 raw-GCN / encoded ABI 语义字段
   - Phase A: fallback ABI 映射 `kernarg_ptr`、`wg_id_x/y`
   - Phase B: explicit descriptor ABI 与 `workitem_id_x/y/z` lane sample 接入
3. Milestone 3: 收口测试与兼容性
   - Phase A: 更新 `TraceTest` 和 `HipRuntimeTest` 的 `WaveLaunch` 断言
   - Phase B: 验证 tensor trace 与 generic fallback 不回归

相对依赖关系：

- formatter API 必须先稳定，再改 executor 调用点
- raw-GCN runtime 断言要等语义字段真正接入后再切换
- tensor/trace compatibility 必须作为最后一轮回归，防止 schema 变更误伤其它 `WaveLaunch` 使用者

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | 盘点当前 `WaveLaunch` formatter、encoded ABI 槽位顺序、runtime trace 断言，确认要从 `sN/vN` 切换到语义字段的最小改动面 | AC-1, AC-2, AC-3 | analyze | - |
| task2 | 在共享 formatter 层引入“语义字段优先、原始窗口回退”的摘要能力，并保持单行 message 与顶层字段不变 | AC-1, AC-4 | coding | task1 |
| task3 | 在 encoded/raw-GCN 路径接入 `kernarg_ptr`、`wg_id_x/y/z`、`workitem_id_x/y/z` 等已知 ABI 语义字段 | AC-2, AC-3 | coding | task2 |
| task4 | 更新 `TraceTest` 与 `HipRuntimeTest` 中受影响的 `WaveLaunch` 回归，把槽位号断言切到语义字段断言，并验证 tensor 兼容 | AC-1, AC-2, AC-3, AC-4 | coding | task3 |
| task5 | 跑 focused trace/runtime 回归，再跑全量 `gpu_model_tests`，确认 `WaveLaunch` schema 细化没有带来回归 | AC-1, AC-2, AC-3, AC-4 | coding | task4 |

## Claude-Codex Deliberation

### Agreements

- `WaveLaunch` 应继续保持单条单行 message，而不是升级成完整寄存器 dump
- 已知 ABI 字段应优先以语义 key 输出，测试也应锁 ABI 含义而不是槽位号
- 对未知语义字段必须保留原始 `sN=` / `vN=` 回退，避免丢失可观测性
- `tensor={agpr_count=...,accum_offset=...}` 兼容性必须作为显式回归保留

### Resolved Disagreements

- 摘要风格：纯语义字段 vs 语义优先 + 原始回退
  - 选择结果：采用“语义优先 + 原始回退”
  - 原因：functional/cycle 路径当前没有完整 ABI recipe；如果强行纯语义化，会让未知 preload 字段失去可观测性，也会把现有 generic trace regressions 打散

### Convergence Status

- Final Status: `converged`

## Pending User Decisions

- None.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

