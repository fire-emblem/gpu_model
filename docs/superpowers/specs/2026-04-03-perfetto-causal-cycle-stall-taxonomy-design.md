# Perfetto Causal Cycle Stall Taxonomy Design

## 背景

当前仓库已经具备：

- `CycleExecEngine`
- `TraceEventKind::Stall`
- `CycleTimelineRenderer::RenderGoogleTrace(...)`
- `timeline.perfetto.json` 导出
- `TraceTest`、`CycleTimelineTest`、`AsyncMemoryCycleTest`、`SharedBarrierCycleTest` 等回归

当前 `cycle` 路径已经开始具备可用的 issue / stall / arrive / commit 时间线，但下一步的重点不再只是：

- 能导出 Perfetto JSON
- 事件顺序大体合理

而是：

1. 在 Perfetto 上，`cycle` 每一拍为什么没有 issue 必须可解释
2. `waitcnt` / `barrier` / front-end competition 的阻塞原因必须稳定可区分
3. focused regression 必须能锁定这些语义，而不是依赖临时自然语言 message

这轮工作的目标不是把 cycle model 直接做成真实硬件拟合器，而是先把 `Perfetto` 观察面提升为“因果可解释”的层级。

## 本轮目标

把 `cycle` 路径的 `Stall` 事件收敛成：

- Perfetto 可稳定观测
- 测试可程序化断言
- 后续 `resume` / issue competition 可观测化能复用

更具体地说，本轮只解决一个问题：

> 当某个 cycle 没有形成预期 issue，Perfetto 必须能回答“为什么”。

## 非目标

本轮明确不做：

- 新一轮 runtime / loader / ELF 能力扩展
- 为了 trace 去扩更多 ISA family 支持
- 第一批次就引入完整 `Resume` event 家族
- 第一批次就导出完整 issue candidate dump
- 第一批次就统一 `functional / encoded / cycle` 三条 trace schema
- 第一批次就让总周期更接近真实硬件
- 第一批次就做完整硬件校准或 microbenchmark 拟合

## 设计目标

### 1. `Stall` message 必须先稳定，再谈更细可观测性

后续如果要继续补：

- `Resume`
- issue competition loser visibility
- 程序级 stall 统计
- 更细 pipe / resource track

都必须建立在“stall 原因名称和基本结构稳定”的前提上。

因此，这一轮优先做 taxonomy，而不是先扩事件种类。

### 2. Perfetto 重点是因果可解释，不是内部状态全量 dump

本轮要保证：

- 为什么没发
- 这一类阻塞与另一类阻塞能区分
- 同一类阻塞在不同 case 中用相同名字表达

本轮不要求：

- 记录所有候选 wave
- 记录所有内部容器状态
- 在一个事件里输出过多调试细节

### 3. 输出必须适合长期测试消费

`Stall` 的表达不能依赖自然语言随手拼接。

必须满足：

- 稳定 key
- 稳定枚举值
- 后续可追加字段
- 老测试可平滑迁移到 `reason=` 断言

## 方案对比

### 方案 A：继续沿用当前自然语言 stall message

优点：

- 改动最小

缺点：

- Perfetto 上能看见“有 stall”，但很难稳定解释
- 文本容易漂移
- focused regression 脆弱

### 方案 B：直接新增大量新的 cycle trace event kind

优点：

- 理论上表达力最强

缺点：

- 第一批 scope 过大
- 很容易把 trace schema 一次性做碎
- 还没有稳定 taxonomy 时，新增 event kind 只是把混乱扩散

### 方案 C：保留 `Stall` event，先统一 `reason=` taxonomy

优点：

- 改动面最小
- 对 Perfetto 可解释性提升最大
- 为下一轮 `Resume` / competition observability 铺路
- 不需要先决定更大的 trace 协议

缺点：

- 第一轮仍然不能完整展示全部 candidate 级细节

### 结论

采用方案 C。

## Stall Taxonomy

本轮先锁定的是 **`reason=` schema**，以及一组已经在 `cycle` 路径中稳定存在或已明确接入的首批 reason。

当前批次显式覆盖或已接入的 reason 包括：

- `waitcnt_global`
- `waitcnt_shared`
- `waitcnt_private`
- `waitcnt_scalar_buffer`
- `barrier_slot_unavailable`
- `warp_switch`

这些值的设计原则是：

- 优先锁定已经在 `cycle` 主路径里稳定存在、且当前 focused tests 能覆盖的 reason
- 先把 `reason=` 包装、Perfetto 命名与 focused regression 稳定下来
- 把其余阻塞语义的规范化留到下一批，而不是在本轮一次性承诺完成

下一批应继续把其余阻塞原因规范化并收敛到同一 taxonomy，例如：

- `no_ready_wave`
- `dependency`
- `barrier_wait`
- `issue_group_conflict`
- `same_wave_conflict`
- `front_end_window`

当前代码里仍可能存在与这些目标名不同的历史 reason token；本轮不要求一次性全部改名完成。

### 已接入 reason 含义

#### `waitcnt_global`

wave 因 `global` memory domain 的 `waitcnt` 条件未满足而阻塞。

#### `waitcnt_shared`

wave 因 `shared` memory domain 的 `waitcnt` 条件未满足而阻塞。

#### `waitcnt_private`

wave 因 `private` memory domain 的 `waitcnt` 条件未满足而阻塞。

#### `waitcnt_scalar_buffer`

wave 因 `scalar-buffer` memory domain 的 `waitcnt` 条件未满足而阻塞。

#### `barrier_slot_unavailable`

当前 barrier generation 需要资源槽位，但 `barrier_slots_per_ap` 相关资源暂不可用。

#### `warp_switch`

当前 cycle 的 issue 需要支付 wave 切换代价，因此 trace 会显式记录一次 context-switch 类 stall。

## Message 格式

### 最小格式

每个 `Stall` event message 至少必须满足：

```text
reason=<taxonomy>
```

例如：

```text
reason=waitcnt_global
reason=dependency
reason=issue_group_conflict
```

### 可选扩展格式

后续允许追加稳定字段，例如：

```text
reason=issue_group_conflict issue_group=0
reason=waitcnt_global wave=3
reason=dependency wave=1 reg=v4
```

### 规则

必须满足：

1. `reason=` 必须存在
2. `reason` 必须是固定 taxonomy 中的值
3. `reason` 必须排在 message 的第一位
4. 追加字段只能增强解释力，不能改变 `reason=` 的基本语义

## 事件模型边界

本轮继续复用现有核心事件家族：

- `Issue`
- `Stall`
- `Arrive`
- `Commit`
- `WaveStats`

这一轮不强制新增新的 event kind。

原因是本轮目标是：

- 先让 `Stall` 语义稳定
- 再决定下一轮是否需要显式 `Resume`

这样可以避免过早扩大 trace 协议面。

## 因果链要求

即使本轮只改 `Stall` taxonomy，设计上也必须服务于以下因果链：

### 1. memory wait 链

```text
Issue(load) -> Stall(waitcnt_*) -> Arrive -> later issue
```

本轮至少要保证：

- `Stall` 上能明确看出是哪一个 `waitcnt` domain

### 2. barrier wait 链

```text
Issue(barrier) -> Stall(barrier_wait) -> later release/resume -> later issue
```

本轮至少要保证：

- `barrier_wait` 不会退化成模糊 idle

### 3. issue competition 链

```text
ready candidates -> Issue(winner) + Stall(issue_group_conflict or same_wave_conflict)
```

本轮至少要保证：

- branch/special 共享组等前端冲突在 trace 上可区分

## 文件落点

本轮实现主要应集中在：

- [src/execution/cycle_exec_engine.cpp](/data/gpu_model/src/execution/cycle_exec_engine.cpp)
  - `Stall` 原因的统一出口
- [src/debug/cycle_timeline.cpp](/data/gpu_model/src/debug/cycle_timeline.cpp)
  - Perfetto 导出时对新 taxonomy 的稳定呈现
- [tests/runtime/trace_test.cpp](/data/gpu_model/tests/runtime/trace_test.cpp)
  - trace schema 和 `reason=` 断言
- [tests/runtime/cycle_timeline_test.cpp](/data/gpu_model/tests/runtime/cycle_timeline_test.cpp)
  - Perfetto 顺序和可见性回归
- [tests/cycle/async_memory_cycle_test.cpp](/data/gpu_model/tests/cycle/async_memory_cycle_test.cpp)
  - `waitcnt_*` stall 回归
- [tests/cycle/shared_barrier_cycle_test.cpp](/data/gpu_model/tests/cycle/shared_barrier_cycle_test.cpp)
  - `barrier_wait` stall 回归
- [tests/cycle/cycle_smoke_test.cpp](/data/gpu_model/tests/cycle/cycle_smoke_test.cpp)
  - 前端 issue / bundle 相关 sanity 回归

## 验收标准

### AC-1：`Stall` 事件具备稳定 `reason=` schema

必须满足：

- `cycle` 路径发出的 `Stall` message 一律包含 `reason=`
- `reason` 值来自当前批次已接入的首批 reason 集合，或后续保留的历史 token
- 旧的模糊自然语言主 message 不再作为 cycle-path focused tests 的主要依赖面

### AC-2：当前已稳定出现的 `waitcnt` domain 阻塞在 Perfetto 上可区分

必须满足：

- 至少 `waitcnt_global` 在 raw trace 与 Perfetto 上有 focused regression 锁定
- 对其余 `waitcnt` domain，本轮优先保证 `reason=` schema 与现有行为兼容，不强行要求每个 domain 都在当前 focused ring 中稳定出现可见 stall

### AC-3：barrier-heavy 路径上的 stall 命名在 raw trace 与 Perfetto 上可稳定观察

必须满足：

- barrier-heavy 路径新增的 focused regression 能锁定至少一种稳定出现的 `reason=` stall 信号
- 当前批次不强行要求 `barrier_wait` 自身已经在 cycle active-window 语义下稳定发出 `Stall` 事件

### AC-4：Perfetto 导出与 focused regression 可稳定消费 `reason=` schema

必须满足：

- `TraceTest` 可以直接断言 `reason=...`
- `CycleTimelineTest` 可以验证关键 stall 类别仍被正确导出
- 相关 cycle focused tests 不依赖易漂移的自然语言句子

## 建议验证环

focused verification 建议至少覆盖：

- `./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*'`
- `./build-ninja/tests/gpu_model_tests --gtest_filter='CycleTimelineTest.*'`
- `./build-ninja/tests/gpu_model_tests --gtest_filter='AsyncMemoryCycleTest.*'`
- `./build-ninja/tests/gpu_model_tests --gtest_filter='SharedBarrierCycleTest.*'`
- `./build-ninja/tests/gpu_model_tests --gtest_filter='CycleSmokeTest.*'`

最终仍应补一次：

- `./build-ninja/tests/gpu_model_tests`

## 后续工作接口

本轮完成后，下一批可以在不推翻 taxonomy 的前提下继续推进：

1. 显式 `Resume` 可观测化
2. issue competition 落败原因的更细粒度可见化
3. stall taxonomy 聚合到 program-level 统计
4. 更细的 resource / pipe track

这些都应建立在本轮的 `reason=` 语义不回退的前提上。
