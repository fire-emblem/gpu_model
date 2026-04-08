# Async Memory Arrive Flow Design

## Context

当前工程里的 `timeline.perfetto.json` 使用 Chrome trace JSON 导出。

目前已经稳定存在的 typed facts 包括：

- issue 侧：
  - `TraceEventKind::MemoryAccess`
  - `message = load_issue` / `store_issue`
- completion 侧：
  - `TraceEventKind::Arrive`
  - `arrive_kind = load / store / shared / private / scalar_buffer`
- wait/resume 侧：
  - `TraceEventKind::WaveArrive`
  - `TraceEventKind::WaveResume`

但当前 Google trace / Perfetto JSON 导出只包含：

- instruction slice: `ph:"X"`
- marker: `ph:"i"`

因此 timeline 上看得到 issue marker 和 arrive marker，却没有“这两个 marker 属于同一次 async memory op”的可视化关联线。

同时仓库已经明确约束：

- trace / timeline 只能消费 typed fact
- 不能在 consumer 侧反推业务关系
- `cycle` 仍然只是 modeled cycle

所以“marker 关联线”必须来自 producer / recorder 已知的关系，而不能由 renderer 根据邻近事件或 `message` 猜配对。

## Goal

为 `timeline.perfetto.json` 增加 async memory issue/completion 的 flow line，第一批只覆盖：

- `TraceEventKind::MemoryAccess(load_issue/store_issue)`
- `TraceEventKind::Arrive`

也就是：

- `MemoryAccess(load_issue) -> Arrive(load)`
- `MemoryAccess(store_issue) -> Arrive(store)`

第一批不处理：

- `WaveArrive`
- `WaveResume`
- `IssueSelect -> WaveStep`
- 其他 marker 之间的 flow

## Non-Goals

- 不在 timeline renderer 中根据 `message`、相邻事件、同 cycle 排序等启发式猜配对
- 不把 `WaveArrive` 混进同一条 flow
- 不新增另一套 trace 主格式
- 不把这个 flow 设计成 wall-clock 或真实硬件依赖关系

## Design Summary

第一批 flow 采用 producer-owned flow id。

也就是：

1. 当 execution 产生一次 async memory issue 时，同时生成一个稳定 flow id
2. 同一次 memory op 的完成 arrive 复用这同一个 flow id
3. recorder 原样保留该 flow metadata
4. Google trace / Perfetto JSON 导出时，把它转成 Chrome trace flow event：
   - start: `ph:"s"`
   - finish: `ph:"f"`

这样关联线的语义来源仍然在 execution，而不是在 consumer。

## Why This Approach

### Option A: producer-owned flow id

优点：

- 完全符合“trace 只消费事实”的仓库约束
- modeled cycle 和 encoded cycle 可以共享同一个抽象
- 后续扩展到 `WaveArrive` / `WaveResume` / `IssueSelect` 时不会推翻第一版

缺点：

- 需要扩 `TraceEvent` / recorder / timeline data

### Option B: recorder-side pairing

让 recorder 按 wave/domain/顺序去配 `MemoryAccess` 与 `Arrive`。

优点：

- 改动面较小

缺点：

- recorder 开始承担业务推断
- 容易在多 outstanding memory ops 下配错

### Option C: renderer-side pairing

让 `cycle_timeline_google_trace.cpp` 根据 marker 邻近关系去补 flow。

优点：

- 实现最快

缺点：

- 最脆弱
- 直接违反仓库对 trace consumer 的约束

本设计采用 Option A。

## Scope Boundary

### Included

- `TraceEvent` 增加可选 flow metadata
- modeled cycle async memory path 产生 `MemoryAccess -> Arrive` flow
- encoded/program-object cycle async memory path 产生相同 flow
- recorder 保留 flow metadata
- `timeline.perfetto.json` 导出 flow events
- focused tests

### Excluded

- `WaveArrive` / `WaveResume` flow
- functional `st/mt` 路径 flow
- text/json trace 展示 flow
- native Perfetto proto exporter 单独建模 flow descriptor

## Data Model

建议在 `TraceEvent` 上增加最小字段：

```cpp
enum class TraceFlowPhase {
  None,
  Start,
  Finish,
};

struct TraceEvent {
  ...
  uint64_t flow_id = 0;
  TraceFlowPhase flow_phase = TraceFlowPhase::None;
};
```

约束：

- `flow_id == 0` 且 `flow_phase == None` 表示无 flow
- 同一条 async memory op 的 issue / arrive 必须共享同一 `flow_id`
- 第一版只允许：
  - `MemoryAccess` 标成 `Start`
  - `Arrive` 标成 `Finish`

不额外引入 flow category 结构；直接复用 event 自身的：

- `presentation_name`
- `canonical_name`
- `category`

并在导出时使用：

- `cat = "flow/async_memory"`
- `name = "async_memory"`

## Execution Semantics

### Modeled cycle path

在 `CycleExecEngine` 中：

- 产生 `MemoryAccess` 时分配一个 flow id
- 把 flow id 捕获进后续 async completion lambda
- completion 时发出的 `Arrive` 复用该 flow id

只对真正异步完成的 memory op 生效：

- global
- shared
- private
- scalar buffer

同步完成或没有 arrive 的路径不产生 flow。

### Encoded / program-object cycle path

在 `ProgramObjectExecEngine::RunCycle()` 中做同样的事：

- issue 时分配 flow id
- `ready_cycle` 完成时发出的 `Arrive` 带同一 flow id

### ID allocation

flow id 只要求在单次 trace 导出内唯一。

建议：

- 在每个 execution engine run 内维护单调递增 `next_flow_id`
- 从 `1` 开始分配

不要求跨 launch 全局唯一。

## Recorder Responsibilities

recorder 不生成 flow，只透传 flow metadata。

要求：

- `events_` 原样保留 flow 字段
- `RecorderEntry` / `RecorderProgramEvent` 导出的 canonical/export fields 能保留 flow metadata
- recorder 不做二次配对

## Timeline Export Responsibilities

在 `cycle_timeline_google_trace.cpp` 中：

- 继续保留现有 `X` slice 和 `i` marker
- 额外遍历 recorder/timeline data 中有 flow 的 marker
- 对 `Start` 输出：

```json
{"name":"async_memory","cat":"flow/async_memory","ph":"s","id":"0x1",...}
```

- 对 `Finish` 输出：

```json
{"name":"async_memory","cat":"flow/async_memory","ph":"f","id":"0x1",...}
```

同一 `id` 的 `s/f` 由 Chrome trace / Perfetto UI 画关联线。

第一版 flow event 的轨道归属直接使用对应 marker 的 row：

- issue start 落在 issue wave slot row
- arrive finish 落在 arrive wave slot row

## Test Plan

第一批只加 focused tests：

1. `TraceTest`
   - modeled cycle async load:
     - `MemoryAccess(load_issue)` 与 `Arrive(load)` 共享同一 flow id
   - modeled cycle async store:
     - `MemoryAccess(store_issue)` 与 `Arrive(store)` 共享同一 flow id
   - encoded cycle async load:
     - 同样共享同一 flow id

2. `CycleTimelineTest`
   - `timeline.perfetto.json` 中出现 `ph:"s"` / `ph:"f"`
   - start / finish 的 `id` 一致
   - 不要求第一版验证 UI 渲染，只验证 JSON 结构

3. Negative checks
   - 无 async arrive 的同步路径不应伪造 flow
   - `WaveArrive` 暂时不带这条 flow

## Risks

### Risk 1: flow id 只靠 wave/domain 可能不唯一

如果同一 wave 有多个 outstanding op，仅靠 `(wave_id, domain)` 会冲突。

规避：

- 第一版直接分配显式单调 flow id
- 不使用推导式 key

### Risk 2: 把 `WaveArrive` 和 `Arrive` 混成一条线

这会把“memory 完成”与“wait 条件满足”两层语义混在一起。

规避：

- 第一版只连 `MemoryAccess -> Arrive`

### Risk 3: renderer 偷做配对

这会破坏当前架构边界。

规避：

- 只有携带 flow metadata 的事件才能生成 flow line
- renderer 不自行猜配对

## Acceptance Criteria

- modeled cycle async load/store 的 `MemoryAccess -> Arrive` 具备稳定 flow id
- encoded cycle async load 的 `MemoryAccess -> Arrive` 具备稳定 flow id
- recorder 不做推断，只透传 flow metadata
- `timeline.perfetto.json` 能导出成对的 `ph:"s"` / `ph:"f"` flow 事件
- 没有 async arrive 的路径不伪造 flow
- `WaveArrive` 第一版不参与这条 flow
