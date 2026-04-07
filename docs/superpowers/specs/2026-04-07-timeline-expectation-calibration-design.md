# Timeline Expectation Calibration Design

## Context

当前项目已经明确了以下硬约束：

- timeline cycle 的生成必须依赖 execution 已产出的 modeled cycle 事实和 recorder 记录。
- trace / timeline / perfetto / text / json 都只能消费事件，不能参与业务逻辑决策。
- instruction slice 的最小 `4 cycle` 可见区间来自 execution/recorder，不能由 timeline/render 补齐。
- 如果 recorder 没有提供 instruction `cycle range`，timeline 必须留空，不能伪造 slice。

在这个前提下，当前还缺一层“预期 timeline 与真实 timeline 的结构化校准”能力。现有测试大多直接看：

- trace 事件序列
- perfetto/json 文本片段
- 个别 focused ordering 断言

这些测试能发现问题，但还不能稳定表达“某个 case 理论上应该出现怎样的 timeline 语义结构”，也不利于后续做批量校准或输出结构化 diff。

本设计引入一套独立于 serializer 的 timeline expectation 模型，用于把：

- 高层可观察语义预期
- recorder 产出的真实 timeline 事实

放到同一个中间抽象上进行比较。

## Goals

- 定义一套稳定、结构化、与 serializer 解耦的 timeline 校准模型。
- 支持从测试侧声明“预期 timeline 语义”，并与 recorder 导出的真实 timeline 快照做结构化对拍。
- 第一版优先覆盖 timeline 语义校准，不扩展到 text/json serializer 对拍。
- 第一版优先覆盖：
  - waitcnt progress
  - wave switch

## Non-Goals

- 不在 trace/timeline 层新增任何业务推断逻辑。
- 不在 comparator 中做宽松模糊匹配、自动修正、自动推断缺失事件。
- 不在第一版处理 perfetto/json 文本字段逐字对拍。
- 不在第一版把测试 helper 暴露为稳定 public API。

## Design Summary

引入三个公开稳定对象：

- `ExpectedTimeline`
- `ActualTimelineSnapshot`
- `TimelineComparator`

其中：

- `ExpectedTimeline` 表达测试定义的“应观察到什么”
- `ActualTimelineSnapshot` 表达从 recorder 提取出的“实际观察到什么”
- `TimelineComparator` 做精确结构化比较并输出 diff

构造 `ExpectedTimeline` 的 helper 第一版先放在测试侧或内部实现，不纳入稳定 debug API。

## Public API Boundary

公开头文件只新增三组：

- `debug/timeline/expected_timeline.h`
- `debug/timeline/actual_timeline_snapshot.h`
- `debug/timeline/timeline_comparator.h`

第一版不新增 public builder API。测试构造器先放到：

- `tests/test_utils/...`
  或
- `src/debug/timeline/internal/...`

由此保证公共抽象面最小化。

## Data Model

### Lane Identity

所有 expectation / actual snapshot 都使用相同的 lane identity：

- `dpc_id`
- `ap_id`
- `peu_id`
- `slot_id`
- `wave_id`

表示同一条 timeline lane 上的可观察事件归属。

建议结构：

```cpp
struct TimelineLaneKey {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  uint32_t wave_id = 0;
};
```

### Event Identity

为避免“只按名字匹配”带来的歧义，所有 slice/marker 比较都显式携带：

- lane
- pc
- name

建议结构：

```cpp
struct TimelineEventKey {
  TimelineLaneKey lane;
  uint64_t pc = 0;
  std::string name;
};
```

### Expected Timeline

建议结构：

```cpp
struct ExpectedSlice {
  TimelineEventKey key;
  uint64_t begin_cycle = 0;
  uint64_t end_cycle = 0;
};

struct ExpectedMarker {
  TimelineEventKey key;
  uint64_t cycle = 0;
  std::optional<TraceStallReason> stall_reason;
  std::optional<TraceArriveProgressKind> arrive_progress;
};

struct OrderingConstraint {
  TimelineEventKey earlier;
  TimelineEventKey later;
};

struct ExpectedTimeline {
  std::vector<ExpectedSlice> required_slices;
  std::vector<ExpectedMarker> required_markers;
  std::vector<TimelineEventKey> forbidden_slices;
  std::vector<OrderingConstraint> ordering;
};
```

语义：

- `required_slices`
  - 必须出现的 instruction slice
- `required_markers`
  - 必须出现的 marker
- `forbidden_slices`
  - 不允许被渲染成 instruction slice 的事件
- `ordering`
  - 必须满足的前后顺序关系

### Actual Timeline Snapshot

建议结构：

```cpp
struct ActualSlice {
  TimelineEventKey key;
  uint64_t begin_cycle = 0;
  uint64_t end_cycle = 0;
};

struct ActualMarker {
  TimelineEventKey key;
  uint64_t cycle = 0;
  TraceStallReason stall_reason = TraceStallReason::None;
  TraceArriveProgressKind arrive_progress = TraceArriveProgressKind::None;
};

struct ActualTimelineSnapshot {
  std::vector<ActualSlice> slices;
  std::vector<ActualMarker> markers;
};
```

`ActualTimelineSnapshot` 只来自 recorder 事实，不允许 consumer 侧推断。

## Builder Responsibilities

### ActualTimelineBuilder

输入：

- `Recorder`

输出：

- `ActualTimelineSnapshot`

规则：

- 只能消费 recorder
- 只有存在 recorder `cycle range` 的 instruction 才能生成 slice
- marker 只能来自 typed event
- 不补 event
- 不补 duration
- 不做 quantize/fallback

### ExpectedTimeline Construction

第一版不定义 public builder。

测试侧使用最简单的显式构造方式：

- 直接构造 `ExpectedTimeline`
- 或使用测试内部的辅助工厂函数

例如：

- `ExpectSlice(...)`
- `ExpectMarker(...)`
- `ExpectOrder(...)`
- `ExpectNoSlice(...)`

第一版不引入通用 DSL，不引入脚本语言。

## Comparison Rules

第一版 comparator 只做精确比较，不做模糊匹配。

### Required Slice Match

必须精确匹配：

- lane
- pc
- name
- `begin_cycle`
- `end_cycle`

### Required Marker Match

必须精确匹配：

- lane
- pc
- name
- `cycle`

如果 expectation 提供了：

- `stall_reason`
- `arrive_progress`

则这些字段也必须匹配。

### Forbidden Slice Match

如果实际 snapshot 中存在同一：

- lane
- pc
- name

的 slice，则比较失败。

这条规则主要用于：

- `s_waitcnt`
- `barrier`
- `arrive`
- `wave_resume`
- `wave_switch_away`

等只允许作为 marker 暴露的事件。

### Ordering Match

对每个 `OrderingConstraint`：

- 在 actual snapshot 中找到对应的 `earlier`
- 在 actual snapshot 中找到对应的 `later`
- 比较两者在 snapshot 序列中的首次出现位置

要求：

- `earlier` 必须先于 `later`

第一版不做“最近匹配”或 wildcard 规则。

## Failure Reporting

`TimelineComparator` 必须输出可读、结构化的失败信息。

建议至少支持以下类型：

- `missing marker`
- `missing slice`
- `unexpected slice`
- `ordering violation`
- `marker field mismatch`
- `slice range mismatch`

建议输出格式类似：

```text
missing marker: lane=DPC_00/AP_00/PEU_00/WAVE_SLOT_00/WAVE_0000 cycle=44 pc=0x108 name=load_arrive_resume
unexpected slice: lane=DPC_00/AP_00/PEU_00/WAVE_SLOT_00/WAVE_0000 pc=0x108 name=s_waitcnt
ordering violation: load_arrive_resume should appear before wave_resume
slice mismatch: lane=DPC_00/AP_00/PEU_00/WAVE_SLOT_00/WAVE_0000 pc=0x100 expected=[8,12) actual=[8,16)
```

## Initial Test Scope

第一版只覆盖两个试点类目：

### 1. Waitcnt Progress

需要能表达并比较：

- `wave_wait`
- `load_arrive_still_blocked`
- `load_arrive_resume`
- `wave_resume`
- `s_waitcnt` 不允许被渲染成 slice
- 顺序：
  - `wave_wait < load_arrive_still_blocked`
  - `load_arrive_still_blocked < load_arrive_resume`
  - `load_arrive_resume < wave_resume`
  - `wave_resume < next wave_step`

### 2. Wave Switch

需要能表达并比较：

- `wave_resume`
- `wave_switch_away`
- 下一条 `wave_step`
- 顺序：
  - `wave_resume < wave_switch_away`
  - `wave_switch_away < next wave_step`

## Module Dependency Direction

硬约束如下：

- execution/state machine 先产生语义事实
- recorder 记录事实
- actual timeline builder 消费 recorder
- comparator 比较 expected vs actual
- perfetto/json/text 只是独立 serializer

统一依赖方向：

`execution -> recorder -> actual timeline snapshot -> comparator`

`test expectation -> expected timeline -> comparator`

禁止方向：

- `serializer -> comparator`
- `perfetto/json/text -> actual timeline snapshot`
- `timeline consumer -> business logic inference`

## Rollout Plan

第一批实现按以下顺序进行：

1. 定义 `ExpectedTimeline` / `ActualTimelineSnapshot` / `TimelineComparator`
2. 实现从 recorder 到 `ActualTimelineSnapshot` 的构建
3. 在测试侧补最小 helper
4. 为 waitcnt progress 新增 expectation-based 回归
5. 为 wave switch 新增 expectation-based 回归
6. 保留现有 perfetto/json focused 测试，作为 serializer 校验而非主校准手段

## Risks

### Risk 1: 过早做成通用 DSL

后果：

- 抽象过重
- 第一版 implementation 成本高

规避：

- 第一版只允许显式结构体构造和很薄的测试 helper

### Risk 2: 重新在 comparator 中引入业务推断

后果：

- 违反 trace/consumer 不参与业务逻辑的硬约束

规避：

- comparator 只比较结构化事实
- actual snapshot 只消费 recorder

### Risk 3: 用 renderer 产物反向证明 renderer 自己

后果：

- 被测对象与比较对象耦合
- 测试价值下降

规避：

- `ActualTimelineSnapshot` 从 recorder 独立构建
- 不直接使用 perfetto/json 文本做主校准

## Acceptance Criteria

- 存在独立于 serializer 的 `ExpectedTimeline` / `ActualTimelineSnapshot` / `TimelineComparator`
- 第一版不暴露 public builder API
- `ActualTimelineSnapshot` 仅消费 recorder 事实
- waitcnt progress expectation-based 测试可稳定通过
- wave switch expectation-based 测试可稳定通过
- 现有 perfetto/json focused 测试继续保留，但不承担 timeline 语义主校准职责
