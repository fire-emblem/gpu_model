# Cycle Timeline Truth Table

## 目标

本表用于定义 `cycle timeline / Perfetto` 当前阶段允许表达的事件事实，明确：

1. 事件来自哪个执行阶段
2. 使用哪个 cycle
3. 在 timeline 中是 `slice`、`marker` 还是 `runtime_event`
4. 是否参与 `InstructionIssue -> Commit` 配对
5. canonical name / category / args 的稳定要求

本表只描述消费层事实，不定义新的业务逻辑。

## 全局约束

1. timeline 只消费 `TraceEvent -> Recorder -> TimelineData`
2. `cycle` 永远是 modeled cycle，不是宿主 wall-clock
3. 只有存在真实 `Commit` 的可执行指令才允许生成 instruction slice
4. marker 只用于表达不可归入 instruction slice 的离散事件
5. 不允许渲染器根据缺失信息补造 `issue / commit / arrive / stall`

## 真值表

| Event / Entry | 事件来源 | cycle 来源 | Timeline 表达 | 参与 slice 配对 | 当前 canonical name / category 要求 | 备注 |
|---|---|---:|---|---|---|---|
| `InstructionIssue` (`TraceEventKind::WaveStep`) | engine 发出真实 issue 事件；recorder 归类为 `RecorderEntryKind::InstructionIssue` | `event.cycle` | 仅作为 slice 起点，不单独画 marker | `是`，必须等待后续 `Commit` | 指令名来自 `display_name`，缺失时回退 `compatibility_message` 的 `op=` | 默认 `begin_cycle=event.cycle`，`end_cycle` 先按最小 `4 cycle` quantize 预填 |
| `Commit` (`TraceEventKind::Commit`) | engine 发出执行完成/提交事件；recorder 归类为 `RecorderEntryKind::Commit` | `event.cycle` | 不单独画 slice；用于闭合前一个 issue 的 slice | `是`，与最早未闭合 issue 配对 | canonical name 保持 `commit`，但 timeline 主要消费其 cycle，不单独显示为指令 slice | 若没有未闭合 issue，当前实现直接忽略，不推断补造 slice |
| `Arrive` (`TraceEventKind::Arrive`) | 异步 memory 或 barrier 到达事件 | `event.cycle`，必须来自真实到达时点 | `marker` | `否` | canonical name 依 `arrive_kind` / `arrive_progress`，如 `load_arrive`, `load_arrive_resume` | 不能把 `arrive` 误画成 instruction slice；`still_blocked / resume` 应保留在名字和 args 中 |
| `Stall` (`TraceEventKind::Stall`) | engine 观测到 wave 当前不可 issue 的原因 | `event.cycle`，表示阻塞被记录的 modeled cycle | `marker` | `否` | canonical name / category 必须稳定，如 `stall_waitcnt_global` / `stall/waitcnt_global` | stall 不是某条指令的延长区间，不得转成 slice |
| `Barrier` (`TraceEventKind::Barrier`) | barrier arrive/release 生命周期事件 | `event.cycle` | `marker` | `否` | barrier arrive/release 应区分 canonical name | `barrier_arrive` 与 `barrier_release` 不能被合并成普通 arrive/stall |
| `WaveLaunch` | wave 进入可观察生命周期 | `event.cycle` | `marker` | `否` | `wave_launch` | 生命周期 marker，不参与 instruction pairing |
| `WaveGenerate` | wave generation/front-end 事件 | `event.cycle` | `marker` | `否` | `wave_generate` | 只表示 front-end 事实 |
| `WaveDispatch` | wave dispatch/front-end 事件 | `event.cycle` | `marker` | `否` | `wave_dispatch` | 只表示 front-end 事实 |
| `SlotBind` | slot 与 wave 绑定事实 | `event.cycle` | `marker` | `否` | `slot_bind` | 用于 resident/logical slot 观察 |
| `IssueSelect` | scheduler 选择某 wave 的事实 | `event.cycle` | `marker` | `否` | `issue_select` | 只能表示 `selected`，绝不等价于真实 `issue` |
| `WaveExit` | wave 生命周期退出 | `event.cycle` | `marker` | `否` | `wave_exit` | 生命周期 marker，不得与 commit 互相替代 |
| `Launch / BlockPlaced / BlockAdmit / BlockLaunch / BlockActivate / BlockRetire` | runtime / block-level 程序事件 | `event.cycle` | `runtime_event`，不挂在 wave slot 上 | `否` | 维持 runtime/block canonical name | 这类事件进入 `TimelineData.runtime_events`，不是 wave-entry marker |

## 当前 renderer 规则

### 1. Slice 规则

当前 [cycle_timeline.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline.cpp) 的 slice 规则是：

1. 遇到 `RecorderEntryKind::InstructionIssue`，推入 open queue
2. 遇到 `RecorderEntryKind::Commit`，弹出最早未闭合 issue
3. 若该 issue 的 op 为 `s_waitcnt`，不生成 slice
4. 其他情况生成一个 `Segment`

对应字段：

- `issue_cycle = issue.begin_cycle`
- `commit_cycle = commit.event.cycle`
- `render_duration_cycles = issue.has_cycle_range ? issue.end_cycle - issue.begin_cycle : quantized(0)`

### 2. Marker 规则

当前 renderer 会把以下 entry 画成 marker：

- `Arrive`
- `Barrier`
- `WaveExit`
- `Stall`
- `WaveLaunch`
- `WaveGenerate`
- `WaveDispatch`
- `SlotBind`
- `IssueSelect`

当前默认 symbol：

- `R` = arrive
- `B` / `|` = barrier arrive / release
- `X` = wave exit
- `S` = stall
- `L` = wave launch
- `G` = wave generate
- `D` = wave dispatch
- `P` = slot bind
- `I` = issue select

### 3. 明确不允许的混淆

当前阶段必须避免以下错误表达：

1. 把 `ready` wave 提前画成 issue slice
2. 把 `IssueSelect` marker 画成真实执行开始
3. 把 `arrive` 当成 commit 或 resume issue
4. 把 `stall` 当成 instruction 持续区间
5. 因缺少 `Commit` 而自行补全 instruction slice

## 第一批需要据此校准的测试

1. [tests/runtime/cycle_timeline_test.cpp](/data/gpu_model/tests/runtime/cycle_timeline_test.cpp)
   - `PerfettoDumpPreservesCycleIssueAndCommitOrdering`
   - `PerfettoDumpPreservesBarrierKernelStallTaxonomy`

2. [tests/runtime/trace_test.cpp](/data/gpu_model/tests/runtime/trace_test.cpp)
   - encoded/functional waitcnt timeline 相关用例
   - instruction slice duration 相关用例
   - same-PEU switch-away / slot-centric 相关用例

3. representative examples
   - waitcnt-heavy
   - barrier-heavy

## 当前结论

本真值表对应当前主线设计：

- `TraceEvent` 和 `RecorderEntry` 是 timeline 的唯一事实来源
- `issue / commit / arrive / stall / barrier / lifecycle` 必须在消费层严格分层
- cycle timeline 的校准工作应优先修正表达错误，而不是修改执行模型本身
