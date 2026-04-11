# Cycle Issue Design Gap Analysis

本文用于落盘当前 `cycle model` 在“wave 指令 issue 选择”上的目标、现状、差距和建议调整方向。

本文不是新的架构总设计，也不是实施计划；它只回答下面几个问题：

- 当前 `cycle issue` 设计目标到底是什么
- 当前代码框架已经做到什么程度
- 还差哪些关键语义没有建模进去
- 如果要继续改，应该优先改哪些点，改成什么形状
- 这些差距在真实例子里会怎样体现

## 1. 适用范围

本文主要约束和分析：

- [`src/execution/cycle_exec_engine.cpp`](../src/execution/cycle_exec_engine.cpp)
- [`src/execution/internal/issue_scheduler.cpp`](../src/execution/internal/issue_scheduler.cpp)
- [`src/gpu_model/execution/internal/wave_state.h`](../src/gpu_model/execution/internal/wave_state.h)

本文会参考但不直接要求与之完全一致：

- [`src/execution/program_object_exec_engine.cpp`](../src/execution/program_object_exec_engine.cpp)
- [`src/execution/functional_exec_engine.cpp`](../src/execution/functional_exec_engine.cpp)
- [`docs/cycle-issue-eligibility-policy.md`](./cycle-issue-eligibility-policy.md)
- [`docs/my_design.md`](./my_design.md)
- `docs/navisim_pact_2022.pdf`
- `docs/HPCA_2021_NVArchSim.pdf`

## 2. 当前设计目标

结合当前主文档、仓库约束和参考资料，`cycle model` 的 issue 选择目标应固定为下面几条。

### 2.1 语义边界必须分层

必须明确区分：

1. `eligible`
2. `selected`
3. `issue`
4. `commit`
5. `arrive / resume`

这几个状态边不能混成一个时间点。

当前仓库已经明确要求：

- `eligible != selected`
- `selected != issue`
- `arrive_resume` 只表示 wave 再次具备可调度资格，不表示该 cycle 必定 issue
- `trace` 只能消费这些事实，不能反向替 execution 补语义

### 2.2 issue 选择必须以 `global_cycle` 为准

仓库硬约束已经固定：

- `global_cycle` 是唯一调度定位时间
- wave 自身的累计 cycle 不能反推全局时间
- 是否允许 issue，要看当前 `global_cycle` 是否满足该 wave 的最早可发时间

因此，cycle issue 选择目标不只是“当前是不是 Active + Runnable”，还应包含：

- `next_issue_earliest_global_cycle`
- front-end 是否真的可见
- wait/barrier/branch 等业务阻塞是否已解除
- 当前选择的资源约束是否允许

### 2.3 policy 只做“在 eligible 集合中选择”

从 [`docs/cycle-issue-eligibility-policy.md`](./cycle-issue-eligibility-policy.md) 和公开参考抽象看，policy 的职责应限制为：

- `RoundRobin`
- `OldestFirst`
- bundle 内的 type/group 约束仲裁

policy 不应承担：

- 恢复 waiting wave
- 计算 resume 语义
- 推断 switch penalty
- 代替 engine 维护 wave 的下一次可发时间

### 2.4 当前目标不是直接做“完整硬件级 CU scheduler”

参考材料能支持更细的 fetch/issue buffer 模型，但当前项目并没有要求这一轮直接做到：

- per-wave instruction buffer
- fetch arbiter
- 多级 dispatch queue
- 完整 scoreboard 依赖图

当前更合适的目标是：

- 保持 `eligible -> selected -> issue -> commit` 的边界清晰
- 让 `RoundRobin / OldestFirst` 在真实 eligible 集合上工作
- 把最关键的 issue 时序约束显式建模出来
- 不让 `trace` 或 timeline 去猜

## 3. 当前代码框架已经做到的部分

### 3.1 已有统一的 scheduler policy 抽象

当前仓库已经有：

- [`src/gpu_model/execution/internal/issue_model.h`](../src/gpu_model/execution/internal/issue_model.h)
- [`src/execution/internal/issue_scheduler.cpp`](../src/execution/internal/issue_scheduler.cpp)

已经支持：

- `EligibleWaveSelectionPolicy::RoundRobin`
- `EligibleWaveSelectionPolicy::OldestFirst`
- `type_limits`
- `group_limits`
- 同 bundle 同一 `wave_id` 最多 issue 一次

这部分已经符合“先建候选，再按 policy 选择”的目标。

### 3.2 cycle 路径已经有 resident slot / active window / issue bundle

[`src/execution/cycle_exec_engine.cpp`](../src/execution/cycle_exec_engine.cpp) 已经具备：

- resident slot
- standby -> active promote
- `BuildResidentIssueCandidates(...)`
- `IssueScheduler::SelectIssueBundle(...)`
- `IssueSelect` marker
- `WaveStep` / `Commit` / `Arrive` / `WaveResume` / `WaveWait` / `WaveSwitchAway`

这说明当前框架并不是完全没有 issue 模型，而是已经有一个可工作的第一版。

### 3.3 `program_object` / `functional` 路径已经有更成熟的 per-wave issue time 状态

[`src/execution/program_object_exec_engine.cpp`](../src/execution/program_object_exec_engine.cpp) 和
[`src/execution/functional_exec_engine.cpp`](../src/execution/functional_exec_engine.cpp) 都已经有：

- `last_issue_cycle`
- `next_issue_cycle`
- pending memory op 的 `ready_cycle`
- wait 恢复后更新 `next_issue_cycle`

也就是说，仓库里已经存在“wave 自身最早下一次可 issue 时间”的状态契约，只是 `cycle_exec_engine` 还没有跟进到同等粒度。

## 4. 与 GPGPU-Sim 的对照

这里的目的不是把当前工程改造成 CUDA 风格的 GPGPU-Sim 克隆，而是借它验证“当前 issue 选择目标里哪些抽象是应该显式存在的”。

主要参考：

- [`third_party/gpgpu-sim_distribution/src/gpgpu-sim/shader.h`](../third_party/gpgpu-sim_distribution/src/gpgpu-sim/shader.h)
- [`third_party/gpgpu-sim_distribution/src/gpgpu-sim/shader.cc`](../third_party/gpgpu-sim_distribution/src/gpgpu-sim/shader.cc)

### 4.1 GPGPU-Sim 明确把“排序”和“真正发射”拆开

在 GPGPU-Sim 里：

- `order_warps()` 负责生成下一轮优先级顺序
- `scheduler_unit::cycle()` 负责逐个 warp 检查能否真正发射
- 真正发射前还要看：
  - `waiting()`
  - `ibuffer_empty()`
  - scoreboard hazard
  - 各执行单元 pipe 是否有空位
  - dual-issue 是否满足不同执行单元约束

这说明它明确区分了：

- 候选顺序
- readiness / hazard 检查
- 执行单元资源检查
- 真正 issue

而不是把这些事压成一个布尔 `ready`。

### 4.2 GPGPU-Sim 的 oldest 语义是动态 age，不是静态 warp 编号

GPGPU-Sim 在 [`shader.cc`](../third_party/gpgpu-sim_distribution/src/gpgpu-sim/shader.cc) 里使用：

- `sort_warps_by_oldest_dynamic_id(...)`

它排序的核心不是静态 `warp_id`，而是 warp 的动态 age。

这和当前仓库的设计目标是一致的：

- `OldestFirst` 应该表达谁更早进入当前可服务集合
- 不应退化成按静态 `WaveTag` 排序

### 4.3 GPGPU-Sim 明确把 issue 和执行单元资源解耦

在 [`scheduler_unit::cycle()`](../third_party/gpgpu-sim_distribution/src/gpgpu-sim/shader.cc) 里，warp 即使通过了排序和 scoreboard，也仍然可能因为：

- `m_mem_out->has_free(...)`
- `m_sp_out->has_free(...)`
- `m_int_out->has_free(...)`
- `m_dp_out->has_free(...)`
- `m_sfu_out->has_free(...)`

而不能发。

这意味着它显式保留了：

- “warp 当前可被考虑”
- “某类执行资源当前不可 issue”

之间的区别。

对当前仓库而言，这对应的启发是：

- `eligible` 不等于“所有后续 issue 资源都一定已满足”
- 后续至少要把 `next_issue_cycle`、switch penalty、必要时 pipe readiness 拆出来

### 4.4 GPGPU-Sim 还有更丰富的 scheduler family，但当前工程不必照搬

GPGPU-Sim 里不仅有：

- `lrr_scheduler`
- `oldest_scheduler`

还显式提供：

- `rrr_scheduler`
- `gto_scheduler`
- `two_level_active_scheduler`
- `swl_scheduler`

以及一组相关参数：

- `gpgpu_num_sched_per_core`
- `gpgpu_max_insn_issue_per_warp`
- `gpgpu_dual_issue_diff_exec_units`

这说明 GPGPU-Sim 把下面几件事拆得很开：

- warp 排序策略
- 每 core 有几个 scheduler
- 一个 warp 每 cycle 最多能发几条
- dual issue 是否必须使用不同执行单元

对当前仓库的意义不是“也要一口气支持这些 scheduler family”，而是：

- 现有 `EligibleWaveSelectionPolicy` 只覆盖了 wave 排序
- `ArchitecturalIssuePolicy` 只覆盖了 bundle type/group 约束
- 还缺一层“wave 自身最早何时可再发”的 timing contract

换句话说，当前仓库的抽象分层还不完整，但并不意味着要直接引入：

- `GTO`
- two-level active
- SWL

这些更激进的调度策略。

### 4.5 GPGPU-Sim 还有显式 stall taxonomy，当前工程仍偏扁平

GPGPU-Sim 在 `scheduler_unit::cycle()` 里至少区分：

- `valid_inst == false`
  - idle 或 control hazard
- `ready_inst == false`
  - RAW / scoreboard / long-latency wait
- `issued_inst == false`
  - pipeline resource stall

也就是说，它不是只有“ready / not ready”两态，而是把：

- 指令本身是否有效
- 依赖是否满足
- pipeline 是否接得住

分成了不同层次。

这对当前仓库是一个很重要的提醒：

- 当前 `IssueSchedulerCandidate.ready` 太像“压缩后的最终布尔值”
- 如果后续要把 cycle issue 做得更稳，至少应该把：
  - `eligible`
  - `selected`
  - `issue_deferred_by_switch`
  - `issue_deferred_by_pipe`
  - `issue_deferred_by_group_conflict`

在 engine 里区分开

即使 scheduler 本身仍只消费一个最小子集。

### 4.6 GPGPU-Sim 不是当前工程的直接目标

需要明确：

- GPGPU-Sim 有 ibuffer、scoreboard、register set、多个执行单元出口和更重的 front-end
- 当前工程的 `cycle model` 还没有也不需要一次性照搬这些结构

因此这里吸收的是“抽象层次”，不是“逐字段复刻”：

- 排序与发射分层
- age key 动态化
- issue 资源显式化
- 真实阻塞原因不被压缩成一个布尔 ready

### 4.7 GPGPU-Sim 抽象与当前仓库对象的一对一映射

为了让后续改造更可执行，可以把当前最相关的抽象先做一层对照：

| GPGPU-Sim 抽象 | 当前仓库里最接近的对象 | 当前状态 | 后续建议 |
|---|---|---|---|
| `scheduler_unit::order_warps()` | `IssueScheduler::SelectIssueBundle(...)` 前的 candidate traversal | 已有 | 保留 |
| `sort_warps_by_oldest_dynamic_id(...)` | `WaveAgeOrderKey(...)` | 仅静态 key | 改成 `eligible_since_cycle` |
| `scheduler_unit::cycle()` 中的 readiness checks | `ResidentSlotReadyToIssue(...)` + `CanIssueInstruction(...)` | 已有但偏粗 | 拆成 eligibility projection |
| `waiting()` / `ibuffer_empty()` / scoreboard / pipe free | `wave.status/run_state/valid_entry` + `slot.busy_until` | 缺少中间层 | 增加 `next_issue_cycle` 和更细 blocked cause |
| `issue_warp(...)` | cycle 路径里的 `WaveStep + Commit + future arrive` | 已有 | 保留，但把 selection 与 actual issue 解耦 |
| `gpgpu_num_sched_per_core` | 当前无直接对应 | 未建模 | 本轮不引入 |
| `gpgpu_max_insn_issue_per_warp` | 当前同 wave 同 bundle 最多一次 | 简化存在 | 暂保留 1，不扩成 dual issue |
| `gpgpu_dual_issue_diff_exec_units` | `ArchitecturalIssuePolicy.type/group limits` 的粗粒度近似 | 仅 bundle 约束 | 本轮不做 per-wave dual issue |

这个映射的目的不是证明两边已经等价，而是明确：

- 当前仓库已经有 `排序` 和 `bundle 仲裁`
- 当前仓库缺的是 `排序之后到真正发射之前` 的那一层状态

因此下一步最该补的是：

- eligibility projection
- wave 自身 earliest-next-issue timing
- scheduler-driven switch timing

而不是直接补更多 scheduler family。

## 5. 当前主要差距

下面这些差距，才是当前 `cycle issue` 逻辑和设计目标之间最重要的偏差。

### 5.1 `cycle_exec_engine` 缺少 per-wave 的 `next_issue_cycle`

当前 `cycle_exec_engine` 里的 `ScheduledWave` 只保存：

- front-end 生命周期
- resident slot 信息
- `dispatch_enabled`
- `wave.valid_entry`

但没有显式保存：

- `last_issue_cycle`
- `next_issue_cycle`
- `eligible_since_cycle`

当前候选构造更多依赖：

- `ResidentSlotReadyToIssue(...)`
- `CanIssueInstruction(...)`

这会导致“这个 wave 在业务上已恢复，但是否真的到了它可再次 issue 的全局时间点”没有单独状态表达。

差距本质：

- 设计目标要的是“按 `global_cycle` 检查 wave 最早可发时间”
- 当前实现更接近“只要前端活着、状态 runnable、入口有效，就认为它 ready”

### 5.2 当前把 PEU 节流近似成了 `busy_until = bundle_commit_cycle`

在 [`src/execution/cycle_exec_engine.cpp`](../src/execution/cycle_exec_engine.cpp) 中，当前 bundle 发出后会做：

- `bundle_commit_cycle = max(cycle + plan.issue_cycles, ...)`
- `slot.busy_until = bundle_commit_cycle`

这意味着当前 `PEU` 的下一次选择，实际上被“上一 bundle 的最长 commit 时间”统一卡住了。

这会把下面几个概念耦合到一起：

- 选择窗口何时再次打开
- 某条指令何时 issue 完成
- 某条指令何时 commit
- 某类 pipe 何时再次可用

而目标语义里，这几件事不应该默认等价。

更准确地说，当前实现是：

- `PEU selection bandwidth` 和 `bundle longest commit` 绑定

而设计目标更接近：

- `PEU selection bandwidth`
- `wave earliest next issue`
- `pipe ready cycle`
- `instruction commit`

至少应拆成两层，不能只用一个 `busy_until` 粗暴代理全部。

### 5.3 scheduler-driven wave switch penalty 还没进入真实 issue 路径

当前 `cycle_exec_engine` 会在 wave 进入：

- `waitcnt`
- `barrier wait`

时发出：

- `WaveSwitchAway`
- `WarpSwitch` stall

但它还没有像 [`src/execution/program_object_exec_engine.cpp`](../src/execution/program_object_exec_engine.cpp) 那样，在“同一 PEU 本轮选中了不同 wave”时，把 `warp_switch_cycles` 作为真实调度成本纳入选择路径。

因此当前 cycle 路径里缺了一类关键状态边：

- 不是因为 wave 进入 wait 才 switch away
- 而是因为 scheduler 本轮改选了另一个 wave，所以产生 switch 成本

这是一个实质差距，不只是 trace 缺 marker。

### 5.4 `OldestFirst` 的 age key 现在还是静态 key，不是动态 ready age

当前 `cycle_exec_engine` 的 `WaveAgeOrderKey(...)` 直接返回：

- `WaveTag(block_id, wave_id)`

这只能表达稳定排序，不能表达：

- 哪个 wave 更早进入 eligible 集合
- 哪个 wave 更久没被服务
- 哪个 wave 是 resume 后最早 ready 的

因此现在的 `OldestFirst` 只是“按静态 wave 编号排序”，不是设计目标里的 oldest-ready-first。

### 5.5 candidate 结构过窄，只能表达 `ready=true/false`

当前 `IssueSchedulerCandidate` 只有：

- `candidate_index`
- `wave_id`
- `age_order_key`
- `issue_type`
- `ready`

这对 scheduler 够用，但对 engine 不够用。

因为 engine 真正需要稳定维护的，至少还包括：

- `eligible_since_cycle`
- `next_issue_earliest_global_cycle`
- 是否因为 switch penalty 暂不可发
- 是否因为 pipe/resource cooldown 暂不可发
- 当前被挡住的主因

现在这些信息没有被显式承载，所以很多状态被压缩成了：

- ready
- blocked reason

表达力不够。

### 5.6 `cycle_exec_engine` 还没有和共享 `WaveExecutionState` 对齐

[`src/gpu_model/execution/internal/wave_state.h`](../src/gpu_model/execution/internal/wave_state.h) 已经提供了统一的 `WaveExecutionState`：

- `pending_memory_ops`
- `waiting_waitcnt_thresholds`
- `wave_cycle_total`
- `wave_cycle_active`
- `last_issue_cycle`
- `next_issue_cycle`

但 `cycle_exec_engine` 还在使用自己的 ad-hoc `ScheduledWave` 状态组合，而不是显式收口到这个共享状态契约。

这会带来两个问题：

1. 同一个仓库里，functional / program-object / cycle 三条执行线对“下一次可 issue 时间”的表达不一致
2. 后续如果继续补 cycle issue 语义，会更容易出现只在某一条路径里修好、另一条路径继续漂移

## 6. 建议调整方向

下面给出的是“当前框架上最合理的收敛方向”，不是一步到位重写整个 cycle front-end。

在吸收 GPGPU-Sim 经验时，建议把“可以借鉴的抽象”和“本轮不建议照搬的机制”区分开：

- 可以借鉴：
  - 排序、资格检查、执行资源检查、真正 issue 分层
  - 动态 age key
  - 更细的 stall taxonomy
  - “每轮排序后再做真正发射检查”的主循环形状
- 本轮不建议照搬：
  - ibuffer
  - 完整 scoreboard
  - 多 scheduler per core
  - dual-issue per warp
  - CUDA 风格寄存器集 / pipe register 结构

当前工程这一轮更适合做的是“补齐分层”，不是“补齐整套硬件前端”。

### 6.0 分阶段调整层次

为了避免范围膨胀，建议把后续改造分成三层。

#### 第一层：最小语义闭环

目标：

- 让 `eligible -> selected -> issue` 在 `global_cycle` 约束下真正成立

范围：

- `next_issue_cycle`
- `eligible_since_cycle`
- scheduler-driven `warp_switch_cycles`
- `IssueSelect` 与 `WaveStep` 时间解耦

这层做完后，应能回答：

- 为什么一个 wave 这拍 eligible 但没 issue
- 为什么 resume 后没有立刻消费
- 为什么 `OldestFirst` 和 `RoundRobin` 会产生不同选择

#### 第二层：资源层次显式化

目标：

- 让 issue 选择不再只靠一个 `busy_until`

范围：

- `selection_ready_cycle`
- `last_bundle_commit_cycle`
- 必要时按 type/group 增加 `resource_ready_cycle`

这层做完后，应能区分：

- scheduler 本轮能不能再选
- 选中了之后能不能立刻发
- 资源冲突是 switch、group 还是 pipe

#### 第三层：更细的 front-end / pipe 结构

目标：

- 在需要时再向 GPGPU-Sim / NaviSim 那类模型靠近

范围：

- fetch / ibuffer
- 更细 scoreboard
- per-pipe ready cycle
- 更复杂的 scheduler family

这层不应与第一层绑在同一轮里推进。

### 6.1 先补 wave 级 issue timing state

最优先的调整不是重写 scheduler，而是给 `cycle_exec_engine` 补齐 wave 自身的 issue timing 状态。

建议目标：

- `cycle_exec_engine` 为每个 wave 引入与 `WaveExecutionState` 对齐的状态
- 至少显式保存：
  - `last_issue_cycle`
  - `next_issue_cycle`
  - `eligible_since_cycle`

建议实现形态：

- 要么直接复用 [`src/gpu_model/execution/internal/wave_state.h`](../src/gpu_model/execution/internal/wave_state.h) 的 `WaveExecutionState`
- 要么在 `cycle_exec_engine` 本地加一个等价状态，但字段语义必须与共享契约对齐

这一步的目的不是“增加更多状态”，而是把当前隐式存在但没被表达的约束显式化。

### 6.2 把“选中”和“真正 issue”之间的时序拆开

建议把 cycle issue 主路径明确拆成下面 4 步：

1. `BuildEligibleCandidates(global_cycle)`
2. `SelectIssueBundle(selection_cycle)`
3. `ResolveActualIssueCycle(...)`
4. `ScheduleCommitAndAsyncArrive(...)`

其中第 3 步至少应考虑：

- `wave.next_issue_cycle`
- `PEU switch_ready_cycle`
- 后续若继续细化，还可并入 `pipe_ready_cycle`

也就是说：

- `IssueSelect` 发生在 `selection_cycle`
- `WaveStep` 发生在 `actual_issue_cycle`
- 如果二者不同，必须允许中间出现真实空档

这才真正符合：

- `selected != issue`

### 6.3 把 `warp_switch_cycles` 变成 scheduler 的真实约束

建议在 `PeuSlot` 上新增一类状态，例如：

- `switch_ready_cycle`

语义：

- 如果本轮选择的 wave 和上次真正 issue 的 wave 不同
- 那么新 wave 的真实 issue 时间至少要满足：
  - `>= global_cycle`
  - `>= switch_ready_cycle`

同时建议在这种路径上统一发出：

- `WaveSwitchAway`
- `WarpSwitch` stall

注意：

- 这不是 trace 层补 marker
- 而是 engine 内部真实采用了 switch penalty 后，对外暴露 typed event

### 6.4 把 `OldestFirst` 改成 dynamic ready age，而不是静态 wave tag

建议：

- `age_order_key` 不再默认用 `WaveTag`
- 改为 `eligible_since_cycle`
- 若同 cycle 进入 eligible，可再用 `WaveTag` 做 tie-break

即：

- 主排序：`eligible_since_cycle`
- 次排序：`WaveTag`

这样 `OldestFirst` 才真正表达：

- 谁最早进入 ready/eligible 集合

而不是：

- 谁的 wave_id 更小

### 6.5 不要继续只靠 `slot.busy_until` 统治整个 issue 路径

`slot.busy_until` 可以保留，但应收窄语义。

建议至少改成下面两类状态分离：

- `selection_ready_cycle`
  - PEU 何时允许重新做下一轮选择
- `last_bundle_commit_cycle`
  - 当前 bundle 最晚 commit 到什么时候

若继续细化，还可再拆：

- `pipe_ready_cycle[type/group]`

当前不一定需要一步做到 per-pipe，但至少要先把：

- “能不能再选”
- “上轮什么时候 commit”

从一个 `busy_until` 里分开。

### 6.6 candidate 构造需要升级成“显式 eligibility projection”

建议不要再把 candidate 仅仅看成 scheduler 输入，而是看成 engine 在本 cycle 对 wave 的 eligibility 投影。

建议字段至少包括：

- `candidate_index`
- `wave_id`
- `issue_type`
- `eligible`
- `age_order_key`
- `eligible_since_cycle`
- `next_issue_earliest_global_cycle`
- `blocked_reason`

其中：

- scheduler 仍然只消费最小子集
- engine 可利用完整投影做 stall reason、diagnostics 和 focused tests

这样做的一个直接好处是：

- 当前仓库仍可保持 `IssueScheduler` 简洁
- 但 engine 不再被迫把所有状态压缩成 `ready=true/false`
- 文档、测试和 trace 也能围绕更稳定的 engine-side facts 建立

### 6.7 调整顺序建议

如果要做实际代码调整，建议按下面顺序推进：

1. 先补 `next_issue_cycle / eligible_since_cycle`
2. 再补 scheduler-driven `warp_switch_cycles`
3. 再把 `IssueSelect` 和真实 `WaveStep` 时间解耦
4. 再把 `OldestFirst` 改成动态 age key
5. 最后再考虑是否继续下沉到 per-pipe ready cycle

这样可以避免一上来就把 front-end、pipe、scoreboard、trace 一起重写。

## 7. 建议中的目标形态

下面给出一个更接近当前仓库风格的目标形态。

### 7.1 wave 级状态

建议每个 cycle wave 至少具备：

```cpp
struct CycleWaveIssueState {
  uint64_t last_issue_cycle = 0;
  uint64_t next_issue_cycle = 0;
  uint64_t eligible_since_cycle = 0;
  bool eligible_since_valid = false;
};
```

### 7.2 PEU 级状态

建议每个 `PeuSlot` 至少具备：

```cpp
struct PeuIssueTimingState {
  uint64_t selection_ready_cycle = 0;
  uint64_t switch_ready_cycle = 0;
  uint64_t last_bundle_commit_cycle = 0;
};
```

### 7.3 每 cycle 的 issue 流程

建议实际流程固定为：

1. 处理 ready timed events
2. 更新 wave 的 waiting/resume/front-end 状态
3. 对 resident active waves 建立 eligibility projection
4. 只有 `next_issue_cycle <= global_cycle` 的 wave 才能进入 eligible 集合
5. scheduler 在 eligible 集合上做 `RoundRobin / OldestFirst`
6. 如果切换 wave，则计算 `switch_ready_cycle`
7. 计算真实 `actual_issue_cycle`
8. 在 `actual_issue_cycle` 发 `WaveStep`
9. 在 `commit_cycle` 发 `Commit`
10. 对 async op 再继续排 future arrive event

这样才真正把：

- readiness
- selection
- issue
- commit

拆成了稳定、可测试的状态边。

## 8. 实际约束例子

下面给出几个必须满足的真实约束例子。

### 8.1 例子一：`resume != issue`

场景：

- `wave0` 在 `cycle=20` 收到 global load arrive
- `wave0` 的 `s_waitcnt` 满足，进入 `resume`
- 同一 `PEU` 上 `wave1` 早在 `cycle=12` 就一直 runnable

约束：

- `wave0` 在 `cycle=20` 只是重新变成 `eligible`
- scheduler 可以在 `cycle=20` 继续选择 `wave1`
- `wave0` 的第一次真正 issue 可以出现在 `cycle=24`、`28` 或更晚
- 不能因为 `resume` 发生了，就把 `wave0` 的消费者指令直接画到 `cycle=20`

这正是：

- `resume != selected`
- `selected != issue`

### 8.2 例子二：scheduler 切 wave 必须付 switch 成本

场景：

- `warp_switch_cycles = 5`
- `cycle=40` 时上一次真正 issue 的是 `wave0`
- `cycle=44` 时 scheduler 决定改选 `wave1`

约束：

- `IssueSelect(wave1)` 可以记录在 `cycle=44`
- 但 `wave1` 的真实 `WaveStep` 不能早于 switch 成本满足前
- 如果目标模型约定切换成本进入 issue 路径，则 `actual_issue_cycle` 应至少 `>= 49`

如果当前实现只是：

- `cycle=44` 选中 `wave1`
- `cycle=44` 立刻 `WaveStep`

那就说明 switch penalty 没有进入真实 issue 路径。

### 8.3 例子三：`OldestFirst` 不能按静态 wave 编号冒充

场景：

- `wave3` 在 `cycle=60` resume
- `wave7` 在 `cycle=52` 就已 eligible 但多轮没选上
- policy = `OldestFirst`

约束：

- 这时应优先看“谁更早进入 eligible 集合”
- `wave7` 应先于 `wave3`
- 不能因为 `wave3.wave_id < wave7.wave_id`
  或 `WaveTag` 更小，就被误当成“更老”

因此：

- `OldestFirst` 的 `age_order_key` 必须是动态 ready age
- 不能继续是静态编号

### 8.4 例子四：bundle 约束不等于 wave 的 next issue time

场景：

- 同一 PEU 上：
  - `wave0` 下一条是 `VectorAlu`
  - `wave1` 下一条也是 `VectorAlu`
- 当前 policy `vector_alu = 1`

约束：

- 这一轮 bundle 最多只能选一个 wave
- 未被选中的那个 wave 应保留 `eligible` 身份
- 它的 `eligible_since_cycle` 不应被清掉
- 下一轮 age/order 仍应把它当作一个持续等待服务的 ready wave

也就是说：

- “本轮没被 bundle 选中”
- 不等于
- “它不再 eligible”

### 8.5 例子五：异步内存返回不能直接解锁整个 PEU 的所有时序

场景：

- `wave0` 发出 shared/global load
- `wave0` 在 `commit_cycle=100` 后进入等待
- 数据在 `ready_cycle=132` 才返回

约束：

- `wave0` 的 `next_issue_cycle` 应至少与恢复语义对齐
- 但同一 `PEU` 上其他 runnable waves 的 issue 选择不应被 `wave0` 的 async wait 路径错误串住
- 不能把“某个 wave 的 commit/arrive 时间”直接等价成“整个 PEU 的 busy_until”

这也是为什么当前只靠一个 `slot.busy_until` 会过粗。

## 9. 本轮建议的最小闭环

如果下一步要落代码，建议只做下面这组最小闭环：

1. 在 `cycle_exec_engine` 中补 `last_issue_cycle / next_issue_cycle / eligible_since_cycle`
2. 候选构造时用 `next_issue_cycle <= global_cycle` 过滤
3. scheduler 切到不同 wave 时，把 `warp_switch_cycles` 纳入真实 issue 路径
4. `IssueSelect` 与 `WaveStep` 明确允许不在同一个 cycle
5. `OldestFirst` 改成按 `eligible_since_cycle` 排序

这一组改动可以直接修正当前最重要的语义差距，同时保持：

- front-end resident/active-window 结构不推翻
- 现有 `IssueScheduler` policy 抽象不推翻
- trace 继续只是 typed event consumer

## 10. 文件级改造落点

为了避免后续实现时范围扩散，建议先把“该改什么文件、各文件承担什么职责”写死。

### 10.1 [`src/gpu_model/execution/internal/wave_state.h`](../src/gpu_model/execution/internal/wave_state.h)

这是最适合承接共享 wave issue timing 契约的位置。

建议修改：

- 在 `WaveExecutionState` 中补：
  - `eligible_since_cycle`
  - `eligible_since_valid`
- 保留现有：
  - `last_issue_cycle`
  - `next_issue_cycle`

建议职责：

- `WaveExecutionState` 成为 functional / program-object / cycle 三条执行路径共享的 wave issue 时间状态来源
- `eligible_since_*` 只表达“进入当前 eligible 集合的时间”，不表达被选中或真正 issue

不建议：

- 把 scheduler policy、blocked reason、trace marker 这些逻辑塞进共享状态头文件
- 让 `wave_cycle_total / active` 去承载全局调度时间语义

### 10.2 [`src/execution/cycle_exec_engine.cpp`](../src/execution/cycle_exec_engine.cpp)

这是本轮实现的主战场，大部分语义修正都应落在这里。

建议修改：

- `ScheduledWave` 对齐或持有 `WaveExecutionState`
- 把当前 candidate 构造拆成两层：
  - engine-side `eligibility projection`
  - scheduler-side `IssueSchedulerCandidate`
- 把当前主循环里的：
  - `if (slot.busy_until > cycle) continue;`
  - `IssueSelect(cycle) + WaveStep(cycle) + slot.busy_until=bundle_commit_cycle`
  改成：
  - 先判断 `selection_ready_cycle`
  - 再计算 `actual_issue_cycle`
  - 再独立安排 `commit_cycle`
- 在 scheduler 选中不同 wave 时，真实引入 `warp_switch_cycles`
- 在 wave `resume`、bundle miss、async arrive 时，维护：
  - `next_issue_cycle`
  - `eligible_since_cycle`

建议职责：

- 维护真实的 `eligible -> selected -> issue -> commit` 时序边
- 维护 PEU 本地 timing state，例如：
  - `selection_ready_cycle`
  - `switch_ready_cycle`
  - `last_bundle_commit_cycle`
- 发出 typed event，但不把 trace 当状态机

不建议：

- 把完整 scoreboard / ibuffer / 多 scheduler per core 一起塞进这一轮
- 把所有 blocked cause 都下推到 `IssueScheduler`

### 10.3 [`src/gpu_model/execution/internal/issue_scheduler.h`](../src/gpu_model/execution/internal/issue_scheduler.h) 与 [`src/execution/internal/issue_scheduler.cpp`](../src/execution/internal/issue_scheduler.cpp)

这两处只应该保持“policy 层最小输入”。

建议修改：

- 继续让 scheduler 只消费：
  - `wave_id`
  - `issue_type`
  - `ready/eligible`
  - `age_order_key`
- `OldestFirst` 继续只看 `age_order_key`，但这个 key 必须从 engine 传入动态 `eligible_since_cycle`

更推荐的做法：

- 不把 `blocked_reason`
- 不把 `switch_ready_cycle`
- 不把 `next_issue_cycle`

直接加进 `IssueSchedulerCandidate` 并强迫 scheduler 理解。

更稳妥的分层是：

- `cycle_exec_engine` 先生成 richer projection
- 再投影成最小 `IssueSchedulerCandidate`
- scheduler 只做排序和 bundle 仲裁

这样能避免 policy 层被 timing/状态细节污染。

### 10.4 [`src/execution/program_object_exec_engine.cpp`](../src/execution/program_object_exec_engine.cpp)

这个文件本轮更适合作为“语义对照源”，不适合作为主要修改面。

建议用途：

- 参考其已有的：
  - `last_issue_cycle / next_issue_cycle`
  - switch penalty 进入 issue 路径
  - `IssueSelect(issue_cycle)` 与 `WaveStep(issue_cycle)` 的时间来源
- 用来约束 cycle 路径不再继续漂移

不建议：

- 为了“统一”而同步重写 program-object 路径
- 在这里再引入第二套只给 cycle 用的分支逻辑

### 10.5 测试文件落点

建议把验证拆到最接近语义边界的现有测试文件里，而不是新建一个“大而全”的 cycle issue mega test。

优先落点：

- [`tests/execution/internal/issue_scheduler_test.cpp`](../tests/execution/internal/issue_scheduler_test.cpp)
  - 保住 policy 单测边界
  - 验证 `OldestFirst` 只看动态 age key
- [`tests/cycle/cycle_smoke_test.cpp`](../tests/cycle/cycle_smoke_test.cpp)
  - 验证 `resume -> IssueSelect -> WaveStep`
  - 验证 selection/issue 分离
  - 验证 same-PEU 轮转和 oldest-first 集成行为
- [`tests/cycle/waitcnt_barrier_switch_focused_test.cpp`](../tests/cycle/waitcnt_barrier_switch_focused_test.cpp)
  - 验证 wait/barrier 恢复语义
  - 验证 scheduler-driven switch penalty 不被误写成 wait-only 事件
- [`tests/cycle/async_memory_cycle_test.cpp`](../tests/cycle/async_memory_cycle_test.cpp)
  - 验证 async return 只影响相关 wave，不全局卡住 PEU
  - 验证 bundle/slot timing 仍保持一致

### 10.6 当前源码证据锚点

下面这些代码位置可以直接作为“当前实现事实”的证据锚点。

- [`src/execution/cycle_exec_engine.cpp`](../src/execution/cycle_exec_engine.cpp)
  - 第 469-470 行：`WaveAgeOrderKey(...)` 直接返回 `WaveTag(...)`
  - 第 1007-1063 行：`BuildResidentIssueCandidates(...)` 只向 scheduler 投喂 `age_order_key + issue_type + ready`
  - 第 1162 行：PEU 选择入口仍由 `slot.busy_until > cycle` 粗门控
  - 第 1220-1243 行：`IssueSelect` 与 `WaveStep` 当前都记录在 `cycle`
  - 第 1244 行与第 1920 行：`commit_cycle` 汇总后回写 `slot.busy_until = bundle_commit_cycle`
- [`src/execution/program_object_exec_engine.cpp`](../src/execution/program_object_exec_engine.cpp)
  - 第 1955-1958 行：已有 scheduler-driven `switch_penalty` 计算
  - 第 1959-1967 行：已有与 switch penalty 对应的 `WaveSwitchAway / WarpSwitch` 事件
  - 第 2489-2494 行：已有 `issue_cycle = next_issue_cycle`、`last_issue_cycle`、`next_issue_cycle = commit_cycle`
- [`src/gpu_model/execution/internal/wave_state.h`](../src/gpu_model/execution/internal/wave_state.h)
  - 第 57-58 行：共享 `WaveExecutionState` 已经有 `last_issue_cycle / next_issue_cycle`
  - 当前还没有 `eligible_since_cycle`
- [`tests/cycle/cycle_smoke_test.cpp`](../tests/cycle/cycle_smoke_test.cpp)
  - 第 345-386 行：已经验证 `resume` 不保证立即消费
  - 第 389-442 行：已经验证 `WaveResume < IssueSelect < WaveStep` 的可观察顺序
- [`tests/cycle/async_memory_cycle_test.cpp`](../tests/cycle/async_memory_cycle_test.cpp)
  - 第 297-324 行：已经验证 resident slot 不能绕过 PEU issue timing
  - 第 402-430 行：已经验证 load 发出后，同 wave 独立 scalar issue 可以先于 arrive
- [`tests/execution/internal/issue_scheduler_test.cpp`](../tests/execution/internal/issue_scheduler_test.cpp)
  - 第 194-219 行：已经验证 `OldestFirst` 依赖 `age_order_key`，而不是 selection cursor

这组证据说明：

- 文档里的差距判断不是抽象推演
- 当前代码确实已经具备一部分接近目标的骨架
- 但真正缺的仍然是：
  - dynamic ready age
  - per-wave next-issue timing
  - selection 与 issue 的真实时间解耦
  - switch penalty 进入 cycle 路径

## 11. Focused Regression / Verification Matrix

下面给出更适合这一轮改造的 focused 验证矩阵。

| 目标语义 | 主改动文件 | 可复用现有测试 | 建议新增/调整测试 | 完成证据 |
|---|---|---|---|---|
| `next_issue_cycle` 真正参与 eligible 过滤 | `src/execution/cycle_exec_engine.cpp` `src/gpu_model/execution/internal/wave_state.h` | [`tests/cycle/cycle_smoke_test.cpp`](../tests/cycle/cycle_smoke_test.cpp) 里已有 `ReadyDoesNotGuaranteeImmediateConsumerIssue`、`ResumeSelectionAndIssueOrderingStayObservable` | 增加“resume 后 wave 可见但因为 `next_issue_cycle` 尚未到而未 issue”的精确 cycle 断言 | `WaveResume < IssueSelect < WaveStep`，且 `WaveStep.cycle >= next_issue_cycle` |
| scheduler-driven switch penalty 进入真实 issue 路径 | `src/execution/cycle_exec_engine.cpp` | [`tests/cycle/waitcnt_barrier_switch_focused_test.cpp`](../tests/cycle/waitcnt_barrier_switch_focused_test.cpp) 已覆盖 wait/switch 事件存在性 | 增加“不同 wave 被选中且 `warp_switch_cycles > 0` 时，`IssueSelect` 与 `WaveStep` 之间出现 penalty gap”的断言 | 对新 wave 有 `IssueSelect.cycle < WaveStep.cycle`，且 gap 满足 switch 约束 |
| `OldestFirst` 使用动态 ready age 而不是静态 wave id | `src/execution/cycle_exec_engine.cpp` `src/execution/internal/issue_scheduler.cpp` | [`tests/execution/internal/issue_scheduler_test.cpp`](../tests/execution/internal/issue_scheduler_test.cpp) 已有 `ExplicitOldestFirstPolicyUsesAgeOrderKeyInsteadOfSelectionCursor` | 新增 cycle 集成测试：较晚编号但更早 eligible 的 wave 必须先被选中 | 在 `OldestFirst` 下，选择顺序随 `eligible_since_cycle` 变化，而不是随 `WaveTag` 变化 |
| bundle 冲突不会清掉持续等待 wave 的 ready age | `src/execution/cycle_exec_engine.cpp` | [`tests/cycle/async_memory_cycle_test.cpp`](../tests/cycle/async_memory_cycle_test.cpp) 已覆盖 resident slot / issue limit 基线 | 新增“同 type limit=1，两 wave 都 ready，未选中 wave 保留 `eligible_since_cycle`，下一轮应优先被服务” | 第二轮选择结果继承上一轮等待 age，而不是重新计龄 |
| async return 不会错误串住整个 PEU 的时序 | `src/execution/cycle_exec_engine.cpp` | [`tests/cycle/async_memory_cycle_test.cpp`](../tests/cycle/async_memory_cycle_test.cpp) 已有 `LoadAllowsIndependentScalarIssueBeforeArrive`、`ResidentSlotsDoNotBypassPeuIssueTiming` | 增加 same-PEU 多 wave 测试：等待 wave 的 arrive 只更新自身 `next_issue_cycle` | 其他 runnable wave 的 `WaveStep` 周期不受等待 wave 的 arrive/commit 粗暴牵连 |
| trace 仍然只是 consumer，不参与调度决策 | `src/execution/cycle_exec_engine.cpp` | [`tests/cycle/cycle_smoke_test.cpp`](../tests/cycle/cycle_smoke_test.cpp) 已有 `FrontendLatenciesAdvanceCycleWithoutTraceSink` | 如实现触及事件顺序，可补一个“无 trace sink 仍得到相同结果/周期”的 focused case | 关闭 trace sink 时，功能结果和关键 cycle 不发生语义性漂移 |

补充建议：

- 单测层先锁 policy 与 age key 规则
- cycle 集成层再锁 `resume/select/issue/commit` 边界
- 不要只看 trace 文本顺序，关键断言应直接比较 event `cycle`、`kind`、`wave_id`、`peu_id`

## 12. 实施完成前的最小验收标准

如果后续进入代码实现阶段，至少应满足下面这些 AC，才算这轮 “cycle issue 选择语义补齐” 完成。

### AC-1 `resume` 只表示重新 eligible，不表示立即 issue

必须存在明确证据表明：

- `WaveResume` 可以早于 `IssueSelect`
- `IssueSelect` 可以早于 `WaveStep`
- `WaveResume` 发生后，consumer 指令不是被硬塞回同 cycle

### AC-2 scheduler 切换 wave 时，`warp_switch_cycles` 是真实 issue 约束

当：

- 上一条真正 issue 的 wave 与本轮选中的 wave 不同
- 且 `warp_switch_cycles > 0`

则必须满足：

- `IssueSelect` 可发生在选择周期
- `WaveStep` 不能早于 switch penalty 满足前

### AC-3 `OldestFirst` 必须是 dynamic ready age，不是静态 `WaveTag`

必须满足：

- 较早进入 eligible 集合的 wave 会先于较晚进入的 wave
- 即使其 `wave_id` 更大，也不能被静态编号反超

### AC-4 bundle miss 不会重置 wave 的等待年龄

当 wave 因：

- type limit
- group limit
- same-wave bundle restriction

在当前轮未被选中时，必须满足：

- 它仍保持 eligible
- `eligible_since_cycle` 不被重置
- 下一轮 oldest-first 仍把它视为持续等待服务的 wave

### AC-5 async wait/arrive 只影响相关 wave，不全局锁死 PEU

必须满足：

- waiting wave 的 `commit/arrive` 不直接充当整个 PEU 的唯一 `busy_until`
- 同一 PEU 上其他 runnable waves 仍可按自身 timing 正常 issue

### AC-6 trace 关闭或缺席时，行为语义不变

必须满足：

- 没有 trace sink 时，cycle model 仍能给出一致的功能结果和关键时序
- 不能靠 trace side effect 才让 `IssueSelect/WaveStep/Commit/Arrive` 顺序成立

### AC-7 本轮完成不应以更大重写为前提

这一轮 AC 不要求：

- ibuffer
- 完整 scoreboard
- multi-scheduler-per-core
- per-warp dual issue

只要求把当前框架内的最小语义闭环补实。

## 13. 暂不建议在这一轮解决的点

下面这些点很重要，但不建议和本轮最小闭环一起做：

- 完整 fetch arbiter
- per-wave instruction buffer
- 完整 scoreboard register ready graph
- 精细到每条 pipe 的 ready cycle 和 execute latency 解耦
- 与某一代 AMD 私有调度器逐拍对齐

原因：

- 这些改动会显著扩大范围
- 会把 front-end、execution、trace、tests 一起放大
- 容易让当前“先把 `eligible -> selected -> issue` 语义边界做实”的目标失焦

## 14. 当前结论

当前 `cycle_exec_engine` 的 issue 逻辑已经具备：

- resident slot
- active window
- policy-based bundle selection
- typed issue/select/commit/arrive/wait/resume 事件

所以问题不是“完全没有 issue 设计”，而是：

- 缺少 wave 级最早下一次可发时间
- 缺少 scheduler-driven switch timing
- 缺少动态 ready age
- 仍然把部分 issue/commit/resource 约束压缩进过粗的 `busy_until`

因此后续调整的重点不应是：

- 把默认 policy 改成“连续发同一个 wave”

而应是：

- 让当前 `eligible -> selected -> issue` 在真实 `global_cycle` 约束下成立
- 让 `RoundRobin / OldestFirst` 作用在正确的 eligible 集合上
- 让 switch / next issue / ready age 成为 engine 内部真实状态，而不是外围推断
