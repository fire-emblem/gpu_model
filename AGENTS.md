# GPU Model Engineering Constraints

本文件记录当前仓库的主架构约束与实现边界。后续功能开发、重构、测试和文档更新都应遵守这些约束。

## 1. Runtime 主架构

当前 runtime 主线只按三部分理解：

1. `HipRuntime`
2. `ModelRuntime`
3. `ExecEngine`

约束：

- `HipRuntime` 是与 AMD HIP runtime 对齐的兼容层。
- `HipRuntime` 对外承接 HIP C 接口 / C ABI 语义。
- `ModelRuntime` 是项目核心实现层，负责 runtime 语义、状态、加载与运行时组织。
- `ExecEngine` 是执行核心，负责真正驱动 functional / cycle / encoded 执行。

禁止：

- 不再新增独立的 `interposer` 子系统概念。
- 不再恢复 `HipInterposerState` 一类中间包装层。
- 不允许 `ModelRuntime` 反向依赖 `HipRuntime`。

## 2. HIP 入口语义

约束：

- `src/runtime/hip_interposer.cpp` 只是 `HipRuntime` 的 C ABI 入口实现载体。
- 文件名中的 `interposer` 是历史遗留命名，不代表当前架构上存在独立的 interposer 模块。
- 所有 HIP 兼容行为应归属到 `HipRuntime` 语义，而不是归属到“interposer 模块”。

## 3. ExecEngine 命名

约束：

- `ExecEngine` 是执行核心的正式命名。
- 不再新增 `RuntimeEngine` 新引用。
- 如果发现旧文档或旧代码中仍有 `RuntimeEngine`，应优先收口到 `ExecEngine`。

## 4. Functional 与 Cycle 模型边界

约束：

- `Functional` 模型允许 `st/mt` 两种宿主执行策略。
- `Cycle` 模型必须保持唯一时序模型，不区分 `cycle st` / `cycle mt`。
- 如果需要表达更激进或更保守的 cycle 行为，必须通过：
  - 资源参数
  - timing 参数
  - issue policy
  - issue limits
  来表达，而不是引入新的 cycle 执行模式。

## 5. Trace 与业务逻辑边界

这是硬约束。

约束：

- trace 只负责消费已经产生的事件，并序列化为：
  - text
  - json/jsonl
  - perfetto/google trace
- trace 不能参与任何业务逻辑决策。
- 没有 trace sink 时，functional / cycle / encoded 行为必须与开启 trace 时一致。

禁止：

- 不允许 trace 层推断状态机边。
- 不允许 trace 层补延迟。
- 不允许 trace 层反推 wait / arrive / resume。
- 不允许为了画 timeline 在 trace 层伪造业务事件。

统一原则：

- 先业务状态变化
- 再记录 typed event
- 最后由 trace 消费

这条原则对 `Functional` 和 `Cycle` 两条主线都成立。

补充约束：

- 必须支持通过全局环境变量关闭 trace。
- 建议统一使用：
  - `GPU_MODEL_DISABLE_TRACE=1`
- 当该开关打开时：
  - runtime 不应创建 trace artifact recorder
  - ExecEngine 不应向外部 trace sink 发事件
  - 非 trace 模块测试必须仍可通过

## 5.1 Trace Cycle 字段语义

这是硬约束。

约束：

- `trace.txt` / `trace.jsonl` / `timeline.perfetto.json` 中的 `cycle` 字段，在当前工程里一律不是物理真实执行时间戳。
- 无论 `Functional st`、`Functional mt` 还是 `Cycle` 模型：
  - `cycle` 都只是对应模型内部推进后的计数时间轴。
  - 它首先用于表达模型中的相对顺序、等待区间、依赖关系和调度结果。
  - 它不能直接等价解释为真实硬件上的绝对时间。
- 当前 `Functional st/mt` 的 `cycle`：
  - 本质是 trace/调度推进时生成的虚拟计数器。
- 当前 `Cycle` 模型的 `cycle`：
  - 本质是 cycle engine 的模型计数时间，不是经过真实硬件校准后的物理时间。
- 如果后续需要表达“真实时间”语义，必须额外定义校准后的 modeled-time / hardware-time 口径，并与当前 trace `cycle` 明确区分。

禁止：

- 不允许在文档、README、测试说明里把当前任意模型 trace 的 `cycle` 直接称为“真实时间戳”。
- 不允许用当前 trace 的 `cycle` 直接证明真实硬件上的 AP / PEU 并行启动、真实重叠执行或真实耗时。
- 不允许在没有额外校准口径的前提下，把当前 trace `cycle` 当成物理时间来做性能结论。

统一原则：

- 当前所有 trace `cycle` 先表达模型语义。
- `Functional` 更偏语义顺序。
- `Cycle` 更偏结构化时序模型，但仍然只是模型时间，不自动等于真实硬件时间。
- 如果一个问题需要“真实何时发生”，必须先说明校准基准，而不是直接引用当前 trace `cycle`。

## 6. Cycle 模型实现原则

约束：

- cycle 模型应是独立、自洽、tick-driven 的状态机。
- 每个 cycle/tick 的推进顺序必须由 engine/state machine 决定，而不是由 trace 驱动。
- future event / arrival / resume / dispatch 都应由 cycle 状态机或事件队列推进。

建议的 tick 顺序：

1. 处理本 cycle ready 的 timed events
2. 推进 AP / block front-end
3. 推进 wave generate / dispatch / slot / active window
4. 推进 PEU arbitration / issue select
5. 调度 future event
6. 进入下一 cycle

## 6.1 Global Cycle 与 Wave Cycle 关系

这是硬约束。

约束：

- `global_cycle` 是唯一的全局调度定位时间。
- 所有资源竞争、AP/PEU 选择、dispatch、arrive、release、issue 判定，都必须以 `global_cycle` 为准。
- `wave` 可以持有自己的 cycle 累计状态，但 `wave_cycle` 不是第二套全局真时间。
- `wave_cycle` 只负责记录该 wave 自身的累计推进量。

建议最少拆分为：

- `wave_cycle_total`
- `wave_cycle_active`

语义：

- `wave_cycle_total`
  - 记录 wave 生命周期内累计经历的推进量
  - 可包含 issue、switch 空泡、等待等被模型计入 wave 生命周期的时间
- `wave_cycle_active`
  - 只记录 wave 真正执行指令或有效推进的时间

统一原则：

- `global_cycle` 负责“定位”
- `wave_cycle` 负责“累计”
- wave 被切回后是否可继续 issue，必须看：
  - `global_cycle`
  - `ready_at_global_cycle`
  - `next_issue_earliest_global_cycle`
- 不允许通过 `wave_cycle` 反推当前全局真时间。

## 6.2 Functional 与 Cycle 的 Resume / Ready 语义

这是硬约束。

约束：

- `Cycle` 模型中：
  - `arrive_resume` 只表示 wave 达到可恢复、可被调度的 `ready/eligible` 状态。
  - `arrive_resume` 不保证同 cycle 或下一个固定 cycle 一定发生 issue。
  - `ready -> selected -> issue` 之间允许存在真实调度间隔。

- `Functional` 模型中：
  - 不展开 `ready -> selected` 之间的额外硬件调度空档。
  - 采用 `4-cycle issue quantum` 语义。
  - `arrive_resume` 发生后，恢复 wave 的消费者指令应在**下一个 issue quantum 起点**发出。
  - 不要求同 cycle issue。

- `Functional st`：
  - 作为单线程、确定性的语义参考模型，必须满足上面的“下一 quantum issue”保证。

- `Functional mt`：
  - 与 `st` 共用同一套 modeled time 规则。
  - 仍保留异步调度和 runnable wave 竞争。
  - 不要求像 `st` 一样对每个恢复 wave 做强同步保证；允许恢复 wave 被其他 runnable wave 抢占。

禁止：

- 不允许在 `Functional` 模型里继续引入额外的 `wave_resume` 基础状态机事件。
- 不允许把 `arrive_resume` 误解为“该 wave 已经被 scheduler 选中 issue”。

统一原则：

- `arrive_resume` 表示条件满足。
- `WaveStep` 表示真正 issue。
- 如果需要观察“恢复后何时真正被消费”，在 `Functional` 中看 `arrive_resume -> next WaveStep`；
  在 `Cycle` 中保留真实调度间隔。

## 7. 前端增强方向

当前 cycle model 后续增强应优先补：

- `block_admit`
- `wave_generate`
- `wave_dispatch`
- `slot_bind`
- `active_promote`
- `wave_issue_select`
- `wave_switch_away`
- `wave_wait`
- `wave_arrive`
- `wave_resume`
- `wave_exit`

以及显式 timing 参数：

- `wave_generation_cycles`
- `wave_dispatch_cycles`

约束：

- 这些必须先在 engine/state machine 中成为真实状态边和真实 cycle 空档。
- trace 只消费这些事件，不定义这些事件。

## 8. Example 结果管理

约束：

- examples 结果目录直接写回各自 example 下的 `results/`
- 不再使用仓库根目录 `.cache/example-results/` 作为默认落盘位置

目的：

- 避免“默认结果”和“实际检查结果”分叉造成误读。
- README、脚本输出、测试断言都应与 `results/` 实际路径保持一致。

## 9. Push 门禁

约束：

- `pre-push` 默认只跑轻量 smoke：
  - `scripts/run_push_gate_light.sh`
- 全量 gate 仍保留：
  - `scripts/run_push_gate.sh`

目的：

- 日常 push 保持快速反馈
- 全量验证仍可手动执行

## 10. 文档策略

约束：

- 主文档必须反映当前代码事实。
- 历史 plans/spec 存档可以做术语收口，但不能改坏原始历史语境。
- 如果主文档与代码冲突，以修正文档到当前实现为优先。

## 11. 修改原则

后续开发默认遵循：

- 先确定职责边界，再写代码
- 新行为先落 engine/state machine，再接 trace
- 优先新增 typed event，不扩展 message-based 猜测逻辑
- 优先做小步、可验证、可回滚的改动
