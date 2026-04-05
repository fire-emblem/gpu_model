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
