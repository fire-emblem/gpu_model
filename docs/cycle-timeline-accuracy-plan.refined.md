# Cycle Timeline Accuracy Calibration Plan

## Goal Description

校准 `cycle timeline / Perfetto` 的时间线语义，使其能够稳定、准确地表达当前执行模型已经产生的事实，重点收口：

- `issue / commit / arrive / stall` 的因果顺序
- slot-centric timeline 的 slice / marker 生成规则
- `ready / selected / issue` 的可观察边界
- barrier-heavy 与 waitcnt-heavy 路径的 timeline 一致性

本计划不重新设计整个 cycle model，不改 trace 作为纯消费层的基本约束，也不扩展新的 Perfetto 格式。目标是先把当前时间线“表达什么”和“何时表达”校准正确，并用 focused regressions 和 representative examples 固化下来。

## Acceptance Criteria

遵循 TDD 原则，每条标准都给出正向和反向验证，确保实现结果可被确定性检查。

- AC-1: `InstructionIssue -> Commit` 配对只为应显示的指令生成 slice，且 `issue_cycle / commit_cycle / render_duration_cycles` 与执行事实一致
  - Positive Tests (expected to PASS):
    - [tests/runtime/cycle_timeline_test.cpp](/data/gpu_model/tests/runtime/cycle_timeline_test.cpp) 中 issue/commit 顺序回归通过，`buffer_load_dword`、普通 ALU 指令可稳定生成 slice
    - [tests/runtime/trace_test.cpp](/data/gpu_model/tests/runtime/trace_test.cpp) 中 encoded functional instruction slice 仍保持 `dur=4`
    - dense scalar / representative vector instruction case 中，相邻 slice 的 issue 间隔与当前模型一致
  - Negative Tests (expected to FAIL):
    - `s_waitcnt` 被误渲染成普通指令 slice
    - 没有 `Commit` 的 issue 事件仍然生成 instruction slice
    - marker 事件被错误并入 instruction slice
  - AC-1.1: `render_duration_cycles` 的来源规则在文档和测试中明确
    - Positive:
      - timeline JSON 中存在稳定的 `issue_cycle / commit_cycle / render_duration_cycles`
      - 关键测试可验证 duration 至少为 4-cycle quantized 结果
    - Negative:
      - duration 由 wall-clock 或渲染器猜测生成
      - 不同 mode 下同类 issue 事件无理由地产生不同 duration 规则

- AC-2: `arrive` 事件的 cycle、命名和可视表达稳定，且 `still_blocked / resume` 语义不混淆
  - Positive Tests (expected to PASS):
    - waitcnt-heavy focused regressions 中 `load_arrive_still_blocked` 与 `load_arrive_resume` 可稳定区分
    - [tests/runtime/cycle_timeline_test.cpp](/data/gpu_model/tests/runtime/cycle_timeline_test.cpp) 与 [tests/runtime/trace_test.cpp](/data/gpu_model/tests/runtime/trace_test.cpp) 中 `arrive` 出现在正确时间顺序上
    - encoded/functional waitcnt representative cases 的 timeline 中都能看到正确 arrive 名称
  - Negative Tests (expected to FAIL):
    - `arrive` 用 resume issue 的 cycle 代替真实异步完成 cycle
    - 多次 arrive 在同一 slot 上发生顺序反转或被覆盖
    - `arrive_progress` 缺失时渲染器自行推断业务语义

- AC-3: `stall` marker 只表达“未 issue 或等待中的可观测原因”，并保持稳定 taxonomy
  - Positive Tests (expected to PASS):
    - `waitcnt_global`、barrier wait、warp switch 等典型 stall 的 canonical name / category / args 稳定
    - barrier-heavy 和 waitcnt-heavy 路径在 raw trace 与 timeline 上都可见一致 stall 原因
    - stall marker 的 cycle 与当前执行模型的阻塞开始/观测时点定义一致
  - Negative Tests (expected to FAIL):
    - stall marker 被误当作 instruction slice 持续区间
    - 同一类 stall 在不同导出路径上使用不一致命名
    - timeline 为了“看起来连续”而补画虚假 stall 区间

- AC-4: `ready / selected / issue` 的边界在 cycle timeline 中不混淆
  - Positive Tests (expected to PASS):
    - focused regressions 能证明 `ready != selected != issue`
    - slot timeline 中不会把 ready wave 提前画成 issue
    - scheduler selection 仅在需要时以 marker 暴露，不冒充真实执行开始
  - Negative Tests (expected to FAIL):
    - `arrive` 后同 cycle 直接被画成 consumer issue，违背当前模型
    - ready wave 在未被选中时就出现 instruction slice
    - selection marker 和 issue slice 顺序颠倒

- AC-5: representative examples 的 timeline 与当前模型事实无明显语义冲突
  - Positive Tests (expected to PASS):
    - 至少 1 组 waitcnt-heavy 和 1 组 barrier-heavy examples 经肉眼校准后无明显时序矛盾
    - example 导出的 `timeline.perfetto.json` 与 focused tests 的命名/字段体系统一
  - Negative Tests (expected to FAIL):
    - example 中出现明显违背模型语义的 issue/arrive/stall 排序
    - example 只能靠额外解释才能理解，且 focused tests 无法覆盖该语义

- AC-6: 文档同步收口，明确 cycle timeline 的正式语义与限制
  - Positive Tests (expected to PASS):
    - [docs/my_design.md](/data/gpu_model/docs/my_design.md)、[docs/module-development-status.md](/data/gpu_model/docs/module-development-status.md)、必要时 [task_plan.md](/data/gpu_model/task_plan.md) 回写本阶段结论
    - 文档明确说明 trace/timeline 只消费事件、`cycle` 不是 wall-clock
  - Negative Tests (expected to FAIL):
    - 文档仍允许 timeline 通过业务推断补事件
    - 文档未明确 `issue / commit / arrive / stall` 的正式含义

## Path Boundaries

Path boundaries 用来约束实现范围，避免遗漏关键目标，也避免无边界扩张。

### Upper Bound (Maximum Acceptable Scope)

完成一套可维护的 cycle timeline 校准闭环：

- 有事件真值表
- 有 focused regressions 覆盖关键顺序与命名
- [src/debug/timeline/cycle_timeline.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline.cpp) 和相关 trace exporter 规则清晰
- representative examples 已做至少两类校准记录
- 正式文档同步更新

### Lower Bound (Minimum Acceptable Scope)

完成最小可信收口：

- 明确 `InstructionIssue / Commit / Arrive / Stall` 的 timeline 真值定义
- 修正最关键的 slice/marker 生成错误
- focused regressions 覆盖至少一组 waitcnt-heavy 和一组 barrier-heavy timeline 顺序问题
- 文档明确写出正式语义

### Allowed Choices

- Can use:
  - 继续使用现有 `Recorder / TraceEvent / CycleTimelineRenderer`
  - 增加 focused tests
  - 调整 [src/debug/timeline/cycle_timeline.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline.cpp) 的 slice/marker 生成规则
  - 调整 trace event 字段填充，但前提是不把业务逻辑挪到 trace 层
- Cannot use:
  - 让 timeline 渲染器根据缺失信息推断业务语义
  - 让 trace/timeline 反向驱动 engine 行为
  - 为了例子显示效果伪造 `issue / commit / arrive / stall`
  - 在本阶段引入新的 Perfetto 主格式或 protobuf 主线恢复

> 注意：本计划的关键设计约束较窄。trace/timeline 只能消费事件、不能承担业务推断，因此上界和下界都围绕“消费层校准”展开，而不是允许多条完全不同的实现路线。

## Feasibility Hints and Suggestions

> 这一节只提供理解和实现提示，不构成强制实现细节。

### Conceptual Approach

建议按“真值表 -> focused tests -> renderer 校准 -> examples 校准 -> 文档回写”的顺序推进：

1. 先定义事件真值表
2. 把当前测试分成：
   - 意图正确但断言过时
   - 意图错误或过度绑定实现
   - 能直接充当校准基线
3. 修 [src/debug/timeline/cycle_timeline.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline.cpp) 时只做消费层收口：
   - issue/commit 配对
   - marker 分类
   - duration 来源
   - canonical name / args 稳定化
4. 再用 examples 做肉眼验证，避免只在 unit test 层自洽

### Relevant References

- [tests/runtime/cycle_timeline_test.cpp](/data/gpu_model/tests/runtime/cycle_timeline_test.cpp) - timeline 顺序、slot 轨道、Perfetto dump 回归
- [tests/runtime/trace_test.cpp](/data/gpu_model/tests/runtime/trace_test.cpp) - canonical trace fields、arrive/stall 命名、mode-stable timeline 回归
- [src/debug/timeline/cycle_timeline.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline.cpp) - issue/commit 配对与 marker/slice 组装主逻辑
- [src/debug/timeline/cycle_timeline_google_trace.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline_google_trace.cpp) - chrome-json/Perfetto 事件导出
- [docs/cycle-timeline-accuracy-calibration-draft.md](/data/gpu_model/docs/cycle-timeline-accuracy-calibration-draft.md) - 原始草稿
- [docs/my_design.md](/data/gpu_model/docs/my_design.md) - 主设计约束，尤其是 trace 只消费模型事件
- [docs/module-development-status.md](/data/gpu_model/docs/module-development-status.md) - 模块状态与缺口跟踪

## Dependencies and Sequence

### Milestones

1. Milestone 1: 建立 cycle timeline 真值表
   - Phase A: 盘点 `InstructionIssue / Commit / Arrive / Stall / Barrier / WaveLifecycle` 当前字段与来源
   - Phase B: 形成“事件含义 / cycle 来源 / 是否画 slice / Perfetto canonical name”对照表

2. Milestone 2: focused tests 审计与重写
   - Phase A: 审 [tests/runtime/cycle_timeline_test.cpp](/data/gpu_model/tests/runtime/cycle_timeline_test.cpp)
   - Phase B: 审 [tests/runtime/trace_test.cpp](/data/gpu_model/tests/runtime/trace_test.cpp) 中 timeline 相关用例
   - Phase C: 删除或重写过度绑定旧实现的断言

3. Milestone 3: renderer 校准
   - Phase A: 校准 `InstructionIssue -> Commit` 配对
   - Phase B: 校准 `arrive / stall / barrier / lifecycle` marker 规则
   - Phase C: 校准 `render_duration_cycles` 和 slot-centric metadata

4. Milestone 4: representative examples 校准
   - Phase A: waitcnt-heavy examples
   - Phase B: barrier-heavy examples
   - Phase C: 记录与 focused tests 的差异并收口

5. Milestone 5: 正式文档回写
   - Phase A: 主设计文档
   - Phase B: 模块状态文档
   - Phase C: 任务计划同步

## Task Breakdown

任务 routing tag 仍保持 `coding` / `analyze` 两类，以兼容 gen-plan schema。

当前仓库执行约束：

- `coding`：由当前主会话直接实现
- `analyze`：由 Codex 执行
- analyze 使用的模型统一为 `gpt-5.4`
- 不引入其他模型或 Claude 路由分支

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | 为 cycle timeline 建立事件真值表，明确 `issue / commit / arrive / stall / barrier / lifecycle` 的正式语义 | AC-1, AC-2, AC-3, AC-4 | coding | - |
| task2 | 审计 [tests/runtime/cycle_timeline_test.cpp](/data/gpu_model/tests/runtime/cycle_timeline_test.cpp) 现有断言，标记保留/重写/删除项 | AC-1, AC-2, AC-3 | analyze | task1 |
| task3 | 审计 [tests/runtime/trace_test.cpp](/data/gpu_model/tests/runtime/trace_test.cpp) 中与 timeline 相关的断言，标记保留/重写/删除项 | AC-1, AC-2, AC-3, AC-4 | analyze | task1 |
| task4 | 根据真值表重写 focused tests，形成稳定的顺序与命名回归 | AC-1, AC-2, AC-3, AC-4 | coding | task2, task3 |
| task5 | 校准 [src/debug/timeline/cycle_timeline.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline.cpp) 中 issue/commit 配对和 marker 分类规则 | AC-1, AC-2, AC-3, AC-4 | coding | task4 |
| task6 | 校准 `render_duration_cycles`、slot metadata 和 timeline 导出字段稳定性 | AC-1, AC-4 | coding | task5 |
| task7 | 用 representative examples 做 waitcnt-heavy / barrier-heavy 肉眼校准并记录差异 | AC-5 | analyze | task5 |
| task8 | 把最终语义、限制和状态回写到正式文档 | AC-6 | coding | task7 |

## Claude-Codex Deliberation

### Agreements

- trace / timeline 必须是消费层，不能承担业务逻辑
- `cycle` 字段必须被定义为 modeled cycle，而不是 wall-clock
- 本阶段应优先修正时间线表达准确性，而不是继续扩展新格式
- waitcnt-heavy 和 barrier-heavy path 是第一批最有价值的校准对象

### Resolved Disagreements

- 无实质分歧。当前草稿与仓库主线约束一致，已直接收敛为计划。

### Convergence Status

- Final Status: `converged`

## Pending User Decisions

- 当前无阻塞性用户决策。
- 默认采用的执行选择：
  - representative examples 先做最小代表集
  - 至少覆盖 1 组 waitcnt-heavy 和 1 组 barrier-heavy
  - 若在这两组中发现表达缺口，再决定是否扩展到更多 examples
- 决策状态：已按默认值收敛，可直接进入下一步执行

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as `AC-`, `Milestone`, `Step`, `Phase`, or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

