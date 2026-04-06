# Cycle Timeline Accuracy Calibration Draft

## 1. 目标

本草稿用于定义下一阶段 `cycle timeline / Perfetto` 准确性校准工作的目标、范围、问题分解和执行计划。

当前目标不是重新设计整个 cycle model，而是先把以下观察面校准正确：

1. `issue / commit / arrive / stall` 的时间顺序必须严格符合当前执行模型事实
2. timeline 中的 slice 和 marker 必须只表达执行模型已经产生的事实，不能引入额外业务推断
3. `cycle` timeline 的字段、命名和渲染规则必须足够稳定，能够支撑回归测试和 examples 分析
4. `st / mt / cycle` 之间要有统一的时间线语义边界，避免把真实 wall-clock 和 modeled cycle 混淆

## 2. 背景与现状

当前仓库已经具备：

- `CycleTimelineRenderer`
- `timeline.perfetto.json`
- slot-centric track hierarchy
- `issue_cycle / commit_cycle / render_duration_cycles`
- `stall reason`
- `arrive kind / arrive progress`
- 多条 focused regressions，覆盖 `waitcnt_global`、barrier-heavy path、encoded functional、resident slot 等典型路径

但当前还没有完全闭环的点包括：

1. timeline 的每个可视事件是否都严格映射到唯一执行事实
2. `commit_cycle` 和 slice `dur` 是否总能准确表达“该指令在 timeline 上应显示的区间”
3. `arrive` 与 `stall` 的顺序、命名和可视表达是否在所有典型模式下都一致
4. `ready / selected / issue` 的语义是否会在 timeline 中被混淆
5. examples 中观察到的直觉问题，是否来自模型行为本身，还是来自 timeline 渲染选择

## 3. 约束

本阶段校准必须遵守以下约束：

1. trace / timeline 只消费事件，不驱动业务逻辑
2. `cycle` 字段表示 modeled cycle，不表示宿主真实时间戳
3. `st / mt / cycle` 下 timeline 的时间轴都必须是模型状态推进结果
4. 不允许为了“图看起来更合理”而伪造 `issue`、`commit`、`arrive` 或 `stall`
5. timeline 上的空泡应保持为空，不得画成假指令 slice

## 4. 核心问题分解

### 4.1 Issue/Commit 配对是否准确

当前 `cycle_timeline.cpp` 通过：

- `InstructionIssue`
- `Commit`

来生成指令 slice。

需要确认：

1. 所有应该进入 slice 的指令都一定有合法配对
2. 不应该进入 slice 的事件不会误生成 slice
3. `s_waitcnt` 这类纯同步控制指令是否继续不画 slice，还是需要单独 marker 语义
4. `render_duration_cycles` 是否应继续等于 quantized issue duration，还是要支持更细粒度来源

### 4.2 Arrive 的时间与可见性

需要确认：

1. `arrive` 的 `cycle` 是否总是来自异步完成时点，而不是 resume 时点
2. `arrive_progress=still_blocked/resume` 是否在 timeline 中被稳定区分
3. 同一 wave 的多次 `arrive` 是否会在 Perfetto 上产生错误覆盖或顺序错乱

### 4.3 Stall 的分类与位置

需要确认：

1. stall marker 是“未 issue 的原因”，还是“某条已 issue 指令的延伸状态”
2. `waitcnt_global`、barrier wait、warp switch 等是否在同一渲染规则下稳定成立
3. stall marker 的 cycle 是阻塞开始时点，还是观测到阻塞的采样时点

### 4.4 Ready / Selected / Issue 边界

cycle model 里已经明确：

- `ready != selected != issue`

因此 timeline 必须避免以下误导：

1. 把 `ready` wave 提前画成 instruction issue
2. 把 scheduler selection 误画成真实执行开始
3. 把 `arrive` 后立即可恢复误画成“同 cycle 已 issue”

### 4.5 Functional / Cycle 之间的统一性

尽管 `Functional` 和 `Cycle` 的内部调度不同，但 timeline 需要满足：

1. 基本字段和命名体系统一
2. `issue / commit / arrive / stall` 的因果语义统一
3. 差异只来自执行模型，不来自导出器行为

## 5. 校准范围

本阶段优先覆盖以下四类场景：

### A. 单 wave 标量/向量指令

目标：

- 锁定连续 issue 间隔
- 锁定 commit 配对
- 锁定基础 slice duration

代表测试：

- dense scalar
- simple vector alu

### B. 单 wave 异步 memory + waitcnt

目标：

- 锁定 `issue -> stall -> arrive -> resume issue`
- 锁定 `arrive_progress`
- 锁定 `waitcnt` 不被误渲染成普通指令 slice

代表测试：

- `BuildCycleOrderingKernel`
- encoded waitcnt cases

### C. 多 wave / 同 PEU 调度

目标：

- 锁定 `wave_switch_away`
- 锁定 slot identity
- 锁定 ready / selected / issue 不混淆

代表测试：

- same-PEU sibling
- multi-wave dense scalar

### D. Barrier-heavy path

目标：

- 锁定 `barrier_arrive / barrier_release`
- 锁定 barrier wait stall 命名
- 锁定 barrier release 后的后续 issue 顺序

代表测试：

- shared barrier cycle kernel
- block reduction / shared reverse / conditional multibarrier

## 6. 输出物定义

本阶段输出物限定为：

1. focused regression tests
2. 必要的 timeline renderer 修正
3. 必要的 trace event 字段修正
4. examples 校准记录
5. 文档更新

不在本阶段输出的内容：

1. 全新的 Perfetto 格式
2. protobuf exporter 主线恢复
3. 大规模 UI 优化
4. 真实硬件完全拟合

## 7. 执行计划

### Step 1: 建立 timeline 真值表

为以下事件建立统一真值定义：

- `InstructionIssue`
- `Commit`
- `Arrive`
- `Stall`
- `Barrier Arrive`
- `Barrier Release`
- `WaveLaunch`
- `WaveExit`
- `WaveSwitchAway`

每个事件明确：

1. 来源执行阶段
2. 应记录的 cycle
3. 是否渲染为 marker
4. 是否参与 slice 配对
5. 在 Perfetto 上的 canonical name / category / args

### Step 2: 检查当前 focused tests 与真值表是否一致

优先审查：

- `tests/runtime/cycle_timeline_test.cpp`
- `tests/runtime/trace_test.cpp`
- `tests/cycle/*`

处理原则：

- 测试意图合理但断言过时：重写测试
- 测试意图不再合理：删除或降级
- 实现不符合真值表：修实现

### Step 3: 校准 `cycle_timeline.cpp`

重点审查：

1. `InstructionIssue -> Commit` 配对
2. `s_waitcnt`、barrier、arrive、stall 的特殊处理
3. `render_duration_cycles` 的来源和 quantize 规则
4. marker 与 slice 是否会重复表达同一事实

### Step 4: 用 representative examples 做肉眼校准

优先 examples：

1. `02`
2. `08`
3. `11`
4. barrier-heavy / waitcnt-heavy representative cases

目标：

- 验证 timeline 是否直观且不违背模型事实
- 记录是否仍存在“看起来不对但语义其实正确”的点

### Step 5: 文档回写

把最终结论回写到：

- `docs/my_design.md`
- `docs/module-development-status.md`
- 必要时 `task_plan.md`

## 8. 验收标准

本阶段验收标准：

1. `cycle_timeline_test.cpp` 的关键顺序回归全部通过
2. `trace_test.cpp` 中与 timeline 命名、slot、arrive、stall 相关回归全部通过
3. 至少一组 waitcnt-heavy 和一组 barrier-heavy representative examples 结果经肉眼校准无明显语义冲突
4. 文档中明确写出 `issue / commit / arrive / stall` 的 timeline 语义

## 9. 风险

### 风险 1：测试断言依赖当前实现细节而不是真实语义

对策：

- 先写真值表，再判断测试该保留还是重写

### 风险 2：修 timeline 时误改执行逻辑

对策：

- 坚持 trace / timeline 只消费事件
- 不在 renderer 内新增业务语义推断

### 风险 3：examples 肉眼直觉与模型语义冲突

对策：

- 优先修正命名和可视表达
- 只有在证据证明执行模型错误时才改 engine

## 10. 当前建议的第一批落点

建议第一批从下面三个点开始：

1. `CycleTimelineTest.PerfettoDumpPreservesCycleIssueAndCommitOrdering`
2. `TraceTest` 中 encoded/functional waitcnt timeline cases
3. `cycle_timeline.cpp` 中 `InstructionIssue -> Commit` slice 生成规则

原因：

- 这三处直接决定 `issue / commit / arrive / stall` 的因果链是否可信
- 修好后再看 barrier-heavy 和 slot-centric 细节，成本更低
