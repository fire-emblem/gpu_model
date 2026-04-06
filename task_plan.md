# 任务计划：当前正式任务主线

## 目标
将历史任务与设计主题提炼为当前正式 task list，只保留仍需持续推进的主线事务；历史计划文件仅作为 archive 参考，不再承担现行待办职责。

## 当前阶段
阶段 4

## 各阶段

### 阶段 1：正式任务收口
- [x] 审核历史 plans/specs，识别仍然有效的未完成主题
- [x] 将已完成的临时问题和过渡方案从当前 task list 中移除
- [x] 将剩余事项归并为正式长期 track
- **状态：** complete

### 阶段 2：正式设计收口
- [x] 将 runtime 分层收口为 `HipRuntime -> ModelRuntime -> ExecEngine`
- [x] 将 trace、functional、cycle、stats 的稳定设计约束写入主设计文档
- [x] 将模块状态文档与正式 task list 完全对齐
- **状态：** complete

### 阶段 3：主线实现推进
- [ ] 完成 `trace canonical event model` 收口
- [ ] 完成 `trace unified entry` 收口
- [ ] 完成 `functional mt` scheduler 公平性、竞争行为与可解释性收口
- [ ] 完成 `cycle` 路径 stall taxonomy 与 `ready -> selected -> issue` 观测语义收口
- [ ] 完成 `ProgramCycleStats` 与当前模型时间语义的一致性校准
- **状态：** pending

### 阶段 4：验证与资产整理
- [ ] 完成 `examples` 剩余分批全量检查
- [ ] 继续清理明显过时、已完成且误导当前主线的历史计划文件
- [ ] 将 archive / active / reference 的文档边界继续收紧
- **状态：** in_progress

### 阶段 5：交付与维护
- [ ] 保持 `task_plan.md`、`docs/my_design.md`、`docs/runtime-layering.md`、`docs/module-development-status.md` 同步
- [ ] 将新增实现结果及时回写到正式设计和状态文档
- **状态：** pending

## 当前正式任务清单
1. `Trace canonicalization`
   - 目标：让 text trace、JSON trace、timeline、Perfetto 全部建立在同一 typed event 解释之上。
   - 当前缺口：producer、test、sink 仍有部分入口未统一到 canonical 语义工厂。

2. `Trace unified entry`
   - 目标：消除 producer/test 侧散落的原始 trace message 拼装，统一为 builder/factory 入口。
   - 当前缺口：仍有历史路径直接依赖 message 文本语义。

3. `Functional MT scheduler semantics`
   - 目标：让 `functional mt` 的 wave 级调度、公平性、等待恢复与竞争行为可解释、可验证。
   - 当前缺口：正确性主干已具备，但公平性与调度解释仍未完全收口。

4. `Cycle observability`
   - 目标：让 `cycle` 路径上的 stall taxonomy、`ready / selected / issue` 关系、slot/timeline 观察面稳定可解释。
   - 当前缺口：结果型正确性已有，但可观测语义与稳定分类还需继续收口。

5. `ProgramCycleStats calibration`
   - 目标：基于“模型 cycle，而非真实物理时间戳”的统一定义，继续校准程序级 cycle 统计。
   - 当前缺口：需要继续统一 functional/cycle 的统计口径与解释边界。

6. `Examples verification`
   - 目标：完成 examples 全量分批检查，并把剩余问题归档为正式缺口，而不是临时口头结论。
   - 当前缺口：`08/11` 已关闭，但 examples 全量批次仍未彻底收口。

7. `Docs asset cleanup`
   - 目标：继续删除明显误导主线的历史任务文件，并把仍有价值的内容提炼进正式设计文档。
   - 当前缺口：archive 仍偏多，部分旧计划虽然已完成但尚未清理。

## 已明确关闭的历史事务
1. `examples/08 mt Perfetto` 指令切片缺失问题已修复。
2. `examples/11` 编译问题已修复。
3. `timeline.perfetto.pb` 已移出正式用户产物路径。
4. `.cache/example-results` 默认结果路径已移除。
5. `HipInterposerState` 已删除，其职责已并入 `HipRuntime`。

## 正式设计约束摘要
1. `HipRuntime` 是 AMD HIP runtime 兼容层；`ModelRuntime` 是项目核心 runtime；`ExecEngine` 是 `ModelRuntime` 内部执行主链。
2. `trace` 只消费和序列化模型事件，不驱动业务逻辑。
3. `st`、`mt`、`cycle` 的 trace `cycle` 都是模型时间，不是物理真实执行时间戳。
4. `functional st` 采用确定性的 issue quantum 语义；`functional mt` 保留 runnable wave 竞争；`cycle` 明确区分 `ready`、`selected`、`issue`。
5. `cycle model` 是唯一的 cycle 模式，不再拆分 cycle `st/mt`。
6. 历史 plans/specs 只能作为背景材料，当前规范以正式设计文档为准。

## 备注
- `docs/other_model_design/` 保持不动，继续作为外部参考。
- 以后若有新主线任务，先写进本文件，再扩展实现与测试。
