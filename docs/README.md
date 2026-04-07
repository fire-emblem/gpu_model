# `docs`

这是项目文档入口页。

请按下面三类理解当前仓库中的文档资产。

## 1. 当前主文档

这些文档应视为当前项目的现行规范或现状说明：

- [README.md](/data/gpu_model/README.md)
- [task_plan.md](/data/gpu_model/task_plan.md)
- [my_design.md](/data/gpu_model/docs/my_design.md)
- [cycle-issue-eligibility-policy.md](/data/gpu_model/docs/cycle-issue-eligibility-policy.md)
- [runtime-layering.md](/data/gpu_model/docs/runtime-layering.md)
- [module-development-status.md](/data/gpu_model/docs/module-development-status.md)
- [memory-hierarchy-interface-reservation.md](/data/gpu_model/docs/memory-hierarchy-interface-reservation.md)

补充说明：

- [isa_coverage_report.md](/data/gpu_model/docs/isa_coverage_report.md)
  - 是生成型覆盖率快照，可作为当前覆盖状态参考
  - 但不应被误读为“完整 ISA 已全部支持”的唯一依据
- [cycle-issue-eligibility-policy.md](/data/gpu_model/docs/cycle-issue-eligibility-policy.md)
  - 固定 `cycle model` 中 `eligible -> selected -> issue` 的语义边界
  - 记录参考资料中的 `round_robin / oldest_first` 与当前实现的映射关系

正式阅读顺序：

1. [task_plan.md](/data/gpu_model/task_plan.md)
   - 看当前还要做什么
2. [my_design.md](/data/gpu_model/docs/my_design.md)
   - 看当前正式设计约束和模块语义
3. [cycle-issue-eligibility-policy.md](/data/gpu_model/docs/cycle-issue-eligibility-policy.md)
   - 看 `cycle issue` 的正式语义边界与参考口径
4. [runtime-layering.md](/data/gpu_model/docs/runtime-layering.md)
   - 看 runtime 正式分层
5. [module-development-status.md](/data/gpu_model/docs/module-development-status.md)
   - 看模块完成度、缺口和推进顺序

## 2. 历史计划 / 实施存档

这些文档保留项目演进过程中的分析、规划和阶段性实施记录。

- [plans/README.md](/data/gpu_model/docs/plans/README.md)
- [superpowers/README.md](/data/gpu_model/docs/superpowers/README.md)

使用约定：

- 默认视为 `archive`
- 不应直接当作当前代码事实
- 如果与当前代码、主文档或测试结论冲突，应以当前代码和主文档为准
- 误导性显著、强依赖临时 branch/worktree 或引用已失效路径的历史文件，可以直接删除

当前仍可从 archive 中提炼、但已经回写进正式文档的主题主要包括：

- LLVM / AMDGPU artifact ingestion 与 code object 主线
- program-object decode / project disasm 的长期方向
- memory pool / segment loading 的接口边界
- PEU / wave issue model 与 `ready / selected / issue` 语义
- naive cycle model 的“少量稳定 knob + 可解释相对差异”原则
- functional `mt` scheduler、公平性与等待恢复语义
- trace canonical / unified entry / stall taxonomy / slot timeline

因此：

- archive 里的这些主题现在可以作为背景材料看
- 但不需要再从 archive 反推当前项目到底“应该怎么做”

## 3. 外部参考材料

这些文档主要用于方案对比和思路参考，不直接定义本仓实现。

- [other_model_design/README.md](/data/gpu_model/docs/other_model_design/README.md)

使用约定：

- 默认视为 `reference`
- 可作为设计背景与对比材料
- 不要求与当前代码同步

## 4. 清理原则

后续若继续整理 docs 资产，按下面顺序判断：

1. 是否仍是当前主线规范的一部分
2. 若不是，是否有历史留档价值
3. 若既不是主线规范，也几乎没有历史价值，才考虑删除

额外约束：

- 根目录 `docs/` 下不再保留泛名、容易误导的计划文件
- 同类计划或实施材料统一放入 `docs/plans/` 或 `docs/superpowers/`
- 若 archive 中某主题仍有现实价值，优先将其蒸馏回当前主文档，而不是继续扩张 archive 的权威性
