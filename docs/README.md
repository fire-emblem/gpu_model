# `docs`

这是项目文档入口页。

请按下面三类理解当前仓库中的文档资产。

## 1. 当前主文档

这些文档应视为当前项目的现行规范或现状说明：

- [README.md](/data/gpu_model/README.md)
- [my_design.md](/data/gpu_model/docs/my_design.md)
- [runtime-layering.md](/data/gpu_model/docs/runtime-layering.md)
- [module-development-status.md](/data/gpu_model/docs/module-development-status.md)
- [memory-hierarchy-interface-reservation.md](/data/gpu_model/docs/memory-hierarchy-interface-reservation.md)

补充说明：

- [isa_coverage_report.md](/data/gpu_model/docs/isa_coverage_report.md)
  - 是生成型覆盖率快照，可作为当前覆盖状态参考
  - 但不应被误读为“完整 ISA 已全部支持”的唯一依据

## 2. 历史计划 / 实施存档

这些文档保留项目演进过程中的分析、规划和阶段性实施记录。

- [plans/README.md](/data/gpu_model/docs/plans/README.md)
- [superpowers/README.md](/data/gpu_model/docs/superpowers/README.md)

使用约定：

- 默认视为 `archive`
- 不应直接当作当前代码事实
- 如果与当前代码、主文档或测试结论冲突，应以当前代码和主文档为准
- 误导性显著、强依赖临时 branch/worktree 或引用已失效路径的历史文件，可以直接删除

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
