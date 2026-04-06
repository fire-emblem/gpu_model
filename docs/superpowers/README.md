# `docs/superpowers`

这里存放围绕项目主线重构、trace/perfetto、functional/cycle 语义收口的实施文档。

目录说明：

- `specs/`
  - 设计约束、命名收口、完成条件
- `plans/`
  - 对应的执行计划、阶段拆分和实施路径

当前使用约定：

- 不是本目录下所有文档都等价于“当前主规范”
- 当前更接近“仍可直接参考”的内容，主要集中在：
  - `2026-04-03-*`
  - `2026-04-04-*`
  这些 trace / Perfetto / canonical event model 相关文档
- 更早的 `2026-03-30` 到 `2026-04-02` 文档，默认优先按“历史实施记录 / archive”理解

阅读规则：

- 同名 `spec` 与 `plan` 共同描述一个重构包
- 若 `plan` 中的任务已完成，以当前代码和最近提交为准
- 历史归档计划仍保留原始任务表述，不要求逐字反映最终代码命名
- 当前项目的现行规范，仍以：
  - [README.md](/data/gpu_model/README.md)
  - [docs/my_design.md](/data/gpu_model/docs/my_design.md)
  - [docs/runtime-layering.md](/data/gpu_model/docs/runtime-layering.md)
  - [docs/module-development-status.md](/data/gpu_model/docs/module-development-status.md)
  为准
