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
  - `2026-04-04-trace-canonical-event-model*`
  - `2026-04-04-trace-unified-entry*`
  - `2026-04-03-perfetto-causal-cycle-stall-taxonomy*`
  - `2026-04-03-perfetto-slot-centric-timeline*`
  - `2026-04-02-functional-mt-wave-scheduler*`
  - `2026-04-01-multi-wave-dispatch-front-end-alignment*`
  - `2026-04-02-program-cycle-stats-calibration*`
  - `2026-04-02-hip-128-block-conditional-multibarrier-validation*`
- 更早或更外围的文档，默认优先按“历史实施记录 / archive”理解
- Phase 1 compatibility-wrapper 与 phase2 legacy-cleanup 阶段的高误导文档已开始直接清理；与当前主线明显脱节、只保留过渡包装层、旧公开名或已完成 cleanup 语义的文件不再继续保留
- `2026-04-12-architecture-restructure-wave1.md`
- `2026-04-12-architecture-restructure-phase3.md`
- `2026-04-13-architecture-final-cleanup.md`
  - 都应按“已完成的实施记录”阅读，不再代表当前允许保留 bridge header / reserved directory 的规范立场

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

补充说明：

- 本目录保留的是“设计与实施过程资料”，不是唯一规范源。
- 真正的当前任务列表以项目根目录 [task_plan.md](/data/gpu_model/task_plan.md) 为准。
- 即便是当前保留的 8 组活跃参考，也应优先视为“补充背景材料”，而不是必须先读它们才能理解当前规范。
- 如果这 8 组主题的稳定结论已经被正式文档充分吸收，对应文件后续也可以继续删除。
- 当前目录已不再保留 ABI closure、wave-launch summary、shared-heavy bring-up 这类已被正式状态文档吸收的历史包。
