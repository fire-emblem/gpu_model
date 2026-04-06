# `docs/plans`

这里存放 2026-03-27 到 2026-03-29 期间的历史规划文档。

用途：

- 保留当时的分析路径、拆解方式和阶段目标
- 作为重构背景资料与决策来源留档

使用约定：

- 这些文档不是当前代码结构的权威描述
- 文中出现的旧模块名、旧目录名、旧类名，视为历史上下文
- 当前主线设计以 [my_design.md](/data/gpu_model/docs/my_design.md) 为准
- 当前阶段性实施状态以 [module-development-status.md](/data/gpu_model/docs/module-development-status.md) 和 `docs/superpowers/*` 下最新 phase 文档为准
- 根目录 `docs/` 下不再保留泛名的实施计划文件（例如 `docs/plan.md`）；同类材料统一放在本目录或 `docs/superpowers/`
- 当前代码目录还未完全达到最终重构形态；看到历史计划中的旧目录、旧类名，不应据此判断“代码回退”或“重构失败”

当前清理结论：

- 本目录中除 `README.md` 外，其余文件默认视为 `archive`
- 如文件明显依赖临时 branch/worktree 或引用当前已不存在路径，可直接删除而不是继续保留

如果历史计划与当前代码不一致，应以当前代码和主线设计为准，而不是回推兼容旧命名。
