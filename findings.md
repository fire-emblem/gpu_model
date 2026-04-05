# 发现与决策

## 需求
- 将项目架构收口为用户认可的目标：
  - `HipRuntime`：与 AMD HIP runtime 对齐的兼容层
  - `ModelRuntime`：项目核心实现
  - `ExecEngine`：执行核心
- 不再把 `interposer` 当作独立架构层。
- push 前门禁改成轻量 smoke，全量 gate 保留手动执行。

## 研究发现
- `HipInterposerState` 原本只是对 `RuntimeSession` 的薄包装，已经不值得单独保留。
- `hip_interposer.cpp` 中的多数关键路径可以安全收口到 `HipRuntime`：
  - kernel 注册
  - launch config push/pop
  - memory ops
  - device query
  - stream/event
  - last error
  - executable launch
- 删除 `HipInterposerState` 后，`HipInterposerStateTest.*:InterposerCTS/*:InterposerFeatureCTS/*` 全量通过。
- 轻量门禁 smoke 覆盖足以快速发现架构层回归：
  - `RuntimeNamingTest.*`
  - `RuntimeProgramCompatibilityAliasTest.*`
  - `HipInterposerStateTest.RunsHipHostExecutableThroughLdPreloadInterposer`
  - `HipRuntimeTest.LaunchKernelCanReadMaterializedDataPool`
  - `ModelRuntimeCoreTest.SimulatesMallocMemcpyLaunchAndSynchronizeFlow`
  - `TraceTest.NativePerfettoProtoContainsHierarchicalTracksAndEvents`
- `ExecEngine` 渐进迁移可安全分三步完成：
  - 先引入公开名字和 shim
  - 再切 high-level 头文件与测试使用
  - 最后切物理实现文件与底层类名
- 当前 docs 命中分析显示：
  - 对外文档基本已收口
  - 剩余旧名主要集中在 `docs/superpowers/plans`、`docs/superpowers/specs`、`docs/plans`
- example runner 默认写回仓库内 `examples/*/results` 是工作树反复变脏的根因
- 已收口为：
  - 默认写到 `.cache/example-results/<example-name>/`
  - 只有显式设置 `GPU_MODEL_EXAMPLE_RESULTS_MODE=repo` 才写回仓库快照目录

## 技术决策
| 决策 | 理由 |
|------|------|
| 删除 `HipInterposerState` | 已无主路径使用点，只会制造额外层级 |
| `hip_interposer.cpp` 归属到 `HipRuntime` 语义 | 满足“HIP 兼容层就是对外 C 接口”目标 |
| `RuntimeEngine` 渐进改名为 `ExecEngine` | 名字更准确，且可通过兼容别名平滑迁移 |
| 保留 `runtime_engine.h` 兼容 shim | 降低一次性全仓重命名风险 |
| 轻量 pre-push + 手动 full gate | 平衡开发效率与验证覆盖 |
| 历史存档文档中的旧名也开始统一替换 | 当前主线已经稳定，继续保留旧名的成本高于保留原貌的收益 |
| example 默认结果改写到 `.cache/example-results` | 避免日常运行污染工作树，同时保留显式快照刷新模式 |
| 当前对外架构文档不再把 `hip_interposer` 当模块名 | 文件名可以保留历史字样，但架构语义必须归到 `HipRuntime` |
| 暂不系统清理 docs 存档中的历史旧名 | 历史计划/spec 属于存档信息，优先保证主代码和关键文档收口 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
| preload 测试在 ASan build 下失败 | 将 `libasan` 优先加入 `LD_PRELOAD` |
| `HipRuntime` 兼容态内存与 `ModelRuntime` 自带 memory 混淆 | 为 `HipRuntime` 显式提供 `compatibility_memory()` |
| `ExecEngine` 迁移初期 include/类型替换容易误伤 | 先引入别名与 shim，再分批替换高层使用点 |

## 资源
- 关键代码：
  - `src/runtime/hip_interposer.cpp`
  - `src/runtime/hip_runtime.cpp`
  - `src/runtime/core/model_runtime.cpp`
  - `src/runtime/exec_engine.cpp`
- 关键脚本：
  - `scripts/run_push_gate.sh`
  - `scripts/run_push_gate_light.sh`
  - `.githooks/pre-push`

## 视觉/浏览器发现
- 本阶段无浏览器/视觉外部信息输入。
