# 进度日志

## 会话：2026-04-05

### 阶段 1：需求与架构确认
- **状态：** complete
- **开始时间：** 2026-04-05
- 执行的操作：
  - 明确最终主线为 `HipRuntime / ModelRuntime / ExecEngine`
  - 明确 `hip_interposer.cpp` 只是 `HipRuntime` 的 C ABI 入口实现载体
  - 确认 `HipInterposerState` 不再保留
- 创建/修改的文件：
  - `docs/runtime-layering.md`
  - `docs/module-development-status.md`
  - `docs/my_design.md`

### 阶段 2：兼容层收口
- **状态：** complete
- 执行的操作：
  - 将 `hip_interposer.cpp` 中关键兼容逻辑迁移到 `HipRuntime`
  - 删除 `HipInterposerState` 头/实现与 CMake 引用
  - 迁移相关 CTS / feature CTS / state tests
- 创建/修改的文件：
  - `src/runtime/hip_interposer.cpp`
  - `src/runtime/hip_runtime.cpp`
  - `include/gpu_model/runtime/hip_runtime.h`
  - `tests/runtime/hip_interposer_state_test.cpp`
  - `tests/runtime/hip_cts_test.cpp`
  - `tests/runtime/hip_feature_cts_test.cpp`

### 阶段 3：门禁与命名收口
- **状态：** complete
- 执行的操作：
  - 将 pre-push 改为轻量 smoke 门禁
  - 保留 `run_push_gate.sh` 作为手动 full gate
  - 引入 `ExecEngine` 公开名字和兼容 shim
  - 将实现文件切到 `src/runtime/exec_engine.cpp`
- 创建/修改的文件：
  - `.githooks/pre-push`
  - `scripts/run_push_gate_light.sh`
  - `scripts/README.md`
  - `include/gpu_model/runtime/exec_engine.h`
  - `include/gpu_model/runtime/runtime_engine.h`
  - `src/runtime/exec_engine.cpp`

### 阶段 4：验证与推送
- **状态：** complete
- 执行的操作：
  - 多次运行 release / asan 定向验证
  - 运行 full gate 并确认通过
  - 运行 light gate 并确认通过
  - 已推送多次主分支更新
- 创建/修改的文件：
  - `results/push-gate/`（门禁日志）
  - `results/push-gate-light/`（轻量门禁日志）

### 阶段 5：交付与收尾
- **状态：** complete
- 执行的操作：
  - 将关键文档中的 `RuntimeEngine` 收口为 `ExecEngine`
  - 将物理实现文件切到 `src/runtime/exec_engine.cpp`
  - 保留 `runtime_engine.h` 兼容 shim
  - 检查工作树，确认当前代码变更已提交并推送，剩余主要是 examples 产物与构建噪音
  - 确认当前不继续系统清理历史存档文档中的旧命名
  - 进一步修正 `runtime-layering.md`，明确 `hip_interposer.cpp` 仅是 `HipRuntime` 的 C ABI 入口实现载体，不是模块名
- 创建/修改的文件：
  - `docs/runtime-layering.md`
  - `docs/module-development-status.md`
  - `docs/my_design.md`
  - `docs/memory-hierarchy-interface-reservation.md`
  - `include/gpu_model/runtime/exec_engine.h`
  - `include/gpu_model/runtime/runtime_engine.h`
  - `src/runtime/exec_engine.cpp`

### 阶段 6：历史文档清理
- **状态：** in_progress
- 执行的操作：
  - 盘点 docs 中旧命名引用
  - 判断当前对外文档已基本收口，剩余主要是历史存档
  - 开始对 `docs/superpowers/plans`、`docs/superpowers/specs`、`docs/plans` 做机械术语替换
- 创建/修改的文件：
  - `docs/superpowers/plans/*`
  - `docs/superpowers/specs/*`
  - `docs/plans/*`

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| release 定向 CTS/interposer | `HipInterposerStateTest.*:InterposerCTS/*:InterposerFeatureCTS/*` | 全部通过 | 44 tests passed | 通过 |
| asan preload 定向 | 5 个 `HipInterposerStateTest.*LdPreloadInterposer` | 全部通过 | 5 tests passed | 通过 |
| full gate | `scripts/run_push_gate.sh` | release/debug/examples 全绿 | 通过 | 通过 |
| light gate | `scripts/run_push_gate_light.sh` | release/debug smoke 全绿 | 通过 | 通过 |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-04-05 | `ASan runtime does not come first` | 1 | 调整 preload 测试命令 |
| 2026-04-05 | `.git/index.lock` 阻塞 commit | 1 | 清理 stale lock 后重试 |
| 2026-04-05 | `exec_engine.h` 自包含错误 | 1 | 恢复为兼容包含并继续迁移 |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 6 |
| 我要去哪里？ | 提交并推送历史文档清理结果 |
| 目标是什么？ | 完成历史文档旧命名清理批次 |
| 我学到了什么？ | `HipInterposerState` 已不再需要，`ExecEngine` 渐进改名可行 |
| 我做了什么？ | 见上方阶段记录 |
