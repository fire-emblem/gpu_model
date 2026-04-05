# 任务计划：Runtime 架构收口、命名统一与 Cycle 完备性增强

## 目标
将当前项目收口到 `HipRuntime / ModelRuntime / ExecEngine` 三个清晰概念，完成门禁轻量化、关键文档同步、历史命名清理，并启动 cycle model 与真实硬件前端行为对齐的增强计划。

## 当前阶段
阶段 7

## 各阶段

### 阶段 1：需求与发现
- [x] 理解用户意图
- [x] 确定约束条件和需求
- [x] 将发现记录到 findings.md
- **状态：** complete

### 阶段 2：规划与结构
- [x] 确定技术方案
- [x] 明确两层 runtime 主线与 `ExecEngine` 目标名
- [x] 记录关键决策及理由
- **状态：** complete

### 阶段 3：实现
- [x] 删除 `HipInterposerState`
- [x] 将兼容职责收口到 `HipRuntime`
- [x] 引入 `ExecEngine` 公开命名与兼容别名
- [x] 将实现文件切到 `src/runtime/exec_engine.cpp`
- [x] 引入轻量 pre-push 门禁
- **状态：** complete

### 阶段 4：测试与验证
- [x] 验证关键 interposer/CTS/feature CTS 回归
- [x] 验证轻量门禁脚本
- [x] 多次通过 pre-push smoke / full gate
- [x] 确认当前无需继续补充一轮 examples 结果审阅
- **状态：** complete

### 阶段 5：交付
- [x] 检查剩余工作树是否仅包含 examples 产物与构建噪音
- [x] 完成关键文档的 `ExecEngine` 命名收口
- [x] 判断当前不继续系统清理历史文档中的旧命名
- [x] 向用户交付当前阶段总结与后续选项
- **状态：** complete

### 阶段 6：历史文档清理
- [x] 盘点 docs 中剩余旧命名引用并区分当前文档与历史存档
- [x] 确认当前对外文档基本已收口
- [x] 开始对历史 plans/spec 存档做机械术语替换
- [x] 提交并推送历史文档清理结果
- **状态：** complete

### 阶段 7：Cycle 完备性增强计划
- [x] 确认 cycle model 不区分 `st/mt`，保持单一硬件时序模型
- [x] 确认 trace 只负责消费 typed event，不承担业务推断
- [ ] 明确 block admit / wave generate / wave dispatch / slot bind / issue / wait / arrive / resume 的状态机边界
- [ ] 设计 `wave_generation_latency` 与 `wave_dispatch_latency` 的配置和事件落点
- [ ] 拆分实现清单，按 engine / state machine / trace regression 组织
- **状态：** in_progress

## 关键问题
1. cycle 前端状态机应先落在哪几个结构体/模块中，才能最少改动现有逻辑。
2. `wave_generation_latency` / `wave_dispatch_latency` 是否和现有 `block_launch_cycles` / `wave_launch_cycles` 合并还是独立保留。

## 已做决策
| 决策 | 理由 |
|------|------|
| 保留 `HipRuntime` 作为 AMD HIP runtime 兼容层 | 与用户的最终目标架构一致 |
| 保留 `ModelRuntime` 作为项目核心实现 | 维持核心实现与兼容层分离 |
| 将 `RuntimeEngine` 目标名收口为 `ExecEngine` | 更贴近执行核心语义，减少与 runtime 语义重叠 |
| `hip_interposer.cpp` 视为 `HipRuntime` 的 C ABI 入口实现载体 | 不再把 interposer 当独立模块 |
| pre-push 改为轻量门禁 | 缩短 push 阻塞时间，同时保留基本保护 |
| 对历史存档文档采用机械术语替换 | 当前主线已稳定，继续保留旧名只会增加理解成本 |
| cycle model 保持唯一模式，不再引入 cycle st/mt | `st/mt` 属于 functional 执行策略，不属于硬件时序模型 |
| trace 只消费 typed event，不做业务推断 | 避免展示层反向定义业务语义 |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| `ASan runtime does not come first` 导致 preload 测试失败 | 1 | 在测试运行命令里将 `libasan` 放到 `LD_PRELOAD` 前面 |
| `GPU_MODEL_GATE_DEBUG_ASAN_GTEST_FILTER` 默认值写法触发 shell unbound variable | 1 | 改为安全的 `:-` 默认展开 |
| `exec_engine.h` 被误改成自包含 include | 1 | 恢复为包含 `runtime_engine.h`，随后再做物理文件收口 |
| 提交时残留 `.git/index.lock` | 1 | 确认无活跃 git 进程后清理 stale lock |

## 备注
- 当前主分支最新已推送提交包含：
  - `Fold interposer state into hip runtime`
  - `Align runtime docs with hip runtime architecture`
  - `Switch pre-push hook to lightweight gate`
  - `Promote ExecEngine as runtime execution type`
  - `Remove RuntimeEngine shim and finalize ExecEngine rename`
  - `Update docs to use ExecEngine terminology`
  - `Normalize archived docs to ExecEngine naming`
  - `Default example outputs to local cache by default`
  - `Ignore local gate and example output directories`
- 当前剩余未提交内容主要是：
  - examples 结果产物
  - build/log/results 噪音
  - 历史文档清理结果
- 仍需避免把 examples 结果产物与构建目录噪音误提交。
