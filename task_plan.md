# 任务计划：Examples 分批检查与 Perfetto 正确性修复

## 目标
完成 examples 全量分批检查，优先修复 `08` 的 `mt` Perfetto 显示问题、`11` 编译问题，并锁定“Perfetto 必须显示每条指令 4 cycle 区间”的约束。

## 当前阶段
阶段 1

## 各阶段

### 阶段 1：需求与发现
- [x] 记录当前 cycle 增强已到的阶段性结果
- [x] 切换到 examples/Perfetto 新主题
- [x] 记录用户明确问题：`08` mt Perfetto 不正确，`11` 编译不过
- **状态：** complete

### 阶段 2：规划与分批检查
- [ ] 枚举 examples 全量检查批次
- [x] 先定位 `08` mt Perfetto 现状与预期偏差
- [x] 先定位 `11` 编译失败根因
- [x] 明确 “每条指令显示 4 cycle 区间” 的实现边界与回归方式
- **状态：** in_progress

### 阶段 3：实现修复
- [x] 修复 `08` mt Perfetto 显示
- [x] 修复 `11` 编译问题
- [x] 收口指令 4-cycle Perfetto 表现
- **状态：** complete

### 阶段 4：分批验证
- [ ] 对 examples 全量分批检查
- [ ] 更新对应 results / 结论
- [ ] 记录仍存在的问题与例外
- **状态：** pending

### 阶段 5：交付
- [ ] 汇总本轮 examples 检查结论
- [ ] 给出后续是否继续扩大 cycle 前端增强的建议
- **状态：** pending

## 关键问题
1. `08` 的 `mt` Perfetto 不正确，根因是在事件生产、timeline 序列化，还是 example 构造本身。
2. `11` 编译失败是否来自 example 代码、build target、还是环境假设。
3. `08` 即使修好指令 slice，也仍不适合作为“单 PEU 多 slot 并发”展示样例，因为其工作分布天然分散到多个 AP/PEU 且都落在 `slot=0`。

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
| 本轮优先切回 examples/Perfetto 正确性 | 这是用户当前最高优先级问题 |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| `ASan runtime does not come first` 导致 preload 测试失败 | 1 | 在测试运行命令里将 `libasan` 放到 `LD_PRELOAD` 前面 |
| `GPU_MODEL_GATE_DEBUG_ASAN_GTEST_FILTER` 默认值写法触发 shell unbound variable | 1 | 改为安全的 `:-` 默认展开 |
| `exec_engine.h` 被误改成自包含 include | 1 | 恢复为包含 `runtime_engine.h`，随后再做物理文件收口 |
| 提交时残留 `.git/index.lock` | 1 | 确认无活跃 git 进程后清理 stale lock |

## 备注
- 当前 cycle 前端阶段性增强已落地但本轮不继续扩张，先回头处理 examples/Perfetto 问题。
