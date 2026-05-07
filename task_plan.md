# 任务计划

## 归档说明

阶段 1-8 的详细任务记录已归档至 `docs/archive/task-plan-phases-1-8.md`。

## 当前主目标

以 **cycle time / cycle model accuracy** 为第一优先级推进，functional model 作为参考基线；runtime 与 ISA 相关补项按 cycle 主线实际依赖按需插入。

## 活跃 Tracks

1. **semantic calibration / cycle accuracy** — 校准 `st/mt/cycle` 执行结果与设计语义
2. **ProgramCycleStats + cycle observability** — 细化 stall taxonomy、ready/selected/issue 解释面
3. **trace canonical event model** — unified entry + disable-trace boundary 收口
4. **runtime API closure** — 按 cycle 主线需求驱动补充
5. **memory pool / mmap + ISA validation** — 按 cycle 主线需求驱动补充
6. **docs / status tracking** — 保持模块交互关系、开发计划和状态文档对齐

## 正式设计约束

1. `HipRuntime -> ModelRuntime -> ExecEngine`：runtime 正式分层，不允许反向依赖
2. trace 只消费和序列化模型事件，不驱动业务逻辑
3. `cycle` 是模型时间，不是物理真实执行时间戳
4. functional st: 确定性 issue quantum；functional mt: runnable wave 竞争；cycle: 区分 ready/selected/issue
5. cycle model 唯一时序模型，不拆分 cycle st/mt
6. `arrive_resume` = eligible，不保证同 cycle issue；`WaveStep` = 真正 issue

## 已完成里程碑

- 阶段 1-8 全部 complete（详见归档）
- 架构重构 Phase 1-6 完成（五层分层 + trace 解耦）
- Cycle Model Calibration 完成（execution 语义 + recorder 协议）
- ROCm 6.2 支持完成（GCN 指令补齐 + 自定义路径）
