# 进度日志

## 归档说明

2026-04-05 至 2026-04-12 的详细会话记录已归档至 `docs/archive/progress-2026-04-05-to-04-12.md`。

## 当前状态 (2026-05-07)

### 已完成里程碑

- **架构重构** (Phase 1-6)：五层架构分层完成，trace 依赖解耦完成
- **Cycle Model Calibration**：execution 语义、recorder 协议、consumer 收口完成
- **Runtime Layering**：`HipRuntime -> ModelRuntime -> ExecEngine` 正式分层
- **ROCm 6.2 支持**：GCN 指令补齐，自定义 ROCm 路径支持
- **ISA Coverage**：1559 unique mnemonics, 82 tracked subset 100% 测试覆盖

### 当前活跃方向

- semantic calibration / cycle accuracy
- ProgramCycleStats + cycle observability
- trace canonical event model (unified entry + disable-trace boundary)
- runtime API closure (按需)
- memory pool / mmap residency + ISA validation (按需)
