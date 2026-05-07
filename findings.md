# 发现与决策

## 归档说明

2026-04-05 至 2026-04-12 的详细发现记录已归档至 `docs/archive/findings-through-2026-04-12.md`。

## 当前需求主题

1. 补齐 runtime 重要 API 实现（按需）
2. 建立 memory pool 与 `mmap` 映射主线（按需）
3. 基于 text asm kernel / encoded code object 做更多 ISA 验证（按需）
4. 校准 `st / mt / cycle` 执行结果与设计语义
5. 继续 semantic calibration / cycle accuracy

## 已关闭的历史发现

以下发现已全部修复或回写到正式文档：

- examples/08 mt Perfetto instruction slice 缺失 → 已修复（Commit 事件补齐）
- examples/11 编译引用旧接口 → 已修复（ExecEngine 更名完成）
- `HipInterposerState` 不再保留 → 已删除并入 `HipRuntime`
- `loguru` 统一日志 → 已完成（src/ 已统一）
- trace 可关闭且不依赖业务逻辑 → 已完成（`GPU_MODEL_DISABLE_TRACE=1`）