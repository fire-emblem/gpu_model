# Runtime Memory API Closure Plan

## Goal Description

收口 `HipRuntime / ModelRuntime / RuntimeSession / DeviceMemoryManager` 的同步 memory API 主路径，建立一组轻量、系统化、无需 kernel launch 的 runtime memory 验证矩阵，作为后续 `mmap residency`、ISA asm-kernel 验证和更大规模 HIP executable 覆盖的前置基线。

本轮只关注：

- `hipMalloc / hipMallocManaged / hipFree`
- `hipMemcpy / hipMemcpyAsync` 的同步兼容语义
- `hipMemset / hipMemsetAsync / hipMemsetD8 / hipMemsetD16 / hipMemsetD32`
- compatibility window 地址分类与 `model_addr` 映射
- `RuntimeSession` 与 `DeviceMemoryManager` 的轻量错误语义和矩阵测试

本轮不扩展：

- kernel launch 功能
- stream 并发语义
- graph / event / async overlap
- 大规模 `.out` / ISA / cycle model 主题

## Acceptance Criteria

- AC-1: 同步 runtime memory API 形成明确、稳定的能力矩阵
  - Positive Tests:
    - `hipMalloc / hipMallocManaged / hipFree` 的 global/managed 分配释放路径可稳定通过 focused tests
    - `hipMemcpy` 对 `HtoD / DtoH / DtoD` 的同步主路径可稳定通过 focused tests
    - `hipMemset / hipMemsetD8 / hipMemsetD16 / hipMemsetD32` 的填充值语义可稳定通过 focused tests
  - Negative Tests:
    - 非法 device pointer 不得被静默接受
    - 不支持或不合法的 memcpy kind 不得伪成功

- AC-2: `RuntimeSession` 与 `DeviceMemoryManager` 的 compatibility 指针语义被系统化锁定
  - Positive Tests:
    - global/managed compatibility window 分类稳定
    - `ResolveDeviceAddress`、allocation lookup、window classify 一致
    - `committed_bytes` 与 allocate/free/reset 行为一致
  - Negative Tests:
    - 非 window 指针不得被误判为 device pointer
    - 已释放指针不得继续映射到旧 allocation

- AC-3: 建立轻量 runtime memory 测试矩阵，避免依赖 kernel launch
  - Positive Tests:
    - 按 API 分类的 focused tests 覆盖 `malloc/free + memcpy + memset + pointer classify`
    - 测试命名和分层能直接体现能力边界
  - Negative Tests:
    - 关键 runtime memory 行为只能通过 HIP executable 或 kernel launch 间接验证

- AC-4: 正式文档与状态看板回写本轮结果
  - Positive Tests:
    - `task_plan.md`、`docs/module-development-status.md`、必要时 `docs/my_design.md` 同步更新
    - 明确记录本轮已完成和仍未覆盖的 runtime memory 边界
  - Negative Tests:
    - 文档仍把 runtime memory 能力描述为“分散 case-by-case 覆盖”

## Path Boundaries

### Upper Bound

- 形成一套清晰的 runtime memory matrix
- 统一 `HipRuntime -> RuntimeSession -> DeviceMemoryManager -> ModelRuntime` 的同步主路径约束
- focused tests 覆盖同步 memory API 的主要成功/失败路径
- 文档同步完成

### Lower Bound

- 至少补齐同步 `malloc/free/memcpy/memset` 的 focused tests
- 锁定 compatibility window / pointer classify / committed_bytes 的核心语义
- 回写正式文档

### Allowed Choices

- Can use:
  - 直接扩展现有 `tests/runtime/device_memory_manager_test.cpp`
  - 直接扩展现有 `tests/runtime/hip_runtime_test.cpp`
  - 直接扩展现有 `tests/runtime/hip_runtime_abi_test.cpp`
  - 轻量重构 `RuntimeSession` / `DeviceMemoryManager` / `HipRuntime` 主路径
- Cannot use:
  - 为了测试方便引入新的 fake 语义而改变对外 API 行为
  - 让 kernel launch 成为 runtime memory matrix 的必要依赖
  - 把 async 语义扩展成真正并发实现

## Dependencies and Sequence

1. 审计当前 runtime memory 实现和已有测试覆盖
2. 建立缺口清单与最小矩阵
3. 先补 focused tests
4. 再补实现缺口
5. 回写正式文档和状态看板

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | 审计当前 runtime memory API 与已有测试矩阵，列出缺口 | AC-1, AC-2, AC-3 | analyze | - |
| task2 | 为同步 `malloc/free/memcpy/memset` 建立 focused tests | AC-1, AC-3 | coding | task1 |
| task3 | 校准 `RuntimeSession / DeviceMemoryManager` 的 compatibility 指针与 committed 语义 | AC-2 | coding | task2 |
| task4 | 清理和补齐 runtime memory 错误语义与 ABI 对齐边界 | AC-1, AC-2 | coding | task3 |
| task5 | 更新正式文档与状态看板 | AC-4 | coding | task4 |

## Implementation Notes

- 实现代码不要出现 `AC-` 等计划术语
- 以轻量 focused tests 为第一优先级
- 只做同步语义，不扩 async overlap
