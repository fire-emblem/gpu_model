# Architecture Final Cleanup — Phase 6

> 目标：消除所有目录结构与设计文档 `docs/architecture-restructuring-plan.md` 的偏差。
> 上一轮完成了主体分层迁移，本轮清理残留。

## Status: COMPLETE ✓

> 所有架构目录结构已对齐；后续 final sweep 已删除剩余 trivial bridge headers，并按严格终态规则拒绝保留 reserved / placeholder 目录。

---

## 已完成 (本次 session)

### Commit dfd3cd5 — opcode_info + encoded_handler 迁移
- [x] `OpcodeExecutionInfo` / `SemanticFamily` 从 `execution/internal/` → `instruction/isa/` (Layer 1-2)
- [x] `EncodedSemanticHandler` 从 `execution/encoded/` → `instruction/semantics/` (Layer 2)
- [x] `ExecutionStats` → `state/execution_stats.h`
- [x] `barrier_state.h`, `wave_utils.h` → `state/wave/`
- [x] 原位置转为 bridge header

### Commit d9bae23 — memory/ 消除 + handlers/ 扁平化
- [x] `src/memory/{cache_model,memory_system,shared_bank_model}.cpp` → `state/memory/`
- [x] 删除 `src/memory/` 空目录
- [x] `execution/internal/handlers/` 扁平化到 `execution/internal/` 根

### Commit 1932fa0 — runtime/ 根目录文件归入子目录
- [x] `launch_config.h`, `kernel_arg_pack.h`, `kernarg_packer.*`, `launch_request.h` → `runtime/config/`
- [x] `runtime_env_config.h` → `runtime/config/`
- [x] `module_load.h`, `module_registry.h`, `runtime_session.h`, `runtime_submission_context.h` → `runtime/model_runtime/`
- [x] `program_cycle_tracker.*`, `program_cycle_stats.h` → `runtime/model_runtime/`
- [x] `device_properties.h`, `device_memory_manager.h`, `mapper.*` → `runtime/model_runtime/`
- [x] `exec_engine.h` → `runtime/exec_engine/`
- [x] `hip_runtime.h` → `runtime/hip_runtime/`
- [x] `model_runtime.h` → `runtime/model_runtime/`
- [x] 所有根目录原位置创建 bridge header
- [x] CMakeLists.txt 路径更新
- [x] 编译通过，push gate 通过
- [x] **提交** (1932fa0)

---

## Final Sweep（2026-04-14）

- [x] 删除剩余 trivial bridge headers：
  - `src/runtime/runtime_config.h`
  - `src/state/ap_state.h`
  - `src/state/register_file.h`
  - `src/execution/functional/functional_execution_mode.h`
- [x] 复核 `src/` 下不再保留仅用于迁移的根级转发头
- [x] 复核最终态不接受 `runtime/kernel_stub/` 这类预留目录
- [x] 复核当前活动计划文档不再宣称“保留 bridge headers 也算合规”

结论：

- bridge headers 只允许作为阶段性迁移手段，不属于最终态；
- `runtime/kernel_stub/` 这类 reserved 目录不再视为“未来可补”的合法终态节点；
- `src/state/execution_stats.h` 等当前保留头必须是实义拥有头，而不是转发头。

---

## 结构一致性最终评估

| 设计目录 | 状态 | 备注 |
|---------|------|------|
| `utils/{logging,config,math}/` | ✅ 完成 | |
| `gpu_arch/{chip_config,register,wave,peu,ap,dpc,device,memory,issue_config}/` | ✅ 完成 | |
| `state/{wave,peu,ap,dpc,device,memory}/` | ✅ 完成 | `state/register/` 当前未形成独立实义接口，不计入本轮终态证明 |
| `instruction/{isa,decode,semantics,operand}/` | ✅ 完成 | 含 semantics 子目录 |
| `execution/{functional,cycle,encoded,internal}/` | ✅ 完成 | |
| `runtime/{hip_runtime,model_runtime,exec_engine,config}/` | ✅ 完成 | 无根级 bridge header 残留 |
| `debug/{trace,recorder,timeline,info}/` | ✅ 完成 | 本轮终态不保留 `debug/replay/` |
| `program/{program_object,executable,encoded,loader}/` | ✅ 完成 | |
| ~~`src/memory/`~~ | ✅ 已删除 | |
| ~~`execution/internal/handlers/`~~ | ✅ 已删除 | |
| ~~`runtime/kernel_stub/`~~ | ✅ 不保留 | reserved/placeholder 目录不属于最终态 |

**当前活跃终态目录结构已按严格规则对齐。** 迁移辅助 bridge headers 不再保留于 `src/`
活跃结构中；未落地的 placeholder / reserved 目录也不再被当作“已完成结构”的一部分。

---

## 验证 Checklist

- [x] `rg -n '#include "(state/ap_state|state/register_file|runtime/runtime_config|execution/functional/functional_execution_mode)\.h"' src tests` 无结果
- [x] 无 `src/memory/` 目录
- [x] 无 `execution/internal/handlers/` 目录
- [x] 无 `runtime/kernel_stub/` 目录
- [x] 结构化残留搜索与 placeholder gate 复核通过
- [x] `./scripts/run_push_gate.sh`（本轮 final sweep 后重新验证；`debug+asan tests passed / release tests passed / all examples passed`）
