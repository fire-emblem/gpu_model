# Architecture Final Cleanup — Phase 6

> 目标：消除所有目录结构与设计文档 `docs/architecture-restructuring-plan.md` 的偏差。
> 上一轮完成了主体分层迁移，本轮清理残留。

## Status: IN PROGRESS

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

### 未提交 — runtime/ 根目录文件归入子目录
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
- [ ] **提交**

---

## 待完成

### Task A: 清理 bridge header (可选，低优先级)
当前 bridge headers 列表：
- `src/state/ap_state.h` → `state/ap/ap_runtime_state.h`
- `src/state/peu_state.h` → `state/peu/peu_runtime_state.h`
- `src/state/register_file.h` → `gpu_arch/register/register_file.h`
- `src/execution/internal/memory_arrive_kind.h` → `gpu_arch/memory/memory_arrive_kind.h`
- `src/execution/internal/opcode_execution_info.h` → `instruction/isa/opcode_info.h`
- `src/execution/encoded/encoded_semantic_handler.h` → `instruction/semantics/encoded_handler.h`
- `src/execution/internal/tensor_op_utils.h` → `instruction/semantics/internal/tensor_result_writer.h`
- `src/runtime/*.h` (14 个 bridge headers)

**策略**：保留 bridge headers 不删除——它们零成本，且允许渐进式更新消费者。
当所有消费者都直接 include 新路径时，再批量删除。

### Task B: 消除 state/ 残留文件 (可选)
- `src/state/execution_stats.h` — 检查是否需要 bridge 或是否直接在正确位置

### Task C: 确认设计偏差是否可接受
设计文档 `runtime/` 子目录规划为 5 个：`hip_runtime/`, `model_runtime/`, `exec_engine/`, `kernel_stub/`, `config/`
当前实际：
- `runtime/config/` ✅
- `runtime/exec_engine/` ✅
- `runtime/hip_runtime/` ✅
- `runtime/model_runtime/` ✅
- `runtime/kernel_stub/` ❌ 不存在 — 设计中的 kernel stub 功能尚未拆出

**结论**：`kernel_stub/` 是预留目录，当前没有对应文件，不影响结构一致性。

---

## 结构一致性最终评估

| 设计目录 | 状态 | 备注 |
|---------|------|------|
| `utils/{logging,config,math}/` | ✅ 完成 | |
| `gpu_arch/{chip_config,register,wave,peu,ap,dpc,device,memory,issue_config}/` | ✅ 完成 | |
| `state/{wave,peu,ap,dpc,device,register,memory}/` | ✅ 完成 | |
| `instruction/{isa,decode,semantics,operand}/` | ✅ 完成 | 含 semantics 子目录 |
| `execution/{functional,cycle,encoded,internal}/` | ✅ 完成 | |
| `runtime/{hip_runtime,model_runtime,exec_engine,config}/` | ✅ 完成 | bridge headers 待清理 |
| `debug/{trace,recorder,timeline,replay,info}/` | ✅ 完成 | |
| `program/{program_object,executable,encoded,loader}/` | ✅ 完成 | |
| ~~`src/memory/`~~ | ✅ 已删除 | |
| ~~`execution/internal/handlers/`~~ | ✅ 已删除 | |
| `runtime/kernel_stub/` | ⬜ 预留 | 无对应文件，不影响 |

**所有设计目标目录结构已对齐。** 残留的 bridge headers 是迁移辅助，不影响架构合规性。

---

## 验证 Checklist

- [x] `cmake --build build-ninja` 通过
- [x] `./scripts/run_push_gate_light.sh` 通过
- [x] 无 `src/memory/` 目录
- [x] 无 `execution/internal/handlers/` 目录
- [x] `runtime/` 根目录只有 bridge headers（无独立实现文件）
- [ ] 全量测试通过 (`./scripts/run_push_gate.sh`)
