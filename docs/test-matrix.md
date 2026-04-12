# 测试矩阵

本文档按 runtime / memory / ISA / semantic 分类整理现有测试，建立轻量级但完备的测试覆盖视图。

## 1. Runtime 测试

| 测试文件 | 覆盖范围 | 优先级 |
|----------|----------|--------|
| `runtime/hip_runtime_abi_test.cpp` | HIP C ABI 入口、LD_PRELOAD | P0 |
| `runtime/hip_cts_test.cpp` | HIP 兼容性测试套件 | P0 |
| `runtime/hip_feature_cts_test.cpp` | HIP 特性测试 | P0 |
| `runtime/exec_engine.cpp` | ExecEngine 入口 | P0 |
| `runtime/execution_stats_test.cpp` | 执行统计 | P1 |
| `runtime/device_memory_manager_test.cpp` | 设备内存管理 | P1 |
| `runtime/trace_test.cpp` | Trace 事件模型 | P0 |
| `runtime/cycle_timeline_test.cpp` | Cycle timeline 渲染 | P0 |
| `runtime/executed_flow_program_cycle_stats_test.cpp` | ProgramCycleStats | P0 |

**运行命令：**
```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeAbiTest.*:HipRuntimeTest.*:ExecEngineTest.*'
```

## 2. Memory 测试

| 测试文件 | 覆盖范围 | 优先级 |
|----------|----------|--------|
| `execution/memory_ops_test.cpp` | 内存操作语义 | P0 |
| `cycle/async_memory_cycle_test.cpp` | 异步内存 flow | P0 |
| `cycle/cache_cycle_test.cpp` | 缓存模型 | P1 |
| `cycle/constant_memory_cycle_test.cpp` | 常量内存 | P1 |
| `cycle/shared_bank_cycle_test.cpp` | 共享存储 bank 冲突 | P1 |

**运行命令：**
```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='AsyncMemoryCycleTest.*:CacheCycleTest.*:MemoryOpsTest.*'
```

## 3. ISA 测试

| 测试文件 | 覆盖范围 | 优先级 |
|----------|----------|--------|
| `isa/opcode_descriptor_test.cpp` | Opcode 描述符 | P0 |
| `instruction/instruction_decoder_test.cpp` | 指令解码 | P0 |
| `instruction/encoded_gcn_inst_format_test.cpp` | GCN 编码格式 | P0 |
| `instruction/encoded_instruction_binding_test.cpp` | 指令绑定 | P0 |
| `loader/amdgpu_code_object_decoder_test.cpp` | 代码对象解码 | P1 |
| `loader/amdgpu_obj_loader_test.cpp` | AMDGPU 对象加载 | P1 |

**运行命令：**
```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='InstructionDecoderTest.*:EncodedGcnInstFormatTest.*:OpcodeDescriptorTest.*'
```

## 4. Semantic 测试

### 4.1 Functional 语义

| 测试文件 | 覆盖范围 | 优先级 |
|----------|----------|--------|
| `functional/divrem_functional_test.cpp` | 整数除法/取模 | P0 |
| `functional/fma_loop_functional_test.cpp` | FMA 循环 | P0 |
| `functional/constant_memory_functional_test.cpp` | 常量内存访问 | P1 |
| `functional/builtin_scalarbit_functional_test.cpp` | 内置标量位操作 | P1 |

### 4.2 Cycle 语义

| 测试文件 | 覆盖范围 | 优先级 |
|----------|----------|--------|
| `cycle/cycle_smoke_test.cpp` | Cycle 模型基础 | P0 |
| `cycle/shared_barrier_cycle_test.cpp` | Barrier 同步 | P0 |
| `cycle/waitcnt_barrier_switch_focused_test.cpp` | Waitcnt/Barrier 切换 | P0 |
| `cycle/cycle_ap_resident_blocks_test.cpp` | AP resident block 管理 | P0 |

### 4.3 Handler 语义

| 测试文件 | 覆盖范围 | 优先级 |
|----------|----------|--------|
| `execution/encoded_semantic_handler_registry_test.cpp` | Handler 注册 | P0 |
| `execution/encoded_semantic_execute_test.cpp` | Handler 执行 | P0 |

**运行命令：**
```bash
# Functional
./build-ninja/tests/gpu_model_tests --gtest_filter='DivremFunctionalTest.*:FmaLoopFunctionalTest.*'

# Cycle
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleSmokeTest.*:SharedBarrierCycleTest.*:CycleApResidentBlocksTest.*'

# Handler
./build-ninja/tests/gpu_model_tests --gtest_filter='EncodedSemanticHandlerRegistryTest.*:EncodedSemanticExecuteTest.*'
```

## 5. 快速验证命令

### 轻量级 Smoke (P0 核心)
```bash
./scripts/run_push_gate_light.sh
```

### Cycle 完整验证
```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleSmokeTest.*:AsyncMemoryCycleTest.*:SharedBarrierCycleTest.*:CycleApResidentBlocksTest.*:WaitcntBarrierSwitchFocusedTest.*'
```

### Functional 完整验证
```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='FunctionalExecEngineWaitcntTest.*:WaitcntFunctionalTest.*:FunctionalWaitcntTest.*'
```

### Runtime 完整验证
```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeAbiTest.*:HipRuntimeTest.*:TraceTest.*:ExecutedFlowProgramCycleStatsTest.*'
```

## 6. 测试统计

| 分类 | 测试套件数 | 测试用例数 |
|------|-----------|-----------|
| Runtime | ~20 | ~150 |
| Memory | ~8 | ~50 |
| ISA | ~15 | ~100 |
| Semantic | ~80 | ~500 |
| **总计** | **123** | **813** |

---

*最后更新: 2026-04-12*
