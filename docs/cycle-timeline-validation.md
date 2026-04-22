# Cycle Timeline Validation Methodology

本文档定义如何评估和验证 `gpu_model` cycle model 输出的正确性。

## 1. Cycle Timeline 核心机制

### 1.1 时间推进模型

```
global_cycle (调度时间)
    |
    +-- events.RunReady(cycle)     // 执行到期的异步事件
    +-- FillDispatchWindow()       // 填充 dispatch 窗口
    +-- BuildCandidates()          // 构建候选 wave
    +-- SelectIssueBundle()        // 选择 issue bundle
    +-- Issue + Commit             // 发射并提交
    +-- Schedule future events     // 调度未来事件
    |
    v
cycle++                           // 推进全局时间
```

### 1.2 关键时间约束

| 约束 | 来源 | 默认值 |
|------|------|--------|
| `kIssueTimelineQuantumCycles` | 发射时间量子 | 4 |
| `default_issue_cycles` | 默认指令成本 | 4 |
| `global_mem_cycles` | 全局内存延迟 | 1024 |
| `scalar_mem_cycles` | 标量内存延迟 | 128 |
| `shared_mem_cycles` | 共享内存延迟 | 24 |
| `tensor_cycles` | Tensor 指令成本 | 16 |
| `warp_switch_cycles` | Wave 切换惩罚 | 1 |

### 1.3 Issue Bundle 选择流程

```
1. BuildResidentIssueCandidates()
   - 检查 timing_ready: next_issue_cycle <= global_cycle
   - 更新 eligible_since_cycle (动态 age)
   - 检查 barrier slot 可用性
   - 分类 blocked_reason

2. IssueScheduler::SelectIssueBundle()
   - 按 RoundRobin/OldestFirst 排序
   - 应用 type_limits / group_limits
   - 返回 selected_candidate_indices

3. 实际发射
   - 计算 actual_issue_cycle = max(cycle, next_issue_cycle, switch_ready_cycle)
   - 发射 IssueSelect 事件
   - 执行 WaveStep
   - 计算 commit_cycle = actual_issue_cycle + issue_cycles
```

## 2. 验证维度

### 2.1 功能正确性（Functional Correctness）

**验证点**：指令执行结果正确

```cpp
// 验证数据结果
EXPECT_EQ(output[i], expected[i]);

// 验证指令计数
EXPECT_EQ(stats.instructions_executed, expected_inst_count);
```

**测试覆盖**：
- `tests/cycle/*_cycle_test.cpp` - 各类 kernel 的 cycle 执行验证
- `tests/runtime/hipcc_parallel_execution_test.cpp` - HIP kernel 对比验证

### 2.2 时序正确性（Timing Correctness）

**验证点**：cycle 时间线符合模型约束

```cpp
// 验证总 cycle
EXPECT_GT(result.total_cycles, 0u);

// 验证 cycle 统计一致性
EXPECT_EQ(result.total_cycles, result.program_cycle_stats->total_cycles);

// 验证 IPC 合理性 (0 < IPC <= 1 对于单发射)
double ipc = stats.IPC();
EXPECT_GT(ipc, 0.0);
EXPECT_LE(ipc, 1.0);  // 单发射模型
```

### 2.3 一致性验证（Consistency）

**st/mt/cycle 模式结果一致性**：

```cpp
// 01-vecadd-cycle-splitting example
for mode in st mt cycle; do
    gpu_model_run_interposed_mode "$so_path" "$exe" "$mode_dir" "$mode"
done

// 验证 cycle 模式与其他模式数据一致
diff <(grep "mismatches" st/stdout.txt) <(grep "mismatches" cycle/stdout.txt)
```

### 2.4 Cache Latency 验证

**精确 cycle 值验证**（`tests/cycle/cache_cycle_test.cpp`）：

```cpp
// 第一次 load 从 DRAM
EXPECT_EQ(cycles[0], 436u);  // issue + DRAM latency

// 第二次 load 命中 L1 cache
EXPECT_EQ(cycles[1], 452u);  // issue + L1 hit latency (16 cycles later)

EXPECT_EQ(result.total_cycles, 452u);
```

### 2.5 Issue 成本验证

**指令分类成本**（`tests/runtime/executed_flow_program_cycle_stats_test.cpp`）：

```cpp
// 纯向量 ALU kernel
EXPECT_EQ(stats.total_cycles, num_insts * 4u);
EXPECT_EQ(stats.vector_alu_cycles, num_insts * 4u);

// 混合 kernel
EXPECT_EQ(stats.global_mem_cycles, num_mem_ops * 1024u);
EXPECT_EQ(stats.barrier_cycles, num_barriers * 4u);
EXPECT_EQ(stats.total_issued_work_cycles, 
          stats.global_mem_cycles + stats.barrier_cycles);
```

## 3. 验证方法

### 3.1 单元测试验证

```bash
# 运行 cycle 相关测试
./build-ninja/tests/gpu_model_tests --gtest_filter=*Cycle*

# 运行 cycle stats 验证
./build-ninja/tests/gpu_model_tests --gtest_filter=*ProgramCycleStats*
```

### 3.2 Example 回归验证

```bash
# 运行所有 examples
for d in examples/0*/; do ./examples/${d}run.sh; done

# 检查 cycle 输出
grep "cycle_total" examples/*/results/*/stdout.txt
```

### 3.3 Trace 验证

**检查 trace 输出结构**：

```bash
# 验证 trace 包含必要字段
grep -q '\[RUN\]' trace.txt
grep -q '\[KERNEL\]' trace.txt
grep -q '\[EVENTS\]' trace.txt
grep -q '\[SUMMARY\]' trace.txt

# 验证 cycle 单调递增
awk '/WaveStep/ {print $2}' trace.txt | sort -n -c
```

### 3.4 Perfetto Timeline 验证

```bash
# 验证 Perfetto JSON 结构
jq '.traceEvents[0].ph' timeline.perfetto.json
jq '.traceEvents[] | select(.name == "WaveStep") | .ts' timeline.perfetto.json
```

## 4. 已知验证点

### 4.1 必须满足的不变量

| 不变量 | 验证方式 |
|--------|----------|
| `total_cycles >= active_cycles` | `EXPECT_GE(stats.total_cycles, stats.active_cycles)` |
| `total_cycles = active_cycles + idle_cycles` | 数值求和验证 |
| `total_issued_work_cycles >= vector_alu_cycles + scalar_alu_cycles + ...` | 分类求和验证 |
| `IPC() <= 1.0` (单发射) | 饱和检查 |
| `WaveOccupancy() <= 1.0` | 完成率检查 |

### 4.2 Stall 分类完整性

当前 stall 分类：
- `stall_barrier` - barrier 等待
- `stall_waitcnt` - 内存计数器等待
- `stall_resource` - 资源竞争
- `stall_dependency` - RAW/WAW/WAR 依赖
- `stall_switch_away` - wave 切换

验证：
```cpp
uint64_t total_stalls = stats.stall_barrier + stats.stall_waitcnt + 
                        stats.stall_resource + stats.stall_dependency +
                        stats.stall_switch_away;
EXPECT_LE(total_stalls, stats.total_cycles);
```

## 5. 不验证的内容

### 5.1 硬件精确时间

**不验证**：cycle 值是否等于真实硬件执行时间

**原因**：
- 当前模型是 naive cycle 模型，未经过硬件校准
- cycle 值用于相对比较，不用于绝对性能预测

### 5.2 调度策略最优性

**不验证**：RoundRobin vs OldestFirst 哪个更优

**原因**：
- 不同策略适用于不同 workload
- 重点是策略行为正确，而非策略优劣

## 6. 扩展验证方向

### 6.1 增加 cycle 精确值断言

对于已知成本的 kernel，增加精确 cycle 断言：

```cpp
// 预期：N 条 ALU 指令 * 4 cycles
EXPECT_EQ(stats.total_cycles, N * 4u);

// 预期：M 条 global load * 1024 cycles
EXPECT_EQ(stats.global_mem_cycles, M * 1024u);
```

### 6.2 增加 cross-mode 一致性测试

```cpp
TEST(CycleConsistencyTest, VecaddStMtCycleMatch) {
  auto st_result = RunKernel(..., ExecutionMode::Functional, "st");
  auto mt_result = RunKernel(..., ExecutionMode::Functional, "mt");
  auto cycle_result = RunKernel(..., ExecutionMode::Cycle);
  
  // 数据一致
  EXPECT_EQ(st_result.data, mt_result.data);
  EXPECT_EQ(mt_result.data, cycle_result.data);
  
  // cycle 统计非零
  EXPECT_GT(cycle_result.total_cycles, 0u);
}
```

### 6.3 增加 timeline 结构验证

验证 Perfetto timeline 结构正确性：
- 每个事件有 ph (phase)、ts (timestamp)、name
- IssueSelect -> WaveStep -> Commit 顺序正确
- 无孤立事件（所有 flow 都有 start 和 end）
