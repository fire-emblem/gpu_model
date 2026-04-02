# Program Cycle Stats Calibration Design

## 背景

现有 `ProgramCycleStats` 已经对外暴露，并且仓库中已有：

- `ProgramCycleTracker`
- `ExecutedFlowProgramCycleStatsTest.*`
- 真实 HIP 程序用例
- `HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks`

但新的 `128 x 128 conditional multibarrier` 用例已经暴露出一个关键问题：

- 当前 `ProgramCycleStats` 的若干 bucket 和记账口径，与期望的“按 wave / 按 cycle / 按 active-lane work”理论模型不一致
- 尤其是 memory op 计数、memory cycle bucket 和 `total_issued_work_cycles` 的求和一致性

## 本轮目标

本轮目标不是重新定义 `ExecutionStats`，而是：

1. 保持现有 `ExecutionStats` 作为粗粒度统计
2. 把 `ProgramCycleStats` 校准成更贴近 active-lane / program-work 语义
3. 先校准 `st/mt functional`
4. 对 `cycle` 先要求排序与量级一致，不要求同一轮完全贴合理论值

## 用户确认后的关键原则

### 1. `ExecutionStats` 和 `ProgramCycleStats` 语义分离

本轮明确采用：

- `ExecutionStats`
  - 保持粗粒度
  - 更接近“执行了多少类操作 / 多少次事件”
- `ProgramCycleStats`
  - 调整为 active-lane / work 近似统计
  - 用于程序级理论 cycle 对照

也就是说，不要求把现有 `ExecutionStats` 改成逐线程 work 记账。

### 2. 先校准 `st/mt`，再约束 `cycle`

本轮先把：

- `SingleThreaded`
- `MultiThreaded`

的 `ProgramCycleStats` 校到接近理论值。

`Cycle` 模式本轮只要求：

- 排序合理
- 量级一致
- 接口兼容

不要求本轮完全贴合同一组理论常数。

### 3. 先修 memory/work bucket，再修 barrier 和总和

本轮优先顺序：

1. memory op 计数语义
2. `shared_mem_cycles / global_mem_cycles / vector_alu_cycles`
3. `barrier_cycles`
4. `total_issued_work_cycles == bucket sum`

## 记账口径

### A. `ExecutionStats`

继续保持当前粗粒度：

- `shared_loads`
- `shared_stores`
- `global_loads`
- `global_stores`
- `barriers`
- `wave_exits`

这些值仍然允许接近“按事件 / 按指令 / 按阶段”的统计，而不是逐 lane work。

### B. `ProgramCycleStats`

改为更接近 active-lane / work 口径：

- memory 类 bucket 按 active lane work 记账
- 计算类 bucket 按 active lane 指令 work 记账
- `barrier_cycles` 按 wave / block 同步开销近似记账
- `total_issued_work_cycles` 必须是各 bucket 的一致性总和

## 方案对比

### 方案 A：把 `ExecutionStats` 也改成逐 lane

优点：

- 对齐彻底

缺点：

- 风险过大
- 会影响大量已有测试
- 不符合本轮用户选择

### 方案 B：只改测试口径，不改生产逻辑

优点：

- 快

缺点：

- 会把当前错误语义“合理化”
- 无法服务后续 cycle model 抽象

### 方案 C：保留 `ExecutionStats` 粗粒度，只校准 `ProgramCycleStats`

优点：

- 与用户要求一致
- 改动面可控
- 更适合程序级 cycle 理论对照

缺点：

- 需要明确区分两套统计含义

### 结论

采用方案 C。

## 128-Block Conditional Multibarrier 作为校准基准

这条用例本轮作为：

- `ProgramCycleStats` 校准基准
- `st/mt` 理论值逼近验证

它不再只是“结果正确”的 regression，而是“统计口径是否合理”的基准测试。

## 理论近似常数

本轮继续使用项目级近似常数：

- 普通指令：`4`
- `MFMA/MMA`：`16`
- shared memory：`32`
- scalar / constant：`128`
- global / private：`1024`

但这些常数应作用在：

- active-lane work
- wave 级执行流

而不是简单的“每条 memory 指令算一次事件”。

## 校准目标

### 1. memory op 计数语义

对于 `ProgramCycleStats`：

- shared/global/private/scalar memory bucket 要和 active-lane work 对应
- 不再沿用粗粒度 `ExecutionStats` 的事件口径

### 2. compute bucket 语义

- `vector_alu_cycles`
- `scalar_alu_cycles`
- `tensor_cycles`

应反映 active-lane / issued-work 近似

### 3. barrier bucket

- `barrier_cycles` 应体现同步阶段的程序级近似代价
- 不应只是简单照抄 `ExecutionStats.barriers * 4`
- 但本轮允许采用 wave/block 粒度的稳定近似，而不要求硬件精度

### 4. total work consistency

最终必须满足：

```cpp
total_issued_work_cycles ==
  scalar_alu_cycles +
  vector_alu_cycles +
  tensor_cycles +
  shared_mem_cycles +
  scalar_mem_cycles +
  global_mem_cycles +
  private_mem_cycles +
  barrier_cycles +
  wait_cycles;
```

## 与 cycle model 的兼容性

后续真正 cycle model 的输入事件源可以不同，但它也应投影到同一 `ProgramCycleStats` 结构。

因此本轮校准的不是：

- functional 内部私有细节

而是：

- `ProgramCycleStats` 作为程序级统计接口的语义

## 验收标准

1. `ExecutionStats` 继续保持粗粒度，不被强行改成逐 lane
2. `ProgramCycleStats` 校准到更接近 active-lane / work 语义
3. `st` 与 `mt` 在 128-block multibarrier case 上接近理论近似
4. `cycle` 至少保持排序和量级合理
5. `total_issued_work_cycles` 与 bucket sum 一致

## 风险

1. 如果 bucket 重新定义不够清楚，后续调用方会混淆 `ExecutionStats` 与 `ProgramCycleStats`
2. 如果 barrier 近似定义太随意，测试会变成硬编码数字而不是语义校准
3. 如果只校准一个大 case，可能会过拟合

## 结论

本轮应该把 `ProgramCycleStats` 明确拉到“程序级 active-lane work 统计”语义上，并用现有 `128 x 128 conditional multibarrier` HIP case 作为主要校准基准；同时保留 `ExecutionStats` 的粗粒度角色不变。
