# Executed Flow Program Cycle Estimator Design

## 背景

当前项目已经具备：

- `SingleThreaded / MarlParallel` 两种 functional 执行模式
- 独立的 naive `CycleExecEngine`
- `ExecutionStats`、trace、timeline 和 wave lifecycle 观察能力

但目前缺少一层“基于 `st/mt functional` 实际执行流”的程序级 cycle 近似统计。

这层能力的目标不是替代真正的 cycle model，而是提供：

1. 一个对 `st/mt` 实际执行流敏感的程序级 cycle 近似值
2. 一个与后续 `CycleExecEngine` 兼容的统一程序级 cycle 结果接口
3. 一组可验证“统计量与理论值近似准确”的程序级用例

## 本轮目标

本轮要做的是：

基于 `st/mt functional` 的实际 wave 执行流，构建一个按 cycle tick 推进的 `ExecutedFlowProgramCycleEstimator`，输出统一的程序级 cycle 统计结果，并用若干代表性程序验证统计量与理论值近似一致；若不一致，则据此校准近似 cost policy。

## 非目标

本轮明确不做：

- 重写现有 `CycleExecEngine`
- 把 estimator 直接当成新的 cycle mode
- 引入完整的 PEU / issue-slot / scoreboard 硬约束模型
- 追求硬件精确 latency
- 把全部统计直接绑定到 trace sink 文本格式

## 理论口径

本轮的“理论值”不是硬件真值，而是项目级近似常数：

- 普通指令：`4 cycle`
- `MFMA/MMA`：`16 cycle`
- shared memory：`32 cycle`
- scalar / constant memory：`128 cycle`
- global / private memory：`1024 cycle`

这些值在本轮视为 `ProgramCycleEstimatorConfig` 的默认参数，而不是不可变的硬编码硬件事实。

## 用户修正后的关键约束

本轮全局 cycle 不能用以下粗口径替代：

- `sum(all wave cycles)`
- `max(per-wave cycles)`

因为这两者都不能正确表达 `st/mt` 下按 wave 执行、按 cycle 推进、存在重叠与等待的程序整体历时。

因此本轮 estimator 必须：

1. 显式按 cycle tick 推进
2. 显式建模每个 wave 的剩余执行 / 等待状态
3. 以“所有 wave 完成前经历的总 tick 数”作为 `program total_cycles`

## 方案对比

### 方案 A：静态指令计数乘常数

做法：

- 数程序里实际执行过的指令
- 按类别乘常数后直接求和

优点：

- 最简单

缺点：

- 不能表达 wave 重叠
- `st/mt` 全局 cycle 容易失真
- 不满足本轮“按 cycle 累加并考虑 wave”的要求

### 方案 B：按 wave 求局部 cycle，再取 `max(per-wave)`

做法：

- 为每个 wave 统计局部 cycle
- 程序全局 cycle 取最大 wave

优点：

- 比静态总和更接近程序历时

缺点：

- 仍然没有显式 cycle 级重放
- 无法稳定表达 barrier / wait / 部分并发重叠
- 用户已明确否定这种口径

### 方案 C：按 cycle tick 推进的 executed-flow estimator

做法：

- 每个 wave 持有剩余执行 / 等待状态
- 全局按 tick 推进
- cycle 内对所有 wave 做一次状态更新
- 直到所有 wave 完成

优点：

- 满足用户要求
- 能自然支持 `st/mt` 比较
- 可向真正 cycle model 演进

缺点：

- 比纯静态统计更复杂

### 结论

采用方案 C。

## 第一版并发规则

用户要求 `st/mt` 的全局 cycle 应近似，而不是完全不同口径。

因此第一版并发规则采用轻量模型：

- runnable wave 可并行推进自身剩余周期
- waitcnt / barrier / memory pending 会阻塞对应 wave
- 本轮暂不引入 PEU / issue-slot 容量上限

同时，为后续与 `CycleExecEngine` 对齐，接口必须预留：

```cpp
struct IssueCapacityModel;
```

第一版可使用“无限轻量容量”实现，后续再替换为更接近 cycle model 的容量约束。

## 统一结果接口

本轮引入统一程序级 cycle 结果结构：

```cpp
struct ProgramCycleEstimate {
  uint64_t total_cycles = 0;
  uint64_t total_issued_work_cycles = 0;

  uint64_t scalar_alu_cycles = 0;
  uint64_t vector_alu_cycles = 0;
  uint64_t tensor_cycles = 0;
  uint64_t shared_mem_cycles = 0;
  uint64_t scalar_mem_cycles = 0;
  uint64_t global_mem_cycles = 0;
  uint64_t private_mem_cycles = 0;
  uint64_t barrier_cycles = 0;
  uint64_t wait_cycles = 0;
};
```

语义：

- `total_cycles`
  - 程序从开始到全部 wave 完成所经历的全局 tick 数
- `total_issued_work_cycles`
  - 所有已执行工作按 cost policy 统计的总 work 量
- 分类字段
  - 用于解释理论值与程序 total cycle 的来源

## 配置接口

本轮默认 cost policy 通过配置对象暴露：

```cpp
struct ProgramCycleEstimatorConfig {
  uint32_t default_issue_cycles = 4;
  uint32_t tensor_cycles = 16;
  uint32_t shared_mem_cycles = 32;
  uint32_t scalar_mem_cycles = 128;
  uint32_t global_mem_cycles = 1024;
  uint32_t private_mem_cycles = 1024;
};
```

要求：

- estimator 不直接硬编码常数到所有调用点
- 用例可以显式引用默认参数
- 未来 cycle model 可复用同一 config 或投影到同一参数集

## 架构分层

### 1. 执行流采样层

职责：

- 从 `SingleThreaded / MarlParallel` functional 执行收集 wave 级事件
- 记录“某 wave 在何时执行了什么类别的工作”

输出建议是结构化事件，而不是依赖 trace 字符串解析：

```cpp
struct ExecutedWaveStep {
  uint32_t block_id;
  uint32_t wave_id;
  ExecutedStepKind kind;
  ExecutedStepClass step_class;
};
```

### 2. Cost policy 层

职责：

- 把某个 executed step 映射成近似 cycle 成本
- 决定它属于哪个统计分类

例如：

- 普通标量/向量 ALU -> `4`
- `MFMA` -> `16`
- shared access -> `32`
- scalar/constant access -> `128`
- global/private access -> `1024`

### 3. Program aggregator 层

职责：

- 为每个 wave 维护剩余工作 / 等待状态
- 每个 tick 推进全部 wave
- 最终输出 `ProgramCycleEstimate`

这层是后续与 `CycleExecEngine` 对齐的关键接口。

## Wave 级状态模型

第一版只需要最小状态：

```cpp
struct WaveCycleState {
  bool completed = false;
  uint64_t remaining_cycles = 0;
  bool waiting = false;
};
```

语义：

- `remaining_cycles > 0`
  - wave 正在消耗当前已发射工作
- `waiting == true`
  - wave 被 barrier / wait / memory pending 阻塞
- `completed == true`
  - wave 已结束

## 每 cycle 推进规则

全局循环：

1. 若所有 wave `completed`，结束
2. 对每个 wave：
   - 如果 `remaining_cycles > 0`，减一
   - 若减到零，允许接受下一段工作
   - 若被等待事件阻塞，维持 waiting
3. 对所有可推进 wave，按当前执行流输入注入下一段工作 cost
4. `total_cycles++`

注意：

- `st/mt` 的差异不来自不同统计口径
- 只来自它们的实际 wave 执行流与重叠关系不同

## 与 `CycleExecEngine` 的兼容性

后续 `CycleExecEngine` 不需要复用 functional 的执行流采样层，但应复用：

- `ProgramCycleEstimate`
- `ProgramCycleEstimatorConfig`
- program-level aggregation output contract

这样可以直接比较：

- `ExecutedFlowProgramCycleEstimator`
- `CycleExecEngine` 的统计结果

并保持上层接口统一。

## 用例设计

第一版至少需要四类程序：

### 1. 纯普通 ALU

目标：

- 验证 `4 cycle` 主口径
- 验证 `st/mt` 全局 cycle 近似一致

### 2. 含 `MFMA`

目标：

- 验证 `tensor = 16 cycle`
- 验证 tensor 指令不会落到普通 ALU 分类

### 3. mixed memory

程序同时覆盖：

- shared
- scalar / constant
- global
- private

目标：

- 验证各 memory cost 分类
- 验证 `total_issued_work_cycles` 的组成与理论值一致

### 4. barrier + multi-wave

目标：

- 验证全局 cycle 是按 wave 和每 cycle 推进得出
- 不是简单求和或简单取最大值
- `st/mt` 程序 total cycle 在可接受误差内近似

## 验收标准

1. 新增统一程序级 cycle 结果接口
2. 新增 executed-flow estimator 配置接口
3. estimator 的全局 cycle 显式按 wave / tick 推进
4. `st/mt` 的程序 cycle 采用同一口径
5. 至少 4 类代表性程序覆盖 ALU / tensor / mixed memory / barrier
6. 若当前 cost 常数导致误差明显，测试应能指出并驱动校准

## 风险

1. 如果执行流采样过粗，可能无法区分真正工作与等待
2. 如果聚合层直接绑死 functional 细节，后续 cycle model 很难复用
3. 如果测试只看总数，不看分类分解，会掩盖错误校准

## 结论

本轮应先实现一个：

- 面向 `st/mt functional`
- 按 wave / tick 推进
- 使用项目级默认近似常数
- 输出统一 `ProgramCycleEstimate`

的 estimator，并把接口设计成未来可与 `CycleExecEngine` 对齐。
