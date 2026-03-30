# Execution Architecture Refactor Assessment

> [!NOTE]
> 历史计划文档。用于保留当时的拆解和决策上下文，不作为当前代码结构的权威描述。当前主线以 `docs/my_design.md` 和 `docs/module-development-status.md` 为准。


## 目标

回答下面四个问题：

1. functional 和 cycle 是否应该共用一套基类代码
2. single-thread 和 multi-thread functional 是否应该完全共用一套代码
3. 当前框架哪里合理，哪里后续维护成本高
4. 项目现在是否需要重构，如果需要，最小重构边界是什么

## 结论先说

### 结论 1

`single-thread functional` 和 `multi-thread functional` 应该继续共用一套执行核心。

当前方向是对的：

- [functional_execution_core.cpp](/data/gpu_model/src/exec/functional_execution_core.cpp) 已经把 `st` 和 `mt` 放到同一个 block/wave 执行核心上
- 两者差别只应该是“谁来驱动 wave 往前走”
- 不应该分裂成两套语义、两套 shared/barrier、两套访存提交流程

所以：

- `st/mt` 共核是必须坚持的
- 后续只需要把“调度/驱动策略”继续抽干净

### 结论 2

`functional` 和 `cycle` 应该共用“执行内核部件”，但不应该硬并成一个大基类 executor。

应该共用的是：

- kernel/block/wave 状态构建
- 常量区 / global / shared / private 的访存辅助逻辑
- `OpPlan` 到状态修改的公共提交逻辑
- barrier release / wait state 的公共状态转换
- trace 文本格式化和部分事件构造

不应该硬共用的是：

- 主调度循环
- event queue 驱动
- scoreboard 依赖检查
- issue bundle 选择
- launch timing / async return 时序推进

原因很简单：

- functional 是“执行一步就立即生效”
- cycle 是“发射、等待、到时提交”
- 两者最不一样的部分，正好就是主循环和提交时机

如果强行做成一个大基类 executor，最后会得到大量 `if (cycle_mode)` / `if (functional_mode)`，维护会更差。

### 结论 3

当前项目需要重构，但只需要做 **局部重构**，不需要大爆炸重写。

当前代码的主要问题不是分层方向错了，而是：

- `functional` 和 `cycle` 在共享部件下面各自复制了不少实现
- 新增一条 memory / barrier / trace 语义时，容易两边都要改
- raw GCN 路径又形成了第三套局部执行逻辑

所以需要的不是推翻，而是把已经重复的部分收拢。

## 当前设计哪里合理

下面这些地方方向是对的，不建议推翻：

- [semantics.h](/data/gpu_model/include/gpu_model/execution/internal/semantics.h) + [semantic_handlers.cpp](/data/gpu_model/src/exec/semantic_handlers.cpp)
  - `Instruction -> OpPlan` 这层已经是正确分层
- [functional_execution_core.cpp](/data/gpu_model/src/exec/functional_execution_core.cpp)
  - `st/mt functional` 已经开始共核
- [issue_model.h](/data/gpu_model/include/gpu_model/execution/internal/issue_model.h) + [issue_scheduler.h](/data/gpu_model/include/gpu_model/execution/internal/issue_scheduler.h)
  - cycle 特有的 issue 逻辑被单独放出来，这是对的
- [scoreboard.h](/data/gpu_model/include/gpu_model/execution/internal/scoreboard.h) + [event_queue.h](/data/gpu_model/include/gpu_model/execution/internal/event_queue.h)
  - 这些本来就应该是 cycle 专属部件，不应该塞回 functional

## 当前设计哪里不合理

### 1. functional 和 cycle 重复维护 block/wave materialize

重复点很明显：

- [functional_execution_core.cpp](/data/gpu_model/src/exec/functional_execution_core.cpp)
- [cycle_executor.cpp](/data/gpu_model/src/exec/cycle_executor.cpp)

两边都自己维护了：

- `ExecutableBlock`
- `MaterializeBlocks`
- `ConstantPoolBase`
- `LoadLaneValue`
- `StoreLaneValue`
- shared/private/global/const 的一批 helper

这会带来两个问题：

- 新增 memory space 语义时，需要双改
- 两个路径的行为会慢慢漂移

### 2. functional 的 public executor 外壳有点多余

现在有：

- [functional_executor.h](/data/gpu_model/include/gpu_model/exec/functional_executor.h)
- [parallel_wave_executor.h](/data/gpu_model/include/gpu_model/exec/parallel_wave_executor.h)

它们最后都只是转发到：

- [functional_execution_core.h](/data/gpu_model/include/gpu_model/exec/functional_execution_core.h)

这不是大问题，但长期看属于“包装层比逻辑层还多”。

### 3. raw GCN 路径形成第三套局部执行模型

- [raw_gcn_executor.cpp](/data/gpu_model/src/exec/raw_gcn_executor.cpp)
- [raw_gcn_semantic_handlers.cpp](/data/gpu_model/src/exec/raw_gcn_semantic_handlers.cpp)

这条路径短期内可以接受，因为它面对的是另一套 ISA 输入。

但它现在自己维护了：

- block/wave 循环
- barrier release
- shared memory 访问
- SGPR/VGPR 写回

这意味着以后 shared/barrier 行为修一次，可能要修三处。

## 最简单、最稳的设计模式

不要用深继承。

最简单、最稳的是：

- **组合**
- **策略模式**
- **小而明确的共享 helper**

### 推荐分层

#### 1. State Builder 层

建议新建一个共享状态构建层，负责：

- 从 `PlacementMap + LaunchConfig` 构造 kernel/block/wave 运行时状态
- 初始化 `WaveState`
- 分配 shared memory
- 构造 `wave_indices_per_peu`

建议名字：

- `exec/execution_state_builder.*`
- `exec/execution_state.h`

这层给 functional 和 cycle 共用。

#### 2. Memory / Barrier Ops 层

建议把下面这些 helper 从 executor 内部抽出来：

- global/shared/private/const load/store/atomic
- constant pool base 解析
- barrier arrive / release
- shared memory / private memory 字节访问

建议名字：

- `exec/execution_memory_ops.*`
- `exec/execution_sync_ops.*`

这层给 functional 和 cycle 共用。

#### 3. Plan Apply 层

`Semantics` 继续负责生成 `OpPlan`。

但建议新增一个共享的 `OpPlanExecutor` 或 `PlanApply` 层，负责：

- 标量/向量寄存器写回
- `exec/cmask/smask` 更新
- branch/exit/barrier 的状态修改
- memory request 的同步提交

functional 直接用“立即提交”版本。

cycle 不直接复用完整提交函数，但应该复用：

- 访存 domain 判定
- 写回 helper
- barrier 状态更新 helper

建议名字：

- `exec/op_plan_apply.*`

#### 4. Run Policy 层

functional 只保留一个执行核心。

单线程和多线程的差别，放到 run policy：

- `SequentialPeuRunPolicy`
- `ParallelPeuRunPolicy`

但没必要为了这个单独做很多 public class。

最小做法是：

- 保留一个 `FunctionalExecutionCore`
- 对外只暴露一个 `Run(mode, worker_threads)` 或 `Run(config)`
- 内部再分 `RunSequential()` / `RunParallel()`

### 为什么不推荐“大基类 executor”

因为 cycle 和 functional 最大的不同点就在：

- 指令何时提交
- memory 何时返回
- barrier 何时释放
- stall 为什么发生

如果把这些塞进一个 `BaseExecutor` 虚函数体系里，最终很容易变成：

- 基类知道太多
- 派生类覆盖太多
- 调试时跨很多层跳转

这个项目更适合：

- 共享数据结构
- 共享 helper
- 各自保留主循环

## 建议的最小重构边界

### Phase A

只整理 functional 内部接口，不碰 cycle 语义：

- 保留 [functional_execution_core.cpp](/data/gpu_model/src/exec/functional_execution_core.cpp) 为唯一 functional 执行核心
- 让 [functional_executor.h](/data/gpu_model/include/gpu_model/exec/functional_executor.h) 和 [parallel_wave_executor.h](/data/gpu_model/include/gpu_model/exec/parallel_wave_executor.h) 合并成一个 functional executor 配置入口

收益：

- `st/mt` 的 public 接口更干净
- 外部不再关心有两个 functional executor class

### Phase B

抽共享状态和 memory/sync helper：

- 从 functional / cycle 里抽出
  - `MaterializeBlocks`
  - `LoadLaneValue`
  - `StoreLaneValue`
  - `ConstantPoolBase`
  - barrier release helper

收益：

- 新增 memory / barrier 功能只改一处
- functional / cycle 行为更一致

### Phase C

抽 `OpPlan` 公共写回 / 提交 helper：

- 先抽“纯状态写回”部分
- 不要一开始就把 cycle 的 event scheduling 强行塞进去

收益：

- 新增一条新指令时，functional / cycle 的行为差异会更集中在 timing，而不是语义

### Phase D

最后再考虑 raw GCN 路径共享 helper：

- 先共享 memory/sync helper
- 暂时不要强推 raw GCN 直接走 `Instruction -> OpPlan`

原因：

- raw GCN 还在快速补 ISA coverage
- 现在强行统一，成本高，回报低

## 不建议现在做的事

- 不建议现在把 functional / cycle / raw_gcn 统一成一个大 executor 基类
- 不建议现在把所有状态都模板化成 timing policy
- 不建议现在为了“看起来统一”去重写 cycle 主循环
- 不建议现在重构 raw GCN 到内部 ISA，再反向执行

这些改动范围太大，短期收益不够。

## 最终建议

项目 **需要重构**，但只需要做下面这一种重构：

- **共享状态和共享 helper 的局部收拢**

项目 **不需要重写**，尤其不需要：

- 重写 functional 主循环
- 重写 cycle 主循环
- 设计一个很重的 executor 继承体系

一句话总结：

- `st/mt functional` 要继续共核
- `functional/cycle` 要共享部件，不共享主循环
- 最简单可维护的方案是 **组合 + 策略模式 + 小 helper 收拢**
