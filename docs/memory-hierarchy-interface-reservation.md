# 轻量级 Memory Hierarchy 接口预留方案

## 文档目的

这份文档记录 `gpu_model` 下一阶段为 `VL1 / SL1 / L2` 轻量级访存建模预留接口的设计结论。

这里的目标不是一次性实现完整 cache/coalescer，而是先把：

- 语义请求
- 合并后事务
- hierarchy 命中与延迟判定
- 真实字节读写

四层边界拆开，使当前项目可以先用固定值或可复现随机值占位，后续再逐步替换成：

- 访存合并发射请求
- `VL1 / SL1 / L2` hit/miss
- fill / eviction

## 设计边界

本方案服务于当前项目“轻量功能模型 + naive / calibrated cycle 分析平台”的目标，不引入 gem5 Ruby/VIPER 那种全系统协议控制器复杂度。

本方案当前只考虑：

- vector global path 对应的 `VL1`
- scalar / scalar-buffer / constant path 对应的 `SL1`
- 共享下游 `L2`
- line 粒度的轻量级合并与 hit/miss

本方案当前不考虑：

- coherence
- MSHR
- GPUVM / page fault / DMA
- 跨 wave 合并
- 真实协议消息与目录状态机

## 当前问题

当前项目已有：

- `MemoryRequest` 语义请求对象
- `MemorySystem` 字节级真实内存容器
- 统一 `CacheModel`
- `SharedBankModel`

但 cycle 主路径中，global memory 的 cache probe 仍直接写在 executor 内部。其结果是：

- `MemoryRequest` 无法表达 path / route / coalescing
- `CycleExecEngine` 同时承担请求语义、cache 路径、hit/miss 和填充逻辑
- 后续一旦加入 `VL1 / SL1 / L2`、coalescing、fill/eviction，主循环会持续膨胀

因此，下一阶段最重要的不是增加更多 latency 参数，而是先把接口边界改对。

## gem5 参考结论

结合 gem5，可以借鉴的不是 Ruby 全栈，而是下面这条逻辑分层：

- vector global path 先过 coalescer，再进入类似 `TCP` 的 L1
- scalar / scalar-buffer path 走独立的 scalar/SQC 路径
- 两条路径最终汇入共享 `L2 / TCC`

对 `gpu_model` 来说，真正值得吸收的是“路径分离 + 事务聚合 + 共享下游 cache”这一接口形状，而不是完整协议控制器实现。

## 目标分层

下一阶段建议把一条内存访问拆成四层：

1. `MemoryRequest`
   - 指令语义层请求
2. `MemoryTransaction`
   - 后端可调度事务
3. `MemoryHierarchyModel`
   - `VL1 / SL1 / L2 / miss` 与 latency 判定
4. `MemorySystem`
   - 真实字节读写

调用关系应变成：

`MemoryRequest -> MemoryTransactionBuilder -> MemoryHierarchyModel -> MemorySystem`

而不是继续在 executor 中直接探测 cache。

## 最小接口草案

### 1. Memory 路径枚举

建议新增：

```cpp
enum class MemoryClient {
  Vector,
  Scalar,
};

enum class CacheRoute {
  VectorL1ThenL2,
  ScalarL1ThenL2,
  L2Only,
  Bypass,
};

enum class CoalescingKind {
  None,
  PerWaveLine,
};

enum class CacheHitLevel {
  None,
  VectorL1,
  ScalarL1,
  L2,
  Miss,
};
```

### 2. 扩展 MemoryRequest

建议在现有 `MemoryRequest` 上增加：

- `request_id`
- `client`
- `route`
- `coalescing`
- `dpc_id`
- `ap_id`
- `peu_id`
- `ordered_return`

用途：

- `MemorySpace` 继续表达语义空间
- `route` 单独表达 cache 路径
- 避免以后通过 opcode 或 `MemorySpace::Constant` 反推 `VL1/SL1/L2` 走向

### 3. 新增 MemoryTransaction

建议新增事务层对象：

```cpp
struct MemoryTransaction {
  uint64_t id = 0;
  uint64_t request_id = 0;
  uint32_t transaction_index = 0;

  MemoryClient client = MemoryClient::Vector;
  CacheRoute route = CacheRoute::Bypass;
  CoalescingKind coalescing = CoalescingKind::None;
  AccessKind kind = AccessKind::Load;

  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;

  uint64_t issue_cycle = 0;
  bool ordered_return = true;

  uint32_t active_lane_count = 0;
  uint32_t merged_line_count = 0;
  std::vector<uint64_t> line_addresses;
};
```

第一阶段可以先让：

- `1 request -> 1 transaction`
- `line_addresses` 仅由活跃 lane 地址按 line 去重得到

### 4. 新增请求完成聚合对象

后续一旦支持 `1 request -> N transactions`，waitcnt 和 scoreboard ready 必须以 request 为单位聚合完成，而不是以单 transaction 为单位。

建议新增：

```cpp
struct InflightMemoryRequest {
  MemoryRequest request;
  uint32_t total_transactions = 0;
  uint32_t completed_transactions = 0;
  uint64_t latest_arrive_cycle = 0;

  bool Done() const { return completed_transactions >= total_transactions; }
};
```

### 5. 新增 MemoryAccessOutcome

建议新增：

```cpp
struct MemoryAccessOutcome {
  CacheHitLevel hit_level = CacheHitLevel::None;
  uint64_t latency_cycles = 0;

  uint32_t vl1_hits = 0;
  uint32_t sl1_hits = 0;
  uint32_t l2_hits = 0;
  uint32_t misses = 0;

  uint32_t merged_line_count = 0;
};
```

### 6. 新增 MemoryTransactionBuilder

建议接口：

```cpp
class MemoryTransactionBuilder {
 public:
  std::vector<MemoryTransaction> Build(const MemoryRequest& request) const;
};
```

设计要求：

- 即使第一阶段只返回一个 transaction，也保持返回 `vector`
- 避免未来从单事务接口改成多事务接口时再大改调用点

### 7. 新增 MemoryHierarchyModel

建议接口：

```cpp
class MemoryHierarchyModel {
 public:
  explicit MemoryHierarchyModel(MemoryHierarchySpec spec = {}) : spec_(spec) {}

  MemoryAccessOutcome Probe(const MemoryTransaction& txn) const;
  void CommitFill(const MemoryTransaction& txn, CacheHitLevel level);

 private:
  MemoryHierarchySpec spec_;
};
```

其中：

- `Probe()` 负责决定 hit level 和 latency
- `CommitFill()` 负责未来的 fill / eviction
- `MemorySystem` 仍只负责真实字节读写

## 配置预留

### 1. 过渡模式

建议在 `MemoryHierarchySpec` 中引入四种模式：

```cpp
enum class MemoryModelMode {
  LegacyUnifiedCache,
  Fixed,
  RandomRange,
  Hierarchy,
};
```

用途：

- `LegacyUnifiedCache`
  - 保持与当前统一 `L1/L2/DRAM` 逻辑兼容
- `Fixed`
  - 用固定值占位
- `RandomRange`
  - 用可复现随机值占位
- `Hierarchy`
  - 未来真正的 `VL1 / SL1 / L2`

### 2. 层级配置

建议新增：

```cpp
struct CacheLevelSpec {
  bool enabled = false;
  uint64_t hit_latency = 0;
  uint32_t line_bytes = 64;
  uint32_t line_capacity = 0;
};

struct CoalescerSpec {
  bool enabled = false;
  uint32_t line_bytes = 64;
  uint32_t max_transactions_per_wave = 0;
  bool ordered_return = true;
};
```

### 3. 占位模式配置

建议新增：

```cpp
struct LatencyRange {
  uint64_t min_cycles = 0;
  uint64_t max_cycles = 0;
};

struct HitWeightSpec {
  uint32_t vector_l1 = 0;
  uint32_t scalar_l1 = 0;
  uint32_t l2 = 0;
  uint32_t miss = 100;
};

struct MemoryPlaceholderSpec {
  MemoryModelMode mode = MemoryModelMode::LegacyUnifiedCache;

  uint64_t fixed_vl1_cycles = 8;
  uint64_t fixed_sl1_cycles = 8;
  uint64_t fixed_l2_cycles = 20;
  uint64_t fixed_miss_cycles = 40;

  LatencyRange vl1_range{};
  LatencyRange sl1_range{};
  LatencyRange l2_range{};
  LatencyRange miss_range{};

  HitWeightSpec vector_hit_weights{};
  HitWeightSpec scalar_hit_weights{};

  uint32_t seed = 1;
};

struct MemoryHierarchySpec {
  CoalescerSpec coalescer;
  CacheLevelSpec vector_l1;
  CacheLevelSpec scalar_l1;
  CacheLevelSpec l2;
  MemoryPlaceholderSpec placeholder;
};
```

## 执行主路径改造原则

当前 cycle executor 中，global memory 的 cache probe 和 fill 仍直接写在主循环里。

下一阶段建议改成：

1. 从 `plan.memory` 构造 `MemoryRequest`
2. 调用 `MemoryTransactionBuilder::Build()`
3. 对每个 transaction 调用 `MemoryHierarchyModel::Probe()`
4. arrive 时：
   - 真正读写 `MemorySystem`
   - 调用 `CommitFill()`
   - 更新 request 级聚合完成状态
5. 只有在 request 全部 transaction 完成后：
   - decrement pending memory ops
   - mark scoreboard ready
   - 触发 request 级 `Arrive` trace

也就是说，后续：

- transaction 是 timing 计算单位
- request 是 waitcnt / scoreboard 完成单位

## 渐进实施顺序

### 第一批

目标：

- 固定接口
- 不改变默认行为

动作：

- 新增上述类型与 spec
- `CycleExecEngine` 改为通过 builder + hierarchy 调用
- 默认走 `LegacyUnifiedCache`

### 第二批

目标：

- 支持占位模式

动作：

- `Fixed`
- `RandomRange`
- `VL1 / SL1 / L2 / miss` 统计字段

### 第三批

目标：

- 支持轻量级事务合并

动作：

- `1 request -> N transactions`
- per-wave line coalescing
- request completion aggregation

### 第四批

目标：

- 支持真实轻量 cache state

动作：

- `VL1 / SL1 / L2` line-based hit/miss
- fill / eviction
- 分路径 probe 与 promote

## 对现有文件的建议落点

建议新增文件：

- `include/gpu_model/memory/memory_client.h`
- `include/gpu_model/memory/memory_transaction.h`
- `include/gpu_model/memory/inflight_memory_request.h`
- `include/gpu_model/memory/memory_access_outcome.h`
- `include/gpu_model/memory/memory_transaction_builder.h`
- `include/gpu_model/memory/memory_hierarchy_model.h`
- `src/memory/memory_transaction_builder.cpp`
- `src/memory/memory_hierarchy_model.cpp`

建议优先改动文件：

- `include/gpu_model/memory/memory_request.h`
- `include/gpu_model/arch/gpu_arch_spec.h`
- `include/gpu_model/execution/cycle_exec_engine.h`
- `src/execution/cycle_exec_engine.cpp`
- `src/runtime/exec_engine.cpp`
- `src/arch/c500_spec.cpp`
- `include/gpu_model/debug/trace_event.h`
- `include/gpu_model/runtime/program_cycle_stats.h`

## 当前最重要的约束

在真正开始写 `VL1 / SL1 / L2` 逻辑之前，必须先坚持下面两点：

1. `MemorySystem` 不负责 cache / hierarchy
2. `CycleExecEngine` 不再直接做 cache probe 细节

只要这两点先做对，后面无论是：

- 固定值
- 随机值
- 访存合并
- hit/miss
- fill/eviction

都只是替换实现，不需要再推翻主路径接口。
