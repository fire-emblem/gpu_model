# HIP 128-Block Conditional Multibarrier Design

## 背景

当前仓库已经有多类基于 `hipcc` 真实程序的并行对照测试，尤其是在：

- `st / mt / cycle` 三模式结果对比
- shared memory + barrier 主线
- block 级数量变化
- 条件控制流下的真实 HIP executable 验证

但还缺一类更“重”的定向用例：

- `grid_dim_x = 128`
- `block_dim_x = 128`
- 含若干次 barrier
- 含条件控制的多阶段 shared / global 变换
- host 侧有可精确重建的预期值检查

## 本轮目标

在现有 `hipcc_parallel_execution_test.cpp` 体系内新增一个真实 HIP 程序用例，验证：

1. `128 blocks x 128 threads` 的较大规模 block 并行
2. 每个 block 内所有线程经过相同数量的 barrier
3. 条件仅控制额外计算与写回路径，不控制 barrier 是否执行
4. host 侧能做逐元素精确比对，并附带逐 block 辅助校验

## 非目标

本轮明确不做：

- 把该用例并入通用 feature CTS 矩阵
- 条件控制某些线程跳过 barrier 的非法/未定义 HIP 场景
- 新建独立测试二进制
- 直接把这条用例当作 cycle model 精确性基准

## 设计约束

### 1. barrier 必须全体一致执行

本轮条件控制流只能影响：

- 额外算术
- shared 数据变换
- 写回值

不能影响：

- block 内线程是否进入某个 `__syncthreads()`

### 2. 主断言是逐元素精确比对

这条用例的主验收必须是：

- `st` 输出数组与期望值逐元素一致
- `mt` 输出数组与期望值逐元素一致
- `cycle` 输出数组与期望值逐元素一致

### 3. 逐 block 校验是辅助而非替代

允许附加：

- block sum
- block min/max
- parity/分类计数

但这些只用于辅助定位，不代替逐元素断言。

## 方案对比

### 方案 A：扩展现有 `conditional_multibarrier`

做法：

- 沿用已有 conditional multibarrier 风格
- 提升到 `grid=128, block=128`
- 保持 shared tile + 多次 barrier + 条件额外计算

优点：

- 最贴合现有测试风格
- 风险最低
- host 期望值公式容易延续现有写法

缺点：

- 如果只是简单放大规模，覆盖新意不足

### 方案 B：block reduce 风格的条件规约

做法：

- 条件控制线程贡献值
- 多次 barrier 完成 block 规约
- 输出 block 结果或回填整段数组

优点：

- 每个 block 的理论值直观

缺点：

- 逐元素覆盖不如 shared tile 变换强

### 方案 C：多阶段 shared tile 条件变换

做法：

- 每个 block 处理一个 `128` 元素 tile
- shared 中做 2 到 3 阶段条件变换
- barrier 固定执行
- 最终输出完整数组

优点：

- 最适合逐元素精确校验
- 同时具有 shared/barrier/条件控制三类压力

缺点：

- host 侧期望值重建需要更认真设计

### 结论

采用方案 C，并在命名/组织上尽量贴近现有 `hipcc_parallel_execution_test.cpp` 风格。

## Kernel 设计

### launch 形状

- `grid_dim_x = 128`
- `block_dim_x = 128`

总线程数：

- `16384`

每个 block 处理一个 `128` 元素 tile。

### shared memory

```cpp
__shared__ int tile[128];
```

### 输入输出

- 输入：`const int* in`
- 输出：`int* out`

### 程序阶段

建议使用三阶段结构：

1. **load stage**
   - `tile[tid] = in[gid]`
2. **barrier 1**
3. **conditional local transform**
   - 条件由 `blockIdx.x` 和 `threadIdx.x` 共同决定
   - 例如：
     - 若 `(blockIdx.x + threadIdx.x) & 1`
       - `tile[tid] += blockIdx.x`
     - 否则
       - `tile[tid] -= threadIdx.x`
4. **barrier 2**
5. **cross-half transform**
   - 前后半区互相读取
   - 例如：
     - `partner = tid ^ 64`
     - 再根据 `blockIdx.x & 1` 选择加/减组合
6. **barrier 3**
7. **writeback**
   - `out[gid] = tile[tid]`

### 条件控制原则

必须保证：

- 所有线程都经历相同数量的 barrier
- 条件只改变数据变换，不改变同步结构

## Host 期望值设计

### 主期望数组

host 侧按与 kernel 一致的数学规则重建：

1. 先构造每个 block 的本地 `tile[128]`
2. 执行第一阶段条件变换
3. 执行第二阶段跨半区变换
4. 把最终 `tile` 写回 `expect`

这样可以做到逐元素精确比对。

### 辅助 block 校验

在逐元素之外，附加每个 block 的辅助检查：

- block sum
- block 首尾元素

这些辅助值用于快速定位：

- 是某个 block 全体错
- 还是某类 lane 变换错

## 测试挂点

本轮直接放在：

- `tests/runtime/hipcc_parallel_execution_test.cpp`

原因：

- 该文件已具备真实 `hipcc` 构建、`st/mt/cycle` 三模式对比和 host 侧期望值重建模式
- 本轮需求是定向高强度验证，而不是通用 feature matrix

## 断言设计

### 核心断言

1. `st` 输出逐元素等于 `expect`
2. `mt` 输出逐元素等于 `expect`
3. `cycle` 输出逐元素等于 `expect`
4. `st / mt / cycle` 的 `barriers` 统计一致
5. `st / mt / cycle` 的 shared/global store/load 统计一致

### program cycle stats 断言

本轮对 `program_cycle_stats` 只做口径自洽断言：

- 对象存在
- `launch.total_cycles == launch.program_cycle_stats->total_cycles`

本轮不把这条 HIP case 直接写成 estimator 精度标定依据。

## 命名建议

测试名建议：

```cpp
TEST(HipccParallelExecutionTest,
     EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks)
```

如果仓库中已有同名/近似名，应在此基础上增强，而不是重复创建语义重叠测试。

## 风险

1. 如果条件公式过于复杂，host 期望值代码会失去可读性
2. 如果只做逐 block 断言，会掩盖 lane 级错误
3. 如果把条件写成 barrier 不一致路径，会导致 HIP 程序本身语义不合法

## 验收标准

1. 新增真实 HIP 程序用例，`grid=128, block=128`
2. 含至少三次 barrier
3. 条件只控制计算/写回路径，不控制 barrier 执行
4. host 侧有逐元素精确期望值检查
5. 同时有逐 block 辅助校验
6. `st / mt / cycle` 三模式结果一致
