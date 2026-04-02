# Shared-Heavy HIP Kernel Closure Design

## 背景

当前主线已经具备：

- `hipcc` 真实 `.out` 的 encoded/raw-GCN 路径
- `shared_reverse`、`softmax_row`、`atomic_count` 等代表性 kernel
- `dynamic_shared_sum` 等 shared-heavy HIP 程序测试

但 shared-heavy 路径仍然容易在真实 HIP `.out` 上撞到 encoded/raw-GCN 覆盖缺口。

最近已经暴露并处理过一批问题：

- `v_pk_mov_b32`
- `ds_read2_b32`
- `s_lshl_b32`
- `s_cmp_lg_u32`
- placeholder fallback 在 `mimg/exp/unknown` 上的恢复

这说明下一步最有价值的切片，不是再抽象讨论 DS family，而是继续围绕真实 shared-heavy HIP case 把闭环补稳。

## 本轮目标

以真实 HIP `.out` 为主线，补齐 shared-heavy 路径的 encoded/raw-GCN 覆盖，使它不再只依赖单个 case 通过。

本轮最小覆盖目标：

1. `dynamic_shared_sum`
2. `shared_reverse`

## 非目标

本轮明确不做：

- 一次性补齐全部 DS / MIMG / EXP family
- 重写 encoded 指令体系
- 改写现有真实 HIP case 的语义
- 新建大而泛的 shared-heavy benchmark 集

## 用户确认后的约束

本轮采用：

- **优先真实 HIP kernel 闭环**
- **不只修单点**

因此本轮必须至少让两组 shared-heavy 真实 HIP case 作为共同验收对象，而不是只修一个 `dynamic_shared_sum`。

## 方案对比

### 方案 A：以真实 HIP case 驱动补齐 shared-heavy 指令链

做法：

- 先盘点真实 HIP case 实际出现的缺失 opcode
- 按“指令簇”补：
  - encoding def / match
  - binding
  - semantic handler
- 每补一批就回归真实 HIP case

优点：

- 收益最高
- 与“任意 HIP 可执行程序”目标最直接相关
- 不会先做抽象、后找不到真实收益

缺点：

- 需要严控不要退化成 case-by-case 零散修补

### 方案 B：先系统化 DS family，再回真实 case

优点：

- 结构更整齐

缺点：

- 容易 scope 膨胀
- 当前不一定是最短路径

### 结论

采用方案 A，但要求按“共享指令簇”而不是“单条 case”组织实现与测试。

## 本轮覆盖对象

### 1. `dynamic_shared_sum`

目标：

- 现有 `HipRuntimeTest.LaunchesHipDynamicSharedExecutableInRawGcnPath` 稳定通过

这条用例的价值：

- 使用 `extern __shared__`
- 含 block barrier
- 含 DS read/write
- 含循环与条件控制

### 2. `shared_reverse`

目标：

- 继续作为第二条 shared-heavy 真实 HIP case 保持通过

这条用例的价值：

- shared + barrier 主线稳定
- 可防止修 `dynamic_shared_sum` 时误伤既有 shared-heavy 路径

## 实现边界

本轮允许修改的方向：

1. `encoded_gcn_encoding_def`
   - 补实际缺失的 shared-heavy 链路 encoding match
2. `encoded_instruction_binding`
   - 补 concrete object / placeholder fallback
3. `encoded_semantic_handler`
   - 补最小必要语义
4. 对应 decode / object loader / runtime tests

本轮不允许：

- 为了过 case 而改 HIP 源程序本身的行为
- 把 unsupported 指令一律 silent no-op 掩盖真实缺口
- 直接删除或放宽真实 HIP case 的断言

## 设计原则

### 1. 优先补“真正挡住闭环”的指令

每补一条指令，都应满足至少一个条件：

- 是 `dynamic_shared_sum` 真正挡住路径的下一条缺口
- 或是 `shared_reverse` / 另一条 shared-heavy case 的共享前置条件

### 2. placeholder 与 concrete object 语义分开

对于指令：

- 已知 family 但本轮不支持执行
  - 可落 placeholder
- 已参与真实 shared-heavy HIP 执行主线
  - 必须落 concrete object + minimal semantic handler

### 3. 不把“真实执行主线”降级成 placeholder

如果某条指令出现在 `dynamic_shared_sum` 的真实执行路径里，并且程序要正确运行：

- 就不能仅用 placeholder 过 object parse
- 必须补到执行可用

## 验收策略

### 第一层：最小单测 / decode regression

用于锁：

- encoding match
- placeholder fallback
- binding 不抛异常

### 第二层：真实 HIP case

至少覆盖：

- `HipRuntimeTest.LaunchesHipDynamicSharedExecutableInRawGcnPath`
- 现有 `shared_reverse` 相关真实 HIP case

### 第三层：必要时的 loader/decode 回归

例如：

- `AmdgpuCodeObjectDecoderTest.*`
- `InstructionArrayParserTest.*`
- `EncodedInstructionBindingTest.*`

## 风险

1. 如果只修 `dynamic_shared_sum`，可能会把 shared-heavy 支持变成单测驱动的偶然成功
2. 如果大量使用 no-op 语义，真实执行结果可能 silently 错误
3. 如果 placeholder 与 concrete object 边界不清楚，会让 decode 能过、执行又崩

## 验收标准

1. `dynamic_shared_sum` raw-GCN 路径稳定通过
2. `shared_reverse` 相关真实 HIP case 继续通过
3. 本轮新增/修正的 decode/binding tests 通过
4. 不放宽已有真实 HIP case 的结果断言
5. shared-heavy 路径不再只依赖单个 case 才成立

## 结论

本轮应继续沿真实 HIP kernel 主线补齐 shared-heavy 路径，以：

- `dynamic_shared_sum`
- `shared_reverse`

两组 case 作为共同验收对象，优先补真正挡住 encoded/raw-GCN 闭环的 shared-heavy 指令链，而不是提前扩成全 family 系统化工程。
