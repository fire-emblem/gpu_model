# Perfetto Dump Rationality Design

## 背景

当前仓库已经具备：

- `TraceArtifactRecorder`
- `timeline.perfetto.json` 导出
- `CycleTimelineRenderer::RenderGoogleTrace(...)`
- `TraceTest` 与 `CycleTimelineTest` 的基础覆盖

但下一步的重点不再只是“能导出 Perfetto JSON”，而是：

1. `function mt` 下的 Perfetto dump 是否结构稳定、语义可解释
2. `cycle` 下的 Perfetto dump 是否与模型的 issue / stall / arrive / barrier 语义一致
3. 真实 HIP 程序在 Perfetto 中是否还能维持合理的时间线形状

## 本轮目标

本轮做两层校验：

1. **结构正确性**
   - `timeline.perfetto.json` 能稳定生成
   - JSON 结构合法
   - `traceEvents` 存在且可解析
   - 关键字段如 `ts / dur / name / tid / pid` 合法且稳定

2. **语义正确性**
   - `function mt`：
     - wave / block 并发关系可解释
     - waiting / resume 的时间顺序合理
   - `cycle`：
     - issue / stall / arrive / commit 顺序合理
     - barrier 相关事件位置与模型一致
   - 真实 HIP case：
     - 轨道规模与事件分布合理
     - 多 barrier 条件路径不会产生明显错误时间线

## 非目标

本轮明确不做：

- 重写 Perfetto / Google Trace 输出格式
- 引入新的 trace file format
- 为所有测试都导出 artifact
- 追求 Perfetto UI 层面的视觉美化

## 代表性 case 选择

### 1. `function mt` 代表性 case

建议使用：

- shared barrier / waitcnt 多 wave case

目标：

- 能看到并发 wave
- 能看到 waiting / resume
- 不会出现明显的轨道错序

### 2. `cycle` 代表性 case

建议使用：

- `AsyncMemoryCycle`
- `SharedBarrierCycle`

目标：

- memory issue 在前
- arrive / commit 在后
- stall / barrier 事件位置合理

### 3. 真实 HIP case

建议直接复用：

- `EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks`

目标：

- block/wave 轨道规模合理
- 多 barrier 条件路径形成可解释 timeline
- 不出现负 duration / 逆序

## 方案对比

### 方案 A：只校 JSON 结构

优点：

- 最简单

缺点：

- 无法发现 timeline 语义错乱

### 方案 B：只校语义

优点：

- 直接命中主要价值

缺点：

- 若 JSON 本身坏了，语义测试会很脆

### 方案 C：结构打底 + 代表性语义校验

优点：

- 覆盖完整
- 改动面仍可控
- 最适合当前阶段

### 结论

采用方案 C。

## 设计原则

### 1. 不改协议，先补验证

先复用现有：

- `TraceArtifactRecorder`
- `CycleTimelineRenderer::RenderGoogleTrace`

通过 tests 对输出做约束，而不是先重写协议。

### 2. 结构校验与语义校验分层

结构校验负责：

- 文件存在
- JSON 有效
- 必需字段存在
- 基本时间约束成立

语义校验负责：

- 事件关系与模型一致
- `mt` / `cycle` 的时间线形状可解释

### 3. 语义校验只锁关键不变量

不要把 Perfetto 测试写成对完整 event 列表逐条精确比对。

应锁定：

- 先后顺序
- 正/负关系
- 数量下界
- 分组存在性

而不是所有 event 的精确序列。

## 结构正确性检查

对 `timeline.perfetto.json` 至少检查：

1. 顶层包含 `"traceEvents"`
2. `traceEvents` 非空
3. 关键事件至少包含：
   - `name`
   - `ph`
   - `ts`
4. 若事件包含 `dur`
   - `dur >= 0`
5. `pid / tid` 不为空时可稳定用于分组

## `function mt` 语义检查

至少检查：

1. 同一 case 中存在多个 wave / thread track
2. waiting 相关事件出现
3. release / resume 之后仍有后续执行事件
4. 同一轨道上的事件 `ts` 单调不逆序

## `cycle` 语义检查

至少检查：

1. issue 事件早于对应 commit / arrive
2. stall 事件不会出现在 commit 之后
3. barrier arrive / release 先后合理
4. 同一轨道上的时间顺序单调

## 真实 HIP case 语义检查

至少检查：

1. `128 x 128 conditional multibarrier` case 能生成 Perfetto 文件
2. 存在多 block / 多 wave 相关轨道或事件
3. barrier 相关事件数量大于零
4. 没有明显非法时间：
   - 负 duration
   - `ts` 大规模倒退

## 建议测试挂点

### `TraceTest`

负责：

- artifact 存在
- JSON 结构有效
- `function mt` 基础可解释性

### `CycleTimelineTest`

负责：

- `cycle` 模式的 timeline 结构与关键顺序不变量

### `HipccParallelExecutionTest`

负责：

- 真实 HIP case 的 Perfetto dump 合理性

## 验收标准

1. `timeline.perfetto.json` 结构校验有自动回归
2. `function mt` 有至少一条 waiting/resume 可解释性回归
3. `cycle` 有至少一条 issue/stall/arrive/commit 顺序回归
4. 真实 HIP case 有至少一条 Perfetto 合理性回归
5. 不改现有 Perfetto 导出协议也能完成本轮校验

## 风险

1. 如果测试锁得太细，会被 harmless event 排序波动打散
2. 如果只测文本存在，不测时间关系，无法发现语义错误
3. 真实 HIP case 的 timeline 事件数较多，测试应避免全量逐条匹配

## 结论

本轮应先在现有导出协议上补一层“结构正确性 + 代表性语义正确性”验证，重点关注：

- `function mt`
- `cycle`
- `128 x 128 conditional multibarrier` 真实 HIP 程序

这能在不重写 trace 协议的前提下，把 Perfetto dump 的“合理性”变成可回归验证的目标。
