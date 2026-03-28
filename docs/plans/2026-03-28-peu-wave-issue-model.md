# PEU / Wave 指令发射模型说明

## 这份文档说明什么

这份文档只说明一件事：

- 在当前项目里，`block`、`AP`、`PEU`、`wave`、`lane` 应该如何组织
- `wave` 在 `PEU` 里应该怎么被选择和发射

重点回答这几个问题：

1. 一个 `block` 里的多个 `wave` 到底是什么关系？
2. 指令发射是在 `AP` 做，还是在 `PEU` 做？
3. 一个 `PEU` 一次最多发几条指令？
4. 如果一个 `wave` 当前不能发射，调度器怎么办？
5. 如果一个 `PEU` 上有超过 `4` 个 `wave`，后面的 `wave` 是不是必须等前面的 `wave` 主动让权？
6. 一个 `AP` 里的 `4` 个 `PEU`，能不能同时发射？

这份文档只讲：

- functional 模型
- naive cycle 模型将来应遵循的调度形状

这份文档不讲：

- 精确硬件拍级流水
- 每个功能单元的精确实现

## 基本概念

### Block

`block` 是一次 kernel launch 中的一个工作块。

当前项目的硬件抽象里：

- 一个 `block` 对应一个 `AP`

所以一个 `block` 内所有线程、所有 `wave`，都属于同一个 `AP`。

### AP

`AP` 是 block 级共享资源域。

一个 `AP` 持有：

- 这个 block 的 `shared/LDS`
- 这个 block 的 barrier 状态
- 这个 block 内 shared atomic 的同步语义
- 这个 block 内所有 `wave`

所以：

- `AP` 是“共享资源和同步”的范围
- 不是“统一平铺 issue 池”

### PEU

`PEU` 是真正的局部调度和发射域。

一个 `PEU` 持有：

- 自己驻留的 `wave` 集合
- 一个本地 round-robin 指针
- 自己当前从哪个 `wave` 取下一条指令

所以：

- `PEU` 是“选哪个 wave 发下一条指令”的地方

### Wave

`wave` 是调度单位。

一个 `wave` 持有自己的：

- PC
- `exec`
- `cmask`
- `smask`
- SGPR / VGPR
- private memory
- `waitcnt` 相关 pending 状态
- branch pending 状态
- waiting-at-barrier 状态

所以：

- wave 是“被调度”的单位

### Lane

`lane` 不是调度单位。

`lane` 只在 wave 被选中之后参与执行：

- wave 取到一条指令
- 再结合 `exec mask`
- 在 wave 内部对最多 `64` 个 lane 做语义执行

所以：

- 项目里不能把 lane 当成 scheduler 的选择对象

## 一个 block 里的 wave 是什么关系

一个 `block` 里的所有 `wave`：

- 共享同一个 `AP`
- 共享同一个 `shared/LDS`
- 共享同一个 barrier 状态
- 共享 block 内 shared atomic 的同步语义

但是：

- 这些 `wave` 不应该放进一个“整个 block 统一的大 issue 池”

正确方式是：

- 它们先属于同一个 `AP`
- 然后再按 `PEU` 分组
- 每个 `PEU` 自己维护本地 wave pool

所以更准确地说：

- `block` 内的 `wave` 是一个 **共享同步域**
- 不是一个 **平铺的大 issue 池**

## 指令发射应该发生在哪一层

结论：

- 指令发射选择应该发生在 `PEU`
- 不应该在整个 `AP` 上做一个统一的波前选择器

原因是：

- `AP` 负责 block 级共享资源
- `PEU` 负责局部调度

这和本地资料是一致的：

- `miaow` 的 issue 逻辑是先形成 ready wave bitmap，再做 local arbiter
- 当前项目文档里也明确把 `SIMD / PEU-local wave selection` 放在 CU front-end

## 一个 PEU 一次最多发几条指令

对当前项目，建议规则是：

- 一个 `PEU`
- 一次发射机会
- 最多选择一个 `wave`
- 发出这个 `wave` 的一条指令

也就是：

- `1 PEU -> 1 wave -> 1 instruction`

### 为什么不是“5 条”

AMD / MIAOW 资料里经常会看到：

- `SIMD`
- `SALU`
- `LSU`
- `SIMF`
- `LDS / branch`

这些代表的是：

- 整个 CU front-end 可能有多类执行资源

但这不等于：

- 对当前项目的一个 `PEU`，要直接建模成“一次同时发 5 条 wave 指令”

对于当前这个轻量 functional / naive cycle 模型，更合适的简化是：

- `PEU` 一次只发一条
- 以后 cycle 层再叠加 issue class 竞争

## 为什么有些 GCN 文档会写“最多 5 条 issue”

这里最容易误解。

文档里提到“最多 5 条 issue”，通常说的是：

- 在 **整个 CU / SQ front-end 的范围**
- 同一个周期里
- 如果不同类别的执行资源都空闲
- 那么理论上可以同时向这些不同类别的资源送出指令

这里的“5”更接近的是：

- 5 类执行路径 / 5 类发射类别

而不是：

- 一个 `PEU` 一次发 5 条
- 一个 `wave` 一次发 5 条
- 整个 `AP` 在所有 `wave` 里统一挑 5 个 `wave`

### 正确理解

应该把它理解成：

- 这是 **类别级并行上限**
- 不是 **单个调度单元的 wave 数量选择规则**

### 例子

假设某一时刻，整个 CU front-end 看到下面几类都各有 ready 指令：

- 一个 `VALU` 指令
- 一个 `VMEM` 指令
- 一个 `SALU` 指令
- 一个 `LDS` 指令
- 一个 `BRANCH` 指令

文档可能会描述成：

- 这一周期理论上最多可以发 `5` 条

但这里的意思是：

- 每一类资源各吃掉一条

不是说：

- 一个 `PEU` 一次发 5 条
- 或整个 `AP` 在所有 `wave` 里统一选出 5 个 `wave`

## 你的问题：整个 AP 里如果有 8 个 wave，是不是从 4 个 PEU 的所有 wave 中统一选 5 个？

结论：

**不是。**

更准确的规则应该是：

1. 先按 `PEU` 分组
2. 每个 `PEU` 只在自己的本地 wave pool 里选 ready wave
3. 每个 `PEU` 一次最多选一个 wave
4. 如果以后 cycle 层要建模“不同类别资源可并行”，那也是在更高一层做 issue-class 竞争

所以不是：

- 整个 `AP` 先把 8 个 wave 摊平
- 再“从小到大”选出 5 个 ready wave

而是：

- `PEU0` 在自己的 wave pool 里选 1 个
- `PEU1` 在自己的 wave pool 里选 1 个
- `PEU2` 在自己的 wave pool 里选 1 个
- `PEU3` 在自己的 wave pool 里选 1 个

至于同一周期能不能在更高层同时让多类资源都工作，那是：

- cycle/front-end issue class 模型的问题
- 不是 `PEU` 本地 wave 选择的问题

### 例子

假设一个 `AP` 里有 8 个 wave：

- `PEU0`: `wave0 wave4`
- `PEU1`: `wave1 wave5`
- `PEU2`: `wave2 wave6`
- `PEU3`: `wave3 wave7`

某一时刻：

- `PEU0` 里只有 `wave4` ready
- `PEU1` 里 `wave1` 和 `wave5` 都 ready，但 round-robin 轮到 `wave5`
- `PEU2` 里没有 ready wave
- `PEU3` 里 `wave3` ready

那么这一步的选择结果应是：

- `PEU0 -> wave4`
- `PEU1 -> wave5`
- `PEU2 -> idle`
- `PEU3 -> wave3`

这才是当前项目应该采用的 wave 选择方式。

不是：

- 把 `wave0..wave7` 合起来排一个总表
- 再从头选出 5 个 ready wave

## wave 什么时候不能发射

一个 `wave` 当前不能发射，常见原因包括：

- 依赖没准备好
- 在等 `waitcnt`
- 在等 memory return
- 在等 branch 结果
- 在等 barrier release

这时正确的调度行为是：

- 不要卡住整个 block
- 也不要卡住整个 `AP`
- 只是在这个 `PEU` 的本地 wave pool 里，把这个 wave 暂时视为 not ready
- 然后继续选下一个 ready wave

如果这个 `PEU` 上所有 `wave` 都不 ready：

- 这个 `PEU` 才 idle

## PEU 内部的 wave pool 怎么看

当前项目的目标硬件里，一个 `PEU`：

- 最多可以驻留 `8` 个 `wave`
- 其中最多 `4` 个处于活跃 issue window

所以一个 `PEU` 内部最好区分两个集合：

1. resident waves
   总共最多 `8`

2. active issue window
   最多 `4`

剩下的 resident wave：

- 还在这个 `PEU` 上
- 但暂时不在活跃 issue window 内

## 当 PEU 上 wave 数量大于 4 时，后面的 wave 怎么办

这个问题最容易被误解。

### 错误理解

错误理解是：

- `wave4` 必须等 `wave0` 主动让权
- `wave5` 必须等 `wave1`

这种理解不对。

### 正确理解

正确规则是：

- 后面的 `wave` 等的是 **active issue window 的空位**
- 不是等某个固定的 earlier wave

### 例子

假设某个 `PEU` 上驻留了 `6` 个 `wave`：

- active issue window: `wave0 wave1 wave2 wave3`
- standby resident waves: `wave4 wave5`

运行过程中：

- `wave0` 因 global memory 等待而长时间不 ready
- `wave1` 因 barrier 等待而不 ready
- `wave2` 还能发
- `wave3` 退出了

这时会发生什么？

- active issue window 出现空位
- 调度器可以把 `wave4` 拉进来

注意这里：

- `wave4` 不是必须等 `wave0`
- 它只需要等到 active window 有位置

所以结论是：

- `>4 wave` 时，后面的 wave 等的是“窗口空位”
- 不是“某个前面的 wave 让权”

## 一个 AP 里的 4 个 PEU 能不能同时发射

结论：

- 可以
- 并且应该允许它们独立推进

不应该人为规定成：

- `PEU0` 发完
- 再 `PEU1`
- 再 `PEU2`
- 再 `PEU3`

这种全局顺序规则太保守。

更合理的规则是：

- 每个 `PEU` 都有自己的本地 wave pool
- 每个 `PEU` 都有自己的 ready-wave 选择
- 它们只在 block 级共享资源上互相耦合

也就是：

- `shared/LDS`
- barrier
- shared atomic

### 例子

假设一个 block 有 8 个 wave，映射关系如下：

- `PEU0`: `wave0 wave4`
- `PEU1`: `wave1 wave5`
- `PEU2`: `wave2 wave6`
- `PEU3`: `wave3 wave7`

在某个时刻：

- `PEU0` 选择 `wave4`
- `PEU1` 选择 `wave1`
- `PEU2` 选择 `wave6`
- `PEU3` 选择 `wave3`

这是允许的。

只有当它们：

- 访问同一个 shared 区域
- 或等待同一个 barrier release

时，才需要通过 block-shared 同步对象协调。

## 哪些状态必须共享，哪些状态必须私有

### block / AP 级共享状态

下面这些状态必须是 block-local 共享的：

- `shared_memory / LDS`
- barrier generation
- barrier arrival count
- block 内等待 barrier 的 wave 集合
- shared atomic 的串行化对象
- `PEU -> wave pool` 映射
- `PEU` 内 RR 指针

### wave 级私有状态

下面这些必须是每个 wave 自己的：

- PC
- `exec`
- `cmask`
- `smask`
- SGPR / VGPR
- private memory
- pending memory counters
- `waitcnt` 状态
- branch pending
- waiting-at-barrier 标记

## 当前项目最适合采用的规则

当前项目最适合的调度规则可以直接写成下面几条：

1. `AP` 持有 block 共享状态
2. `PEU` 持有本地 resident wave pool
3. `PEU` 内用 round-robin 选择 ready wave
4. 一次 issue 机会，一个 `PEU` 最多发一个 wave 的一条指令
5. wave 内再按 `exec mask` 跑 64 lane
6. 当前 wave 发不出，就切到同 `PEU` 的下一个 ready wave
7. 一个 `PEU` 上所有 wave 都发不出时，该 `PEU` 才 idle
8. `4` 个 `PEU` 默认允许独立推进

## 当前项目已经做到哪一步

当前项目已经开始往这个方向走了：

- `Mapper` 已经把 wave 分配到 `peu_id`
- shared functional core 里已经有：
  - `wave_indices_per_peu`
  - `next_wave_rr_per_peu`
- 现在 `MarlParallel` 已经在 block 内按 `PEU` 分组 wave

还没有完全做完的部分有：

1. `>4` resident wave 的 active-window / standby-window 显式建模
2. 更完整的 `wait / stalled / resume` 状态机
3. shared / barrier / atomic 的更细粒度同步
4. cycle 路径复用同样的 front-end / PEU 调度结构

## 最后，直接回答你的问题

### 一个 block 里的 wave 是一个统一的大执行池吗？

不是。

准确说法是：

- 一个 block 对应一个 `AP`
- 这个 `AP` 是 block 级共享同步域
- 真正的 issue 池应该按 `PEU` 拆开

### 一个 PEU 一次能发 5 条吗？

对当前项目，不建议这样建模。

建议规则是：

- 一个 `PEU` 一次只发一个 wave 的一条指令

### wave 数量大于 4 时，后面的 wave 必须等前面的某个 wave 让权吗？

不是。

正确规则是：

- 后面的 wave 等的是 active issue window 的空位
- 不是等某个固定 earlier wave

### 4 个 PEU 能同时发射吗？

可以。

应该默认允许它们独立推进。

只有 block 级共享资源会把它们同步起来。
