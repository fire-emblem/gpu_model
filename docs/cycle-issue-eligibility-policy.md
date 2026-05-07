# Cycle Issue Eligibility And Policy

本文定义当前工程中 `cycle model` 的 `issue eligible` 与 `issue policy` 语义。

目标：

- 固定 `eligible -> selected -> issue` 的状态边界
- 固定参考资料与当前实现之间的映射关系
- 说明为什么当前模型不应默认保证“同一个 wave 连续发射直到阻塞”

非目标：

- 本文不声明当前 `cycle` 时间已校准为真实硬件时间
- 本文不修改当前默认 `issue policy`
- 本文不要求当前实现必须精确复刻某一代 AMD 硬件的私有调度细节

## 1. 适用范围

本文约束：

- `ExecutionMode::Cycle`
- `program_object` 的 `cycle` 主线

本文不直接约束：

- `Functional st`
- `Functional mt`

`Functional` 仍可作为语义参考，但其 `4-cycle issue quantum` 规则不等价于这里的 cycle 调度选择逻辑。

## 2. 术语

### 2.1 `eligible`

`eligible` 表示某个 wave 在当前 `global_cycle` 具备被调度器考虑的资格。

`eligible` 不是“最终一定会 issue”，只表示“可以进入本轮选择集合”。

### 2.2 `selected`

`selected` 表示调度器在当前 cycle 的候选集合中选中了该 wave，准备将它放入当前 issue bundle。

### 2.3 `issue`

`issue` 表示当前 wave 的指令真正作为本 cycle 的执行行为被发出，并消耗当前模型中的 issue 资源。

### 2.4 `policy`

`policy` 只负责在当前 `eligible` 集合上做选择。

`policy` 不负责：

- 生成 readiness
- 恢复 waiting wave
- 补 barrier / waitcnt / arrive
- 绕过 front-end / slot / group / pipe 资源限制

## 3. 硬约束

当前项目对 cycle issue 语义采用下面的固定顺序：

1. 先由 engine/state machine 根据 `global_cycle` 推导 wave 当前是否 `eligible`
2. 再由 `issue policy` 在 `eligible` 集合中选择一个或多个 wave
3. 最后执行真正的 `issue`

禁止把这三步混在一起理解。

尤其是：

- `eligible != selected`
- `selected != issue`
- 某个 wave 在某 cycle `eligible`，不代表该 cycle 必定 `issue`
- 某个 wave 连续两次 `issue`，也不代表模型存在“sticky same-wave”硬规则

## 4. `eligible` 的最小定义

当前工程里，一个 wave 要进入本 cycle 的 issue 候选集合，至少应满足下面条件：

1. wave 已进入当前模型允许参与 issue 的前端状态
   - 例如已完成 `generate / dispatch / launch / active-window` 所需前置阶段
2. wave 尚未 `exit / completed`
3. 当前 `pc` 能映射到合法指令
4. wave 当前不处于业务阻塞态
   - 例如 `waitcnt`
   - `barrier wait`
   - 其它明确的 waiting 状态
5. 当前 cycle 对应的本地资源允许它被考虑
   - 例如 slot 可用
   - active window 中可见
   - AP / PEU 前端状态允许
6. 当前 bundle 的 issue 约束允许该类型继续被加入
   - `type_limits`
   - `group_limits`

补充说明：

- `eligible` 必须来自 execution/state machine 的真实状态
- 不能由 trace/timeline/perfetto 反推

## 5. 参考资料中的 policy 语义

### 5.1 AMD / LLVM 参考

*(参考文件 `src/spec/llvm_amdgpu_refs/AMDGPUUsage.html` 已移除)* 在 `WG_RR_EN` 处明确给出：

- 若 `WG_RR_EN = 1`，wavefronts 对同一 SIMD 采用 `round-robin`
- 否则采用 `oldest age order`

这说明公开资料里更稳定的抽象是：

- 先有“同一 SIMD 上的候选 wavefronts”
- 再在这些候选 wavefronts 之间按 policy 选择

而不是“默认锁定一个 wave 连续发射直到阻塞”。

### 5.2 gem5 参考

[`third_party/gem5/src/gpu-compute/rr_scheduling_policy.hh`](../third_party/gem5/src/gpu-compute/rr_scheduling_policy.hh) 把 `RR` 定义为：

- 从 ready list 前端取一个 wave
- 取出后从 ready list 删除
- 由调用方负责把仍然 eligible 的 wave 重新放回 ready list 尾部

[`third_party/gem5/src/gpu-compute/of_scheduling_policy.hh`](../third_party/gem5/src/gpu-compute/of_scheduling_policy.hh) 把 `OF` 定义为：

- 在 ready list 中选择最老的 wave

[`third_party/gem5/src/gpu-compute/schedule_stage.cc`](../third_party/gem5/src/gpu-compute/schedule_stage.cc) 也清楚地把流程拆成：

- ready list 选 wave
- 进入 `schList`
- 再进 `dispatchList`
- 再做资源仲裁

这同样支持：

- ready / eligible 是一层
- policy 选择是一层
- dispatch / issue / resource arbitration 又是后续层

### 5.3 MIAOW 参考

[`third_party/miaow/src/verilog/rtl/issue/arbiter.v`](../third_party/miaow/src/verilog/rtl/issue/arbiter.v) 用 `last_issued_wf_id` 驱动旋转优先级。

[`third_party/miaow/src/verilog/rtl/issue/instruction_arbiter.v`](../third_party/miaow/src/verilog/rtl/issue/instruction_arbiter.v) 还显式把：

- 当前正在选择的 wave
- 上一拍刚发过的 wave

临时排除在本轮可发集合之外。

这类实现偏向公平轮转或防止连续重复选择，同样不支持“默认连续发射同一 wave”这类硬规则。

## 6. 当前仓库中的实现映射

### 6.1 policy 抽象

当前统一抽象在：

- [`src/execution/internal/issue_logic/issue_scheduler.h`](../src/execution/internal/issue_logic/issue_scheduler.h)
- [`src/execution/internal/issue_logic/issue_scheduler.cpp`](../src/execution/internal/issue_logic/issue_scheduler.cpp)

核心对象：

- `ArchitecturalIssueType`
- `ArchitecturalIssueLimits`
- `ArchitecturalIssuePolicy`
- `EligibleWaveSelectionPolicy`

其中：

- `type_limits` 定义每类 issue type 的数量上限
- `group_limits` 定义共享 issue group 的数量上限
- `type_to_group` 定义 issue type 到 issue group 的映射

### 6.2 spec/default policy

架构级默认 policy 由：

- [`src/execution/internal/cost_model/cycle_issue_policy.h`](../src/execution/internal/cost_model/cycle_issue_policy.h)

提供：

- `CycleIssuePolicyForSpec(...)`
- `CycleEligibleWaveSelectionPolicyForSpec(...)`
- `CycleIssuePolicyWithLimits(...)`

当前 `mac500` 默认值在：

- [`src/gpu_arch/chip_config/mac500_spec.cpp`](../src/gpu_arch/chip_config/mac500_spec.cpp)

其中默认 policy 仍来自 spec，而不是写死在 scheduler 内部。当前 `mac500` 还把：

- `branch`
- `special`

放到了同一个 issue group，这意味着它们在默认 bundle 里互相冲突。

当前 `mac500` 的 wave-order selection 默认值是：

- `EligibleWaveSelectionPolicy::RoundRobin`

### 6.3 scheduler 选择逻辑

当前选择器位于：

- [`src/execution/internal/issue_scheduler.cpp`](../src/execution/internal/issue_scheduler.cpp)

其核心规则是：

1. 先根据 `EligibleWaveSelectionPolicy` 生成候选遍历顺序
2. `RoundRobin`
   - 从 `selection_cursor` 开始做一次 RR 旋转
3. `OldestFirst`
   - 按 `age_order_key` 升序选择
4. 只考虑 `candidate.ready == true` 的候选
5. 同一 bundle 内同一 `wave_id` 最多被选一次
6. 满足 `type_limits` 与 `group_limits` 才能进入 bundle
7. 对 `RoundRobin`，若本轮至少选择了一个候选，则下一轮 cursor 前移

因此当前 scheduler 的本质是：

- 基于 `eligible candidates` 的显式 wave-order policy + bundle policy 选择器
- 默认 wave-order policy 为 `RoundRobin`
- 不是 sticky same-wave 选择器
- 也不是隐式、未命名的近似 RR

### 6.4 cycle / encoded-cycle 路径

当前两条主线都采用“先建候选，再做 policy 选择”的结构：

- [`src/execution/cycle_exec_engine.cpp`](../src/execution/cycle_exec_engine.cpp)
- [`src/execution/program_object_exec_engine.cpp`](../src/execution/program_object_exec_engine.cpp)

两者都会：

1. 基于当前 cycle 和 wave 状态构建候选
2. 调用 `IssueScheduler::SelectIssueBundle(...)`
3. 对选中的 wave 执行真正 issue

因此：

- 当前工程在大方向上已经符合“eligible 集合 + policy 选择”的参考抽象
- 当前两条 cycle 路径现在都通过统一的 wave-order policy 入口驱动默认 `RoundRobin`
- 当前仍需继续校准的是 candidate 构造、资源时序与 future `OldestFirst` age key 定义，而不是把默认规则改成“连续发射同一 wave”

## 7. 对 timeline / trace 解释的影响

本文直接影响对 cycle timeline 的解释方式：

- 同一 `PEU` 上多个 wave 交错 issue，默认是正常现象
- 只要它们都 `eligible`，RR / OF 一类策略都可能导致交错
- 单个 wave 的 issue 流不连续，不能自动判定为 bug

更可疑的信号通常是：

- issue 起点破坏当前模型要求的时间量化规则
- wave 明明不该 `eligible` 却被发射
- wave 明明应该恢复为 `eligible` 却长期进不了候选集
- group/type/resource 限制没有按 policy 生效

## 8. 对 `01-vecadd` 的解释口径

以 `01-vecadd` 为例：

- 同一 `PEU` 上存在多个 resident / active / runnable waves
- 因此 issue 交错本身是符合本文定义的
- 不能把“没有连续打一整个 wave”当作直接 bug

后续分析 `01-vecadd` 时，应优先检查：

1. 当前 cycle 下 wave 是否真的 `eligible`
2. active window / resident slot / AP front-end 是否正确
3. `IssueScheduler` 的 RR 起点推进是否正确
4. `type_limits / group_limits` 是否按预期限制 bundle
5. 阻塞恢复后的 wave 是否重新回到候选集合

而不是先假设“硬件应该连续发同一个 wave”。

## 9. 当前项目结论

当前项目的正式口径是：

- `cycle model` 默认不保证同一 wave 连续发射
- 当前正式抽象是“在 eligible waves 上按 policy 选择 issue”
- 参考资料支持至少两类主流 policy：
  - `round_robin`
  - `oldest_first`
- 当前仓库已显式引入 `EligibleWaveSelectionPolicy` 来表达这两类 wave-order policy
- 当前默认 wave-order policy 为 `RoundRobin`
- 本文的重点仍是固定语义边界，而不是宣称当前默认值已完成真实硬件校准

如果后续需要更换默认 policy，应视为：

- cycle 模型校准变更

而不是：

- trace/render/perfetto 展示层变更
