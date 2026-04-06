# 任务计划：当前正式任务主线

## 目标
以当前模块设计为基础，推进 runtime API、memory pool / `mmap`、ISA 指令验证、`st/mt/cycle` 语义校准、日志与 trace 收口，并将模块交互关系、开发计划和模块开发状态统一记录到正式文档。

## 当前阶段
阶段 3

## 各阶段

### 阶段 1：正式任务收口
- [x] 审核历史 plans/specs，识别仍然有效的未完成主题
- [x] 将已完成的临时问题和过渡方案从当前 task list 中移除
- [x] 将剩余事项归并为正式长期 track
- **状态：** complete

### 阶段 2：正式设计收口
- [x] 将 runtime 分层收口为 `HipRuntime -> ModelRuntime -> ExecEngine`
- [x] 将 trace、functional、cycle、stats 的稳定设计约束写入主设计文档
- [x] 将模块状态文档与正式 task list 完全对齐
- **状态：** complete

### 阶段 3：主线实现推进
- [ ] 落定 runtime API / memory pool / `mmap` 第一阶段框架设计
- [ ] 建立不同 `memcpy` / `memset` 行为的主测试清单
- [ ] 建立 ISA text-asm kernel 验证体系
- [ ] 完成 `st/mt/cycle` 结果与设计语义校准
- [ ] 统一日志到 `loguru`
- [ ] 继续收口 trace canonical / unified / disable-trace 边界
- **状态：** in_progress

### 阶段 4：验证与资产整理
- [ ] 建立轻量级 runtime / memory / ISA / semantic 测试矩阵
- [ ] 完成 `examples` 剩余分批全量检查
- [ ] 继续清理明显过时、已完成且误导当前主线的历史计划文件
- [ ] 将 archive / active / reference 的文档边界继续收紧
- **状态：** pending

### 阶段 5：交付与维护
- [ ] 保持 `task_plan.md`、`docs/my_design.md`、`docs/runtime-layering.md`、`docs/module-development-status.md` 同步
- [ ] 将新增实现结果及时回写到正式设计和状态文档
- **状态：** pending

## 当前正式任务清单
1. `Runtime API closure`
   - 目标：先落定 `HipRuntime / ModelRuntime` 的重要 API 框架，尤其是不同 `memcpy` / `memset` 变体与无需 kernel launch 的行为矩阵。
   - 当前缺口：同步 `malloc/free/memcpy/memset` 主路径现已补齐一轮 focused matrix，覆盖 `RuntimeSession`、`DeviceMemoryManager`、LD_PRELOAD ABI 纯 memory 路径、非法 `hipMemcpyKind` 和非法 compatibility pointer 返回值；后续仍需继续补 async 边界、null host pointer / byte-count 边界，以及更完整的 runtime property/error matrix。

2. `Memory pool and mmap-backed residency`
   - 目标：建立统一 memory pool 语义，并把关键 pool 的底层存储逐步收口到 `mmap` 主线。
   - 当前缺口：pool 分类已存在，但 compatibility virtual window、统一设备内存管理器、`reserve large range + commit on demand` 策略和分阶段落地顺序还未正式写清。

3. `ISA validation expansion`
   - 目标：基于 text asm 生成 kernel 程序，对更多 ISA 指令做“不 crash + 结果正确”的自动验证。
   - 当前缺口：目前更多是按代表性 kernel 和 focused opcode 回归覆盖，缺少更系统的 asm-kernel 验证框架。

4. `Semantic calibration`
   - 目标：让 `st / mt / cycle` 的执行结果和时间语义与正式设计保持一致。
   - 当前缺口：核心语义已稳定，`cycle timeline` 消费层当前已补齐 `arrive_progress(still_blocked/resume)` 透传、runtime/front-end typed marker canonical name 稳定性，以及 representative waitcnt-heavy / barrier-heavy example 校准；后续还需要持续校准更多指令族、memory wait、scheduler 行为与 program stats。

5. `Trace and logging consolidation`
   - 目标：统一日志到 `loguru`，并继续收口 trace canonical / unified / disable-trace 边界。
   - 当前缺口：日志系统仍不统一；text/json trace 虽已可全局关闭，但还需继续确保其完全不依赖业务逻辑。

6. `Lightweight test matrix`
   - 目标：建立完备但轻量级的 runtime / memory / ISA / semantic 测试矩阵，优先覆盖无需完整 HIP kernel 的路径。
   - 当前缺口：已有大量 focused test，但缺少“按模块能力分层”的轻量矩阵整理；异常路径测试暂不作为第一批主线。

7. `Design and status tracking`
   - 目标：持续把模块交互关系、开发计划、已实现/待补强状态写入正式文档，用文档跟踪模块开发状态。
   - 当前缺口：主文档已收口，但这批新需求尚未完全映射到模块状态和优先级。

8. `HipRuntime compatibility naming cleanup`
   - 目标：从语义上移除“interposer 是独立模块”的历史遗留，把它严格收口为 `HipRuntime` 的 LD_PRELOAD C ABI 入口载体。
   - 当前缺口：仍残留在文件名、测试名、CMake target、日志模块名和部分运行时内部命名中。

## 串行 / 并行划分

### 必须串行的关键路径

1. `Runtime API closure`
   - 原因：runtime/memory 接口边界不稳定时，后续 ISA 测试、semantic calibration、trace/log 收口都会反复返工。

2. `Memory pool and mmap-backed residency`
   - 依赖：`Runtime API closure`
   - 原因：memory pool 与地址解析是 runtime memory 行为和 kernel materialize 的共同底座。

3. `ISA validation expansion`
   - 依赖：`Runtime API closure`、`Memory pool and mmap-backed residency`
   - 原因：text-asm kernel 验证需要稳定的 program load、寄存器/内存副作用落地和基础 memory 主线。

4. `Semantic calibration`
   - 依赖：`ISA validation expansion`
   - 原因：没有系统化指令验证，`st/mt/cycle` 的结果对齐和 cycle 参考值校准没有可靠基线。

### 可并行 branch 开发

1. `Trace and logging consolidation`
   - 推荐在 `Runtime API closure` 基本稳定后并行推进
   - 依赖较弱：不应阻塞 runtime/memory/ISA 主线，但不能早于关键接口命名与模块边界稳定。

2. `Lightweight test matrix`
   - 可与 `Runtime API closure`、`ISA validation expansion` 并行推进
   - 形式上依赖各模块接口，但测试框架、目录组织、门禁策略可以提前搭建。

3. `Design and status tracking`
   - 全程并行
   - 每个里程碑完成后都应回写正式文档。

4. `Examples verification`
   - 可在 `Semantic calibration` 进入中后期并行推进
   - 不应早于 runtime/memory/ISA 基础收口，否则 examples 只会放大底层噪声。

## 执行依赖图

```text
Design and status tracking
    |
    +--> Runtime API closure ----> Memory pool / mmap ----> ISA validation ----> Semantic calibration
    |                                   |                        |                      |
    |                                   +----> Lightweight test matrix <---------------+
    |                                   |
    |                                   +----> Trace and logging consolidation
    |
    +--------------------------------------------------------------------> Examples verification
```

## 依赖说明

- `Runtime API closure -> Memory pool / mmap`
  - runtime memory API 的行为边界必须先稳定，pool 才能有可靠的对外约束。

- `Memory pool / mmap -> ISA validation`
  - text-asm kernel 的执行结果验证需要稳定的全局内存、kernarg、code/data residency 主线。

- `ISA validation -> Semantic calibration`
  - 先证明单条/小组合指令在 `st/mt/cycle` 下都正确，之后再讨论 scheduler、stall、program stats 的校准。

- `Trace/log` 不得反向阻塞关键路径
  - 它可以并行推进，但不能定义执行语义，也不能成为 kernel/runtime correctness 的前置条件。

- `Examples verification` 不是最前置
  - examples 是综合验证层，不是底层能力的替代。

- `HipRuntime compatibility naming cleanup` 可并行于主线推进
  - 但应先做语义和文档收口，再做文件名/target/test 名称清理，避免和正在进行的 `include -> src` 合并冲突。

## 终极目标达成条件

终极目标不是“能跑一些 case”，而是同时满足下面四层要求：

1. `Functional correctness`
   - `st/mt` 在支持边界内，对真实 HIP `.out` 程序给出与真实 GPU 一致的最终结果
   - host 侧验证可直接通过

2. `Reference-cycle usefulness`
   - `st/mt` 能输出稳定、可解释、可比较的参考 cycle 值
   - 更换算法、访存组织或 schedule 策略时，参考 cycle 能体现方向正确的变化

3. `Cycle-model refinement`
   - `cycle` 不是第一层 correctness 来源，而是在参考 cycle 基础上进一步模拟资源、stall、dispatch 和 latency
   - `cycle` 输出应比 `st/mt` 更细，但不能与 correctness 主线冲突

4. `Optimization relevance`
   - 最终模型要能作为算子优化和编译器优化的可信参考
   - 也就是说，测试不只验证“不 crash”，还要验证：
     - 结果正确
     - 相对 cycle 趋势合理
     - 调度/访存变化可以在统计和 timeline 上被解释

## 仍需补齐的开发项

### A. Runtime / Memory

- `hipMalloc / hipFree / hipMemcpy* / hipMemset*` 的同步主路径完整化
- fake device pointer -> model address -> pool storage 的统一映射
- compatibility virtual address windows
- 统一设备内存管理器
- `reserve big virtual range + commit on demand`
- `mmap` backed `Global` pool 参考实现
- `Kernarg / Code / RawData / Constant` pool 的统一接口
- segment-oriented `DeviceLoadPlan`
- module 常驻、替换、冲突、卸载语义
- `bss` / zero-init / relocation / symbol binding

### B. Program / ABI

- `.out` / ELF / code object / metadata 的更完整 ingestion
- hidden args / implicit args / system SGPR/VGPR 完整覆盖
- by-value struct / aggregate kernel arg 对齐与打包
- wave launch 初始状态的语义摘要与可验证输出

### C. ISA / Execution

- text asm -> binary -> kernel program 的统一验证框架
- scalar/vector/memory/control/sync/tensor 的指令族系统化覆盖
- decode / disasm / execute 一致性
- encoded mainline 与 modeled mainline 的语义对齐
- shared-heavy / atomic-heavy / barrier-heavy 的稳定主线

### D. Semantic calibration

- `functional st` 的确定性参考语义继续锁定
- `functional mt` 的 wave 级并发、公平性、等待恢复继续锁定
- `ProgramCycleStats` 继续与模型时间语义对齐
- `cycle` 的 `ready / selected / issue`、stall taxonomy、slot timeline 继续细化
- `st/mt/cycle` 对同一程序的结果和趋势关系继续校准

### E. Observation / Tooling

- `loguru` 统一日志
- text/json trace 可关闭且不影响执行事实
- canonical typed trace event + unified trace entry
- 轻量测试矩阵、真实 HIP baseline、examples 综合验证的三层门禁

## 仍需补齐的测试项

### 1. 轻量 runtime / memory tests

- 不依赖 kernel launch
- 覆盖：
  - `Malloc/Free`
  - `MemcpyHtoD / DtoH / DtoD`
  - `Memset`
  - pool address mapping
  - compatibility window range classify
  - `mmap` residency 基本读写

### 2. ISA asm-kernel tests

- 用 text asm 生成单功能 kernel
- 分层覆盖：
  - scalar ALU
  - vector ALU
  - global/shared/private/scalar-buffer memory
  - branch/control
  - barrier/waitcnt
  - atomics
  - MFMA/tensor

### 3. Semantic calibration tests

- 相同 kernel 在 `st/mt/cycle` 下结果一致
- `st` 与 `mt` 的参考 cycle 关系稳定
- 算法或 schedule 变化时，参考 cycle 变化方向合理
- `cycle` 相比 `st/mt` 能给出更细的 stall / slot / issue 解释

### 4. Real HIP executable tests

- 真实 `.out` host 验证通过
- baseline kernel families 持续覆盖：
  - vecadd / fma / bias chain
  - shared reverse / reduce / softmax
  - atomic count
  - conditional multibarrier
  - MFMA

### 5. 综合门禁

- 文档改动：不编译
- 代码改动：轻量 smoke
- 关键模块改动：对应 focused matrix
- 周期性手动 full gate：
  - release
  - debug/asan
  - real HIP examples

## 已明确关闭的历史事务
1. `examples/08 mt Perfetto` 指令切片缺失问题已修复。
2. `examples/11` 编译问题已修复。
3. `timeline.perfetto.pb` 已移出正式用户产物路径。
4. `.cache/example-results` 默认结果路径已移除。
5. `HipInterposerState` 已删除，其职责已并入 `HipRuntime`。

## 正式设计约束摘要
1. `HipRuntime` 是 AMD HIP runtime 兼容层；`ModelRuntime` 是项目核心 runtime；`ExecEngine` 是 `ModelRuntime` 内部执行主链。
2. `trace` 只消费和序列化模型事件，不驱动业务逻辑。
3. `st`、`mt`、`cycle` 的 trace `cycle` 都是模型时间，不是物理真实执行时间戳。
4. `functional st` 采用确定性的 issue quantum 语义；`functional mt` 保留 runnable wave 竞争；`cycle` 明确区分 `ready`、`selected`、`issue`。
5. `cycle model` 是唯一的 cycle 模式，不再拆分 cycle `st/mt`。
6. 历史 plans/specs 只能作为背景材料，当前规范以正式设计文档为准。
7. 重要 runtime API、memory pool、ISA 验证、语义校准、日志与 trace 收口，必须以模块化轻量测试矩阵持续跟踪。

## 备注
- `docs/other_model_design/` 保持不动，继续作为外部参考。
- 以后若有新主线任务，先写进本文件，再扩展实现与测试。
