# 任务计划：当前正式任务主线

## 目标
以当前模块设计为基础，将 `cycle time` 与 `cycle model` 准确性作为第一优先级，持续推进 `ProgramCycleStats`、stall taxonomy、`ready/selected/issue` 与 timeline 解释面；runtime API、memory pool / `mmap`、ISA 指令验证改为按 cycle 主线需求驱动的补充项，并将模块交互关系、开发计划和模块开发状态统一记录到正式文档。

## 当前阶段
阶段 6

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
- [x] 完成 `cycle time` / `cycle model` 准确性主线校准
- [x] 细化 `ProgramCycleStats`、stall taxonomy、`ready/selected/issue` 与 slot timeline 解释面
- [x] 建立 cycle-first 的 representative kernel / example 校准清单
- [x] 按 cycle 主线需求补 runtime API / memory pool / ISA 缺口（审计完成：无 P0 阻塞项）
- [x] 统一日志到 `loguru`（审计完成：`src/` 已统一使用 loguru）
- [x] 继续收口 trace canonical / unified / disable-trace 边界
- **状态：** complete

### 阶段 4：验证与资产整理
- [ ] 建立轻量级 runtime / memory / ISA / semantic 测试矩阵
- [x] 完成 `examples` 剩余分批全量检查
  - 已验证: 01-vecadd-basic, 02-fma-loop, 03-shared-reverse, 04-atomic-reduction, 05-softmax-reduction
  - 已验证: 06-mma-gemm, 07-vecadd-cycle-splitting, 08-conditional-multibarrier, 09-dynamic-shared-sum, 10-block-reduce-sum
  - 已验证: 11-perfetto-waitcnt-slots
  - 所有 11 个 examples 全部通过验证
- [x] 继续清理明显过时、已完成且误导当前主线的历史计划文件
  - 已归档: docs/archive/plans-2026-03/, docs/archive/superpowers-plans-completed/, docs/archive/superpowers-specs-completed/
- [x] 将 archive / active / reference 的文档边界继续收紧
  - 已完成: 所有历史 plans/specs 已归档到 docs/archive/
- [x] 扩展 ProgramCycleStats 支持 gem5/Nsight Compute 风格的性能计数器
  - 已完成: 添加指令计数、内存操作计数、wave 统计、派生指标
- **状态：** in_progress

### 阶段 5：交付与维护
- [ ] 保持 `task_plan.md`、`docs/my_design.md`、`docs/runtime-layering.md`、`docs/module-development-status.md` 同步
- [ ] 将新增实现结果及时回写到正式设计和状态文档
- **状态：** pending

### 阶段 6：全项目架构优化分析
- [x] 恢复当前规划上下文并核对未同步工作树背景
- [x] 盘点顶层模块、公共头、实现层和构建层边界
- [x] 识别当前最主要的架构债：公共/内部边界泄漏、runtime 总控类、状态所有权重叠、artifact ingestion 职责交叉、execution 共享域模型不纯、构建边界过弱
- [x] 将分析落盘到 `docs/architecture/project_architecture_refactor_analysis.md`
- [x] 将分析文档接入 `docs/README.md`，并补各阶段完成判定，避免分析文档成为孤立结论
- [ ] 后续如进入实现，按 Phase 1 `边界清理` 优先推进
- **状态：** in_progress

## 当前正式任务清单
1. `Semantic calibration`
   - 目标：把 `cycle time` 与 `cycle model` 准确性提到第一优先级，让 `st / mt / cycle` 的结果、时间语义和 program-level 解释面保持一致。
   - 当前状态：**已完成** - 核心语义已稳定，execution 语义、recorder 协议、consumer 收口均已完成。

2. `Trace and logging consolidation`
   - 目标：统一日志到 `loguru`，并继续收口 trace canonical / unified / disable-trace 边界。
   - 当前状态：**已完成** - `src/` 已统一使用 loguru；trace canonical event model 已完成。

3. `Lightweight test matrix`
   - 目标：建立完备但轻量级的 cycle / semantic / observation 测试矩阵，并继续维持 representative kernels / examples 的 calibration baseline。
   - 当前缺口：已有大量 focused test，但缺少”按 cycle accuracy 目标分层”的矩阵整理。

4. `Runtime API closure`
   - 目标：在不再作为默认最前置任务的前提下，按 cycle 主线需要补 runtime / memory API 框架。
   - 当前状态：审计完成，无 P0 阻塞项。P1 缺口（`hipMemGetInfo`, `hipHostMalloc/Free` 等）可按需补充。

5. `Memory pool and mmap-backed residency`
   - 目标：建立统一 memory pool 语义，并把关键 pool 的底层存储逐步收口到 `mmap` 主线。
   - 当前状态：审计完成，无 P0 阻塞项。P1 缺口（`mmap`-backed Global pool）可按需补充。

6. `ISA validation expansion`
   - 目标：基于 text asm 生成 kernel 程序，对更多 ISA 指令做”不 crash + 结果正确”的自动验证。
   - 当前状态：审计完成，无 P0 阻塞项。P1 缺口（Buffer/Flat load/store）可按需补充。

7. `Design and status tracking`
   - 目标：持续把模块交互关系、开发计划、已实现/待补强状态写入正式文档。
   - 当前状态：主文档已收口，阶段3完成项已回写。

8. `HipRuntime compatibility naming cleanup`
   - 目标：从语义上移除”interposer 是独立模块”的历史遗留。
   - 当前缺口：仍残留在文件名、测试名、CMake target、日志模块名中。

9. `Architecture boundary cleanup`
   - 目标：把“正式设计已经确定的边界”真正落实到代码结构、公共头、状态所有权和构建边界中。
   - 当前缺口：
     - 公共头仍直接依赖 `internal/*`
     - `ExecEngine / ModelRuntime / RuntimeSession` 仍偏总控类
     - `MemorySystem / DeviceMemoryManager / ModelRuntime` 仍有所有权重叠
     - `ObjectReader / encoded_program_object.cpp` 仍把外部工具调用、解析、组装耦在一起

## 串行 / 并行划分

### 必须串行的关键路径

1. `Semantic calibration`
   - 原因：用户已明确 `cycle time` 与 `cycle model` 准确性是当前第一优先级，后续所有补项都应服务于它。

2. `Trace and logging consolidation`
   - 依赖：`Semantic calibration`
   - 原因：需要先明确 cycle 语义与统计口径，再决定观察层该如何稳定暴露这些事实。

3. `Examples verification`
   - 依赖：`Semantic calibration`、`Trace and logging consolidation`
   - 原因：examples 是 cycle accuracy 的综合验证层，应在语义与观察面基本稳定后推进。

4. `Runtime API closure / Memory pool / ISA validation`
   - 依赖：具体阻塞项
   - 原因：三者不再默认构成最前置串行主链；只有当某个 cycle 校准 case 被明确阻塞时，才插入小步补项并回到 cycle 主线。

### 可并行 branch 开发

1. `Runtime API closure / Memory pool / ISA validation`
   - 改为按需并行补项
   - 依赖具体 cycle case：只补当前 cycle 校准真正依赖的那部分，不先展开完整 closure。

2. `Lightweight test matrix`
   - 可与 `Semantic calibration` 并行推进
   - 形式上依赖各模块接口，但测试框架、目录组织、门禁策略可以先按 cycle-first 目标整理。

3. `Design and status tracking`
   - 全程并行
   - 每个里程碑完成后都应回写正式文档。

4. `Examples verification`
   - 可在 `Semantic calibration` 进入中后期并行推进
   - 不应早于 cycle 语义与 trace/ProgramCycleStats 观察面基本稳定，否则 examples 只会放大解释噪声。

## 执行依赖图

```text
Design and status tracking
    |
    +--> Semantic calibration / cycle accuracy ----> Trace+ProgramStats observability ----> Examples verification
    |                    |                                       |
    |                    +--------------> Lightweight test matrix+
    |                    |
    |                    +--------------> Runtime / Memory / ISA supplements (on demand)
    |
    +--------------------------------------------------------------------> docs / status updates
```

## 依赖说明

- `Semantic calibration` 是当前主驱动
  - 后续补项都必须先回答“它是否直接提升 cycle time / cycle model 准确性”，否则不应抢占主线。

- `Runtime / Memory / ISA` 改为依赖驱动
  - 如果某个 cycle 校准 case 被 runtime 行为、residency 语义或 ISA baseline 卡住，再做有边界的补项，而不是先展开完整 closure。

- `Trace/log` 不得反向阻塞关键路径
  - 它可以并行推进，但不能定义执行语义，也不能成为 cycle correctness 的前置条件。

- `Examples verification` 不是最前置
  - examples 是 cycle accuracy 的综合验证层，不是主线语义校准的替代。

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
6. `trace canonical event model` 已完成：`TraceEventView` 提供统一 typed-first 解释。
7. `trace unified entry` 已完成：语义工厂函数统一 trace 构建入口。
8. `async memory arrive flow` 已完成：flow_id/flow_phase 支持 Perfetto flow 事件。
9. `perfetto stall taxonomy` 已完成：`TraceStallReason` 统一 stall 分类。
10. `perfetto slot-centric timeline` 已完成：`TraceSlotModelKind` 区分 functional/cycle slot 语义。
11. `program cycle stats calibration` 已完成：`ProgramCycleStats` 与模型时间语义对齐。
12. `multi-wave dispatch front-end alignment` 已完成：waiting-wave dispatch eligibility 统一。
13. `functional mt wave scheduler` 已完成：`ApSchedulerState` 实现 wave-granularity 调度。
14. `cycle model calibration followup` 已完成：execution 语义、recorder 协议、consumer 收口。

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
