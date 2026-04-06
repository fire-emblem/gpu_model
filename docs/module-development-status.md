# Module Development Status

## 文档用途

这份文档是项目主线开发看板。

后续开发要求：

1. 新功能开始前，先确认它属于哪个顶层模块
2. 代码合入后，必须同步修改本文件中的模块状态
3. 如果目标边界发生变化，先改本文件，再改代码

## 当前主目标

当前主目标以 **functional model** 为主线，同时保持 **naive cycle model** 可持续演进和可验证。

主线术语约定：

- `runtime`: `HipRuntime / ModelRuntime / ExecEngine`
- `program`: `ProgramObject / ExecutableKernel / EncodedProgramObject`
- `instruction`: decode / disasm / instruction object / lowering
- `execution`: `FunctionalExecEngine / CycleExecEngine / EncodedExecEngine / WaveContext`
- `arch`: architecture spec / topology / device properties
- 历史已删除名：`ModelRuntimeApi / RuntimeHooks / HostRuntime`

目标定义：

- 支持 `1` 个 model device
- 支持 `1` 个 context
- 支持 `1` 个 stream
- 支持命令行直接执行 `hipcc` 编译得到的 HIP 可执行程序
- host 侧 CPU 逻辑原生执行
- runtime API 进入 model runtime
- kernel 进入 model 执行
- 结果正确返回 host

目标能力范围：

- HIP runtime 层
- module / ELF / code object 加载层
- GCN ISA 二进制解析 / 反汇编 / 执行层
- wave / block / device 执行层
- 内存 / 寄存器 / 同步 / trace 层
- cycle / issue / waitcnt / timeline 观察层

## 当前正式任务主线

当前正式 task tracks 只保留以下八项：

1. `runtime API closure`
2. `memory pool / mmap residency`
3. `ISA validation expansion`
4. `semantic calibration`
5. `trace canonical event model`
6. `trace unified entry + disable-trace boundary`
7. `unified logging`
8. docs / status tracking

说明：

- 历史计划中的已完成事项不再单独作为当前待办保留。
- 若某模块状态已完成，但仍有“解释边界”“测试门禁”“观察语义”未闭环，可并入上述 track，而不是再新建历史式临时任务。

## 当前重构结论

截至 `2026-04-01`：

- `runtime/program` 主线重构已基本完成
- `instruction/execution` 的公开命名、实现主路径、测试主路径已完成收口
- `exec/*` 与 `decode/*` 旧顶层代码目录已迁空并删除
- 全量 `gpu_model_tests` 当前结果为 `559 passed`

当前可以认为：

- 主线目录重构已经基本完成
- `instruction/encoded/internal/*` 与 `execution/internal/*` 现视为长期内部层
- 剩余尾项主要转为历史文档整理和后续功能覆盖，而不是旧目录回收

原因：

- `exec/*` 与 `decode/*` 旧目录主干已经迁空并删除
- `instruction/encoded/internal/*` 与 `execution/internal/*` 作为长期内部层保留
- 仍有部分 GCN-specific 文件名和术语保留在底层 encoded 子层
- 历史计划与参考文档仍保留大量旧术语，不等于当前代码事实

因此，当前状态应判断为：

- 主线公开接口重构：`已完成`
- 主线目录层级重构：`已完成`
- 长期内部层划分：`已确定`
- 历史文档最终整理：`进行中`
- 整体代码重构按主线目标：`已完成`

## 第一阶段边界

第一阶段必须支持：

- 单卡
- 单 context
- 单 stream
- 同步执行
- device property 查询
- host/device 同步内存拷贝
- global / constant / data / kernarg / shared / private 内存
- scalar register file
- vector register file，按 wave `64` lane 建模
- `1D / 2D / 3D` grid 和 block
- ELF / HIP fatbin / code object 解析
- metadata 二进制解析与赋值
- LLVM AMDGPU 调用约定下 wave 初始特殊寄存器装载
- 全部 GCN ISA 的解析 / 执行 / 反汇编
- trace / log
- wave 启动时打印初始化寄存器值、wave id、block id、dpc/ap/peu
- module load
- functional `st` 和 `mt`
- tensor core / MFMA 计算
- barrier / wave barrier / 基本同步

第一阶段明确暂不支持：

- 多卡
- 多 context
- 多 stream 并发
- async memcpy
- async stream overlap
- graph
- peer-to-peer
- image / sampler / texture
- 复杂 debug API

## 状态定义

- `Done`
  - 该模块已满足第一阶段目标边界
- `Partial`
  - 已有主干代码，但还不足以支撑“任意 HIP 可执行程序”
- `Not Started`
  - 基本没有进入实现
- `Deferred`
  - 当前阶段明确暂缓

## 顶层模块状态

| ID | 模块 | 第一阶段目标 | 当前状态 | 已完成 | 仍缺失 |
|---|---|---|---|---|---|
| `M0` | 架构规格与设备建模 | `c500` 设备规格、单卡、wave64、DPC/AP/PEU 层级、device property 对外可查询 | `Done` | 已有 `c500` 架构注册、`8 DPC x 13 AP/DPC x 4 PEU/AP` 拓扑、wave64、层级放置，以及统一的 device property / attribute 查询接口与 model-native runtime facade | 仍需后续新架构参数化扩展，但不阻塞第一阶段 |
| `M1` | Runtime 基础（`HipRuntime / ModelRuntime / ExecEngine`） | 单卡、单 context、单 stream、同步 runtime 入口、基本错误码与设备选择 | `Partial` | 已有 `ModelRuntime` facade、`ExecEngine` 主路径、`HipRuntime` 路径、`hipMalloc/hipFree/hipMemcpy/hipMemcpyAsync/hipMemset/hipMemsetAsync/hipLaunchKernel/hipGetDeviceCount/hipGetDevice/hipSetDevice/hipGetDeviceProperties/hipDeviceGetAttribute`、基础 stream/event 空实现，以及 model-native 统一 `LoadModule` 入口；`.out/.o` 已可显式选择 encoded program object 或 lowered modeled program object 路径；`LD_PRELOAD` 下的 `HipRuntime` C ABI 入口已显式固化“默认流 + 单显式 stream”边界，并对销毁后/无效 stream handle 返回错误 | 还缺 property / attribute 更完整覆盖；还缺 context 边界文档化与统一限制；还缺更完整同步 runtime 子集梳理 |
| `M2` | ProgramObject / ELF / Code Object 加载 | 支持 module load、ELF 解析、fatbin / `.out` / code object 装载、const/data 段装载、metadata 二进制解析 | `Partial` | 已有 `ProgramObject`/bundle、ELF/code object loader、device load plan/materialize、artifact path 路径；已能从 code object/`.out` 解析 kernel descriptor、metadata、kernarg size、hidden arg layout、descriptor symbol；统一 `LoadModule` 已支持 `Auto/AmdgpuObject/ProgramBundle/ExecutableImage/ProgramFileStem`；kernel metadata 已开始收敛到 typed 结构并被 loader / encoded launch / runtime 复用；runtime 侧已具备按 module 名装载、卸载、枚举 module/kernel 的基础生命周期接口 | 还缺完整 ELF section/program header/relocation 覆盖；还缺 metadata 字段的进一步系统化 typed 覆盖；还缺更完整的模块常驻/替换/冲突语义约束 |
| `M3` | GCN 二进制 decode / disasm | 基于连续 `.text` 二进制高效解析全部 GCN ISA，输出结构化 decode 与反汇编 | `Partial` | encoded GCN decode/disasm 主体已迁入 `instruction/encoded/*`；已具备 encoded instruction 提取、format classify、encoding def、decoder、formatter；已支持 `text bytes -> encoded instruction array -> decoded instruction array -> instruction object array` 主路径；compute-focused真实 HIP kernel 的 decode/disasm 已覆盖到 `vecadd/fma_loop/bias_chain/shared_reverse/softmax_row/mfma`；shared-heavy 真实 HIP case 的 decode/binding 缺口已继续收口到 `shared_reverse + dynamic_shared_sum + block_reduce_sum + softmax_row` 四锚点；decode 旧顶层目录已删除 | 还缺“全部 GCN ISA” encoding 覆盖；还缺更系统的 bitfield/union 定义；还缺 graphics/image/export/interp family 的深入覆盖；还缺高性能批量 decode 路径校验 |
| `M4` | GCN ISA 语义执行 | 支持全部 GCN ISA 的 functional 执行，包括标量、向量、访存、控制流、同步、LDS、MFMA | `Partial` | instruction ISA functional/cycle 已覆盖较多基础指令；encoded instruction 路径已支持真实 `.out` compute kernel 主线；已支持 `vecadd/fma_loop/bias_chain/shared_reverse/softmax_row/mfma`；decode 阶段已完成 `op_type -> opcode -> concrete instruction object` 工厂实例化；`instruction/*` 与 `execution/*` 主命名已成为公开主路径；`exec/*` 旧目录主干已迁入 `instruction/*` 与 `execution/*`；此前 6 个 instruction/execution/MFMA 相关失败测试已清零；shared-heavy 真实 HIP 执行主线已收口到 `shared_reverse + dynamic_shared_sum + block_reduce_sum + softmax_row` 四锚点，不再只依赖单一 case | 距离“全部 GCN ISA 执行”仍有差距；graphics family/descriptor family 仍主要占位；仍需继续做系统化 opcode 覆盖与归类 |
| `M5` | LLVM AMDGPU ABI / wave 启动 | 正确读取 kernarg、hidden args、special SGPR/VGPR、block/thread/grid 维度、wave 启动初值 | `Partial` | 已有 descriptor + metadata 驱动的 wave 初始 SGPR/VGPR preload；已支持 kernarg segment ptr、workgroup id、workitem id、hidden block/group args、`x/y/z` grid-block launch 维度、encoded `.out` launch ABI 主线；kernarg 打包已抽成公共模块；visible arg offset/aggregate、`3D hidden args`、fallback encoded ABI kernarg 约定已有回归覆盖；真实 `hipcc` 生成的 `3D hidden-arg` 与 `3D builtin-id` encoded 路径现已在 decode / interposer / runtime 三层 focused regression 下可执行验证；`WaveLaunch` trace 已能对已知 ABI 字段输出 `kernarg_ptr/wg_id_x/y/z/workitem_id_x/y/z` 语义摘要，并对未知字段保留原始寄存器回退 | 还缺更完整的 system SGPR/VGPR 集合；还缺更多 target-specific ABI 差异；还缺 wave 启动寄存器 trace dump 进一步细化 |
| `M6` | Functional 执行核心 | 单线程和多线程共用一套 functional core；支持 wave/block/device 层级执行；支持 `st/mt` 切换 | `Partial` | 已有共享 `FunctionalExecEngine` 核心；`st/mt` 已共核；已有 PEU-local wave pool、round-robin、block 内 shared/barrier kernel 的 `mt` 路径；marl 已接入；`1D/2D/3D` launch 配置、placement 和 `xyz` builtin 主线已打通；真实 `hipcc` 生成的 `3D vecadd + 条件边界 + 小计算量` 程序已能经 `.out -> lowered modeled program object` 路径在 `st/mt` 下对比执行；内部执行支撑件已稳定收敛到 `execution/internal/*`；已新增显式 wave run state，`FunctionalExecEngine` 的 `waitcnt` 进入等待、memory wait 恢复、barrier release 恢复现已收敛到共享状态契约，waiting-wave 扫描入口也已在 `st/mt` 下统一复用，相关 focused regression 已锁定；同 `PEU` 上显式进入 `Waiting` 的 wave 现在也已有 focused regression 证明不会阻塞 ready sibling，barrier release 后的早到 wave 会重新进入 dispatch；functional `st/mt` 现已从 trace 计数时间收口到 modeled time：纯 scalar `100` 指令、dense global load overlap、implicit drain、wait/resume quantum 语义都已有 focused regression；`Functional st` 作为确定性的参考模型，`arrive_resume` 后在下一 issue quantum 起点消费，`Functional mt` 则保留 runnable wave 竞争；`ProgramCycleStats` 实际运行统计与大 block / 多 wave / 多 block / 非对称 wave work 的默认成本口径回归已锁定；`MarlParallel` 在大 block 且 worker_threads 偏小时已收口到稳定的 block-level 并行执行，避免多 block 大规模回归死锁 | 还缺更多 wait reason 扩展；还缺对任意 HIP 程序的大规模稳定性验证 |
| `M7` | 内存系统与地址空间 | global/shared/private/constant/kernarg/data/managed 独立地址空间，host/device 拷贝与 map 映射 | `Partial` | 已有多 memory pool、managed、kernarg、constant、device load materialize、host/device 基本 memcpy、fake device ptr 到 model addr 映射 | 还缺统一的 `mmap` backed pool 主线；还缺不同 memcpy 方向与同步语义的轻量测试矩阵；还缺 data/const/bss/relocation 更完整装载；还缺 map/unmap 语义完善 |
| `M8` | 同步、barrier、atomic | block barrier、wave barrier、global/shared/private 基本同步与常用 atomic | `Partial` | 已有 `s_barrier`、wave barrier、shared/global atomic add、shared memory barrier kernel 测试、functional `mt` 条件变量等待；barrier wait/release 现已统一落到显式 `run_state/wait_reason` 状态机，functional、cycle、raw-GCN 主线的 shared/barrier 回归已重新收口；`waitcnt` 的 `global/shared/private/scalar-buffer` memory-domain 与 barrier 现已共享同一 waiting/runnable 状态契约 | 还缺更多 atomic 指令覆盖；还缺更完整 waitcnt 领域与同步语义；还缺 encoded GCN 路径的系统同步覆盖（legacy raw GCN 路径仍需兼容）；还缺更完整同步 CTS |
| `M9` | Tensor / MFMA | 支持 tensor core / MFMA 指令解析、反汇编、执行与结果验证 | `Partial` | 已有 `v_mfma_f32_16x16x4f32` 最小路径和 probe/test；encoded semantic 已覆盖 `f32/f16/bf16/i8` 多个 MFMA 变体与 shape；kernel descriptor / metadata 已能暴露 typed tensor ABI（如 `agpr_count/accum_offset`）；真实 `hipcc` 生成的 MFMA executable 已在 runtime / interposer / CTS 主线上验证 | 还缺 MFMA 指令族更系统覆盖；还缺寄存器布局、累加器语义与更多 datatype/shape 支持；还缺从“代表性真实 kernel”走向更广覆盖的 tensor 回归矩阵 |
| `M10` | Trace / Log / Debug | 支持详细 log、instruction trace、wave launch trace、寄存器值打印、层级信息打印 | `Partial` | 已有 trace sink、file/json trace、ASCII timeline、Google trace、instruction trace、cycle timeline；`FunctionalExecEngine / EncodedExecEngine / CycleExecEngine` 三条当前执行 backend 已统一发出 `WaveLaunch` 事件并带初始 `WaveContext` 状态摘要；`WaveLaunch` 单行摘要已提升为“ABI 语义优先、原始寄存器回退”，raw-GCN/encoded 路径可直接观察 `kernarg_ptr`、`wg_id_*`、`workitem_id_*` 等字段；usage 脚本已能稳定导出 encoded decode 与 HIP interposer 主线结果；functional trace 已新增 `WaveStats` 快照，可观察 wave launch/init/active/end 生命周期进度，且已进一步拆分 `runnable/waiting` 并由 barrier / waitcnt regression 锁定；统一 wait-state machine 的 `Stall + WaveStats` 语义现已被 focused trace 回归直接覆盖；cycle 路径 `Stall` 事件现已统一使用稳定的 `reason=` schema，Perfetto/timeline focused 回归已锁定 `waitcnt_global` 与 barrier-heavy path 上的 stall 可观测性；runtime `LaunchResult` 现已统一暴露 `ProgramCycleStats`，便于直接查看 functional `st/mt` 的实际程序 cycle 统计；`timeline.perfetto.json` 现已新增结构正确性、`function mt` waiting/resume 语义，以及真实 `128 x 128 conditional multibarrier` HIP case 的 sanity 回归 | 还缺 `loguru` 统一日志主线；还缺更完整的 wave 启动初始寄存器 dump；还缺 text/json trace 完全与业务逻辑解耦的收口；还缺 encoded / functional / runtime 三条路径的统一 trace 格式进一步收敛 |
| `M11` | 命令行 `.out` 执行闭环 | `LD_PRELOAD` 后，任意第一阶段边界内 HIP 可执行程序可直接命令行执行 | `Partial` | 已有 host `main()` 原生执行 + `HipRuntime` C ABI 入口 + kernel 进入 model 的闭环；真实 `.out` 已验证 `vecadd/fma_loop/bias_chain/by_value_aggregate/atomic_count/shared_reverse/dynamic_shared_sum/block_reduce_sum/softmax_row/mfma`；兼容路径与 `LD_PRELOAD` 路径均有 CTS 覆盖；基础 property 查询与 model-native module API 已打通 | 还缺更完整 runtime API；还缺“任意 HIP 程序”所需的完整 decode/exec/runtime 覆盖 |
| `M12` | 测试与状态门禁 | 用例矩阵、真实 HIP 程序、encoded decode、runtime、CTS、回归门禁 | `Partial` | 已有 gtest 统一测试；`HipRuntimeTest.*`、真实 HIP `.out` 兼容路径测试、encoded decode usage、主 CTS 和 feature CTS 均已打通；当前 fresh 全量 gate 已通过；`LD_PRELOAD` 路径现为通过态；shared-heavy `shared_reverse + dynamic_shared_sum + block_reduce_sum + softmax_row` 现已形成四锚点门禁，其中 `shared_reverse` 与 `dynamic_shared_sum` 已在 decode / hip runtime / model runtime 三层收口，并进入 HIP CTS 与 feature CTS 快速矩阵，`block_reduce_sum` 已补齐 decode / hip runtime / model runtime focused regression，并进入主 HIP CTS 快速矩阵，`softmax_row` 已具备 decode / hip runtime / model runtime focused regression、主 HIP CTS、parallel execution regression 和 example 闭环；atomic-heavy `atomic_count` 现也已进入主 HIP CTS 快速矩阵；ABI-heavy `by_value_aggregate` 现已补齐真实 HIP decoder regression，并进入主 HIP CTS 快速矩阵；`3D hidden args / builtin ids` 现也已补齐 decode / hip runtime / model runtime focused regression；executed-flow `ProgramCycleStats` 现已有 tracker-focused synthetic tests、runtime-focused 成本口径 tests，以及 `128 x 128 conditional multibarrier` HIP baseline 校准回归 | 还缺轻量级 runtime/memory/ISA/semantic 分层测试矩阵；还缺 text-asm kernel 指令验证框架；还缺 decode/disasm/ABI/property/module-load 专项测试归档；还缺状态与模块看板绑定的验收标准 |
| `M13` | Cycle model | 完整 cycle 建模、issue/latency/waitcnt/event/timeline | `Partial` | 已有 naive cycle 主干、issue model、waitcnt 领域阻塞、event queue、timeline、Google trace、cache/bank conflict/waitcnt cycle 测试；cycle front-end 现已具备每个 `AP` 最多 `2` 个 resident blocks`、每个 `PEU` `active_window = 4`，以及显式 `resident_wave_slots_per_peu = 8` 的参考模型，并已有 focused regression 锁定 standby promotion 与 resident backfill 语义，因此 cycle mode 已可达到单 `PEU` `>4` resident waves；modeled 与 encoded 两条 cycle 主循环现已开始用 `IssueScheduler` 对同一 `PEU` 的多 wave candidate 做 bundle selection，并保持 shared-reverse / softmax / atomic 等 encoded cycle regression 结果稳定；`CycleTimingConfig` / `ExecEngine` / arch spec 现已纵向打通 `ArchitecturalIssuePolicy`，使默认 policy、runtime `SetCycleIssuePolicy(...)`、`SetCycleIssueLimits(...)` 与 modeled/encoded 两条 cycle front-end 的 bundle 选择保持一致，并已有 focused regression 锁定“policy-only 放宽 vector bundle”“limits override 优先于 policy type_limits”，以及“仅覆盖 issue limits 时仍保留 spec-level issue grouping”的行为；`c500` 默认 policy 现已把 `branch` 与 `special` 合并到同一 issue group，并由 spec-level 与 scheduler-level focused regression 锁定默认共享组语义；`waitcnt/dependency/front_end_wait` 阻塞 wave 现保持 resident 且留在 in-window，不再错误让出 front-end 窗口；barrier waiting resident waves 会让出 active slots，并在 barrier release 后重新进入 active window；barrier wait/release 状态机已与 shared/barrier cycle kernel、block reduction、softmax reduction、transpose 等回归重新对齐；`barrier_slots_per_ap = 16` 现已通过 barrier-generation/context 资源 helper 接入 modeled/encoded cycle barrier arrival/release 路径，并有 helper-focused regression 锁定 acquire/release 语义；当前 cycle 结果型证明也已形成闭环：front-end latency 推进不依赖 trace、resident/backfill/promote 正确、dense global load overlap 正确、`endk` 隐式 drain 正确、`ready != selected != issue` 也已有 focused regression 锁定；functional `st/mt` 现已新增与 cycle mode 结果形状兼容的 executed-flow program cycle 统计，可用于整程序运行统计对照与后续校准，且已覆盖大 block / 多 wave / 多 block 的 mixed-cost 与非对称 workload 回归；`global/shared/scalar-buffer` 不同 memory-arrive + waitcnt，以及 single/double barrier 的程序统计增长也已由 focused regression 锁定；代表性 case 的排序、分类统计求和、自 `st/mt` 模式一致性已由 focused regression 锁定，可用于查看 program-level cycle 统计趋势；当前还新增了 `128 x 128 conditional multibarrier` HIP baseline，用于在真实 `hipcc` 程序上对照 `st/mt/cycle` 的 program cycle 统计形状与多 barrier 条件路径稳定性；cycle timeline / Perfetto dump 现已新增 issue/arrive/commit/stall 顺序合理性的 focused 回归；cycle 路径 `Stall` 事件现已统一使用稳定的 `reason=` schema，Perfetto/timeline focused 回归已锁定 `waitcnt_global` 与 barrier-heavy path 上的 stall 可观测性 | 仍缺更完整的架构资源冲突、更多 memory domain/pipe 细节、更完整的 cycle-path reason taxonomy 规范化与 focused coverage、与真实硬件差异说明和参数化建模文档 |

## 当前阶段总评
当前项目距离“更完整的功能覆盖与长期维护形态”仍有差距，但主线重构和当前边界内能力已经完成。

同时需要明确：

- 如果以“公开主路径命名、顶层目录层级、主测试路径、当前全量验证结果是否收口”为标准，本轮重构已经完成
- 如果以“彻底减少底层 GCN-specific 术语、整理全部历史文档”为标准，仍有收尾项

当前最关键的缺口不是单点 bug，而是下面八个面：

1. `Runtime APIs`
   - 重要 API 覆盖、不同 memcpy/memset 语义、同步主路径框架
2. `Memory`
   - memory pool / `mmap` backed residency / map-unmap 语义
3. `Instruction validation`
   - text-asm kernel 指令验证框架
4. `Functional`
   - `mt` scheduler 的公平性、竞争行为和解释边界
5. `Cycle`
   - stall taxonomy、`ready/selected/issue` 可观测性、timeline 解释面
6. `Program statistics`
   - `ProgramCycleStats` 与当前模型时间语义对齐
7. `Trace + Log`
   - canonical/unified/disable-trace/loguru 收口
8. `Verification + Docs`
   - 轻量测试矩阵、examples、文档状态跟踪

当前阶段额外约束：

- 异常路径测试暂不作为第一批主线
- 先落接口框架、模块边界和主测试 list，再逐步推进实现

补充说明：

- `WaveStats`、wait/resume 状态机、waitcnt wait reason、resident-block bring-up、conditional multibarrier example bring-up、executed-flow stats bring-up 这些主题已视为“基础主干已落地”，后续只作为正式主线的背景，不再单独维持计划文件。

## 从历史计划提炼出的稳定 backlog 主题

下列主题虽然不再单独保留为“当前活跃计划文件”，但仍然是需要持续观察的稳定 backlog：

1. `M2`
   - LLVM / AMDGPU artifact ingestion
   - segment-oriented load plan
   - module 生命周期与段装载语义
2. `M3`
   - encoded decode / project disasm 完整化
   - bitfield / encoding definition 体系继续完善
3. `M4 + M12`
   - shared-heavy 真实 HIP 路径继续作为 coverage / regression 锚点维持
   - 但不再单独保留 bring-up 计划文件
4. `M5`
   - ABI 闭环、wave launch 语义摘要、hidden/system args 继续作为稳定 backlog 维持
   - 但不再单独保留早期 closure 设计文档
5. `M6`
   - `functional mt` 的 wave 级调度、公平性、显式 wait/resume 原语
6. `M13`
   - PEU / wave issue model
   - cycle 顶层 timing / issue / stall 分类
   - `ready / selected / issue` 观测语义
7. `M10 + M12`
   - canonical trace construction
   - typed event interpretation
   - slot-centric timeline / Perfetto 观察面
8. `M12 + M13`
   - `128 x 128 conditional multibarrier` 作为 program stats 与 barrier-heavy 稳定性基线

这些主题已经提炼进正式设计文档；后续优先直接修改正式文档，不再依赖历史计划充当规范。

## 当前推进顺序

后续开发优先按下面顺序推进：

### Step 1

先补 runtime + memory 基础：

- runtime 重要 API
- 不同 memcpy / memset 变体
- memory pool / `mmap` 主线
- 无 kernel launch 的主测试 list 与框架

理由：

- 这是后续 kernel、program load、ISA 测试和日志/trace 收口的公共底座

### Step 2

补 instruction validation：

- text-asm kernel 程序生成
- 不 crash 验证
- 结果正确性验证

理由：

- 需要把 ISA 实现从“代表性 case”推进到“系统化指令验证”

### Step 3

补 `functional mt` + `cycle` 语义校准：

- `functional mt` scheduler 公平性与等待恢复
- `cycle` 的 stall taxonomy
- `ready / selected / issue`
- `st / mt / cycle` 结果语义对齐

理由：

- 指令与 runtime 基础稳定后，才能做真正可信的跨模型校准

### Step 4

补 trace / log 主线：

- canonical typed event model
- unified trace entry
- disable-trace 边界
- `loguru` 统一日志

理由：

- 需要统一观察面，但不能让观察层反向污染业务逻辑

### Step 5

补测试矩阵、examples 与文档状态跟踪：

- 轻量测试矩阵
- examples 分批全量验证
- 模块开发状态持续写回文档

理由：

- 否则实现会继续分散在 case-by-case 修补里

## 串行与并行开发关系

### 串行关键路径

1. `M1 -> M7`
   - runtime API 与 memory pool / `mmap` 必须先收口
2. `M1/M7 -> M3/M4`
   - memory/runtime 主线稳定后，才能建立可靠的 ISA asm-kernel 验证
3. `M3/M4 -> M6/M13`
   - 指令验证形成系统基线后，才能做 `st/mt/cycle` 语义校准

### 可并行分支

1. `M10`
   - trace canonical / unified / disable-trace / `loguru`
   - 可在 `M1` 稳定后并行推进，但不能反向定义执行语义
2. `M12`
   - 轻量测试矩阵框架、门禁组织、目录收口
   - 可与 `M1`、`M3` 并行推进
3. docs 跟踪
   - 全程并行

### 依赖图

```text
M1(runtime) ---> M7(memory) ---> M3/M4(ISA decode/exec validation) ---> M6/M13(semantic calibration)
      |               |                     |                                   |
      |               +------> M12(test matrix) <-------------------------------+
      |
      +------> M10(trace/log)   [parallel branch, non-blocking for correctness]
      |
      +------------------------------------------------------------> examples/docs integration
```

### 原则

- correctness 主线优先，observation 主线并行但不阻塞
- unit / lightweight tests 前置，examples 后置
- 模块交互边界先稳定，再扩张综合验证

## 当前必须补充但用户需求里没有明确写出的点

为了让“任意 HIP 可执行程序”更接近可执行，还需要补下面这些点：

- by-value struct / aggregate kernel arg 的对齐和打包规则
- metadata 中 hidden args / implicit args 的完整覆盖
- ELF relocation / symbol 绑定 / section 到 device 地址的关系
- module 生命周期中的替换、覆盖、冲突与常驻策略
- tensor / MFMA 从代表性 probe 扩展到更系统的 shape 和 datatype 覆盖
- `bss` / zero-init 段处理
- 更完整的 global/shared/private/scalar-buffer atomic 指令族
- 更完整的错误码和 runtime 返回行为
- kernel / module / device 属性查询接口
- encoded decode / disasm / exec 的一致性测试（legacy raw GCN 路径需保持兼容）
- wave launch 时的系统寄存器初始化 trace

## 面向终极目标的阶段门槛

### Gate A：结果正确闭环

通过条件：

- `st` 在支持边界内对真实 HIP `.out` 给出正确结果
- `mt` 在同一边界内给出与 `st` 一致的结果
- host 侧验证不依赖 trace / log

阻塞项：

- runtime/memory 主路径不稳
- ABI/hidden args 不完整
- 指令执行覆盖不足

### Gate B：参考 cycle 可用

通过条件：

- `st/mt` 能给出稳定的 `ProgramCycleStats`
- 更换算法、访存组织、schedule 后，参考 cycle 有方向正确的变化
- 关键差异能被 waitcnt / memory / issue / barrier 语义解释

阻塞项：

- ISA asm-kernel 验证不足
- `functional mt` scheduler 语义未锁定
- `ProgramCycleStats` 口径漂移

### Gate C：cycle 模式增强

通过条件：

- `cycle` 不改变 correctness
- `cycle` 比 `st/mt` 提供更细的 slot / stall / dispatch / latency 解释
- `cycle` 与 `st/mt` 保持趋势一致，但允许更细粒度估算

阻塞项：

- `ready/selected/issue` 语义未稳定
- stall taxonomy 不稳定
- slot timeline 不稳定

### Gate D：优化参考可信

通过条件：

- baseline kernels 与真实 GPU 的结果一致
- 参考 cycle 能区分算法与 schedule 的优劣
- trace/log/timeline 可以解释结论，而不是制造结论

阻塞项：

- 真实 HIP baseline 不足
- 测试矩阵不成体系
- 文档状态与代码现状脱节

## 仍需补齐的测试体系

### T1. 轻量 runtime / memory matrix

- 无 kernel launch
- `malloc/free/memcpy/memset`
- pool mapping
- `mmap` residency

### T2. asm-kernel ISA matrix

- text asm -> binary -> kernel
- 指令族逐类覆盖
- 不 crash + 结果正确

### T3. semantic calibration matrix

- `st/mt/cycle` 结果一致性
- reference cycle 趋势
- scheduler / stall / wait 语义

### T4. real HIP executable matrix

- 真实 `.out`
- baseline kernels
- host 侧精确验证

### T5. gate hierarchy

- docs-only skip
- light smoke
- focused module matrix
- periodic full gate

## 下一次状态更新规则

后续每完成一个模块中的关键子目标，必须更新本文件：

- 修改模块状态
- 修改“已完成”
- 修改“仍缺失”
- 如果推进顺序需要调整，先改本文件，再改代码
