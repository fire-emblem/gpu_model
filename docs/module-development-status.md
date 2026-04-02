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

- `runtime`: `HipRuntime / ModelRuntime / RuntimeEngine`
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
| `M1` | Runtime 基础（`HipRuntime / ModelRuntime / RuntimeEngine`） | 单卡、单 context、单 stream、同步 runtime 入口、基本错误码与设备选择 | `Partial` | 已有 `ModelRuntime` facade、`RuntimeEngine` 主路径、`HipRuntime` 路径、`hipMalloc/hipFree/hipMemcpy/hipMemcpyAsync/hipMemset/hipMemsetAsync/hipLaunchKernel/hipGetDeviceCount/hipGetDevice/hipSetDevice/hipGetDeviceProperties/hipDeviceGetAttribute`、基础 stream/event 空实现，以及 model-native 统一 `LoadModule` 入口；`.out/.o` 已可显式选择 encoded program object 或 lowered modeled program object 路径；`LD_PRELOAD` interposer 已显式固化“默认流 + 单显式 stream”边界，并对销毁后/无效 stream handle 返回错误 | 还缺 property / attribute 更完整覆盖；还缺 context 边界文档化与统一限制；还缺更完整同步 runtime 子集梳理 |
| `M2` | ProgramObject / ELF / Code Object 加载 | 支持 module load、ELF 解析、fatbin / `.out` / code object 装载、const/data 段装载、metadata 二进制解析 | `Partial` | 已有 `ProgramObject`/bundle、ELF/code object loader、device load plan/materialize、artifact path 路径；已能从 code object/`.out` 解析 kernel descriptor、metadata、kernarg size、hidden arg layout、descriptor symbol；统一 `LoadModule` 已支持 `Auto/AmdgpuObject/ProgramBundle/ExecutableImage/ProgramFileStem`；kernel metadata 已开始收敛到 typed 结构并被 loader / encoded launch / runtime 复用；runtime 侧已具备按 module 名装载、卸载、枚举 module/kernel 的基础生命周期接口 | 还缺完整 ELF section/program header/relocation 覆盖；还缺 metadata 字段的进一步系统化 typed 覆盖；还缺更完整的模块常驻/替换/冲突语义约束 |
| `M3` | GCN 二进制 decode / disasm | 基于连续 `.text` 二进制高效解析全部 GCN ISA，输出结构化 decode 与反汇编 | `Partial` | encoded GCN decode/disasm 主体已迁入 `instruction/encoded/*`；已具备 encoded instruction 提取、format classify、encoding def、decoder、formatter；已支持 `text bytes -> encoded instruction array -> decoded instruction array -> instruction object array` 主路径；compute-focused真实 HIP kernel 的 decode/disasm 已覆盖到 `vecadd/fma_loop/bias_chain/shared_reverse/softmax_row/mfma`；shared-heavy 真实 HIP case 的 decode/binding 缺口已继续收口到 `shared_reverse + dynamic_shared_sum + block_reduce_sum + softmax_row` 四锚点；decode 旧顶层目录已删除 | 还缺“全部 GCN ISA” encoding 覆盖；还缺更系统的 bitfield/union 定义；还缺 graphics/image/export/interp family 的深入覆盖；还缺高性能批量 decode 路径校验 |
| `M4` | GCN ISA 语义执行 | 支持全部 GCN ISA 的 functional 执行，包括标量、向量、访存、控制流、同步、LDS、MFMA | `Partial` | instruction ISA functional/cycle 已覆盖较多基础指令；encoded instruction 路径已支持真实 `.out` compute kernel 主线；已支持 `vecadd/fma_loop/bias_chain/shared_reverse/softmax_row/mfma`；decode 阶段已完成 `op_type -> opcode -> concrete instruction object` 工厂实例化；`instruction/*` 与 `execution/*` 主命名已成为公开主路径；`exec/*` 旧目录主干已迁入 `instruction/*` 与 `execution/*`；此前 6 个 instruction/execution/MFMA 相关失败测试已清零；shared-heavy 真实 HIP 执行主线已收口到 `shared_reverse + dynamic_shared_sum + block_reduce_sum + softmax_row` 四锚点，不再只依赖单一 case | 距离“全部 GCN ISA 执行”仍有差距；graphics family/descriptor family 仍主要占位；仍需继续做系统化 opcode 覆盖与归类 |
| `M5` | LLVM AMDGPU ABI / wave 启动 | 正确读取 kernarg、hidden args、special SGPR/VGPR、block/thread/grid 维度、wave 启动初值 | `Partial` | 已有 descriptor + metadata 驱动的 wave 初始 SGPR/VGPR preload；已支持 kernarg segment ptr、workgroup id、workitem id、hidden block/group args、`x/y/z` grid-block launch 维度、encoded `.out` launch ABI 主线；kernarg 打包已抽成公共模块；visible arg offset/aggregate、`3D hidden args`、fallback encoded ABI kernarg 约定已有回归覆盖；真实 `hipcc` 生成的 `3D hidden-arg` 与 `3D builtin-id` encoded 路径已可执行验证；`WaveLaunch` trace 已能对已知 ABI 字段输出 `kernarg_ptr/wg_id_x/y/z/workitem_id_x/y/z` 语义摘要，并对未知字段保留原始寄存器回退 | 还缺更完整的 system SGPR/VGPR 集合；还缺更多 target-specific ABI 差异；还缺 wave 启动寄存器 trace dump 进一步细化 |
| `M6` | Functional 执行核心 | 单线程和多线程共用一套 functional core；支持 wave/block/device 层级执行；支持 `st/mt` 切换 | `Partial` | 已有共享 `FunctionalExecEngine` 核心；`st/mt` 已共核；已有 PEU-local wave pool、round-robin、block 内 shared/barrier kernel 的 `mt` 路径；marl 已接入；`1D/2D/3D` launch 配置、placement 和 `xyz` builtin 主线已打通；真实 `hipcc` 生成的 `3D vecadd + 条件边界 + 小计算量` 程序已能经 `.out -> lowered modeled program object` 路径在 `st/mt` 下对比执行；内部执行支撑件已稳定收敛到 `execution/internal/*`；已新增显式 wave run state，`FunctionalExecEngine` 的 `waitcnt` 进入等待、memory wait 恢复、barrier release 恢复现已收敛到共享状态契约，waiting-wave 扫描入口也已在 `st/mt` 下统一复用，相关 focused regression 已锁定；同 `PEU` 上显式进入 `Waiting` 的 wave 现在也已有 focused regression 证明不会阻塞 ready sibling，barrier release 后的早到 wave 会重新进入 dispatch；functional `st/mt` 现已新增基于 executed-flow 的 `ProgramCycleStats` 实际运行统计，并有纯 ALU / const / shared / private / waitcnt / barrier，以及大 block / 多 wave / 多 block / 非对称 wave work 的默认成本口径回归锁定；`MarlParallel` 在大 block 且 worker_threads 偏小时已收口到稳定的 block-level 并行执行，避免多 block 大规模回归死锁 | 还缺更多 wait reason 扩展；还缺对任意 HIP 程序的大规模稳定性验证 |
| `M7` | 内存系统与地址空间 | global/shared/private/constant/kernarg/data/managed 独立地址空间，host/device 拷贝与 map 映射 | `Partial` | 已有多 memory pool、managed、kernarg、constant、device load materialize、host/device 基本 memcpy、fake device ptr 到 model addr 映射 | 还缺 data/const/bss/relocation 更完整装载；还缺 host/device 独立地址空间模型文档化；还缺 map/unmap 语义完善；还缺 `3D` launch 对应地址与 builtins 闭环 |
| `M8` | 同步、barrier、atomic | block barrier、wave barrier、global/shared/private 基本同步与常用 atomic | `Partial` | 已有 `s_barrier`、wave barrier、shared/global atomic add、shared memory barrier kernel 测试、functional `mt` 条件变量等待；barrier wait/release 现已统一落到显式 `run_state/wait_reason` 状态机，functional、cycle、raw-GCN 主线的 shared/barrier 回归已重新收口；`waitcnt` 的 `global/shared/private/scalar-buffer` memory-domain 与 barrier 现已共享同一 waiting/runnable 状态契约 | 还缺更多 atomic 指令覆盖；还缺更完整 waitcnt 领域与同步语义；还缺 encoded GCN 路径的系统同步覆盖（legacy raw GCN 路径仍需兼容）；还缺更完整同步 CTS |
| `M9` | Tensor / MFMA | 支持 tensor core / MFMA 指令解析、反汇编、执行与结果验证 | `Partial` | 已有 `v_mfma_f32_16x16x4f32` 最小路径和 probe/test；encoded semantic 已覆盖 `f32/f16/bf16/i8` 多个 MFMA 变体与 shape；kernel descriptor / metadata 已能暴露 typed tensor ABI（如 `agpr_count/accum_offset`）；真实 `hipcc` 生成的 MFMA executable 已在 runtime / interposer / CTS 主线上验证 | 还缺 MFMA 指令族更系统覆盖；还缺寄存器布局、累加器语义与更多 datatype/shape 支持；还缺从“代表性真实 kernel”走向更广覆盖的 tensor 回归矩阵 |
| `M10` | Trace / Log / Debug | 支持详细 log、instruction trace、wave launch trace、寄存器值打印、层级信息打印 | `Partial` | 已有 trace sink、file/json trace、ASCII timeline、Google trace、instruction trace、cycle timeline；`FunctionalExecEngine / EncodedExecEngine / CycleExecEngine` 三条当前执行 backend 已统一发出 `WaveLaunch` 事件并带初始 `WaveContext` 状态摘要；`WaveLaunch` 单行摘要已提升为“ABI 语义优先、原始寄存器回退”，raw-GCN/encoded 路径可直接观察 `kernarg_ptr`、`wg_id_*`、`workitem_id_*` 等字段；usage 脚本已能稳定导出 encoded decode 与 HIP interposer 主线结果；functional trace 已新增 `WaveStats` 快照，可观察 wave launch/init/active/end 生命周期进度，且已进一步拆分 `runnable/waiting` 并由 barrier / waitcnt regression 锁定；统一 wait-state machine 的 `Stall + WaveStats` 语义现已被 focused trace 回归直接覆盖；runtime `LaunchResult` 现已统一暴露 `ProgramCycleStats`，便于直接查看 functional `st/mt` 的实际程序 cycle 统计；`timeline.perfetto.json` 现已新增结构正确性、`function mt` waiting/resume 语义，以及真实 `128 x 128 conditional multibarrier` HIP case 的 sanity 回归 | 还缺更完整的 wave 启动初始寄存器 dump；还缺标准化 debug 日志等级；还缺 encoded / functional / runtime 三条路径的统一 trace 格式进一步收敛 |
| `M11` | 命令行 `.out` 执行闭环 | `LD_PRELOAD` 后，任意第一阶段边界内 HIP 可执行程序可直接命令行执行 | `Partial` | 已有 host `main()` 原生执行 + HIP interposer + kernel 进入 model 的闭环；真实 `.out` 已验证 `vecadd/fma_loop/bias_chain/atomic_count/shared_reverse/dynamic_shared_sum/block_reduce_sum/softmax_row/mfma`；`HipInterposerState` 注册路径与 `LD_PRELOAD` 路径两条主线均有 CTS 覆盖；基础 property 查询与 model-native module API 已打通 | 还缺更完整 runtime API；还缺“任意 HIP 程序”所需的完整 decode/exec/runtime 覆盖 |
| `M12` | 测试与状态门禁 | 用例矩阵、真实 HIP 程序、encoded decode、runtime、CTS、回归门禁 | `Partial` | 已有 gtest 统一测试；`HipRuntimeTest.*`、`HipInterposerStateTest.*`、encoded decode usage、主 CTS 和 feature CTS 均已打通；当前 fresh 全量 `gpu_model_tests` 结果为 `559 passed`；`LD_PRELOAD` interposer 用例已转为通过；shared-heavy `shared_reverse + dynamic_shared_sum + block_reduce_sum + softmax_row` 现已形成四锚点门禁，其中 `shared_reverse` 与 `dynamic_shared_sum` 已在 decode / interposer / runtime 三层收口，并进入 HIP CTS 与 feature CTS 快速矩阵，`block_reduce_sum` 已补齐 decode / interposer / runtime focused regression，并进入主 HIP CTS runtime/interposer 快速矩阵，`softmax_row` 已具备 decode / interposer / runtime focused regression、主 HIP CTS、parallel execution regression 和 example 闭环；atomic-heavy `atomic_count` 现也已进入主 HIP CTS runtime/interposer 快速矩阵；executed-flow `ProgramCycleStats` 现已有 tracker-focused synthetic tests、runtime-focused成本口径 tests，以及 `128 x 128 conditional multibarrier` HIP baseline 校准回归 | 还缺以“任意 HIP 可执行程序”为目标的分层门禁矩阵文档；还缺 decode/disasm/ABI/property/module-load 专项测试归档；还缺状态与模块看板绑定的验收标准 |
| `M13` | Cycle model | 完整 cycle 建模、issue/latency/waitcnt/event/timeline | `Partial` | 已有 naive cycle 主干、issue model、waitcnt 领域阻塞、event queue、timeline、Google trace、cache/bank conflict/waitcnt cycle 测试；cycle front-end 现已具备每个 `AP` 最多 `2` 个 resident blocks 的参考模型，并已有 focused regression 锁定 `PEU` 侧 `active_window = 4`、standby promotion 与 resident backfill 语义，因此 cycle mode 已可达到单 `PEU` `>4` resident waves；`waitcnt/dependency/front_end_wait` 阻塞 wave 现保持 resident 且留在 in-window，不再错误让出 front-end 窗口；barrier waiting resident waves 会让出 active slots，并在 barrier release 后重新进入 active window；barrier wait/release 状态机已与 shared/barrier cycle kernel、block reduction、softmax reduction、transpose 等回归重新对齐；functional `st/mt` 现已新增与 cycle mode 结果形状兼容的 executed-flow program cycle 统计，可用于整程序运行统计对照与后续校准，且已覆盖大 block / 多 wave / 多 block 的 mixed-cost 与非对称 workload 回归；`global/shared/scalar-buffer` 不同 memory-arrive + waitcnt，以及 single/double barrier 的程序统计增长也已由 focused regression 锁定；代表性 case 的排序、分类统计求和、自 `st/mt` 模式一致性已由 focused regression 锁定，可用于查看 program-level cycle 统计趋势；当前还新增了 `128 x 128 conditional multibarrier` HIP baseline，用于在真实 `hipcc` 程序上对照 `st/mt/cycle` 的 program cycle 统计形状与多 barrier 条件路径稳定性；cycle timeline / Perfetto dump 现已新增 issue/arrive/commit/stall 顺序合理性的 focused 回归 | 仍缺更完整的架构资源冲突、更多 memory domain/pipe 细节、与真实硬件差异说明和参数化建模文档 |

## 当前阶段总评
当前项目距离“更完整的功能覆盖与长期维护形态”仍有差距，但主线重构和当前边界内能力已经完成。

同时需要明确：

- 如果以“公开主路径命名、顶层目录层级、主测试路径、当前全量验证结果是否收口”为标准，本轮重构已经完成
- 如果以“彻底减少底层 GCN-specific 术语、整理全部历史文档”为标准，仍有收尾项

当前最关键的缺口不是单点 bug，而是五个大面：

1. `M2 + M3`
   - 完整 `ProgramObject` / ELF / code object / metadata / encoded binary decode 覆盖
2. `M4`
   - 全 GCN ISA decode / disasm / exec 覆盖，特别是非 compute families
3. `M1 + M11`
   - `HipRuntime / ModelRuntime / RuntimeEngine` 第一阶段闭环补全
4. `M5 + M6 + M7 + M8`
   - ABI、memory、sync、wave 启动状态收敛成稳定 `FunctionalExecEngine` 主干
5. `M13`
   - cycle model 从“可用的 naive 分析工具”继续走向“更稳定的优化评估平台”

## 严格推进顺序

后续开发按下面顺序推进，不要跳跃：

### Step 1

先补 `M1` 的第一阶段 runtime 边界：

- 单卡/单 context/单 stream 约束固化
- property 查询
- module load 基础 API
- 必需的同步 runtime API 子集

理由：

- 没有稳定 runtime 边界，后面“任意 HIP 可执行程序”这个目标无法验收

### Step 2

补 `M2`：

- module 生命周期
- ELF / fatbin / code object / metadata 二进制解析
- const/data/raw-data/kernarg 段装载

理由：

- runtime 进来了以后，必须先保证模块和镜像装载可靠

### Step 3

补 `M3`：

- 基于连续二进制的完整 GCN decode
- bitfield/union 定义完善
- project-side disassembler 完整化

理由：

- 没有全量 decode，就不可能有“任意 HIP 程序”

### Step 4

补 `M4`：

- 按 GCN ISA 大类把 encoded exec 全部补齐
- 不允许 case-by-case 继续散长
- parse / disasm / exec 用统一可扩展表驱动

理由：

- 这是第一阶段的真正主路径核心

### Step 5

补 `M5 + M6`：

- LLVM AMDGPU ABI 特殊寄存器初始化
- wave launch trace
- `1D/2D/3D` launch
- `st/mt` functional 稳定性

理由：

- 真程序能不能正确跑，取决于 ABI 和 launch 初态是否对

### Step 6

补 `M7 + M8 + M9`：

- memory 空间和 map/memcpy 语义补齐
- barrier/atomic/wait 同步补齐
- tensor/mfma 补齐

理由：

- 这是大多数复杂 kernel 的执行基础

### Step 7

补 `M10 + M12`：

- 统一 trace/log 格式
- 增加 launch 初始寄存器 trace
- 增加 decode/disasm/ABI/runtime/module-load/real-hip-program 测试门禁

理由：

- 没有统一可观察性和门禁，后续功能会持续回归

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

## 下一次状态更新规则

后续每完成一个模块中的关键子目标，必须更新本文件：

- 修改模块状态
- 修改“已完成”
- 修改“仍缺失”
- 如果推进顺序需要调整，先改本文件，再改代码
