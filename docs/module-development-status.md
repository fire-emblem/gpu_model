# Module Development Status

## 文档用途

这份文档是项目主线开发看板。

后续开发要求：

1. 新功能开始前，先确认它属于哪个顶层模块
2. 代码合入后，必须同步修改本文件中的模块状态
3. 如果目标边界发生变化，先改本文件，再改代码

## 当前主目标

当前主目标只讨论 **functional model**。

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
| `M0` | 架构规格与设备建模 | `c500` 设备规格、单卡、wave64、DPC/AP/PEU 层级、device property 对外可查询 | `Partial` | 已有 `c500` 架构注册、层级放置、wave64、PEU/AP/DPC/GPU 关系 | 还缺标准化 device property 查询接口与返回结构；还缺对外暴露 model device 信息的统一 runtime API |
| `M1` | Host runtime 与 HIP runtime 基础 | 单卡、单 context、单 stream、同步 runtime 入口、基本错误码与设备选择 | `Partial` | 已有 `HostRuntime`、`RuntimeHooks`、`hipMalloc/hipFree/hipMemcpy/hipLaunchKernel/hipGetDeviceCount/hipGetDevice/hipSetDevice`、基础 stream/event 空实现 | 还缺 `hipGetDeviceProperties` / `hipDeviceGetAttribute` 等 property 查询；还缺 context/stream 边界文档化与统一限制；还缺更完整同步 runtime 子集梳理 |
| `M2` | Module / ELF / Code Object 加载 | 支持 module load、ELF 解析、fatbin / `.out` / code object 装载、const/data 段装载、metadata 二进制解析 | `Partial` | 已有 program image、bundle、ELF/code object loader、device load plan/materialize、artifact path 路径、部分 metadata 解析 | 还缺完整 ELF section/program header/relocation 覆盖；还缺 metadata 二进制字段的系统化解析；还缺正式 module load API 闭环；还缺常驻 module 生命周期管理完善 |
| `M3` | GCN 二进制 decode / disasm | 基于连续 `.text` 二进制高效解析全部 GCN ISA，输出结构化 decode 与反汇编 | `Partial` | 已有 raw instruction 提取、format classify、encoding def、decoder、formatter、部分指令 bitfield 识别；已支持 `text bytes -> raw instruction array -> decoded instruction array` 主路径 | 还缺“全部 GCN ISA” encoding 覆盖；还缺更系统的 bitfield/union 定义；还缺完整二进制到 decode 的高覆盖率测试；还缺高性能批量 decode 路径校验 |
| `M4` | GCN ISA 语义执行 | 支持全部 GCN ISA 的 functional 执行，包括标量、向量、访存、控制流、同步、LDS、MFMA | `Partial` | internal ISA functional/cycle 已覆盖较多基础指令；raw GCN 路径已支持部分真实指令；已有 MFMA probe 和部分 raw semantic handlers；已支持 decode 阶段的 `op_type -> opcode -> concrete instruction object` 工厂实例化 | 距离“全部 GCN ISA 执行”差距仍大；raw GCN exec 仍是子集；很多真实 HIP 程序会因为缺指令语义失败；对象已实例化但不少对象仍委托到旧 handler 逻辑；tensor/mfma 仍需系统扩展与验证 |
| `M5` | LLVM AMDGPU ABI / wave 启动 | 正确读取 kernarg、hidden args、special SGPR/VGPR、block/thread/grid 维度、wave 启动初值 | `Partial` | 已有 kernarg 构造、部分 hidden arg、`blockIdx/localId/globalId` 初始化、部分 raw SGPR/VGPR 初始化 | 还缺完整 LLVM AMDGPU ABI 特殊寄存器初始化；还缺 `z` 维；还缺更多 hidden/system SGPR 约定；还缺 wave 启动寄存器 trace dump |
| `M6` | Functional 执行核心 | 单线程和多线程共用一套 functional core；支持 wave/block/device 层级执行；支持 `st/mt` 切换 | `Partial` | 已有共享 `FunctionalExecutionCore`；`st/mt` 已共核；已有 PEU-local wave pool、round-robin、block 内 shared/barrier kernel 的 `mt` 路径；marl 已接入 | 还缺 `1D/2D/3D` 完整 launch 支持；还缺更完整 wait/resume 抽象；还缺对任意 HIP 程序的大规模稳定性验证；还缺与 raw GCN path 的共享执行部件收拢 |
| `M7` | 内存系统与地址空间 | global/shared/private/constant/kernarg/data/managed 独立地址空间，host/device 拷贝与 map 映射 | `Partial` | 已有多 memory pool、managed、kernarg、constant、device load materialize、host/device 基本 memcpy、fake device ptr 到 model addr 映射 | 还缺 data/const/bss/relocation 更完整装载；还缺 host/device 独立地址空间模型文档化；还缺 map/unmap 语义完善；还缺 `3D` launch 对应地址与 builtins 闭环 |
| `M8` | 同步、barrier、atomic | block barrier、wave barrier、global/shared/private 基本同步与常用 atomic | `Partial` | 已有 `s_barrier`、wave barrier、shared/global atomic add、shared memory barrier kernel 测试、functional `mt` 条件变量等待 | 还缺更多 atomic 指令覆盖；还缺更完整 waitcnt 领域与同步语义；还缺 raw GCN 路径的系统同步覆盖；还缺更完整同步 CTS |
| `M9` | Tensor / MFMA | 支持 tensor core / MFMA 指令解析、反汇编、执行与结果验证 | `Partial` | 已有 `v_mfma_f32_16x16x4f32` 最小路径和 probe/test | 还缺 MFMA 指令族系统覆盖；还缺寄存器布局、累加器语义、更多 datatype 支持；还缺真实 kernel 验证 |
| `M10` | Trace / Log / Debug | 支持详细 log、instruction trace、wave launch trace、寄存器值打印、层级信息打印 | `Partial` | 已有 trace sink、file/json trace、ASCII timeline、Google trace、instruction trace、cycle timeline | 还缺 wave 启动初始寄存器 dump；还缺标准化 debug 日志等级；还缺 raw GCN / functional / runtime 三条路径的统一 trace 格式 |
| `M11` | 命令行 `.out` 执行闭环 | `LD_PRELOAD` 后，任意第一阶段边界内 HIP 可执行程序可直接命令行执行 | `Partial` | 已有 host `main()` 原生执行 + HIP interposer + kernel 进入 model 的闭环；已有多个 `.out`/feature CTS | 还缺 module API 路径；还缺 property 查询；还缺更多 runtime API；还缺“任意 HIP 程序”所需的完整 decode/exec/runtime 覆盖 |
| `M12` | 测试与状态门禁 | 用例矩阵、真实 HIP 程序、raw GCN、runtime、CTS、回归门禁 | `Partial` | 已有 gtest 统一测试、100+ HIP feature/runtime CTS、raw/interposer/parallel regression | 还缺以“任意 HIP 可执行程序”为目标的分层门禁矩阵；还缺 decode/disasm/ABI/property/module-load 专项测试；还缺状态与模块看板绑定的验收标准 |
| `M13` | Cycle model | 完整 cycle 建模、issue/latency/waitcnt/event/timeline | `Deferred` | 当前已有 naive cycle 主干、issue model、event queue、timeline | 本轮暂不展开，后续单独讨论 |

## 当前阶段总评

当前项目离“任意第一阶段边界内 HIP 可执行程序可命令行执行”还有明显差距。

当前最关键的缺口不是单点 bug，而是四个大面：

1. `M2 + M3`
   - 完整 module / ELF / code object / metadata / raw binary decode
2. `M4`
   - 全 GCN ISA decode / disasm / exec 覆盖
3. `M1 + M11`
   - HIP runtime 第一阶段闭环补全
4. `M5 + M6 + M7 + M8`
   - ABI、memory、sync、wave 启动状态收敛成稳定 functional 主干

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

- 按 GCN ISA 大类把 raw exec 全部补齐
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
- `bss` / zero-init 段处理
- 更完整的 global/shared/private/scalar-buffer atomic 指令族
- 更完整的错误码和 runtime 返回行为
- kernel / module / device 属性查询接口
- raw GCN decode / disasm / exec 的一致性测试
- wave launch 时的系统寄存器初始化 trace

## 下一次状态更新规则

后续每完成一个模块中的关键子目标，必须更新本文件：

- 修改模块状态
- 修改“已完成”
- 修改“仍缺失”
- 如果推进顺序需要调整，先改本文件，再改代码
