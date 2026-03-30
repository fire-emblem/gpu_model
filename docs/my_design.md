# GPU Model 目标设计

## 1. 设计目标

本项目的目标是让 `pcc` 或 `hipcc` 编译生成的 `.out` 可执行程序，在不依赖真实 GPU 执行 kernel 的前提下，能够通过项目内的 model runtime 完成：

- Host 代码原生执行
- HIP runtime 调用被拦截并转发到 model runtime
- device program 被解析为静态程序对象
- kernel launch 被转换为统一的执行请求
- kernel 在 functional model 或 cycle model 中执行
- 执行结果回写到 host 可见内存
- 输出 trace、timeline 和性能统计信息

长期主线应收敛为：

`runtime -> program -> instruction -> execution -> wave`

其中：

- `runtime` 负责 API 入口和 launch 驱动
- `program` 负责程序对象和模块装载
- `instruction` 负责指令表示、解析与建模
- `execution` 负责 wave 级执行和功能/周期模型
- `wave` 是执行时的核心状态单元

## 2. 运行时分层

运行时应分成两层，而不是把所有职责堆在一个类里。

### 2.1 HIP 兼容层

这一层负责实现 HIP runtime 兼容接口。

职责：

- 提供 `hipMalloc`、`hipMemcpy`、`hipLaunchKernel` 等基础接口
- 处理 `LD_PRELOAD` 模式下对真实程序的拦截
- 维护 fake device pointer 到 model address 的映射
- 将 HIP 参数整理为 model runtime 可消费的统一请求

设计要求：

- 这一层必须尽量薄
- 只做参数适配、符号映射和入口转发
- 不重复实现 module load、decode、memory、execution 语义

长期命名建议：

- `HipRuntime`

### 2.2 Model Runtime 层

这一层负责项目内部公开的 runtime API。

职责：

- 提供统一的项目内 runtime 接口
- 对测试、工具和非 HIP 入口暴露模型能力
- 转发 load、launch、memory、trace 请求到内部执行引擎

长期命名建议：

- `ModelRuntime`

### 2.3 Runtime Engine 层

这一层负责真正的执行总控。

职责：

- 接收统一的 launch request
- 选择程序执行路径
- 驱动 module load、program object、execution engine
- 管理 memory、trace、launch result、timing config
- 在 functional 与 cycle 模式之间做模式分发

长期命名建议：

- `RuntimeEngine`

## 3. 程序对象层

程序对象层的职责是：在 kernel launch 之前，把 `.out` / ELF / code object 中的 device program 解析为稳定的静态程序对象。

这层不负责执行，只负责“把程序准备好”。

### 3.1 Program Object

程序对象层应能够从 `.out` / ELF 中读取：

- `.text`
- `.data`
- `.rodata`
- metadata
- kernel descriptor
- symbol / section / segment 信息

这些信息应组成统一的 `ProgramObject`。

设计要求：

- 程序解析应尽量发生在 launch 之前，而不是每次 launch 时临时解析
- 一个可执行程序可能包含多个 kernel 入口，程序对象应能统一持有这些静态信息
- ELF / code object 中的 device 段，应以 image 的方式映射到 device memory

长期核心对象建议：

- `ProgramObject`
- `EncodedProgramObject`
- `ExecutableKernel`
- `ObjectReader`

### 3.2 模块装载

模块装载层应支持：

- `.out`
- ELF / code object
- 程序包或程序镜像格式

它应负责：

- 识别输入格式
- 生成 `ProgramObject`
- 完成 const/data/kernarg 等段的静态装载准备
- 提供 module 生命周期管理

## 4. 指令层

指令层的职责是把程序对象中的代码段转换为可执行、可建模、可追踪的指令表示。

### 4.1 编码指令层

对于真实机器编码，首先应形成与编码直接绑定的指令表示。

它应支持：

- 从二进制 `.text` 解码指令流
- 根据 bitfield 识别指令格式
- 根据指令类型判断长度并递增 PC
- 生成可按 PC 访问的有序指令序列
- 从 text asm 通过 `llvm-mc` 转成二进制后复用同一条流程
- 允许测试直接注入指令序列

这一层的核心概念是“encoded instruction”，而不是历史上含糊的 `raw`。

设计要求：

- 指令序列必须支持按 `pc` 跳转和访问
- 指令实例是不可变对象
- encoding descriptor、decoded instruction、executable binding 应分层

### 4.2 建模指令层

项目内部还需要一层与执行模型更贴近的 `modeled instruction` 表示。

它的职责是：

- 为 functional / cycle model 提供更稳定的内部语义对象
- 不直接依附于 encoded 指令的机器表示
- 允许未来做 lowering、共享 effect layer 和 coverage 管理

### 4.3 指令对象设计

指令对象应至少包含：

- opcode
- opname
- operands
- mode flags
- 执行语义入口

指令体系应有一个基类，并按实际语义分派子类，例如：

- 标量计算
- 向量计算
- tensor / MFMA 计算
- 全局访存
- 共享访存
- 私有访存
- 常量访存
- 跳转控制
- 同步控制

### 4.4 Operand 设计

operand 只描述静态信息，不直接持有执行态寄存器值。

它应描述：

- 立即数
- 寄存器类型
- 寄存器编号
- 其他编码字段

真正的寄存器文件和值，应存放在执行态的 wave context 中。

## 5. 架构层

架构层负责描述目标 GPU 的静态规格，而不是执行逻辑本身。

### 5.1 设备规格

例如 `c500`：

- wave size = 64
- shared memory = 64 KB
- 每个 wave 的 scalar register = 128
- 每个 wave 的 vector register = 256 组，每组 64 lane
- `8 DPC x 13 AP/DPC x 4 PEU/AP`

### 5.2 层级对象

架构层应定义并提供：

- DPC
- AP
- PEU
- wave
- thread

它们的静态层级关系和拓扑放置规则。

### 5.3 内存与寄存器资源

架构层需要声明：

- global memory
- shared memory
- private memory
- constant memory
- register file

执行层再基于这些规格实例化实际运行态。

## 6. 执行引擎层

执行引擎层是程序对象与 wave 执行之间的桥梁。

职责：

- 管理 GPU model 实例
- 管理 block / wave 的运行态构造
- 管理 barrier 与同步状态
- 根据模式选择 functional 或 cycle 执行路径
- 复用共享的 memory / sync / state / trace helper

长期执行引擎应至少包括：

- `FunctionalExecEngine`
- `CycleExecEngine`
- `EncodedExecEngine`

设计原则：

- 功能正确性与周期推进要分层
- functional 和 cycle 可以共享状态构造与 effect apply
- 但不应强行并成一个大基类 executor

## 7. Wave 执行上下文

wave 是执行时的核心单位。

长期对外语义统一称为 `WaveContext`。

执行语义统一使用 `WaveContext`。
不再保留单独的 `WaveState` 对外层名称。

### 7.1 WaveContext 应包含的内容

- 指令流引用
- `dpcId`
- `apId`
- `peuId`
- `waveId`
- `blockId`
- `blockIdx(x,y,z)`
- `exec mask`
- `cmask`
- `smask`
- predicate / branch state
- scalar register file
- vector register file
- private memory 基地址与每线程范围
- shared memory 基地址与每 block 范围
- wave 状态

wave 状态至少应区分：

- 可执行
- 等待 barrier
- 等待内存
- 同步阻塞
- 已退出

### 7.2 Launch 初始化

kernel launch 时应完成：

- grid/block 维度读取
- kernel 参数装载
- 隐式和显式 SGPR/VGPR 初始化
- `dpc/ap/peu/wave/thread` 层级放置
- `exec mask` 和 builtin id 初始化
- LLVM AMDGPU ABI 约定下的特殊寄存器装载

## 8. 功能模型与周期模型

### 8.1 Functional Model

functional model 的目标是结果正确。

要求：

- 支持单线程和多线程两种驱动方式
- 两者共用同一套执行核心
- 共用 wave state、memory、sync、effect apply 逻辑

### 8.2 Cycle Model

cycle model 的目标不是 RTL 级精确，而是：

- 提供相对 cycle 差异分析
- 提供 stall / bubble / utilization 解释
- 提供可比较的 timeline

初始周期模型可以采用简化配置：

- 普通指令固定 4 cycle
- shared memory 访问固定 32 cycle
- scalar / constant 访问固定 128 cycle
- global / private 访问固定 1024 cycle

后续可扩展：

- waitcnt 领域
- memory domain 区分
- issue 限制
- 资源冲突
- 参数化与校准

## 9. 调试、Trace 与性能观测

项目必须内建统一的调试和观测能力，而不是事后附加。

### 9.1 日志

应有统一日志系统，支持：

- 开关控制
- 等级控制
- 面向 runtime / program / instruction / execution 的分层输出

### 9.2 指令与 Wave Trace

应支持：

- wave 初始化 trace
- 指令执行 trace
- 可选寄存器值打印
- `pc / wave id / block id / dpc / ap / peu` 输出
- wave 切换原因说明

### 9.3 性能观测

应支持：

- pipeline utilization
- bubble 比例
- memory bound / compute bound 分类
- instruction cycle 统计
- AP / PEU 利用率统计

### 9.4 Timeline

应支持导出可被 Perfetto 或同类工具导入的 timeline。

timeline 至少应能展示：

- 不同 PEU 上的 wave 指令流
- 指令的周期区间
- barrier / wait / stall
- wave 内指令是连续发射还是出现空泡

## 10. 长期目录与模块收口目标

当前项目长期应收敛到以下主模块：

- `runtime`
- `program`
- `instruction`
- `execution`
- `arch`

不再长期保留为独立主模块的历史概念包括：

- `loader`
- `decode`
- 混装的 `isa`
- 以 `raw_gcn` 命名的一整套路径

对应长期命名方向：

- `raw_*` -> `encoded_*`
- `canonical/internal` -> `modeled_*`
- `ProgramImage` -> `ProgramObject`
- `KernelProgram` -> `ExecutableKernel`
- `HostRuntime` -> `RuntimeEngine`
- `RuntimeHooks` -> `HipRuntime`
- `ModelRuntimeApi` -> `ModelRuntime`
- `FunctionalExecutionCore` -> `FunctionalExecEngine`
- `FunctionalExecutor` / `ParallelWaveExecutor` -> `FunctionalExecEngine`
- `CycleExecutor` -> `CycleExecEngine`
- `RawGcnExecutor` -> `EncodedExecEngine`

## 11. 第一阶段收口原则

第一阶段不是简单重命名，而是把主路径压直。

第一阶段优先目标：

- 统一 runtime / program / instruction / execution 主链
- 统一核心类型命名
- 保留兼容层，避免一次性打爆所有调用点
- 测试目录按模块同步迁移

第一阶段不要求一次完成：

- 全量 ISA 覆盖
- relocation / bss 完整实现
- cycle 校准闭环
- 所有历史文档一次改完

但第一阶段结束后，仓库主路径必须已经体现本设计的层次，而不是继续沿用旧的分裂结构。
