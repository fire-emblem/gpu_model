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

补强重点：

- runtime 重要 API 的完备性
- 不同 `memcpy` / `memset` 语义与错误路径
- 可脱离 kernel launch 的 lightweight runtime tests

当前阶段边界：

- 先不把异常路径测试作为第一批主线
- 先把同步语义下的 `malloc/free/memcpy/memset` 主路径框架和测试矩阵写清

长期命名建议：

- `ModelRuntime`

### 2.3 Exec Engine 层

这一层负责真正的执行总控。

职责：

- 接收统一的 launch request
- 选择程序执行路径
- 驱动 module load、program object、execution engine
- 管理 memory、trace、launch result、timing config
- 在 functional 与 cycle 模式之间做模式分发

长期命名建议：

- `ExecEngine`

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
- `ExecutableKernel`
- `ObjectReader`

当前正式约束补充：

- 程序主线公开对象只保留一个 `ProgramObject`。
- 不再区分 `canonical path` / `encoded path` 两条公开程序执行路径。
- 来自 `.out` / ELF / code object 的真实编码、descriptor、decoded instruction、instruction object 等静态结果，都统一挂在 `ProgramObject` 上。
- builder 或测试若需要构造程序，也应最终收口为同一个 `ProgramObject` 输入，而不是再引入第二套“源码路径对象”。

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

当前进一步约束：

- 项目应继续把真实 LLVM / AMDGPU / HIP artifact 当作外部生产物，把本项目定位为 artifact consumer，而不是内嵌完整编译器。
- 第一优先级是稳定消费 `.out` / ELF / code object / metadata / section bytes，而不是回退到手写 asm 或临时 lowering 作为主线。
- 所有 loader 应逐步收口到统一的 segment-oriented `DeviceLoadPlan` 语义，包括：
  - `Code`
  - `ConstantData`
  - `RawData`
  - `KernargTemplate`
- memory pool 至少应有明确的逻辑种类：
  - `Global`
  - `Constant`
  - `Shared`
  - `Private`
  - `Managed`
  - `Kernarg`
  - `Code`
  - `RawData`
- 后续关键 pool 的底层实现应逐步收口到统一的 `mmap` backed residency，以便：
  - 清晰表达 segment 映射
  - 支持 map/copy/zero-fill 语义
  - 让 runtime memory tests 不依赖 kernel launch

### 3.3 统一设备内存管理

所有内存空间后续应由统一的设备内存管理器管理，而不是各层分散直接持有。

推荐两层结构：

1. `DeviceMemoryManager`
   - 统一管理：
     - compatibility virtual address windows
     - allocation table
     - 各 memory pool
     - pointer classify / pointer lookup / address translate

2. `MemoryPool`
   - 管理具体 pool 的容量、分配、释放、读写、range 检查

推荐 pool 至少包括：

- `GlobalPool`
- `ManagedPool`
- `KernargPool`
- `CodePool`
- `RawDataPool`
- 后续扩展：
  - `ConstantPool`
  - `SharedPool`
  - `PrivatePool`

### 3.4 Compatibility 虚拟地址窗口

对 HIP C API / `LD_PRELOAD` 兼容路径，项目应自行规定 compatibility 虚拟地址窗口，而不是依赖宿主 `mmap` 返回的随机高位地址。

建议：

- `GlobalCompatWindow`
- `ManagedCompatWindow`
- 后续按需扩展其他 pool

判断规则：

1. 先按虚拟地址区间判断 pool kind
2. 再查 allocation table 判断该地址是否真是合法分配
3. 最后映射到 `model_addr`

因此：

- `window classify` 负责“它属于哪个 pool”
- `allocation lookup` 负责“它是否合法、大小是多少、映射到哪里”

### 3.5 大范围预留 + 按需提交

对 `Global` / `Managed` 兼容路径，推荐使用：

- 先预留大片虚拟地址空间
- 后续按需提交物理页或映射实际存储

也就是：

- `reserve big virtual range`
- `commit on demand`

这样可以同时满足：

- fake device pointer 语义稳定
- compatibility 指针属性可快速判断
- `malloc/free` 不依赖随机离散 `mmap`
- 后续 pool 统一管理更容易扩展

推荐第一阶段落地顺序：

1. `Global` pool
2. `Kernarg` pool
3. `Code` / `RawData` pool
4. `Constant` pool

第一阶段先不追求所有 pool 都完整 `mmap` 化，而是先建立统一接口和 `Global` 的参考实现。

当前阶段已落地的最小骨架：

- `GlobalCompatWindow` / `ManagedCompatWindow` 固定虚拟地址窗口
- 进程启动时先 `reserve` 整个 window
- `Allocate*` 时按页 `commit`
- `Free/Reset` 时回收 `commit`
- window 当前 `committed_bytes` 可被查询和测试验证

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
- decode / disasm 主线应继续朝“append-only 的 encoding definition + decoder + formatter”收口，而不是散落的 opcode 分支
- 项目内 disasm 应来自项目自身对 encoded bytes 的解释，而不是长期依赖外部 `objdump` 文本作为主语义来源
- 对指令实现层，后续应建立 text asm -> binary -> kernel program 的统一验证主线：
  - 先验证 decode / execute 不 crash
  - 再验证结果寄存器 / 内存副作用正确
  - 并对 `st / mt / cycle` 做一致性和语义校准

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

设计原则：

- 功能正确性与周期推进要分层
- functional 和 cycle 可以共享状态构造与 effect apply
- 但不应强行并成一个大基类 executor
- 公开 launch 主线只保留 `LaunchProgramObject -> ExecEngine`
- 不再暴露单独的 `LaunchEncodedProgramObject` 公开入口

### 6.1 Functional MT 约束

- `functional mt` 的并行单位应是 wave，而不是更粗粒度的 block。
- `MarlParallel` 的目标不是复刻硬件拍级并发，而是提供 wave 级并发、等待恢复和调度竞争的可解释模型。
- block-local shared state、barrier release、memory wait / resume 都必须通过显式状态机或同步原语表达，而不是隐式依赖宿主线程调度偶然正确。
- `st / mt / cycle` 的执行结果必须以正式设计语义为准，而不是以 trace 或宿主线程调度偶然形态为准。

### 6.2 PEU / Wave Issue 约束

- `AP` 是 block 级共享资源域，不是统一 issue 池。
- `PEU` 才是真正的局部调度和发射域。
- `wave` 是调度单位，`lane` 不是调度器的选择对象。
- 必须明确区分：
  - `ready`
  - `selected`
  - `issue`
- 一个 wave 可恢复，不代表同 cycle 必然被选中发射；这一点在 cycle model 中必须成立，在 functional model 中应尽量使用简化但不冲突的语义近似。

### 6.3 Cycle 顶层原则

- cycle model 的目标不是硬件精准复刻，而是稳定、可解释地反映相对 cycle 差异。
- 应保持少量、稳定、可说明的 timing knobs，而不是堆积过多微架构细节。
- 至少应持续区分：
  - issue cycle
  - latency cycle
- stall 原因必须可归类，timeline 必须能解释 bubble 来源。
- 默认应保持 `default_issue_cycles = 4`，仅对少量真正影响分析结论的类别或指令做 override。

## 7. 当前正式设计约束

本节是从历史 plans/specs 中提炼出的当前稳定设计结论。后续实现与测试若和本节冲突，应优先修改实现或更新本节，而不是回退到历史文档解释当前行为。

### 7.1 Runtime 正式分层

- 最外层对外接口是 `HipRuntime`，语义上与 AMD HIP runtime 对齐。
- `HipRuntime` 只负责 C ABI 兼容、参数适配、符号映射、fake pointer 映射和入口转发。
- `ModelRuntime` 是项目内部统一 runtime 主入口。
- `ExecEngine` 是 `ModelRuntime` 内部执行主链，不单独作为对外 runtime 层。
- `src/runtime/hip_interposer.cpp` 只是 `HipRuntime` 的 C ABI 实现载体，不代表独立模块。

### 7.2 Trace 正式约束

- `trace` 的职责只是消费模型事件并序列化为 text/json/timeline/Perfetto。
- 任何执行语义、时序推进、等待恢复、统计记账都必须发生在 runtime/execution/state machine 中，不能放到 trace 层推断。
- `timeline cycle` 的生成必须依赖 execution 已产出的 modeled cycle 事实和 recorder 记录，不能依赖 Perfetto/text/json consumer 侧反推。
- 当前正式对外 timeline 产物是 `timeline.perfetto.json`。
- `timeline.perfetto.pb` 不再作为正式用户产物。
- producer 与 test 侧的 trace 构造应继续统一到单一 semantic factory / builder 入口。
- consumer 侧应统一消费 typed event / recorder facts，而不是各自从 `message` 二次猜语义。
- `TraceEvent.message` 只能作为兼容/展示字段继续存在，不应再充当主语义契约。
- text/json trace 必须可关闭，且关闭时不能依赖业务逻辑降级或改变执行结果。

统一依赖方向：

- `execution producer -> recorder facts -> timeline data -> text/json/perfetto`
- 如果 producer 语义还分裂，必须先修 producer，不能在 timeline / Perfetto 层补偿。

### 7.2.1 日志正式约束

- 项目日志应统一到 `loguru`。
- runtime / program / instruction / execution / trace 各模块都应使用统一日志入口，而不是各自分散输出。
- 关键调试信息应围绕：
  - runtime API 入参与错误路径
  - memory pool / address mapping
  - program load / segment materialize
  - wave launch / wait / resume / issue
  - cycle stall / selection / commit
  进行结构化输出。
- 但第一批实现优先级仍低于 runtime/memory 主框架；先保证关键交互关系和接口边界稳定，再统一日志实现。

### 7.3 模型时间语义

- `st`、`mt`、`cycle` 三条路径上的 trace `cycle` 都是模型计数时间。
- 这些 `cycle` 用于表达模型内部顺序、等待、发射与完成关系，不表示真实物理执行时间戳。
- `ProgramCycleStats` 也必须遵守同一时间语义边界。

### 7.4 Functional 与 Cycle 的语义边界

- `functional model` 的目标是功能正确和可解释的模型时间推进。
- `cycle model` 的目标是 tick-driven 的资源/调度/等待建模与结果趋势分析。
- `cycle model` 只有一个模式，不再拆成 cycle `st/mt`。
- `functional st` 是最确定的参考语义：
  - 满足恢复条件后，在下一 issue quantum 起点消费下一条可执行指令。
- `functional mt` 共享相同的模型时间规则，但保留 runnable wave 竞争与 CPU 并行调度差异。
- `cycle` 中必须允许 `ready != selected != issue`。

### 7.5 Wave / 调度 / 时间线约束

- wave 是执行态核心单元。
- 每个 wave 需要持有自己的推进状态；全局执行再基于统一模型时间、资源状态和调度器推进。
- `resume` 的语义不是“事件刚发生”，而是“该 wave 重新满足继续执行条件”；真正执行下一条指令仍受对应模型的调度规则约束。
- 相邻指令的 issue 间隔、wait/arrive/commit、barrier release、front-end latency 等都属于执行模型事实，再由 trace 消费。
- `cycle timeline` 必须遵守：
  - 只有真实 `InstructionIssue -> Commit` 配对的指令才允许生成 instruction slice
  - `Arrive / Stall / Barrier / WaveLifecycle / IssueSelect / SlotBind / WaveGenerate / WaveDispatch` 都是 marker，不得伪造成 instruction duration slice
  - `IssueSelect` 只表示 selected，不表示 real issue
  - `Arrive` 记录异步完成时点，不得偷换成 resume issue 时点
  - runtime/block 事件只留在 runtime 轨道，不挂到 wave slot 轨道
- timeline 的最细观察面应优先围绕 slot / lane-of-execution，而不是只围绕 wave 名称。
- 对 `cycle`，最细轨道需要保留 resident slot 语义；对 `st/mt`，最细轨道允许使用逻辑 lane，但仍需共享统一 hierarchy 和字段命名。
- bubble 应保持为空白区间，而不是伪造“空指令”填充。
- stall 原因必须是稳定 taxonomy，而不是临时自然语言文本。

### 7.6 当前正式主线

当前仍需持续推进的正式主线只有：

- trace canonical typed event model
- trace unified entry
- functional `mt` scheduler semantics
- cycle observability / stall taxonomy
- `ProgramCycleStats` calibration
- examples full-batch verification
- runtime API closure
- memory pool / `mmap` mapping
- ISA asm-kernel validation
- unified logging

其中还应默认遵守以下更细约束：

- `functional mt`：执行单位是 wave，不是 block；默认 worker budget 是全局共享资源，不应按 AP 复制一套 OS 线程池。
- `dispatch`：需继续统一 `ready -> selected -> issue` 语义边界，blocked sibling 不应拖死 ready sibling。
- `ProgramCycleStats`：与 `ExecutionStats` 保持语义分离，前者面向 active-lane / program-work 解释，后者保留粗粒度事件统计。
- `128 x 128 conditional multibarrier` 真实 HIP case 继续作为 program stats 与多 barrier 稳定性的主 baseline 之一。

历史 plans/specs 中若包含更细的步骤、旧命名或过渡方案，应视为 archive，而不是当前规范。

### 7.7 面向终极目标的设计分层

终极目标应明确分成三层，而不是混成一个“大而全”的目标：

1. `Result-correct functional layer`
   - 以 `st/mt` 为主
   - 目标是得到与真实 GPU 一致的最终执行结果
   - host 测试可直接验证通过

2. `Reference-cycle functional layer`
   - 仍以 `st/mt` 为基础
   - 目标是输出稳定、可比较、对优化有指导意义的参考 cycle
   - 算法或 schedule 变化时，cycle 趋势应有解释力

3. `Refined cycle-simulation layer`
   - 即 `cycle mode`
   - 目标是在 reference-cycle 之上继续细化资源、dispatch、stall、latency 建模
   - 不是 correctness 的来源，而是更细粒度的 timing 模型

因此优先级必须遵守：

- 先 correctness
- 再 reference cycle
- 最后 refined cycle simulation

而不是反过来用 cycle mode 去定义 correctness 主线。

## 8. Wave 执行上下文

wave 是执行时的核心单位。

长期对外语义统一称为 `WaveContext`。

执行语义统一使用 `WaveContext`。
不再保留单独的 `WaveState` 对外层名称。

### 8.1 WaveContext 应包含的内容

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

### 8.2 Launch 初始化

kernel launch 时应完成：

- grid/block 维度读取
- kernel 参数装载
- 隐式和显式 SGPR/VGPR 初始化
- `dpc/ap/peu/wave/thread` 层级放置
- `exec mask` 和 builtin id 初始化
- LLVM AMDGPU ABI 约定下的特殊寄存器装载

## 9. 功能模型与周期模型

### 9.1 Functional Model

functional model 的目标是结果正确。

要求：

- 支持单线程和多线程两种驱动方式
- 两者共用同一套执行核心
- 共用 wave state、memory、sync、effect apply 逻辑

### 9.2 Cycle Model

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

对于下一阶段轻量级 `VL1 / SL1 / L2`、访存合并与 hit/miss 的接口预留方案，详见：

- [memory-hierarchy-interface-reservation.md](/data/gpu_model/docs/memory-hierarchy-interface-reservation.md)

## 10. 调试、Trace 与性能观测

项目必须内建统一的调试和观测能力，而不是事后附加。

### 10.1 日志

应有统一日志系统，支持：

- 开关控制
- 等级控制
- 面向 runtime / program / instruction / execution 的分层输出

### 10.2 指令与 Wave Trace

应支持：

- wave 初始化 trace
- 指令执行 trace
- 可选寄存器值打印
- `pc / wave id / block id / dpc / ap / peu` 输出
- wave 切换原因说明

### 10.3 性能观测

应支持：

- pipeline utilization
- bubble 比例
- memory bound / compute bound 分类
- instruction cycle 统计
- AP / PEU 利用率统计

### 10.4 Timeline

应支持导出可被 Perfetto 或同类工具导入的 timeline。

timeline 至少应能展示：

- 不同 PEU 上的 wave 指令流
- 指令的周期区间
- barrier / wait / stall
- wave 内指令是连续发射还是出现空泡

当前 cycle timeline / Perfetto 的正式消费层约束：

- timeline 只消费执行模型已经产生的事件，不补造业务事件
- trace/timeline 上的 `cycle` 一律表示 modeled cycle，不是宿主 wall-clock
- 只有真实的 `InstructionIssue -> Commit` 配对才能生成 instruction slice
- `Arrive`、`Stall`、`Barrier`、`WaveLaunch/WaveExit`、`IssueSelect`、`WaveGenerate/WaveDispatch/SlotBind` 都只允许作为 marker 或 runtime event 暴露
- `ready / selected / issue` 必须保持可观察边界，不能在 timeline 中互相冒充

## 11. 长期目录与模块收口目标

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
- `HostRuntime` -> `ExecEngine`
- `RuntimeHooks` -> `HipRuntime`
- `ModelRuntimeApi` -> `ModelRuntime`
- `FunctionalExecutionCore` -> `FunctionalExecEngine`
- `FunctionalExecutor` / `ParallelWaveExecutor` -> `FunctionalExecEngine`
- `CycleExecutor` -> `CycleExecEngine`
- `RawGcnExecutor` -> `EncodedExecEngine`

## 12. 第一阶段收口原则

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
