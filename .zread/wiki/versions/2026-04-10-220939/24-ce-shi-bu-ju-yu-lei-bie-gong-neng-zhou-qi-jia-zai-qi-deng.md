本页面向初学者，系统梳理本项目测试套件的分类与作用域，帮助你在“功能正确性”“周期模型”“加载器/镜像”“运行时与 Trace”“指令/ISA”等不同层级中快速定位用例与验证目标，避免误用测试层级导致的信号混淆。整体分类与来源均以 tests/CMakeLists.txt 中的目标编译清单为准，并辅以代表性测试文件佐证。Sources: [CMakeLists.txt](tests/CMakeLists.txt#L1-L7) [CMakeLists.txt](tests/CMakeLists.txt#L12-L37) [CMakeLists.txt](tests/CMakeLists.txt#L57-L77) [CMakeLists.txt](tests/CMakeLists.txt#L78-L89) [CMakeLists.txt](tests/CMakeLists.txt#L94-L123)

## 测试总览与编译产物
全部测试以单一可执行文件 gpu_model_tests 构建，统一链接到核心库 gpu_model 与 GTest，便于一次性运行全覆盖回归；清单中直接列出各类别测试源，保证分类清晰、可溯源。编译目标同时设置了包含目录与编译期宏（用于在测试中解析仓库根路径）。Sources: [CMakeLists.txt](tests/CMakeLists.txt#L1-L6) [CMakeLists.txt](tests/CMakeLists.txt#L125-L147)

下图给出“测试类别 → 校验层级”的关系图，基于 CMake 清单与代表性测试内容抽象而成，便于初学者按层定位问题（先功能后周期，遇到装载失败看加载器，遇到行为命名或类型对齐看指令/ISA 与执行层，涉及 Trace/可视化看运行时与 Trace）：
mermaid
flowchart TD
  A[tests 目录] --> B[functional 功能测试]
  A --> C[cycle 周期测试]
  A --> D[loader 加载器/镜像测试]
  A --> E[runtime 运行时与 Trace 测试]
  A --> F[instruction 指令对象/解码/格式化]
  A --> G[isa ISA 描述/Opcode]
  A --> H[execution 执行层与内部策略]
  A --> I[program 程序对象模型]
  A --> J[arch 架构注册/规格]
  B -->|ExecEngine.Launch (Functional)| E0[gpu_model/runtime/exec_engine]
  C -->|ExecutionMode::Cycle| E1[gpu_model/execution/cycle_exec_engine]
  D -->|AsmParser/Bundle IO| E2[gpu_model/loader/*]
  E -->|Trace/Perfetto| E3[gpu_model/debug/*]
  F -->|Decoder/Formatter| E4[gpu_model/instruction/*]
  G -->|OpcodeDescriptor| E5[gpu_model/isa/*]
  H -->|Issue/Waitcnt/Barrier| E6[gpu_model/execution/internal/*]
  I -->|ProgramObject| E7[gpu_model/program/*]
  J -->|ArchRegistry| E8[gpu_model/arch/*]
Sources: [CMakeLists.txt](tests/CMakeLists.txt#L8-L37) [CMakeLists.txt](tests/CMakeLists.txt#L78-L89) [CMakeLists.txt](tests/CMakeLists.txt#L94-L123)

## 测试运行的默认环境与开关
测试入口 test_main 对运行环境做了统一配置：默认多线程功能模式（更快）、默认关闭 Trace 与日志；可通过环境变量覆盖，例如 GPU_MODEL_DISABLE_TRACE=0 开启 Trace、GPU_MODEL_DISABLE_LOGURU=0 开启日志、GPU_MODEL_FUNCTIONAL_MODE=st 切换单线程功能模式。程序名还会注入 GPU_MODEL_LOG_PROGRAM 以标记日志来源。Sources: [test_main.cpp](tests/test_main.cpp#L6-L21)

## 类别一：功能测试（functional）
- 目的与边界：验证“语义正确性”和“数据面效果”，不关心精确时序；常以构建内核（InstructionBuilder/装载后的 ExecutableKernel）+ ExecEngine.Launch 驱动，断言内存结果或 Trace 是否包含关键事件。代表用例通过 builder 组装 vecadd，发起多 Block/多线程执行并核对结果。Sources: [vecadd_functional_test.cpp](tests/functional/vecadd_functional_test.cpp#L13-L32) [vecadd_functional_test.cpp](tests/functional/vecadd_functional_test.cpp#L43-L78)

- 典型形态：功能测试也会在需要时附带轻量 Trace 验证，例如确认出现 MemoryAccess 与 WaveExit 事件；但不要求时序数值稳定（避免与周期测试职责重叠）。Sources: [vecadd_functional_test.cpp](tests/functional/vecadd_functional_test.cpp#L80-L111)

- 覆盖面与入口参考：functional 目录覆盖从映射/栅栏/私有/常量内存到 2D/3D/转置等语义主题，均收敛到 ExecEngine 统一入口。Sources: [CMakeLists.txt](tests/CMakeLists.txt#L38-L57)

表：功能 vs 周期的取舍（面向初学者）
- 功能测试：关注结果正确性、可附带事件存在性，不断言总周期；适合快速回归与语义迭代。Sources: [vecadd_functional_test.cpp](tests/functional/vecadd_functional_test.cpp#L70-L78) [CMakeLists.txt](tests/CMakeLists.txt#L38-L57)
- 周期测试：关注精确时序、调度/队列/前端开销等，断言周期数字或事件序列顺序；适合模型标定与性能回归。Sources: [cycle_smoke_test.cpp](tests/cycle/cycle_smoke_test.cpp#L100-L111) [cycle_smoke_test.cpp](tests/cycle/cycle_smoke_test.cpp#L142-L168)

## 类别二：周期模型测试（cycle）
- 目的与边界：在 ExecutionMode::Cycle 下断言总周期、开始/结束周期、设备 gap 等时序指标；同一语义在功能测试通过后，再以周期测试锚定“时序真实度”。Sources: [cycle_smoke_test.cpp](tests/cycle/cycle_smoke_test.cpp#L104-L111) [cycle_smoke_test.cpp](tests/cycle/cycle_smoke_test.cpp#L113-L141)

- 典型形态：通过定制前端延迟轮廓、校验连续内核的 device gap、Block 排队与 Wave 启动等前端/调度现象；也会结合 ArchRegistry 读设备规格驱动数据规模（如 AP 数量溢出触发排队）。Sources: [cycle_smoke_test.cpp](tests/cycle/cycle_smoke_test.cpp#L147-L168) [cycle_smoke_test.cpp](tests/cycle/cycle_smoke_test.cpp#L170-L191)

- 事件级验证：利用 TraceSink 采集事件，按 Wave/PEU/PC 过滤定位首个/后续事件，断言 Issue/Barrier/Dispatch 等顺序，支持排队时刻与跨 AP 包裹验证。Sources: [cycle_smoke_test.cpp](tests/cycle/cycle_smoke_test.cpp#L27-L47) [cycle_smoke_test.cpp](tests/cycle/cycle_smoke_test.cpp#L50-L66)

## 类别三：加载器与镜像测试（loader）
- 目的与边界：覆盖汇编文本解析、标签/PC 空间解析、WaitCnt 语法、对象/包（bundle）与可执行镜像的 IO 路径，确保“程序如何被装载与还原”为稳定基座。Sources: [asm_parser_test.cpp](tests/loader/asm_parser_test.cpp#L14-L33) [asm_parser_test.cpp](tests/loader/asm_parser_test.cpp#L115-L132)

- 汇编解析与执行集成：解析 vecadd 汇编后以功能模式执行并断言结果，证明“解析→构建→执行”的端到端链路稳定；同时验证标签按编码尺寸对齐（PC 递增为 8 字节步进）。Sources: [asm_parser_test.cpp](tests/loader/asm_parser_test.cpp#L81-L113) [asm_parser_test.cpp](tests/loader/asm_parser_test.cpp#L45-L56)

- 用例组织与外部工具链：asm_cases 保存装载端到端测试所需的汇编夹具，鼓励使用可被 llvm-mc 装配的真实语法，并建议“扩展发现机制而非在 C++ 测试里内嵌长汇编”。Sources: [README.md](tests/asm_cases/README.md#L1-L10)

## 类别四：运行时与 Trace/可视化测试（runtime）
- 目的与边界：验证 ModelRuntime/HipRuntime 行为、会话与内存生命周期、并行模式、执行统计与时间线记录；同时核查 Trace 的结构与字段（尤其是 Perfetto 导出）是否使用“规范化名称”，避免依赖消息解析。Sources: [CMakeLists.txt](tests/CMakeLists.txt#L94-L123) [trace_perfetto_test.cpp](tests/runtime/trace_perfetto_test.cpp#L12-L22)

- Perfetto 结构与命名验证：构造极小内核执行，导出 timeline.perfetto.json，断言 traceEvents 结构存在；同时针对 WaveLaunch/WaveExit 与前端/运行时标记验证“规范化名称”而非 message 文本，确保前后兼容。Sources: [trace_perfetto_test.cpp](tests/runtime/trace_perfetto_test.cpp#L54-L90) [trace_perfetto_test.cpp](tests/runtime/trace_perfetto_test.cpp#L92-L160) [trace_perfetto_test.cpp](tests/runtime/trace_perfetto_test.cpp#L162-L200)

- 外部工具链前置条件：部分运行时与装载路径测试会探测 LLVM/AMDGPU 或 HIP 主机工具链（如 llvm-mc、hipcc 等），未安装时可自动跳过或走替代路径；相关工具链检测与夹具装配封装在测试工具方法中。Sources: [hip_runtime_test.cpp](tests/runtime/hip_runtime_test.cpp#L27-L41) [hip_runtime_test.cpp](tests/runtime/hip_runtime_test.cpp#L67-L86) [trace_perfetto_test.cpp](tests/runtime/trace_perfetto_test.cpp#L20-L22)

## 类别五：指令/ISA 与执行层基础测试（instruction/isa/execution/internal）
- 指令与 ISA：覆盖解码器、格式化器、指令对象执行与 ISA Opcode 描述等，作为“语义管道之前”的契约护栏；清单中列出 decoder/formatter/object_execute 与 isa/opcode_descriptor 等测试条目。Sources: [CMakeLists.txt](tests/CMakeLists.txt#L8-L12) [CMakeLists.txt](tests/CMakeLists.txt#L29-L37) [CMakeLists.txt](tests/CMakeLists.txt#L36-L37)

- 执行层与内部策略：对 Issue 模型、Eligibility、Scheduler、Barrier 资源池、WaitCnt/Tensor 工具等进行单元化验证，确保周期模型的策略件按预期组合。Sources: [CMakeLists.txt](tests/CMakeLists.txt#L20-L28) [CMakeLists.txt](tests/CMakeLists.txt#L35-L35)

- 命名与类型契约：通过 ExecutionNamingTest 对关键类型/方法签名做静态断言，约束 IExecutionEngine/CycleExecEngine 等接口一致性，减少重构回归。Sources: [execution_naming_test.cpp](tests/execution/execution_naming_test.cpp#L18-L42) [execution_naming_test.cpp](tests/execution/execution_naming_test.cpp#L44-L69)

## 类别六：程序对象/架构注册（program/arch）
- 程序对象：验证 Program 源读取与对象类型，服务于“加载前/后”的边界稳定；与 loader 类别共同覆盖从文本/对象→可执行的过渡。Sources: [CMakeLists.txt](tests/CMakeLists.txt#L90-L93)

- 架构注册：最小化测试 arch_registry 接入路径，确保通过名称获取规格等基础能力无回归，常与周期测试中基于规格的参数化结合使用。Sources: [CMakeLists.txt](tests/CMakeLists.txt#L7-L7) [cycle_smoke_test.cpp](tests/cycle/cycle_smoke_test.cpp#L170-L178)

## 目录布局与命名约定（面向初学者）
tests 目录按“验证层级/主题”分子目录：functional（语义）、cycle（时序）、loader（装载/镜像）、runtime（运行时与 Trace）、instruction/isa/execution（指令/描述/执行策略）、program/arch（程序对象/架构）；CMake 清单与 asm_cases 约定共同定义了组织方式与夹具来源。Sources: [CMakeLists.txt](tests/CMakeLists.txt#L8-L37) [CMakeLists.txt](tests/CMakeLists.txt#L57-L89) [CMakeLists.txt](tests/CMakeLists.txt#L90-L123) [README.md](tests/asm_cases/README.md#L1-L10)

表：测试类别速览（目的/入口/示例/工具链）
- 功能（functional）：语义正确性；ExecEngine.Launch；示例 vecadd_functional_test；一般无外部工具链要求。Sources: [vecadd_functional_test.cpp](tests/functional/vecadd_functional_test.cpp#L43-L78)
- 周期（cycle）：时序与调度；ExecutionMode::Cycle；示例 cycle_smoke_test；无硬性工具链，但常配合 TraceSink 分析。Sources: [cycle_smoke_test.cpp](tests/cycle/cycle_smoke_test.cpp#L100-L111) [cycle_smoke_test.cpp](tests/cycle/cycle_smoke_test.cpp#L170-L178)
- 加载器（loader）：汇编/对象/包解析与 IO；AsmParser/Bundle IO；示例 asm_parser_test；推荐安装 LLVM 工具链以跑端到端。Sources: [asm_parser_test.cpp](tests/loader/asm_parser_test.cpp#L14-L33) [README.md](tests/asm_cases/README.md#L5-L10)
- 运行时与 Trace（runtime）：会话/统计/Perfetto；ModelRuntime/HipRuntime/Trace；示例 trace_perfetto_test；可能依赖 llvm-mc/hipcc。Sources: [trace_perfetto_test.cpp](tests/runtime/trace_perfetto_test.cpp#L54-L90) [hip_runtime_test.cpp](tests/runtime/hip_runtime_test.cpp#L27-L41)
- 指令/ISA/执行内部：解码/格式化/Issue 策略；对应子模块；示例 execution_naming_test/opcode_descriptor_test；无外部工具链。Sources: [execution_naming_test.cpp](tests/execution/execution_naming_test.cpp#L18-L42) [CMakeLists.txt](tests/CMakeLists.txt#L36-L37)

## 选择合适测试层级的决策流
当你新增/修改组件时，可按下述决策流选择测试入口，避免“拿错层级”：
mermaid
flowchart LR
  S[开始：变更点] --> P1{是否解析/装载相关?}
  P1 -- 是 --> L[loader: 解析/PC/元数据/镜像IO 测试]
  P1 -- 否 --> P2{是否语义变化?}
  P2 -- 是 --> F[functional: 结果正确性 + 事件存在性]
  P2 -- 否 --> P3{是否调度/性能/前端时序?}
  P3 -- 是 --> C[cycle: 周期数字/顺序/排队]
  P3 -- 否 --> P4{是否 ISA/指令/格式化?}
  P4 -- 是 --> I[instruction/isa: 解码/格式化/描述]
  P4 -- 否 --> P5{是否运行时/Trace/可视化?}
  P5 -- 是 --> R[runtime: 会话/统计/Perfetto/工具链]
  P5 -- 否 --> E[execution/internal: Issue/Barrier/WaitCnt 工具]
Sources: [CMakeLists.txt](tests/CMakeLists.txt#L8-L37) [CMakeLists.txt](tests/CMakeLists.txt#L57-L89) [CMakeLists.txt](tests/CMakeLists.txt#L94-L123)

## 质量与覆盖：与 ISA 覆盖报告的衔接
ISA 覆盖报告统计了解码/执行/加载器维度的覆盖率，为用例编写提供定量参考：例如“decode/exec 100%，loader 集成 22%”提示应优先补齐装载端到端用例；本页分类与覆盖报告可交叉使用，定位“哪个层级缺口最大”。Sources: [isa_coverage_report.md](docs/isa_coverage_report.md#L11-L17)

## 进阶阅读与后续路径
- 想进一步理解不同执行模式与引擎职责，请阅读：[执行模式与 ExecEngine 工作流](11-zhi-xing-mo-shi-yu-execengine-gong-zuo-liu)。Sources: [execution_naming_test.cpp](tests/execution/execution_naming_test.cpp#L18-L38)
- 若需扩展装载路径与镜像格式，请转至：[加载器与镜像格式支持（AMDGPU object/HIP fatbin）](14-jia-zai-qi-yu-jing-xiang-ge-shi-zhi-chi-amdgpu-object-hip-fatbin)。Sources: [asm_parser_test.cpp](tests/loader/asm_parser_test.cpp#L14-L33)
- 建立回归闭环与示例协同机制，请看：[示例与测试协同回归策略](25-shi-li-yu-ce-shi-xie-tong-hui-gui-ce-lue)。Sources: [CMakeLists.txt](tests/CMakeLists.txt#L1-L6)
- 需要量化验证覆盖，请参考：[ISA 覆盖率生成与报告解读](26-isa-fu-gai-lu-sheng-cheng-yu-bao-gao-jie-du)。Sources: [isa_coverage_report.md](docs/isa_coverage_report.md#L1-L17)