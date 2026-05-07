# GPU Model 全项目架构优化与重构分析

> 注：本文是架构分析快照与分阶段建议文档，不是当前代码事实的唯一规范源。
> 若其中的具体文件路径、已完成度判断或待办项与当前代码或主文档冲突，请以
> `task_plan.md`、`docs/architecture-restructuring-plan.md`、`docs/runtime-layering.md`
> 和当前代码为准。

## 1. 目的与边界

本文面向当前整个 `gpu_model` 仓库，分析：

- 现有代码架构哪些地方已经基本合理
- 哪些地方存在结构性设计债
- 哪些模块需要重构才能更符合设计模式与工程最佳实践
- 这些重构建议应如何分阶段推进

本文不是功能设计文档，也不是一次性大重写计划。

本文遵守当前仓库已经明确的正式约束：

- runtime 主线保持 `HipRuntime -> ModelRuntime -> ExecEngine`
- `trace / recorder / timeline` 只消费执行事实，不反向定义业务语义
- `cycle` 继续保持单一时序模型，不引入 `cycle st/mt`
- 不恢复独立 `interposer` 子系统概念

换句话说，本文讨论的是：

- 在不破坏当前正式架构方向的前提下
- 如何把“已经收口的设计意图”进一步落到更稳的代码结构上

## 2. 当前架构的积极面

在讨论问题之前，需要先明确当前项目并不是“完全失控”的代码库。

当前值得保留的正面结构包括：

- 公开主线命名已经基本收口到 `HipRuntime / ModelRuntime / ExecEngine`
- `ProgramObject` 作为程序静态表示已经成为正式主对象
- `FunctionalExecEngine / CycleExecEngine / ProgramObjectExecEngine` 的角色边界至少在命名层已经可识别
- `TraceSink`、`Recorder`、`Timeline` 三层已经明确了“trace 不能驱动业务逻辑”的方向
- `DeviceMemoryManager` 已经把 compatibility pointer window 的方向做出来了
- `ExecEngine` 已经采用 `pImpl`，说明仓库已经接受“对外 API 与实现解耦”的思路

所以本文的主判断不是“推翻现有架构”，而是：

- 方向大体正确
- 但代码层还有不少“半收口状态”
- 导致依赖方向、职责分离、状态所有权和构建边界还不够稳

## 3. 总体判断

当前项目最核心的架构问题，不是单个模块“写得太长”，而是下面四类结构性问题叠加：

1. 公共接口层与内部实现层还没有真正隔离
2. runtime / execution / loader 仍存在多个“总控类”承担过多职责
3. 状态所有权在 `MemorySystem / DeviceMemoryManager / ModelRuntime / RuntimeSession` 等模块之间仍有重叠
4. 工程构建层没有把模块边界强制下来，很多跨层依赖只是“现在恰好能编过”

如果不优先处理这四类问题，后续即使单点功能继续推进，也会持续遇到：

- 修改一个模块，连带多个模块一起重编
- 设计约束写在文档里，但代码仍能轻易穿透边界
- 同一语义在多个地方各自持有一份“差不多”的状态
- 测试虽然多，但很多测试只能验证结果，难以约束结构退化

## 4. P0：最需要优先重构的架构问题

### 4.1 公共头文件与 `internal/*` 仍然互相泄漏

这是当前最应该优先处理的问题。

直接证据：

- [functional_exec_engine.h](../../src/gpu_model/execution/functional_exec_engine.h#L6) 直接包含 `gpu_model/execution/internal/semantics.h`
- [cycle_exec_engine.h](../../src/gpu_model/execution/cycle_exec_engine.h#L6) 直接包含 `gpu_model/execution/internal/execution_engine.h`
- [gpu_arch_spec.h](../../src/gpu_model/arch/gpu_arch_spec.h#L7) 直接包含 `gpu_model/execution/internal/issue_model.h`
- [exec_engine.h](../../src/gpu_model/runtime/exec_engine.h#L7) 直接包含 `gpu_model/execution/cycle_exec_engine.h`
- [program_object_exec_engine.h](../../src/gpu_model/execution/program_object_exec_engine.h#L6) 直接依赖 `CycleTimingConfig`

这意味着：

- `src/gpu_model/*` 这一层虽然承担了“公开头”角色
- 但它实际上仍然把执行内部契约暴露到了上层

结果是：

- `arch` 层依赖 `execution/internal`
- `runtime` 层依赖具体 cycle 实现头
- `program-object` 执行入口被具体 timing 类型绑住

这不符合稳定依赖方向。

更理想的做法是：

- 把稳定的公共值类型单独提到非 `internal` 头
- 例如：
  - `issue_policy_types.h`
  - `cycle_timing_config.h`
  - `execution_context_fwd.h`
- 公共头只依赖这些稳定类型
- `internal/*` 只允许实现文件和真正内部头使用

建议重构：

- 从 [gpu_arch_spec.h](../../src/gpu_model/arch/gpu_arch_spec.h) 中抽离 issue policy 相关值类型，避免 `arch -> execution/internal`
- 从 [cycle_exec_engine.h](../../src/gpu_model/execution/cycle_exec_engine.h) 中抽离 `CycleTimingConfig`
- 让 [program_object_exec_engine.h](../../src/gpu_model/execution/program_object_exec_engine.h) 只依赖稳定配置头，而不是直接依赖 cycle engine 头
- 让 [functional_exec_engine.h](../../src/gpu_model/execution/functional_exec_engine.h) 只前置声明 `ExecutionContext` 或依赖稳定 context 头，而不是整个 `semantics.h`

这是后续所有重构的基础，因为如果边界不先收紧，后面的职责拆分很容易重新耦回去。

### 4.2 `ExecEngine`、`RuntimeSession`、`ModelRuntime` 都偏“总控类”

当前 runtime 主线上，已经同时存在三个高负荷总控对象：

- [ExecEngine](../../src/gpu_model/runtime/exec_engine.h)
- [ModelRuntime](../../src/gpu_model/runtime/model_runtime.h)
- [RuntimeSession](../../src/gpu_model/runtime/runtime_session.h)

问题不在于“有三个类”，而在于它们都同时承担了多类职责。

#### 4.2.1 `ExecEngine` 职责过重

直接证据：

- [exec_engine.cpp](../../src/runtime/exec_engine.cpp) 有 `725` 行
- 它同时包含：
  - 架构选择
  - metadata 校验
  - trace sink 解析
  - cycle timing resolve
  - launch 参数修正
  - execution backend 分发
  - program-object 路径和 direct kernel 路径切换
- 还直接包含了：
  - loader 头
  - trace event 工厂
  - execution backend 头
  - `execution/internal/cycle_issue_policy.h`

这说明 `ExecEngine` 目前像一个“大型应用服务 + 运行时装配器 + 执行调度器”的混合体。

更好的目标形态是：

- `ExecEngine` 保留 facade 身份
- 其内部拆成几个更窄的服务：
  - `LaunchRequestValidator`
  - `LaunchPreparationService`
  - `CycleTimingResolver`
  - `ExecutionDispatcher`
  - `TraceSessionBuilder`

这样有两个好处：

- 改 trace 初始化时，不需要重新理解 kernel launch 语义主线
- 改 cycle timing merge 逻辑时，不会影响 program-object / functional 分发结构

#### 4.2.2 `RuntimeSession` 承担了过多 compatibility 相关状态

直接证据：

- [runtime_session.h](../../src/gpu_model/runtime/runtime_session.h#L38) 暴露的方法横跨：
  - device pointer 管理
  - stream/event
  - launch config stack
  - error 状态
  - executable image load
  - compatibility arg pack
  - trace artifact recorder
- [runtime_session.h](../../src/gpu_model/runtime/runtime_session.h#L113) 还持有 thread-local error / stream 状态和 trace artifact recorder
- [hip_runtime.cpp](../../src/runtime/hip_runtime.cpp#L24) 到 [hip_runtime.cpp](../../src/runtime/hip_runtime.cpp#L322) 大量方法都只是继续转发到 `GetRuntimeSession()`

这带来的问题是：

- `HipRuntime` 虽然语义上很薄
- 但真正的 compatibility 复杂度被挤到了 `RuntimeSession`
- `RuntimeSession` 又通过全局 `GetRuntimeSession()` 成为隐式共享状态容器

这不利于：

- 多 context / 多 stream 扩展
- 显式依赖注入
- 在测试里构造多个互不干扰的 runtime 会话

建议重构：

- 保留 `RuntimeSession` 作为 compatibility facade
- 但内部拆为：
  - `CompatibilityDeviceState`
  - `CompatibilityStreamState`
  - `CompatibilityLaunchState`
  - `CompatibilityArtifactState`
- `GetRuntimeSession()` 只保留在 HIP ABI bridge 层使用
- 非 ABI 核心逻辑应优先显式依赖对象，而不是依赖全局 session

#### 4.2.3 `ModelRuntime` 的所有权边界仍不够干净

直接证据：

- [model_runtime.h](../../src/gpu_model/runtime/model_runtime.h#L91) 同时持有 `owned_runtime_` 和 `runtime_engine_`
- [model_runtime.h](../../src/gpu_model/runtime/model_runtime.h#L95) 还自己维护 `allocations_`
- 同时它又负责：
  - module registry
  - load plan materialize
  - kernel/program launch
  - device property 查询

这里的问题不是“功能太多”，而是：

- `ModelRuntime` 看起来应该是一个稳定 runtime facade
- 但它仍持有部分资源所有权和分配跟踪
- 与 `DeviceMemoryManager`、`MemorySystem`、`RuntimeSession` 的边界仍有交叉

建议重构：

- `ModelRuntime` 继续保留 facade 角色
- 但让内存分配追踪只由一个对象持有
- 如果 compatibility allocation 归 `DeviceMemoryManager`
- 那 `ModelRuntime::allocations_` 应逐步退场

### 4.3 全局单例与隐式状态过多，限制可测试性和后续扩展

直接证据：

- [runtime_config.h](../../src/gpu_model/runtime/runtime_config.h#L61) 使用 `RuntimeConfigManager::Instance()`
- [runtime_session.h](../../src/gpu_model/runtime/runtime_session.h#L126) 暴露 `GetRuntimeSession()`
- [hip_ld_preload.cpp](../../src/runtime/hip_runtime/hip_ld_preload.cpp) 通过静态 `HipApi()` 驱动整个 ABI 路径

这种设计对 ABI bridge 是方便的，但如果进入核心运行时主线，就会带来典型问题：

- 测试顺序相关
- 状态污染难以定位
- 无法显式创建两套 runtime 世界
- 未来如果扩到多 context / 多 device，会先撞到单例状态假设

建议遵循的原则是：

- “全局单例只留在 ABI 适配边缘”
- “核心运行时主线尽量显式传递 session / config / service”

更具体地说：

- `hip_ld_preload.cpp` 可以继续通过静态 `HipRuntime`
- 但 `HipRuntime -> RuntimeSession -> ModelRuntime -> ExecEngine` 内部应逐步改成可显式构造、可注入、可复用的对象图

## 5. P1：中优先级但值得系统处理的问题

### 5.1 内存模型的状态所有权有重叠

当前至少有三层与内存所有权有关：

- [MemorySystem](../../src/gpu_model/memory/memory_system.h)
- [DeviceMemoryManager](../../src/gpu_model/runtime/device_memory_manager.h)
- [ModelRuntime](../../src/gpu_model/runtime/model_runtime.h)

直接证据：

- [memory_system.h](../../src/gpu_model/memory/memory_system.h) 直接持有各 pool 的字节存储
- [device_memory_manager.h](../../src/gpu_model/runtime/device_memory_manager.h) 持有 compatibility windows 和 allocation map
- [model_runtime.h](../../src/gpu_model/runtime/model_runtime.h#L95) 仍保留 `allocations_`

这说明当前已经有“正确方向”，但尚未完全做到：

- `MemorySystem` 负责底层地址空间和读写
- `DeviceMemoryManager` 负责 compatibility pointer 视图
- `ModelRuntime` 不再重复持有 allocation state

建议重构：

- 明确单一事实来源：
  - compatibility pointer / host-visible mapping -> `DeviceMemoryManager`
  - model pool storage / model_addr range -> `MemorySystem`
  - runtime facade 只调度，不重复维护 allocation table
- 后续如果要支持更多 pool 或 map/unmap，优先扩展这两个对象的边界，而不是再新挂一层 map

### 5.2 artifact ingestion 路径职责交叉

这是 `program / loader` 方向最明显的结构债。

直接证据：

- [encoded_program_object.cpp](../../src/program/encoded_program_object.cpp#L62) 定义 `RunCommand(...)`
- [encoded_program_object.cpp](../../src/program/encoded_program_object.cpp#L80) 自带临时目录生命周期
- [encoded_program_object.cpp](../../src/program/encoded_program_object.cpp#L121) 同时做 fatbin 提取、bundle 识别、code object materialize
- 同一文件后面还继续做：
  - section/symbol parsing
  - metadata/note parsing
  - descriptor 绑定
  - `ProgramObject` 组装
- 而 [object_reader.cpp](../../src/program/object_reader.cpp#L80) 还单独承担了 asm-stem 路径的文本/metadata/const 读取

这类结构的问题在于：

- “文件输入源”
- “外部工具调用”
- “ELF/metadata 解析”
- “ProgramObject 构造”

四类职责目前没有明确分层。

建议重构成更清晰的 pipeline：

1. `ArtifactLocator`
   - 判定输入是 asm stem / AMDGPU object / host executable
2. `CodeObjectExtractor`
   - 只负责 fatbin 抽取、bundle 选择、临时文件
3. `AmdgpuArtifactParser`
   - 只负责 section/symbol/note/descriptor 解析
4. `ProgramObjectBuilder`
   - 把 parser 结果装配成 `ProgramObject`

这样做后：

- `ObjectReader` 只做 façade
- 外部工具调用和解析逻辑不再耦在同一个编译单元里
- 将来如果改成纯库内解析，不会影响 `ProgramObject` 装配接口

### 5.3 三套执行引擎的共享域模型还不够纯

当前仓库已经意识到需要共享 wave/block/issue 相关状态，这点是对的。

但共享抽象还没纯化完。

直接证据：

- [wave_state.h](../../src/gpu_model/execution/internal/wave_state.h#L1) 被定位为共享状态头
- 但它仍直接包含：
  - `gpu_model/debug/trace/event_factory.h`
  - `gpu_model/execution/internal/issue_eligibility.h`
- 同时 `FunctionalExecEngine`、`CycleExecEngine`、`ProgramObjectExecEngine` 仍各自保留大量状态机和辅助逻辑
- 更早一轮旧审查也曾指出三套执行引擎之间存在重复

这里的核心问题不是“有没有重复代码”，而是：

- 共享状态对象还混入了 trace / eligibility 等偏实现层概念
- 导致它难以真正成为 engine-neutral domain model

建议重构方向：

- 保持 `WaveExecutionState / PendingMemoryOp / BlockBarrierState` 这类共享域对象
- 但把它们从 trace/event factory 依赖里剥离出去
- 让共享状态层只表达：
  - 调度状态
  - 等待状态
  - 执行累计状态
  - 资源可用性状态
- trace 事件拼装和 blocked reason 映射放在更外层 adapter/helper

这样未来无论是 functional 还是 cycle 扩展，都不会继续把共享层“实现化”。

### 5.4 trace / recorder / export 仍有展示策略与数据模型混写

直接证据：

- [trace/sink.h](../../src/gpu_model/debug/trace/sink.h#L27) 在同一个公共头里直接暴露：
  - `NullTraceSink`
  - `CollectingTraceSink`
  - `FileTraceSink`
  - `JsonTraceSink`
- [recorder.h](../../src/gpu_model/debug/recorder/recorder.h) 同时持有原始 `TraceEvent` 与 recorder-level decorated entry
- [recorder_export.cpp](../../src/debug/recorder/recorder_export.cpp#L75) 之后同时承担：
  - text/json 序列化
  - trace header/snapshot 写法
  - event 过滤
  - summary/presentation 拼接

这意味着当前 trace 栈大体能用，但还有两种职责没完全分开：

- “记录什么事实”
- “如何展示给用户”

建议重构：

- `TraceSink` 头只保留接口和最小测试用 sink
- 文件型 sink 放到实现层
- `Recorder` 只做 structured capture
- text/json/perfetto renderer 分别承担 presentation
- 像 [recorder_export.cpp](../../src/debug/recorder/recorder_export.cpp) 里“text trace 只保留 wave_step / wave_exit”这类展示策略，最好放进更清楚的 formatter/policy 层

### 5.5 参数团和长签名已经开始逼出“请求对象”需求

直接证据：

- [exec_engine.h](../../src/gpu_model/runtime/exec_engine.h#L33) 和 [exec_engine.h](../../src/gpu_model/runtime/exec_engine.h#L39) 的 `SetLaunchTimingProfile(...)` 已经有两个长参数版本
- [program_object_exec_engine.h](../../src/gpu_model/execution/program_object_exec_engine.h#L20) 的 `Run(...)` 需要十多个参数
- [hip_runtime.h](../../src/gpu_model/runtime/hip_runtime.h#L51) 和 [model_runtime.h](../../src/gpu_model/runtime/model_runtime.h#L47) 的 launch API 也已经很宽

这通常说明：

- 值类型已经出现
- 但还没有被提升成更稳定的请求对象或配置对象

建议重构：

- 将 launch timing 改成单一 `LaunchTimingSpec`
- `ProgramObjectExecEngine::Run(...)` 改收一个 `ProgramExecutionRequest`
- 如果需要收集更多 runtime 上下文，再定义 `ExecutionServices` 或 `ExecutionDependencies`

这样做的好处不是“为了好看”，而是：

- 参数扩展不再破坏大量调用点
- 单测更容易构造最小请求
- 数据团会自然变成可命名对象，职责也更清晰

### 5.6 构建层没有真正帮忙守住模块边界

直接证据：

- [CMakeLists.txt](../../CMakeLists.txt) 目前只构建一个大静态库 `gpu_model`
- 所有测试都链接同一个 [gpu_model_tests](../../tests/CMakeLists.txt) 大测试二进制
- 公共 include 根统一指向 `src`

这会导致两个问题：

- 模块边界即使穿透，构建系统也很难第一时间报出来
- 编译和测试成本会持续集中放大

建议重构：

- 第一阶段不必拆成很多共享库
- 但至少可以先拆成若干 CMake target：
  - `gpu_model_arch`
  - `gpu_model_program`
  - `gpu_model_execution`
  - `gpu_model_runtime`
  - `gpu_model_debug`
- 同时把测试逐步拆成按模块分组的可执行目标

这样即使最终还保留总库，也能让依赖方向在构建层变得可检查。

## 6. 推荐的目标架构形态

在不改变当前正式主线命名的前提下，更推荐的整体结构是：

### 6.1 API / Application / Domain / Adapter 四层

```text
HIP ABI / CLI / Tests
  -> HipRuntime / ModelRuntime / ExecEngine
    -> launch / module / memory / trace application services
      -> execution / program / memory / debug domain objects
        -> file system / mmap / readelf / objcopy / perfetto / env adapters
```

关键原则：

- API 层不直接依赖 `internal/*`
- application service 负责编排，不持有重复状态
- domain object 不依赖 trace/export/presentation
- adapter 负责：
  - 环境变量
  - 文件系统
  - 外部命令
  - ABI bridge

### 6.2 推荐保留的设计模式

当前项目里已经有一些模式雏形，建议加强而不是推翻：

- `Facade`
  - `HipRuntime`
  - `ModelRuntime`
  - `ExecEngine`
- `Strategy`
  - functional / cycle / encoded backend 分发
- `Adapter`
  - HIP ABI 到 runtime 主线
- `Builder / Assembler`
  - `ProgramObject` 组装
- `Value Object`
  - `LaunchConfig`
  - `RuntimeSubmissionContext`
  - `CycleTimingConfig`（建议抽独立头）

### 6.3 不建议采用的方向

不建议：

- 直接把所有执行引擎合并成一个超大模板类
- 为了“统一”而把 trace 逻辑再塞回共享状态层
- 在这轮里继续增加新的历史兼容层命名
- 在没有边界清理之前就贸然做大规模目录移动

## 7. 建议的重构顺序

### Phase 1：边界清理

目标：

- 先把“什么是公共契约，什么是内部细节”切干净

建议动作：

- 抽离 `CycleTimingConfig`
- 抽离 issue policy 相关稳定值类型
- 清掉公共头对 `internal/*` 的直接包含
- 收紧 `src/gpu_model/*` 的对外暴露范围

### Phase 2：runtime 总控拆分

目标：

- 缩小 `ExecEngine`、`RuntimeSession`、`ModelRuntime` 的变更半径

建议动作：

- 拆 `ExecEngineImpl::Launch`
- 拆 `RuntimeSession` 的 compatibility 状态管理
- 消除 `ModelRuntime` 与 `DeviceMemoryManager` 的 allocation ownership 重叠

### Phase 3：program / loader 管线拆分

目标：

- 让 artifact ingestion 更可测试、更可替换

建议动作：

- 外部工具调用与 parser 分离
- `ObjectReader` 收敛为 façade
- `ProgramObjectBuilder` 明确化

### Phase 4：execution 共享域模型纯化

目标：

- 把 execution 共享状态层从“半实现层”变成真正 domain 层

建议动作：

- 让 `wave_state.h` 不再依赖 trace/event factory
- 提取共享：
  - wait/resume state
  - async memory state
  - issue timing state
  - barrier state

### Phase 5：构建与测试边界收紧

目标：

- 让模块边界在构建层和测试层也可见

建议动作：

- CMake 分 target
- 测试分模块二进制
- 增加“禁止公共头包含 internal 头”的 lint/check

### 各阶段完成判定

为了避免后续把这些 phase 重新做成“只有方向、没有落点”的口号，建议每个阶段至少满足下面的完成判定再进入下一阶段。

- Phase 1 完成判定：
  - `src/gpu_model/*` 公共头不再直接包含 `execution/internal/*`
  - `CycleTimingConfig`、issue policy 稳定值类型已经挪到非 `internal` 头
  - 相关头可以独立自包含通过最小编译
- Phase 2 完成判定：
  - `ExecEngine` 只保留 façade 与装配职责，launch 主流程已拆成更窄的内部服务
  - `RuntimeSession` 不再同时管理 device/stream/launch/artifact 全部 compatibility 细节
  - `ModelRuntime` 不再重复持有 allocation table 或等价状态
- Phase 3 完成判定：
  - 外部工具调用、临时目录管理、artifact 解析、`ProgramObject` 组装已经分层
  - `ObjectReader` 收敛为 façade，而不是继续堆逻辑
  - 至少存在 extractor / parser / builder 级别的 focused tests
- Phase 4 完成判定：
  - `wave_state.h` 一类共享状态头不再依赖 trace/event factory
  - shared execution state 只表达调度/等待/资源/累计状态，不混 presentation 逻辑
  - functional / cycle / program-object 三条执行线都能复用同一套共享域对象
- Phase 5 完成判定：
  - CMake 已拆出最小可维护的模块 target
  - 测试已按模块或主题拆成多组二进制，而不是单一大测试入口
  - CI 或本地 gate 已能检查“公共头不得依赖 internal 头”这一类结构约束

## 8. 具体文件级优先落点

如果从当前代码直接开工，建议优先从下面这些文件开始：

- [exec_engine.h](../../src/gpu_model/runtime/exec_engine.h)
  - 去掉对具体 execution 头的直接暴露依赖
- [cycle_exec_engine.h](../../src/gpu_model/execution/cycle_exec_engine.h)
  - 抽出 `CycleTimingConfig`
- [gpu_arch_spec.h](../../src/gpu_model/arch/gpu_arch_spec.h)
  - 解除对 `execution/internal/issue_model.h` 的直接依赖
- [program_object_exec_engine.h](../../src/gpu_model/execution/program_object_exec_engine.h)
  - 用 request object 替代长参数列表
- [runtime_session.h](../../src/gpu_model/runtime/runtime_session.h)
  - 拆 compatibility 状态职责
- [model_runtime.h](../../src/gpu_model/runtime/model_runtime.h)
  - 清理 allocation ownership 重叠
- [encoded_program_object.cpp](../../src/program/encoded_program_object.cpp)
  - 拆 external-tool orchestration / parsing / building
- [wave_state.h](../../src/gpu_model/execution/internal/wave_state.h)
  - 去除 trace/event factory 依赖，纯化共享域模型
- [trace/sink.h](../../src/gpu_model/debug/trace/sink.h)
  - 收紧公共 API，只保留最小接口

## 9. 结论

当前项目的主要架构债，并不是“用了错误的总设计”，而是：

- 已经有了正确的正式设计方向
- 但代码层仍保留不少半收口、半迁移、半抽象的结构

最该优先处理的不是“把所有大文件拆小”，而是：

1. 公共接口与内部实现边界
2. runtime 主线的状态所有权
3. loader/program 的职责分层
4. execution 共享域模型纯化
5. 构建层的边界约束

如果这五件事先做对，后续无论是继续提升 cycle accuracy、补 runtime API、扩 ISA coverage，还是补 trace 观察面，成本都会显著下降。

如果不先做这些结构收口，那么后续功能开发会继续依赖“开发者记忆中的架构边界”，而不是依赖代码本身能表达和约束的边界。
