本页面向入门开发者，聚焦“设备属性与配置查询路径”的端到端链路：从 HIP 兼容层 API（如 hipGetDeviceProperties/hipDeviceGetAttribute）到内部 HipRuntime/RuntimeSession/ModelRuntime 的分层职责，再到 ArchRegistry 和具体架构规格（c500）如何提供底层数值；同时说明运行时配置（执行模式、并行度、Trace/Log 开关）通过环境变量被解析并影响查询结果或执行流程。您当前位于“运行时接口与兼容层”章节的“设备属性与配置查询路径”页面 [You are currently here]。Sources: [runtime_session.h](src/gpu_model/runtime/runtime_session.h#L57-L66)

## 一、查询总览与分层职责（从第一性原理出发）
从第一性原理看，设备属性查询本质是“前端 API 语义 → 运行时门面 → 会话态路由 → 模型后端 → 架构规格”的数据下行与汇聚映射；配置查询则是“环境变量 → 运行时配置解析 → 执行/追踪策略”的控制上行影响。该过程遵循清晰的层次边界：hip_runtime_abi 只做 C ABI 兼容与校验，HipRuntime 做门面与会话代理，RuntimeSession 保持线程本地状态并协调内存与追踪，ModelRuntime 聚合设备与模块、向架构注册表取数，最终由 GpuArchSpec（c500）提供规格来源。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L309-L339)

Mermaid 架构关系图（阅读提示：自左向右表示调用方向，自上而下表示抽象到具体）
graph TD
  A[HIP 兼容层 API<br/>hipGetDevice/hipGetDeviceCount<br/>hipGetDeviceProperties/hipDeviceGetAttribute] --> B[HipRuntime 门面]
  B --> C[RuntimeSession 会话态<br/>设备/事件/Launch 配置/Trace]
  C --> D[ModelRuntime 后端<br/>GetDeviceProperties/GetDeviceAttribute]
  D --> E[ArchRegistry::Get(\"c500\")]
  E --> F[GpuArchSpec(c500)]
  D -.-> G[BuildRuntimeDeviceProperties<br/>（由 GpuArchSpec 转 RuntimeDeviceProperties）]
  A -.-> H[环境变量解析<br/>执行模式/Trace/Log 等]
  H --> C
Sources: [model_runtime.cpp](src/runtime/core/model_runtime.cpp#L144-L153)

## 二、设备基础查询：设备数量/选择/当前设备
- hipGetDeviceCount/hipGetDevice/hipSetDevice 在 ABI 层完成参数校验后，直接转发到 HipRuntime，再由 RuntimeSession 委派到 ModelRuntime。ModelRuntime 当前实现为单设备：GetDeviceCount=1；仅接受 device_id=0 的 SetDevice。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L309-L329)

- ModelRuntime 中对应实现：GetDeviceCount 固定返回 1；SetDevice 仅当 device_id==0 返回 true，GetDevice 返回 current_device_。这确保了上层 API 在无多设备情况下的稳定行为。Sources: [model_runtime.cpp](src/runtime/core/model_runtime.cpp#L116-L131)

- RuntimeSession 将上述查询透明转发到 ModelRuntime，保持会话一致性，不引入额外状态。Sources: [runtime_session.cpp](src/runtime/core/runtime_session.cpp#L64-L83)

表：设备基础查询 API 与内部实现映射
- HIP API → HipRuntime → RuntimeSession → ModelRuntime → 备注
- hipGetDeviceCount → GetDeviceCount → GetDeviceCount → GetDeviceCount=1 → 单设备
- hipGetDevice → GetDevice → GetDevice → GetDevice → 返回 current_device_
- hipSetDevice → SetDevice → SetDevice → SetDevice(仅0有效) → 非0返回 hipErrorInvalidDevice
Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L309-L329)

## 三、设备属性对象构建链：从架构规格到 RuntimeDeviceProperties
- 架构规格注册：ArchRegistry 将字符串 "c500" 绑定到 detail::MakeC500Spec()，属性提供者即 GpuArchSpec(c500)。Sources: [arch_registry.cpp](src/arch/arch_registry.cpp#L15-L19)

- c500 规格内容：包含 wave_size、处理器拓扑（dpc/ap/peu）、缓存/共享存储与时间参数等，是设备属性的“单一事实来源”。Sources: [c500_spec.cpp](src/arch/c500_spec.cpp#L7-L26)

- 属性装配：ModelRuntime::GetDeviceProperties 固定取 "c500" 规格并调用 BuildRuntimeDeviceProperties，将 GpuArchSpec 映射为 RuntimeDeviceProperties（如 warp_size、multi_processor_count、l2_cache_size 等）。Sources: [model_runtime.cpp](src/runtime/core/model_runtime.cpp#L144-L153)

- BuildRuntimeDeviceProperties 细节：从 spec.name/wave_size/total_ap_count/cache_model 等字段构建 name/warp_size/multi_processor_count/l2_cache_size 等最终用于对外的属性对象。Sources: [model_runtime.cpp](src/runtime/core/model_runtime.cpp#L16-L35)

## 四、hipGetDevicePropertiesR0600 字段填充映射
- ABI 层将 RuntimeDeviceProperties 拷贝映射到 hipDeviceProp_tR0600 的各字段，包括总显存、SM 数、时钟、L2、共享内存、寄存器、统一寻址与托管内存能力等，同时写入 gcnArchName="c500"。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L339-L378)

表：RuntimeDeviceProperties → hipDeviceProp_tR0600（部分字段）
- name → name（字符串拷贝）
- total_global_mem → totalGlobalMem
- warp_size → warpSize
- multi_processor_count → multiProcessorCount
- l2_cache_size → l2CacheSize
- unified_addressing → unifiedAddressing
- managed_memory → managedMemory
- max_threads_per_multiprocessor → maxThreadsPerMultiProcessor
Sources: [device_properties.h](src/gpu_model/runtime/device_properties.h#L42-L75)

## 五、hipDeviceGetAttribute 枚举查询路径与分派表
- 输入 hipDeviceAttribute_t 在 ABI 层被转换为内部 RuntimeDeviceAttribute，随后调用 HipRuntime→RuntimeSession→ModelRuntime 完成取值。未知/不支持的枚举返回 hipErrorInvalidValue。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L381-L489)

- ModelRuntime::GetDeviceAttribute 根据枚举从 RuntimeDeviceProperties 选择字段并返回，可覆盖线程/网格/共享内存/时钟/L2/托管内存等常见查询项。Sources: [model_runtime.cpp](src/runtime/core/model_runtime.cpp#L155-L221)

表：部分枚举到字段的映射
- hipDeviceAttributeWarpSize → warp_size
- hipDeviceAttributeMaxThreadsPerBlock → max_threads_per_block
- hipDeviceAttributeMaxGridDimX/Y/Z → max_grid_size[0/1/2]
- hipDeviceAttributeL2CacheSize → l2_cache_size
- hipDeviceAttributeUnifiedAddressing → unified_addressing
- hipDeviceAttributeManagedMemory → managed_memory
Sources: [model_runtime.cpp](src/runtime/core/model_runtime.cpp#L189-L221)

Mermaid 流程图：hipDeviceGetAttribute 路径
flowchart LR
  A[hipDeviceGetAttribute] --> B[参数校验/设备号检查]
  B --> C[映射为 RuntimeDeviceAttribute]
  C --> D[HipRuntime::GetDeviceAttribute]
  D --> E[RuntimeSession::GetDeviceAttribute]
  E --> F[ModelRuntime::GetDeviceAttribute]
  F --> G[RuntimeDeviceProperties 字段选择]
  G --> H[返回 int 值/错误码]
Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L381-L389)

## 六、配置查询路径：执行模式、功能模式与并行度
- 执行模式解析：ABI 层在 hipLaunchKernel 前解析 GPU_MODEL_EXECUTION_MODE（cycle/functional），影响后续执行引擎选择。未设置默认 functional。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L47-L58)

- 运行时集中配置管理：RuntimeConfigManager::ReloadFromEnv 解析 GPU_MODEL_EXECUTION_MODE、GPU_MODEL_FUNCTIONAL_MODE（st/mt/别名）、GPU_MODEL_FUNCTIONAL_WORKERS（线程数），并打印配置变更，支持动态设置。Sources: [runtime_config.cpp](src/runtime/config/runtime_config.cpp#L130-L160)

- 线程数默认规则：若功能模式为 MT 且未显式提供，默认取 CPU 核心数的 90%（向下取整且至少为 1）。Sources: [runtime_config.cpp](src/runtime/config/runtime_config.cpp#L39-L43)

- 轻量环境加载路径：runtime_env_config 提供同等环境变量解析的轻量入口（禁用 Trace/功能模式/工作线程数），与集中管理器保持一致的语义。Sources: [runtime_env_config.cpp](src/runtime/config/runtime_env_config.cpp#L14-L41)

Mermaid 概念关系图：配置影响执行
graph TD
  ENV[环境变量<br/>GPU_MODEL_EXECUTION_MODE<br/>GPU_MODEL_FUNCTIONAL_MODE<br/>GPU_MODEL_FUNCTIONAL_WORKERS] --> PARSE[配置解析<br/>RuntimeConfig / EnvConfig]
  PARSE --> MODE[ExecutionMode / FunctionalExecutionMode]
  MODE --> LAUNCH[hipLaunchKernel 路由参数]
  LAUNCH --> ENGINE[ExecEngine 执行策略/并行度]
Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L614-L625)

## 七、配置查询路径：Trace/Log 控制
- Trace 使能与输出目录：RuntimeSession::ResolveTraceArtifactRecorderFromEnv 读取 GPU_MODEL_DISABLE_TRACE 与 GPU_MODEL_TRACE_DIR。默认禁用；当 GPU_MODEL_DISABLE_TRACE=0 且提供 GPU_MODEL_TRACE_DIR 时创建 TraceArtifactRecorder 并在每次 kernel launch 后 Flush 并输出摘要。Sources: [runtime_session.cpp](src/runtime/core/runtime_session.cpp#L416-L444)

- ABI 层集成：hipLaunchKernel 调用 ResolveTraceArtifactRecorderFromEnv，并在执行后 Flush 时间线与附加 launch_summary。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L626-L636)

- 日志开关与级别：RuntimeConfig 支持 GPU_MODEL_DISABLE_LOGURU、GPU_MODEL_LOG_LEVEL、GPU_MODEL_LOG_FILE 等，按需控制模块化日志输出。Sources: [runtime_config.cpp](src/runtime/config/runtime_config.cpp#L176-L199)

表：常用环境变量与效果
- GPU_MODEL_EXECUTION_MODE=cycle|functional → 选择周期/功能执行
- GPU_MODEL_FUNCTIONAL_MODE=mt|st → 多/单线程功能执行
- GPU_MODEL_FUNCTIONAL_WORKERS=N → 多线程工作者数量
- GPU_MODEL_DISABLE_TRACE=0 → 显式启用 Trace
- GPU_MODEL_TRACE_DIR=/path → Trace 产物输出目录
- GPU_MODEL_DISABLE_LOGURU=0 → 启用 loguru 日志
Sources: [runtime_config.cpp](src/runtime/config/runtime_config.cpp#L130-L189)

## 八、典型交互与故障定位要点
- 设备 ID 校验失败：当 hipSetDevice(deviceId!=0) 或 hipGetDeviceProperties/hipDeviceGetAttribute 传入越界设备号，将返回 hipErrorInvalidDevice。确认当前仅支持 device_id=0。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L325-L338)

- 属性枚举不支持：未知的 hipDeviceAttribute_t 将返回 hipErrorInvalidValue，可对照内部分派表增补映射。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L481-L489)

- 属性值来源一致性：所有属性由 RuntimeDeviceProperties 提供，并通过 BuildRuntimeDeviceProperties→ArchRegistry(c500) 保持一致性。若需调整属性，请从 GpuArchSpec(c500) 与装配函数入手。Sources: [model_runtime.cpp](src/runtime/core/model_runtime.cpp#L16-L35)

Mermaid 步骤图：hipGetDeviceProperties 故障快速排查
flowchart TD
  S[开始] --> A[检查 deviceId 是否为 0]
  A -->|否| E[返回 hipErrorInvalidDevice]
  A -->|是| B[HipRuntime→RuntimeSession 路由]
  B --> C[ModelRuntime::GetDeviceProperties]
  C --> D[ArchRegistry::Get(\"c500\") / 缺失则抛错]
  D --> F[BuildRuntimeDeviceProperties 装配]
  F --> G[ABI 拷贝到 hipDeviceProp_tR0600]
  G --> H[返回 hipSuccess]
Sources: [model_runtime.cpp](src/runtime/core/model_runtime.cpp#L144-L153)

## 九、面向扩展的认知锚点（何处修改能改变“查询结果”）
- 改变“值从何来”：修改 c500 架构规格（GpuArchSpec）与 BuildRuntimeDeviceProperties 的映射逻辑，影响所有属性查询的一致来源。Sources: [c500_spec.cpp](src/arch/c500_spec.cpp#L7-L26)

- 改变“值如何暴露”：在 hip_runtime_abi 的属性转换与分派表中增补字段映射或新增枚举处理分支。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L339-L378)

- 改变“执行/追踪策略”：通过环境变量（或 RuntimeConfig API）调整执行模式、并行度、Trace/Log，使 launch 行为与可观测性发生变化。Sources: [runtime_config.cpp](src/runtime/config/runtime_config.cpp#L227-L249)

## 相关阅读与下一步
- 若需理解 ABI 兼容层的整体设计与对齐，请阅读：[HipRuntime C ABI 与 API 对齐](18-hipruntime-c-abi-yu-api-dui-qi)。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L593-L637)
- 若需掌握运行时门面与会话态的生命周期，请继续：[ModelRuntime 外观与会话生命周期](19-modelruntime-wai-guan-yu-hui-hua-sheng-ming-zhou-qi)。Sources: [runtime_session.h](src/gpu_model/runtime/runtime_session.h#L112-L124)
- 若需调整日志模块与使用约定，请转到：[日志主线与 loguru 使用约定](21-ri-zhi-zhu-xian-yu-loguru-shi-yong-yue-ding)。Sources: [runtime_config.cpp](src/runtime/config/runtime_config.cpp#L185-L199)
- 如需启用并解读 Trace 产物，请参阅：[Trace 格式、字段与开关策略](22-trace-ge-shi-zi-duan-yu-kai-guan-ce-lue)。Sources: [runtime_session.cpp](src/runtime/core/runtime_session.cpp#L416-L444)