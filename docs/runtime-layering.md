# Runtime Layering

## 目标

runtime 侧主线按两层来理解：

1. `HipRuntime`（与 AMD HIP runtime 对齐的兼容层，C ABI 入口）
2. `ModelRuntime`（项目核心实现）

`ExecEngine` 是 `ModelRuntime` 内部执行主链的一部分，不单独作为对外 runtime 层。

## Layer 1: HipRuntime

这一层的目标是尽量保持和 AMD HIP runtime 一致。

当前包括：

- `src/runtime/hip_interposer.cpp`
- `src/runtime/hip_runtime.cpp`

职责：

- 提供 `hipMalloc / hipMemcpy / hipLaunchKernel / hipGetDeviceProperties / hipDeviceGetAttribute ...`
- 提供真实 HIP 程序进入项目 runtime 的 C ABI 入口
- 处理 host function 到 kernel name 的映射
- 处理 fake device pointer 到 model address 的映射
- 把 HIP 参数格式转换成 `ModelRuntime` 可以消费的格式

约束：

- 这层不应该拥有独立的 kernel 执行逻辑
- 这层不应该重复实现 program load / launch / memory 语义
- `src/runtime/hip_interposer.cpp` 只是 `HipRuntime` 的 C ABI 入口实现载体，不再作为独立“interposer 模块”理解

## Layer 2: ModelRuntime

这一层的目标是提供项目核心实现与统一内部 runtime 主线。

当前包括：

- `include/gpu_model/runtime/model_runtime.h`
- `include/gpu_model/runtime/module_load.h`
- `src/runtime/runtime_engine.cpp`（目标命名：`ExecEngine`）
- `src/runtime/core/*`

职责：

- device 选择和 property 查询
- memory allocation / memcpy / memset
- program object / encoded program load
- 统一 `LoadModule` 请求分发
- `ExecutableKernel` launch
- trace / launch result / last load result
- 统一进入 `ExecEngine`

### ExecEngine

`ExecEngine` 承接 `ModelRuntime` 的执行主链，负责：
- `ProgramObject / EncodedProgramObject` 装载与 materialize
- 构建 `ExecutableKernel` 与 launch plan
- 驱动 `FunctionalExecEngine / CycleExecEngine / EncodedExecEngine`
- 组织 `WaveContext` 生命周期与运行时状态输出

## 历史已删除名（仅用于阅读旧记录）

- `ModelRuntimeApi` -> `ModelRuntime`
- `RuntimeHooks` -> `HipRuntime`
- `HostRuntime` -> `ExecEngine`
- `HipInterposerState` -> 已删除，其兼容职责已并入 `HipRuntime`

### 仓库内测试 / 工具

优先使用：

- `ModelRuntime`
- `HipRuntime`
- `ExecEngine`

### 真实 HIP `.out`

调用路径：

- host program
- HIP runtime symbol
- `HipRuntime` C ABI entry
- `ModelRuntime`
- `ExecEngine`
- loader / decode / execution / memory

## 当前状态

当前已经完成：

- `ModelRuntime` facade
- `HipRuntime` 到 `ModelRuntime` 的主路径复用
- 基础 device property 查询
- model-native 统一 `LoadModule` 入口和 `ExecutableKernel` launch 主路径
- `HipInterposerState` 已删除，兼容职责已并入 `HipRuntime`

当前还缺：

- 更完整的 property / attribute 覆盖
- 更明确的 `ProgramObject / EncodedProgramObject` API 分层
- 历史遗留文件名与构建目标的进一步收口
- 文档和测试中的主线名称已统一采用 `ModelRuntime`
