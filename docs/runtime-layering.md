# Runtime Layering

## 目标

runtime 侧主线按三层来理解：

1. `HipRuntime`（HIP compatibility layer）
2. `ModelRuntime`（model-facing runtime facade）
3. `RuntimeEngine`（runtime core engine）

这样可以把：

- HIP ABI 兼容
- 真实 `.out` 命令行执行
- 仓库内测试/工具直接调用
- program / instruction / execution / memory / trace 核心逻辑

分开。

## Layer 1: HipRuntime

这一层的目标是尽量保持和 HIP runtime 一致。

当前包括：

- `src/runtime/hip_interposer.cpp`
- `src/runtime/hip_interposer_state.cpp`

职责：

- 提供 `hipMalloc / hipMemcpy / hipLaunchKernel / hipGetDeviceProperties / hipDeviceGetAttribute ...`
- 处理 `LD_PRELOAD` 下的真实 HIP 程序 ABI 拦截
- 处理 host function 到 kernel name 的映射
- 处理 fake device pointer 到 model address 的映射
- 把 HIP 参数格式转换成 `ModelRuntime` 可以消费的格式

约束：

- 这层不应该拥有独立的 kernel 执行逻辑
- 这层不应该重复实现 program load / launch / memory 语义

## Layer 2: ModelRuntime

这一层的目标是提供项目自己的 runtime facade，统一仓库内 API 入口。

当前包括：

- `include/gpu_model/runtime/model_runtime.h`
- `include/gpu_model/runtime/module_load.h`

职责：

- device 选择和 property 查询
- memory allocation / memcpy / memset
- program object / encoded program load
- 统一 `LoadModule` 请求分发
- `ExecutableKernel` launch
- trace / launch result / last load result
- 统一进入 `RuntimeEngine`

## Layer 3: RuntimeEngine

这一层承接执行主链，负责：

- `ProgramObject / EncodedProgramObject` 装载与 materialize
- 构建 `ExecutableKernel` 与 launch plan
- 驱动 `FunctionalExecEngine / CycleExecEngine / EncodedExecEngine`
- 组织 `WaveContext` 生命周期与运行时状态输出

当前实现文件名已切换到主线术语：

- `src/runtime/hip_runtime.cpp`
- `src/runtime/runtime_engine.cpp`

## 历史已删除名（仅用于阅读旧记录）

- `ModelRuntimeApi` -> `ModelRuntime`
- `RuntimeHooks` -> `HipRuntime`
- `HostRuntime` -> `RuntimeEngine`

### 仓库内测试 / 工具

优先使用：

- `ModelRuntime`
- `HipRuntime`
- `RuntimeEngine`

### 真实 HIP `.out`

调用路径：

- host program
- HIP runtime symbol
- interposer
- `HipInterposerState`
- `ModelRuntime`
- `RuntimeEngine`
- loader / decode / execution / memory

## 为什么不合并成一个入口

因为两类问题不同：

- 进程内 API 调用
- 外部程序 ABI 拦截

它们应该共享同一套核心实现，但不应该强行变成一个入口。

正确做法是：

- 保留两个入口
- 合并底层实现

## 当前状态

当前已经完成：

- `ModelRuntime` facade
- HIP interposer 到 `ModelRuntime` 的主路径复用
- 基础 device property 查询
- model-native 统一 `LoadModule` 入口和 `ExecutableKernel` launch 主路径

当前还缺：

- 更完整的 property / attribute 覆盖
- 更明确的 `ProgramObject / EncodedProgramObject` API 分层
- 历史遗留命名对应实现文件的进一步收口
- 文档和测试中的主线名称已统一采用 `ModelRuntime`
