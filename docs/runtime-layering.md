# Runtime Layering

## 目标

runtime 侧现在按两层来理解：

1. HIP compatibility layer
2. model-native runtime layer

这样可以把：

- HIP ABI 兼容
- 真实 `.out` 命令行执行
- 仓库内测试/工具直接调用
- loader / exec / memory / trace 核心逻辑

分开。

## Layer 1: HIP Compatibility Layer

这一层的目标是尽量保持和 HIP runtime 一致。

当前包括：

- `src/runtime/hip_interposer.cpp`
- `src/runtime/hip_interposer_state.cpp`

职责：

- 提供 `hipMalloc / hipMemcpy / hipLaunchKernel / hipGetDeviceProperties / hipDeviceGetAttribute ...`
- 处理 `LD_PRELOAD` 下的真实 HIP 程序 ABI 拦截
- 处理 host function 到 kernel name 的映射
- 处理 fake device pointer 到 model address 的映射
- 把 HIP 参数格式转换成 model-native runtime 可以消费的格式

约束：

- 这层不应该拥有独立的 kernel 执行逻辑
- 这层不应该重复实现 module/load/launch/memory 语义

## Layer 2: Model-Native Runtime Layer

这一层的目标是提供项目自己的 runtime API。

当前包括：

- `include/gpu_model/runtime/model_runtime_api.h`
- `include/gpu_model/runtime/runtime_hooks.h`
- `src/runtime/runtime_hooks.cpp`
- `src/runtime/host_runtime.cpp`

职责：

- device 选择和 property 查询
- memory allocation / memcpy / memset
- module / image / code object load
- kernel launch
- trace / launch result / last load result
- 统一进入 loader / decode / exec / memory 主链

当前关系：

- `ModelRuntimeApi` 是更明确的 model-native facade
- `RuntimeHooks` 仍保留，作为现有测试和工具的稳定 C++ 接口
- `ModelRuntimeApi` 当前内部委托到 `RuntimeHooks`

## 推荐调用路径

### 仓库内测试 / 工具

优先使用：

- `ModelRuntimeApi`

兼容保留：

- `RuntimeHooks`

### 真实 HIP `.out`

调用路径：

- host program
- HIP runtime symbol
- interposer
- `HipInterposerState`
- `ModelRuntimeApi`
- `RuntimeHooks`
- `HostRuntime`
- loader / decode / exec / memory

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

- `ModelRuntimeApi` facade
- HIP interposer 到 model runtime 的主路径复用
- 基础 device property 查询

当前还缺：

- 更完整的 property / attribute 覆盖
- 更明确的 module API 分层
- `RuntimeHooks` 到 `ModelRuntimeApi` 的进一步职责收缩
- 文档和测试中逐步统一采用 `ModelRuntimeApi` 作为 model-native 层名称
