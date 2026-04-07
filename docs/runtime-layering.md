# Runtime Layering

## 目标

runtime 侧主线按两层来理解：

1. `HipRuntime`（与 AMD HIP runtime 对齐的兼容层，C ABI 入口）
2. `ModelRuntime`（项目核心实现）

`ExecEngine` 是 `ModelRuntime` 内部执行主链的一部分，不单独作为对外 runtime 层。

## Layer 1: HipRuntime

这一层的目标是尽量保持和 AMD HIP runtime 一致。

当前包括：

- `src/runtime/hip_runtime_abi.cpp`
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
- `src/runtime/hip_runtime_abi.cpp` 只是 `HipRuntime` 的 C ABI / LD_PRELOAD 入口实现载体
- `gpu_model_hip_runtime_abi` target / `libgpu_model_hip_runtime_abi.so` 是兼容入口动态库，不代表独立模块层

## Layer 2: ModelRuntime

这一层的目标是提供项目核心实现与统一内部 runtime 主线。

当前包括：

- `src/gpu_model/runtime/model_runtime.h`
- `src/gpu_model/runtime/module_load.h`
- `src/runtime/exec_engine.cpp`
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
- `ProgramObject` 装载与 materialize
- 构建 `ExecutableKernel` 与 launch plan
- 驱动 `FunctionalExecEngine / CycleExecEngine / EncodedExecEngine`
- 组织 `WaveContext` 生命周期与运行时状态输出

## 关键交互关系

### 1. 无 kernel launch 的 runtime / memory 主线

- `HipRuntime`
  - 接收 `hipMalloc / hipFree / hipMemcpy* / hipMemset*` 等 API
- `ModelRuntime`
  - 统一参数校验、错误码、设备/上下文边界
- `ExecEngine`
  - 统一进入 memory system / memory pool / pointer mapping 主链
- `Memory pools`
  - 承担 `Global / Constant / Shared / Private / Kernarg / Code / RawData` 等存储语义

这条路径必须可独立于 kernel launch 被测试覆盖。

第一阶段实现边界：

- 只关注：
  - `hipMalloc / hipFree`
  - `hipMemcpyHtoD / DtoH / DtoD`
  - `hipMemset`
- 以单卡、单 context、单 stream、同步语义为准
- `async memcpy / stream / event` 只保留最小兼容，不进入第一优先级主线

推荐分层：

- `HipRuntime`
  - HIP API 入口与参数适配
- `ModelRuntime`
  - 统一 runtime 语义与对外 facade
- `ExecEngine`
  - memory op 分发与 pointer mapping
- `DeviceMemoryManager`
  - 统一管理 compatibility 虚拟地址窗口、allocation table 和各 memory pool
- `MemoryPoolManager`
  - pool 注册、分配、释放、地址解析
- `MappedStorage`
  - `mmap` backed 底层存储抽象

补充约束：

- compatibility pointer 的属性判断，不应依赖宿主随机虚拟地址高位。
- 项目应自行规定若干 compatibility virtual address windows。
- 判断顺序应为：
  1. 指针是否落在某个已知 window
  2. 若是，则得到 pool kind
  3. 再查 allocation table 判断是否合法分配及其边界

- 对 `Global` / `Managed`，推荐先做：
  - 大片虚拟地址空间预留
  - 按需物理页提交
  - 后续 `malloc/free` 只做窗口内虚拟分配与回收

### 2. kernel launch 主线

- `HipRuntime`
  - 负责 HIP C ABI 与 launch 参数适配
- `ModelRuntime`
  - 形成统一 launch request
- `ExecEngine`
  - 选择 `ProgramObject`
  - 完成 materialize 与 executable launch
- `FunctionalExecEngine / CycleExecEngine / EncodedExecEngine`
  - 执行 wave/block/device 级语义

### 3. trace / log 主线

- 日志主线应统一收口到 `loguru`
- trace 主线应只消费执行结果，不反向参与业务决策
- text/json trace 必须可全局关闭，并且关闭后不影响执行事实

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
- 更完整的 runtime memory API 子集与不同 memcpy 行为矩阵
- memory pool 与 `mmap` backed residency 的第一阶段框架落地
- 更明确的 `ProgramObject` API 分层
- 历史遗留文件名与构建目标的进一步收口
- 与正式设计文档持续同步的 runtime 边界维护

## 正式解释规则

1. runtime 设计与职责边界，以本文件和 [my_design.md](/data/gpu_model/docs/my_design.md) 为准。
2. 历史计划中若出现旧名、旧层级或中间过渡包装，不应据此反向解释当前代码。
3. 真实 HIP `.out` 的主路径固定理解为：
   - host program
   - HIP runtime symbol
   - `HipRuntime` C ABI entry
   - `ModelRuntime`
   - `ExecEngine`
   - loader / program / execution / memory / trace
4. `ExecEngine` 是 `ModelRuntime` 内部执行主链，不是与 `HipRuntime` / `ModelRuntime` 并列的第三层对外 runtime。

## 当前仍残留的历史命名

当前已完成的语义清理：

- `src/runtime/hip_runtime_abi.cpp` 取代了历史 `hip_interposer.cpp` 主入口文件名
- `gpu_model_hip_runtime_abi` / `libgpu_model_hip_runtime_abi.so` 取代了历史 target / 库名
- `tests/runtime/hip_runtime_abi_test.cpp` 取代了历史 `hip_interposer_state_test.cpp`
- `RuntimeSession` 内部的 `interposer_*` 分配/事件/参数布局命名已改为 compatibility / runtime-abi 语义
- `src/gpu_model/*` 已取代历史 `include/gpu_model/*` 头文件根

后续清理顺序：

1. 先收口语义和文档
2. 再清理测试名、日志名、target 名
3. 最后再看是否需要物理文件名重命名
