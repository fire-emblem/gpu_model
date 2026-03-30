# gpu_model

`gpu_model` 是一个面向 AMD/GCN 风格 GPU kernel 的轻量级 C++ 功能模型与 naive cycle 模型。

当前项目重点不是硬件级精准复刻，而是给：

- 算子库优化
- 编译器 codegen 比较
- 硬件参数变更评估
- HIP/AMDGPU kernel 行为验证

提供一个可执行、可追踪、可扩展的分析平台。

## Architecture Spine

当前主线术语按下面 5 层统一：

- `runtime`: `HipRuntime -> ModelRuntime -> RuntimeEngine`
- `program`: `ProgramObject / ExecutableKernel / EncodedProgramObject`
- `instruction`: decode 后的 instruction object 与语义分发
- `execution`: `FunctionalExecEngine / CycleExecEngine / EncodedExecEngine / WaveContext`
- `arch`: 架构参数、设备拓扑与属性建模

迁移状态（2026-03）：

- 文档主线已切到上述术语
- 历史旧名 `ModelRuntimeApi / RuntimeHooks / HostRuntime` 已从主线术语移除
- 阅读旧记录时请按 `ModelRuntime / HipRuntime / RuntimeEngine` 对照理解

## 当前能力

当前主线已经具备：

- `c500` 架构参数注册与运行时选择
  - 默认拓扑为 `8 DPC x 13 AP/DPC x 4 PEU/AP`
- functional execution
- naive cycle execution
- functional execution mode switch
  - `SingleThreaded`
  - `MarlParallel`
- trace / debug / ASCII timeline / Google trace
- AMDGPU object / HIP fatbin / HIP `.out` 的加载入口
- HIP command-line interposer
  - host `main()` 原生执行
  - HIP runtime API 被 `LD_PRELOAD` 拦截
  - kernel launch 转到 model 执行
  - 返回 host 继续执行
- global / constant / kernarg / raw-data / managed pool 支持
- descriptor + metadata 驱动的 encoded program object launch
- encoded code object decode / disassemble / execute 主路径
  - `.text` 指令 bytes 提取
  - `text bytes -> encoded instruction array -> decoded instruction array`
  - decode 阶段实例化 instruction object
  - encoded GCN 指令直接执行
- 真实 HIP `.out` 功能主线已验证通过的代表性 kernel
  - `vecadd`
  - `fma_loop`
  - `bias_chain`
  - `shared_reverse`
  - `softmax_row`
  - `mfma_probe`
- 大规模 gtest / CTS / usage regression 已打通

## 当前分层

代码当前大致分为：

- `arch/`
  架构规格与注册
- `isa/`
  instruction 定义与 decode/语义映射（legacy 名称：canonical internal ISA）
- `state/`
  wave / register / execution state
- `memory/`
  global / shared / private / constant memory
- `loader/`
  asm、AMDGPU object、HIP artifact、encoded code object decode（legacy 文档中常写 raw code object）
- `decode/`
  GCN format bitfield / encoding def / formatter
- `exec/`
  issue model、semantic handlers、functional/cycle executor、parallel-wave executor scaffold
- `runtime/`
  HipRuntime、ModelRuntime、RuntimeEngine
- `debug/`
  trace、timeline、debug info

## Runtime 分层

runtime 侧主线按三层理解：

- HIP compatibility layer
  - `HipRuntime`（`hip*` ABI + interposer）
  - `LD_PRELOAD` interposer
- model-native runtime layer
  - `ModelRuntime`
- runtime core layer
  - `RuntimeEngine`

历史已删除名（仅用于阅读旧记录）：

- `ModelRuntimeApi` -> `ModelRuntime`
- `RuntimeHooks` -> `HipRuntime`
- `HostRuntime` -> `RuntimeEngine`

详细说明见：

- [docs/runtime-layering.md](/data/gpu_model/docs/runtime-layering.md)

## 当前执行形态

### 1. 直接 model 运行

适用于：

- 手写 instruction asm（legacy: canonical asm）
- 内部测试 kernel
- cycle trace 研究

### 2. 加载 HIP / AMDGPU artifact

适用于：

- `hipcc` 生成的 `.o`
- `hipcc` 生成的 `.out`
- fatbin 中的 device code object

### 3. 命令行运行真实 HIP `.out`

当前已经支持：

- host CPU 原生执行 `main()`
- `hipMalloc/hipMallocManaged/hipMemcpy/hipLaunchKernel/...` 被 interposer 拦截
- 常见同步 runtime API 已支持基础拦截
  - `hipMemset` / `hipMemsetD8` / `hipMemsetD32`
  - `hipGetDeviceCount` / `hipGetDevice` / `hipSetDevice`
  - `hipStreamCreate` / `hipStreamDestroy` / `hipStreamSynchronize`
  - `hipGetLastError` / `hipPeekAtLastError`
- `hipMemcpyAsync` 当前仅有 compatibility 拦截/同步退化路径，不属于第一阶段 async 能力支持
- kernel launch 进入 model
- host 继续执行并做结果校验

## 当前重要限制

当前仍然存在的主要限制：

- 还没有完成“全部 GCN ISA” 的 decode / disasm / exec 覆盖
- graphics / image / export / interp 等 family 仍主要是占位
- 部分 loader 路径仍保留了 `llvm-objdump` / tool-assisted 提取作为兼容入口
- runtime API 还没有补齐到“任意 HIP 程序”所需的完整子集
- cycle model 已经可用，但仍然是用于相对比较的 naive 近似模型，不是硬件精确模型

所以现阶段更准确的理解是：

- host-side `.out` command-line path 已稳定可用
- descriptor + metadata 驱动的 encoded binary launch 已打通
- compute-focused HIP kernel 覆盖已经较广
- 剩余工作重点在完整 ISA 覆盖、graphics family、runtime completeness 和更系统的 cycle 建模

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

## 测试

运行全部 gtest：

```bash
./build/tests/gpu_model_tests
```

默认 `CTS/Feature CTS` 只跑 `quick` 代表性子集。

运行完整矩阵：

```bash
GPU_MODEL_TEST_PROFILE=full ./build/tests/gpu_model_tests
```

运行单组测试：

```bash
# runtime/program 主线命名示例
./build/tests/gpu_model_tests --gtest_filter=HipRuntimeTest.LaunchesHipVecAddExecutableAndValidatesOutput
```

## 常用入口

### 内建 vecadd 示例

```bash
./build/vecadd_main
```

### cycle trace 示例

```bash
./build/fma_loop_cycle_trace_main --mode cycle --grid 2 --block 65 --n 8 --iterations 2
```

### 真实 HIP `.out` 命令行路径

先构建 interposer：

```bash
cmake --build build --target gpu_model_hip_interposer -j
```

然后运行 usage：

```bash
./usages/hip-command-line-interposer/run.sh
```

### encoded code object decode / formatter

新增示例会把 AMDGPU/HIP 产物里的 encoded instructions 打出来（legacy 文档中常写 raw instructions）：

```bash
./build/code_object_dump_main <path-to-amdgpu-object-or-hip-out> [kernel_name]
```

当前输出会同时展示：

- 原始 decode 的 `fmt`
- parser/工厂实例化后的 `op_type`
- 具体指令对象 `class`

### 顶层执行检查脚本

当前稳定的执行检查可直接跑：

```bash
./scripts/run_exec_checks.sh
```

它会串起来执行：

- `raw-code-object-decode` usage
- `hip-command-line-interposer` usage

## usages

可复现脚本见：

- [usages/README.md](/data/gpu_model/usages/README.md)

当前比较关键的 usage：

- [usages/hip-fatbin-launch/README.md](/data/gpu_model/usages/hip-fatbin-launch/README.md)
- [usages/hip-command-line-interposer/README.md](/data/gpu_model/usages/hip-command-line-interposer/README.md)

## 顶层开发状态

功能主线开发状态与推进顺序见：

- [docs/module-development-status.md](/data/gpu_model/docs/module-development-status.md)

## 工程参考资料

面向 decode / disasm / ABI / loader 的长期工程参考资料见：

- [src/spec/README.md](/data/gpu_model/src/spec/README.md)

## 近期路线

接下来主线应继续收敛到：

1. 基于 GCN ISA encoding 定义补齐剩余 decode/disasm 覆盖
2. 继续扩展 encoded semantic handler / instruction object 执行覆盖（legacy raw 路径需保持兼容）
3. 收敛 descriptor / metadata / module-load 的正式接口
4. 完善 runtime property 查询与 module API
5. 在现有 naive cycle 基础上继续增强 wait / issue / timeline 分析能力

## Marl

当前仓库已 vendor：

- `third_party/marl`

当前 `RuntimeEngine` 已支持 functional 执行模式切换：

- `SingleThreaded`
- `MarlParallel`

现阶段 `MarlParallel` 先作为 scheduler-backed 执行骨架接入，
保证：

- 构建链闭合
- 运行时模式可切换
- 结果与当前 single-thread functional 路径一致

下一阶段再把真正的 wave 并行、block 内 barrier/atomic/wait 协调逐步落到这个骨架上。
