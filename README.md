# gpu_model

`gpu_model` 是一个面向 AMD/GCN 风格 GPU kernel 的轻量级 C++ 功能模型与 naive cycle 模型。

当前项目重点不是硬件级精准复刻，而是给：

- 算子库优化
- 编译器 codegen 比较
- 硬件参数变更评估
- HIP/AMDGPU kernel 行为验证

提供一个可执行、可追踪、可扩展的分析平台。

## 当前能力

当前主线已经具备：

- `c500` 架构参数注册与运行时选择
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
- descriptor + metadata 驱动的 raw code object launch
- raw code object decode / disassemble / execute 主路径
  - `.text` 原始指令 bytes 提取
  - `text bytes -> raw instruction array -> decoded instruction array`
  - decode 阶段实例化 instruction object
  - raw GCN 指令直接执行
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
  canonical internal ISA
- `state/`
  wave / register / execution state
- `memory/`
  global / shared / private / constant memory
- `loader/`
  asm、AMDGPU object、HIP artifact、raw code object decode
- `decode/`
  GCN format bitfield / encoding def / formatter
- `exec/`
  issue model、semantic handlers、functional/cycle executor、parallel-wave executor scaffold
- `runtime/`
  host runtime、runtime hooks、HIP interposer
- `debug/`
  trace、timeline、debug info

## 当前执行形态

### 1. 直接 model 运行

适用于：

- 手写 canonical asm
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
  - `hipMemcpyAsync`
  - `hipMemset` / `hipMemsetD8` / `hipMemsetD32`
  - `hipGetDeviceCount` / `hipGetDevice` / `hipSetDevice`
  - `hipStreamCreate` / `hipStreamDestroy` / `hipStreamSynchronize`
  - `hipGetLastError` / `hipPeekAtLastError`
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
- descriptor + metadata 驱动的 raw binary launch 已打通
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

运行单组测试：

```bash
./build/tests/gpu_model_tests --gtest_filter=RuntimeHooksTest.LaunchesHipVecAddExecutableAndValidatesOutput
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

### raw code object decode / formatter

新增示例会把 AMDGPU/HIP 产物里的 raw instructions 打出来：

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
2. 继续扩展 raw semantic handler / instruction object 执行覆盖
3. 收敛 descriptor / metadata / module-load 的正式接口
4. 完善 runtime property 查询与 module API
5. 在现有 naive cycle 基础上继续增强 wait / issue / timeline 分析能力

## Marl

当前仓库已 vendor：

- `third_party/marl`

当前 `HostRuntime` 已支持 functional 执行模式切换：

- `SingleThreaded`
- `MarlParallel`

现阶段 `MarlParallel` 先作为 scheduler-backed 执行骨架接入，
保证：

- 构建链闭合
- 运行时模式可切换
- 结果与当前 single-thread functional 路径一致

下一阶段再把真正的 wave 并行、block 内 barrier/atomic/wait 协调逐步落到这个骨架上。
