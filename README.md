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
- trace / debug / ASCII timeline / Google trace
- AMDGPU object / HIP fatbin / HIP `.out` 的加载入口
- HIP command-line interposer
  - host `main()` 原生执行
  - HIP runtime API 被 `LD_PRELOAD` 拦截
  - kernel launch 转到 model 执行
  - 返回 host 继续执行
- global / constant / kernarg / raw-data / managed pool 支持
- raw code object decode scaffolding
  - `.text` 原始指令 words 提取
  - GCN format classify
  - 最小 encoding def
  - 最小 formatter

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
  issue model、semantic handlers、functional/cycle executor
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
- kernel launch 进入 model
- host 继续执行并做结果校验

## 当前重要限制

项目还没有完成最终目标里的这一步：

- 不再依赖 `llvm-objdump` 文本
- 改为完全基于 `.text` 二进制 bytes
- instruction decode
- raw instruction execute

当前 device 路径仍然是：

- raw code object extraction
- 部分 project-side format / encoding scaffolding
- 现有执行主路径仍依赖当前 device ingestion 兼容层

所以现阶段应理解为：

- host-side `.out` command-line path 已打通
- raw binary decode 框架已启动
- full raw execute 仍在继续建设

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

## usages

可复现脚本见：

- [usages/README.md](/data/gpu_model/usages/README.md)

当前比较关键的 usage：

- [usages/hip-fatbin-launch/README.md](/data/gpu_model/usages/hip-fatbin-launch/README.md)
- [usages/hip-command-line-interposer/README.md](/data/gpu_model/usages/hip-command-line-interposer/README.md)

## 近期路线

接下来主线应继续收敛到：

1. 基于 GCN ISA encoding 定义的完整 decode
2. project-side disassembler
3. raw instruction IR
4. raw semantic handler / raw execute
5. 逐步替换当前 device 文本兼容路径
