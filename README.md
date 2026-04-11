# gpu_model

`gpu_model` 是一个面向 AMD/GCN 风格 GPU kernel 的轻量级 C++ 功能模型与 naive cycle 模型。

A lightweight C++ functional and naive cycle model for AMD/GCN-style GPU kernels.

## Quick Start

```bash
# Build (recommended: Ninja preset)
cmake --preset dev-fast
cmake --build --preset dev-fast

# Run tests
./build-ninja/tests/gpu_model_tests

# Run an example
./examples/01-vecadd-basic/run.sh
```

### 环境要求

- CMake >= 3.22
- C++17 编译器 (GCC / Clang)
- Ninja (推荐) 或 Make
- hipcc (仅用于编译 HIP 源码，模型运行不依赖 GPU)

## 项目定位

当前项目重点不是硬件级精准复刻，而是给算子库优化、编译器 codegen 比较、硬件参数变更评估、HIP/AMDGPU kernel 行为验证提供一个可执行、可追踪、可扩展的分析平台。

## 核心能力

- **三种执行模式**: `st` (单线程功能执行)、`mt` (多线程功能执行)、`cycle` (naive cycle 模型)
- **Example 默认策略**: 非对比型 example 默认只跑 `mt`，对比/可视化 example 显式保留多模式
- **Example 编译缓存**: example 默认通过 `tools/hipcc_cache.sh` 复用 `hipcc` 输出，可用 `GPU_MODEL_USE_HIPCC_CACHE=0` 关闭
- **真实 HIP 程序支持**: 通过 `LD_PRELOAD` 拦截 HIP API，host 原生执行 + kernel 在模型中运行
- **Trace 可视化**: 输出 `timeline.perfetto.json`，支持 Chrome Trace Viewer 分析
- **GCN ISA 解码**: 支持 AMDGPU object / HIP fatbin / HIP `.out` 加载与执行

## 架构概览

五层架构：`runtime -> program -> instruction -> execution -> wave`

| 层级 | 核心组件 |
|------|----------|
| runtime | HipRuntime, ModelRuntime, ExecEngine |
| program | ProgramObject, ExecutableKernel, EncodedProgramObject |
| instruction | 指令解码与语义分发 |
| execution | FunctionalExecEngine, CycleExecEngine, WaveContext |
| arch | GpuArchSpec, 设备拓扑 |

详细架构见 [CLAUDE.md](CLAUDE.md) 和 [docs/runtime-layering.md](docs/runtime-layering.md)。

## Examples

详见 [examples/README.md](examples/README.md)，按难度编号组织：

| 编号 | 例子 | 验证重点 |
|------|------|----------|
| 01 | vecadd-basic | 基础路径接通 |
| 02 | fma-loop | 循环 + 浮点累积 |
| 03 | shared-reverse | shared memory + barrier |
| 04 | atomic-reduction | global atomic 归约 |
| 05 | softmax-reduction | 多阶段归约 |
| 06 | mma-gemm | MFMA 能力检测 |
| 07 | vecadd-cycle-splitting | 写法对比分析 |
| 08 | conditional-multibarrier | 条件分支 + 多次 barrier |
| 09 | dynamic-shared-sum | 动态 shared memory |
| 10 | block-reduce-sum | 多 block 归约 |
| 11 | perfetto-waitcnt-slots | Trace 调试可视化 |
| 12 | schedule-strategy-comparison | 调度策略对比 |
| 13 | algorithm-comparison | 算法对比 |

## Scripts

| 脚本 | 用途 |
|------|------|
| `./scripts/install_quality_tools.sh` | 安装质量检查工具 |
| `./scripts/run_quality_checks.sh` | 先生成 `compile_commands.json`，再并行运行重复率 / 圈复杂度 / `cppcheck` 静态检查，并生成 `results/quality/summary.txt` |
| `./scripts/run_push_gate_light.sh` | 快速回归 (推荐日常使用) |
| `./scripts/run_exec_checks.sh` | 执行检查 |
| `./scripts/run_real_hip_kernel_regression.sh` | HIP kernel 回归 |

详见 [scripts/README.md](scripts/README.md)。

## 文档导航

- **现行规范**: [README.md](README.md) → [docs/my_design.md](docs/my_design.md) → [docs/runtime-layering.md](docs/runtime-layering.md)
- **开发状态**: [docs/module-development-status.md](docs/module-development-status.md)
- **工程参考**: [src/spec/README.md](src/spec/README.md)
- **历史存档**: [docs/plans/](docs/plans/), [docs/superpowers/](docs/superpowers/)

## 当前限制

- GCN ISA 未完全覆盖，graphics/image family 仍为占位
- cycle 模型为 naive 近似，用于相对比较而非硬件精确仿真
- `trace.txt` / `timeline.perfetto.json` 中的 `cycle` 为模型时间，非物理时间

## 近期路线

1. 补齐 GCN ISA decode/disasm 覆盖
2. 扩展 instruction semantic handler
3. 收敛 descriptor/metadata/module-load 接口
4. 完善 runtime API 覆盖
5. 增强 cycle 建模能力
