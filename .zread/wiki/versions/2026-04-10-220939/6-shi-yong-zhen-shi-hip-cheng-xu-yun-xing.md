本页目标：教你将未改动的真实 HIP 可执行程序直接接入本项目的 GPU 模型运行，核心机制是通过 LD_PRELOAD 注入 libgpu_model_hip_runtime_abi.so 截获 HIP Runtime ABI 调用，并以功能/周期模式执行，同时产出统一的 Trace 与统计产物，便于后续分析与可视化。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L47-L58) [common.sh](examples/common.sh#L119-L128) [common.sh](examples/common.sh#L138-L151)

## 快速路径：一键回归脚本
若你已完成构建，可直接运行回归脚本验证真实 HIP 工作流覆盖：scripts/run_real_hip_kernel_regression.sh 会构建必要目标、跑共享压力环、原子计数聚焦测试、以及示例 04 Atomic Reduction，三类均通过时输出 [real-hip] ok 概要。Sources: [run_real_hip_kernel_regression.sh](scripts/run_real_hip_kernel_regression.sh#L11-L19) [run_real_hip_kernel_regression.sh](scripts/run_real_hip_kernel_regression.sh#L21-L29) [run_real_hip_kernel_regression.sh](scripts/run_real_hip_kernel_regression.sh#L34-L39)

脚本会将结果写入 build/real-hip-kernel-regression 目录，并通过 grep 校验关键成功标记（共享环 ok、GTest PASSED、示例结果比对通过）。这保证了 HIP 可执行程序路径、ABI 注入与 Trace 产物链条均工作正常。Sources: [run_real_hip_kernel_regression.sh](scripts/run_real_hip_kernel_regression.sh#L8-L9) [run_real_hip_kernel_regression.sh](scripts/run_real_hip_kernel_regression.sh#L30-L33)

## 手动运行你的 HIP 程序（推荐流程）
- 第一步：编译你的 HIP 源码为可执行文件。可使用内置 hipcc 缓存工具以加速重复构建：gpu_model_compile_hip_source 会优先调用 tools/hipcc_cache.sh（可通过 GPU_MODEL_USE_HIPCC_CACHE=0 关闭）。Sources: [common.sh](examples/common.sh#L30-L38) [hipcc_cache.sh](tools/hipcc_cache.sh#L17-L25)

- 第二步：准备 ABI 注入库与运行模式。构建目标 gpu_model_hip_runtime_abi 并找到生成的 libgpu_model_hip_runtime_abi.so；示例脚本固定从构建目录加载该动态库。Sources: [common.sh](examples/common.sh#L40-L49) [run.sh](examples/04-atomic-reduction/run.sh#L13-L16)

- 第三步：以 LD_PRELOAD 注入并运行。推荐复用 gpu_model_run_interposed_mode，它会设置核心环境变量、配置 LD_LIBRARY_PATH 指向 ROCm 系统库目录、并将标准输出和所有 Trace 产物收集到结果目录。Sources: [common.sh](examples/common.sh#L84-L92) [common.sh](examples/common.sh#L114-L133) [common.sh](examples/common.sh#L135-L136)

- 第四步：验证结果与产物。脚本会比对输出关键字并断言 Trace 工件存在：trace.txt、trace.jsonl、timeline.perfetto.json、launch_summary.txt。你也可直接检查这些文件以进行后续分析与可视化。Sources: [common.sh](examples/common.sh#L154-L159) [common.sh](examples/common.sh#L138-L151)

一个最小可运行样例可参考示例 04 的流程：编译 atomic_reduction.hip → 注入运行（模式 mt）→ 断言输出 “atomic_reduction value=257 expected=257”。Sources: [run.sh](examples/04-atomic-reduction/run.sh#L16-L25)

## 端到端执行流（步骤图）
下图展示从“编译 HIP → 运行时 ABI 截获 → 构建加载计划 → 内核执行 → 产出 Trace”的关键路径，便于将你的可执行程序无缝接入模型。

```mermaid
flowchart TD
  A[hipcc 编译 HIP 程序] --> B[通过 LD_PRELOAD 注入 libgpu_model_hip_runtime_abi.so]
  B --> C[ABI 截获: __hipRegisterFunction / Push/Pop 配置]
  C --> D[解析执行模式 GPU_MODEL_EXECUTION_MODE=functional/cycle]
  D --> E[构建可执行加载计划 BuildExecutableLoadPlan]
  E --> F[LaunchExecutableKernel 执行(功能/周期)]
  F --> G[记录 Trace 与 timeline.perfetto.json]
  G --> H[launch_summary.txt 汇总(OK/周期统计)]
```
Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L145-L159) [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L160-L178) [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L47-L58) [hip_runtime.cpp](src/runtime/hip_runtime.cpp#L143-L160) [common.sh](examples/common.sh#L138-L151)

## 运行模式与环境变量
为了适配不同的调试/性能目标，运行时支持多种模式与相关参数。下表汇总了主要环境变量及其行为，均可通过 gpu_model_run_interposed_mode 自动设置，或手动导出后自行运行你的可执行程序。

- 模式/变量总览
  - GPU_MODEL_EXECUTION_MODE: functional 或 cycle，决定整体执行引擎。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L47-L58)
  - GPU_MODEL_FUNCTIONAL_MODE: st 或 mt；functional 下的单/多线程调度策略。Sources: [common.sh](examples/common.sh#L93-L103) [common.sh](examples/common.sh#L121-L122)
  - GPU_MODEL_FUNCTIONAL_WORKERS: 多线程模式下的工作线程数，默认取 GPU_MODEL_MT_WORKERS 或 4。Sources: [common.sh](examples/common.sh#L100-L104) [common.sh](examples/common.sh#L128-L130)
  - GPU_MODEL_CYCLE_FUNCTIONAL_MODE: cycle 模式下的功能执行前置策略（如 st），影响内核语义执行的并发形态。Sources: [common.sh](examples/common.sh#L104-L107)
  - GPU_MODEL_TRACE_DIR/GPU_MODEL_DISABLE_TRACE: Trace 输出位置与开关，默认开启并将产物写到指定目录。Sources: [common.sh](examples/common.sh#L121-L125) [common.sh](examples/common.sh#L123-L124)
  - GPU_MODEL_LOG_MODULES/GPU_MODEL_LOG_LEVEL: 模块化日志与等级控制，示例默认聚焦 hip_runtime_abi/info。Sources: [common.sh](examples/common.sh#L124-L126)
  - LD_PRELOAD: 注入 libgpu_model_hip_runtime_abi.so（自动串联 libasan 以兼容 ASan 运行）。Sources: [common.sh](examples/common.sh#L62-L71) [common.sh](examples/common.sh#L126-L127)
  - LD_LIBRARY_PATH: 自动补全 ROCm 库搜索路径（/opt/rocm/lib[64]）。Sources: [common.sh](examples/common.sh#L51-L60) [common.sh](examples/common.sh#L131-L133)

- 手动注入最小示例（等价于脚本）
  - 导出变量并执行你的 HIP 程序：
    - export GPU_MODEL_EXECUTION_MODE=functional; export GPU_MODEL_FUNCTIONAL_MODE=mt
    - export GPU_MODEL_TRACE_DIR=/path/to/results; export GPU_MODEL_DISABLE_TRACE=0
    - export LD_PRELOAD=/path/to/libgpu_model_hip_runtime_abi.so
    - ./your_hip_program
  - 推荐使用 gpu_model_run_interposed_mode 自动处理 ASan/ROCm/Trace 目录等细节。Sources: [common.sh](examples/common.sh#L119-L136) [run.sh](examples/04-atomic-reduction/run.sh#L21-L25)

## 产物与验证要点
每次运行会生成以下工件，可用于功能正确性与性能/时间线分析：
- trace.txt/trace.jsonl：结构化事件与摘要快照，包含 “GPU_MODEL TRACE”、“[EVENTS]”、“[SUMMARY]” 等标记。Sources: [common.sh](examples/common.sh#L145-L151) [common.sh](examples/common.sh#L138-L144)
- timeline.perfetto.json：可导入 Perfetto 的时间线数据，配合页面[可视化 Trace（Perfetto）](5-ke-shi-hua-trace-perfetto) 深入分析。Sources: [common.sh](examples/common.sh#L150-L151)
- launch_summary.txt：按内核追加记录 launch_index、kernel、execution_mode、functional_mode、ok、总周期等统计。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L69-L96)

对于示例 04 Atomic Reduction，正确性会在标准输出出现 “atomic_reduction value=257 expected=257”，并由脚本自动断言。你可以以此作为集成校验基线。Sources: [run.sh](examples/04-atomic-reduction/run.sh#L23-L25) [run_real_hip_kernel_regression.sh](scripts/run_real_hip_kernel_regression.sh#L27-L33)

## 原理速览：ABI 截获与内核绑定
模型通过拦截编译产物中的 HIP 注册与调用流程工作：在进程启动后，__hipRegisterFunction 记录 host 函数与设备内核名的映射；__hipPushCallConfiguration/__hipPopCallConfiguration 捕获每次 Launch 的网格/块/共享内存配置。随后由 HipRuntime 解析可执行文件并生成加载计划，调用 LaunchExecutableKernel 在功能/周期模式下执行。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L145-L159) [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L160-L178) [hip_runtime.cpp](src/runtime/hip_runtime.cpp#L156-L169)

执行模式由环境变量 GPU_MODEL_EXECUTION_MODE 决定（functional 或 cycle）；运行结束后，记录器会将汇总统计以追加方式写入 launch_summary.txt，便于多内核/多次发射的场景分析。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L47-L58) [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L69-L96)

## 故障排查（速查表）
- 进程启动报 “missing tool: hipcc” 或编译失败：请确认主机存在 HIP Toolchain；脚本在缺少 hipcc/sha256sum 时会直接失败。Sources: [common.sh](examples/common.sh#L22-L28) [hipcc_cache.sh](tools/hipcc_cache.sh#L7-L15)
- 没有生成 Trace 产物或断言失败：检查 GPU_MODEL_DISABLE_TRACE 是否为 0；确认结果目录下存在 trace.txt/trace.jsonl/timeline.perfetto.json/launch_summary.txt，示例脚本会在缺失时失败。Sources: [common.sh](examples/common.sh#L121-L125) [common.sh](examples/common.sh#L138-L151) [common.sh](examples/common.sh#L154-L159)
- 动态库加载异常（找不到 ROCm 依赖）：确保 LD_LIBRARY_PATH 包含 /opt/rocm/lib 或 /opt/rocm/lib64；工具会自动探测并拼接。Sources: [common.sh](examples/common.sh#L51-L60) [common.sh](examples/common.sh#L131-L133)
- 开启 ASan 的程序注入失败：LD_PRELOAD 会自动在 ABI 库前拼接 libasan 路径以兼容，你也可手动处理。Sources: [common.sh](examples/common.sh#L62-L71)
- 多线程功能模式下吞吐不达预期：调整 GPU_MODEL_FUNCTIONAL_WORKERS（默认 4），并确认 functional_mode=mt 生效。Sources: [common.sh](examples/common.sh#L100-L104) [common.sh](examples/common.sh#L121-L130)

## 建议的下一步
- 深入理解时间线与事件：参见[可视化 Trace（Perfetto）](5-ke-shi-hua-trace-perfetto)，将 timeline.perfetto.json 导入 Perfetto 分析。Sources: [common.sh](examples/common.sh#L150-L151)
- 若需了解 ABI 对齐与可拦截范围：参见[HipRuntime C ABI 与 API 对齐](18-hipruntime-c-abi-yu-api-dui-qi)，对应的拦截入口见本页所引 hip_runtime_abi.cpp。Sources: [hip_runtime_abi.cpp](src/runtime/hip_runtime_abi.cpp#L145-L159)
- 若你首次使用示例与环境校验：补充阅读[运行示例与验证](4-yun-xing-shi-li-yu-yan-zheng)，快速形成端到端心智模型。Sources: [run_real_hip_kernel_regression.sh](scripts/run_real_hip_kernel_regression.sh#L21-L29)