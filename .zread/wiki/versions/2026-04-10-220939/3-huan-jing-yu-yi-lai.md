本页定位为“最小可用环境”的权威说明：告诉你在 Linux 上需要哪些构建与运行时依赖、如何快速配置它们、有哪些关键环境变量，以及如何用最小步骤验证环境是否正确。你当前所在位置：快速上手 → 环境与依赖 [You are currently here]。Sources: [README.md](README.md#L21-L27)

## TL;DR 环境清单（初学者优先）
- 必需（构建期）：
  - CMake ≥ 3.22
  - C++17 编译器（GCC/Clang）
  - Ninja（推荐；默认预设使用）
- 可选（示例编译期）：
  - hipcc（仅用于编译 HIP 示例或真实 HIP 程序；模型本身不依赖 GPU）
- 结论：无须 GPU 硬件即可运行模型与测试。Sources: [README.md](README.md#L21-L27) [CMakePresets.json](CMakePresets.json#L3-L16)

## 环境分层总览（先解释后示意）
从第一性原理拆解环境：构建工具链负责生成二进制与测试；HIP 工具链仅在编译示例或真实 HIP 程序时参与；运行期通过环境变量控制执行模式与追踪，必要时用 LD_PRELOAD 拦截 HIP API 实现“host 原生 + kernel 在模型中运行”的路径，输出可视化 Trace。Sources: [README.md](README.md#L37-L40) [examples/common.sh](examples/common.sh#L119-L136)

```mermaid
graph TD
  Dev[Developer (Linux)] --> Build[CMake 3.22+<br/>Ninja + GCC/Clang]
  Build --> Artifacts[tests/gpu_model_tests<br/>libgpu_model_hip_runtime_abi.so]
  Examples[HIP Examples] -->|hipcc 或 hipcc_cache.sh| Fatbin[HIP可执行/对象]
  Fatbin --> Run[运行期]
  Run -->|LD_PRELOAD=.../hip_runtime_abi.so| Interpose[HIP API 拦截]
  Interpose --> ModelExec[模型执行: st/mt/cycle]
  ModelExec --> Trace[trace.txt / trace.jsonl / timeline.perfetto.json]
  classDef tool fill:#eef,stroke:#88f;
  class Build,Examples,Run,Interpose,ModelExec,Trace tool;
```
Sources: [README.md](README.md#L37-L40) [CMakePresets.json](CMakePresets.json#L10-L16) [examples/README.md](examples/README.md#L7-L13)

## 目录要点（与环境相关的最小子集）
- src/：核心实现（构建目标由 CMake 驱动）  
- examples/：面向用户的 HIP 示例，默认启用 hipcc_cache 缓存编译产物，可通过环境变量关闭  
- scripts/：构建与回归相关脚本（如轻量/全量门禁、最小执行检查）  
- tools/：hipcc_cache.sh 等辅助工具  
以上子集与环境关系最为紧密，其余 third_party/ 目录存在但非入门必需。Sources: [examples/README.md](examples/README.md#L7-L13) [scripts/README.md](scripts/README.md#L5-L13) [README.md](README.md#L55-L83)

项目结构（节选，便于定位环境相关目录）：
.
├── CMakePresets.json
├── README.md
├── examples/
├── scripts/
└── tools/
Sources: [CMakePresets.json](CMakePresets.json#L10-L16) [examples/README.md](examples/README.md#L1-L13) [scripts/README.md](scripts/README.md#L1-L13)

## 安装与配置（Linux，最小步骤）
- 安装构建工具链：确保 CMake ≥ 3.22、GCC/Clang 和 Ninja 可用；项目默认预设使用 Ninja 生成器（dev-fast）。首次配置与构建可用：cmake --preset dev-fast && cmake --build --preset dev-fast。Sources: [CMakePresets.json](CMakePresets.json#L10-L16) [README.md](README.md#L9-L19)
- 准备 HIP 工具链：若要编译并运行 HIP 示例或真实 HIP 程序，需提供 hipcc；模型运行不依赖 GPU 硬件。Sources: [README.md](README.md#L21-L27)
- 基础验证：构建测试二进制与 ABI 组件，运行最小例子并校验输出与 Trace 产物（脚本会自动调用示例 01）。可执行：scripts/run_exec_checks.sh。Sources: [scripts/run_exec_checks.sh](scripts/run_exec_checks.sh#L12-L24)

## 运行期环境变量速查（可复制到终端逐项调试）
- 构建/路径
  - GPU_MODEL_BUILD_DIR：覆盖默认构建目录检测（默认优先使用 build-ninja）。Sources: [examples/common.sh](examples/common.sh#L3-L14)
  - GPU_MODEL_TRACE_DIR：Trace 输出目录（示例运行脚本会按模式组织目录）。Sources: [examples/common.sh](examples/common.sh#L119-L127)
- 示例编译与缓存
  - GPU_MODEL_USE_HIPCC_CACHE：是否启用 hipcc 缓存（默认 1）；设 0 关闭缓存直调用 hipcc。Sources: [examples/common.sh](examples/common.sh#L30-L38)
  - GPU_MODEL_HIPCC_CACHE_DIR：hipcc 缓存目录（默认 /tmp/gpu_model_hipcc_cache）。Sources: [tools/hipcc_cache.sh](tools/hipcc_cache.sh#L4-L6)
- 执行模式与并行度
  - GPU_MODEL_EXECUTION_MODE：functional 或 cycle（示例运行脚本按 st/mt/cycle 映射设置）。Sources: [examples/common.sh](examples/common.sh#L119-L127)
  - GPU_MODEL_FUNCTIONAL_MODE：functional 下的 st/mt；cycle 下可借助 GPU_MODEL_CYCLE_FUNCTIONAL_MODE 指定内部参考功能模式（默认 st）。Sources: [examples/common.sh](examples/common.sh#L104-L107)
  - GPU_MODEL_FUNCTIONAL_WORKERS：mt 下的 worker 数；若未设置，脚本用 GPU_MODEL_MT_WORKERS（默认 4）派发。Sources: [examples/common.sh](examples/common.sh#L101-L103) [examples/common.sh](examples/common.sh#L128-L130)
- 日志与追踪
  - GPU_MODEL_DISABLE_TRACE：是否禁用 Trace（默认 0 开启）。Sources: [examples/common.sh](examples/common.sh#L121-L127)
  - GPU_MODEL_LOG_MODULES / GPU_MODEL_LOG_LEVEL：日志模块与级别（示例默认 hip_runtime_abi / info）。Sources: [examples/common.sh](examples/common.sh#L123-L127)
- 动态链接环境
  - LD_PRELOAD：注入模型 HIP 运行时拦截库；脚本会自动串联 libasan（若存在）与目标 so。Sources: [examples/common.sh](examples/common.sh#L62-L71) [examples/common.sh](examples/common.sh#L119-L127)
  - LD_LIBRARY_PATH：若系统存在 /opt/rocm/lib{,64}，脚本会自动添加到 LD_LIBRARY_PATH。Sources: [examples/common.sh](examples/common.sh#L51-L60) [examples/common.sh](examples/common.sh#L131-L135)

表格（环境变量摘要）
- 变量 | 默认 | 作用域
- GPU_MODEL_USE_HIPCC_CACHE | 1 | 示例编译缓存开关
- GPU_MODEL_HIPCC_CACHE_DIR | /tmp/gpu_model_hipcc_cache | hipcc 缓存路径
- GPU_MODEL_EXECUTION_MODE | functional/cycle | 运行期模式
- GPU_MODEL_FUNCTIONAL_MODE | st/mt | 功能执行子模式
- GPU_MODEL_FUNCTIONAL_WORKERS | 未设 | mt 并行度（用 GPU_MODEL_MT_WORKERS 回退）
- GPU_MODEL_CYCLE_FUNCTIONAL_MODE | st | cycle 内部参考功能模式
- GPU_MODEL_DISABLE_TRACE | 0 | Trace 总开关
- GPU_MODEL_TRACE_DIR | 由脚本生成 | Trace 输出目录
- GPU_MODEL_LOG_MODULES | hip_runtime_abi | 日志模块
- GPU_MODEL_LOG_LEVEL | info | 日志级别
Sources: [examples/common.sh](examples/common.sh#L96-L136) [tools/hipcc_cache.sh](tools/hipcc_cache.sh#L4-L6)

## HIP 工具链与缓存策略
- hipcc 的使用场景：当你需要编译 examples/* 下的 HIP 源码或运行真实 HIP 程序时，脚本通过 hipcc 或 hipcc_cache.sh 生成目标产物；模型本体与测试构建不依赖 hipcc。Sources: [README.md](README.md#L21-L27) [examples/README.md](examples/README.md#L7-L13)
- 缓存机制：hipcc_cache.sh 以编译参数、hipcc 版本与输入文件哈希作为 key，实现输出缓存；需 sha256sum 与 hipcc 可用；可用 GPU_MODEL_HIPCC_CACHE_DIR 指定缓存目录，或将 GPU_MODEL_USE_HIPCC_CACHE=0 关闭缓存直连 hipcc。Sources: [tools/hipcc_cache.sh](tools/hipcc_cache.sh#L7-L15) [tools/hipcc_cache.sh](tools/hipcc_cache.sh#L57-L76)

## 最小自检流程（建议在配置完成后执行）
- 构建与运行快速检查：scripts/run_exec_checks.sh 会构建必要目标（gpu_model_tests、gpu_model_hip_runtime_abi），随后运行 examples/01-vecadd-basic 并校验 “vecadd validation ok” 以及 Trace 产物存在。Sources: [scripts/run_exec_checks.sh](scripts/run_exec_checks.sh#L12-L24) [examples/README.md](examples/README.md#L49-L57)

## 常见故障与定位提示
- CMake 生成器错误或 Ninja 缺失：项目默认 CMake 预设使用 Ninja 生成器（dev-fast）。如遇生成器相关错误，先确认 Ninja 是否已安装或改用其他生成器并相应调整预设。Sources: [CMakePresets.json](CMakePresets.json#L10-L16)
- “missing tool: hipcc” 或 “missing tool: sha256sum”：来自 hipcc_cache.sh 的前置检查，请安装对应工具或设置 GPU_MODEL_USE_HIPCC_CACHE=0 绕过缓存直连 hipcc。Sources: [tools/hipcc_cache.sh](tools/hipcc_cache.sh#L7-L15) [examples/common.sh](examples/common.sh#L30-L38)
- 真实 HIP 路径运行失败（符号找不到）：确认 LD_PRELOAD 与 LD_LIBRARY_PATH 是否正确设置；脚本会尝试自动拼接 /opt/rocm/lib{,64} 与 libasan。Sources: [examples/common.sh](examples/common.sh#L51-L71) [examples/common.sh](examples/common.sh#L119-L136)

## 相关脚本与可选环境变量（回归用）
- 轻量门禁 GPU_MODEL_GATE_LIGHT_GTEST_FILTER 与全量门禁 GPU_MODEL_GATE_DEBUG_ASAN_GTEST_FILTER 可覆盖默认的测试过滤，建议在阅读“常用脚本与回归套件”前保留默认值以降低复杂度。Sources: [scripts/README.md](scripts/README.md#L18-L19) [scripts/README.md](scripts/README.md#L26-L31)

## 推荐后续阅读（完成环境配置后）
- 运行示例与验证：学习如何按 st/mt/cycle 运行与验证结果，进一步理解 Trace 产物格式与成功标准。[运行示例与验证](4-yun-xing-shi-li-yu-yan-zheng) Sources: [examples/README.md](examples/README.md#L16-L21)
- 可视化 Trace（Perfetto）：将 timeline.perfetto.json 导入进行可视化分析。[可视化 Trace（Perfetto）](5-ke-shi-hua-trace-perfetto) Sources: [README.md](README.md#L38-L40)
- 使用真实 HIP 程序运行：通过 LD_PRELOAD 钩子让 host 原生执行、kernel 在模型中运行。[使用真实 HIP 程序运行](6-shi-yong-zhen-shi-hip-cheng-xu-yun-xing) Sources: [README.md](README.md#L37-L39)
- 常用脚本与回归套件：了解轻量/全量门禁、禁用 Trace 的冒烟等常用脚本。[常用脚本与回归套件](7-chang-yong-jiao-ben-yu-hui-gui-tao-jian) Sources: [scripts/README.md](scripts/README.md#L11-L39)