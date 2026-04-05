# scripts

这里放仓库级辅助脚本。

当前比较关键的入口：

- `install_git_hooks.sh`
  - 安装仓库内 git hooks
  - 会把 `core.hooksPath` 指到仓库里的 `.githooks`

- `run_push_gate_light.sh`
  - `git push` 前的轻量门禁
  - 默认只并行跑两条 smoke pipeline：
    - `Debug + ASan` smoke filter
    - `Release` smoke filter
  - 默认不跑全量 `gpu_model_tests`
  - 默认不跑 `examples`
  - 可用环境变量 `GPU_MODEL_GATE_LIGHT_GTEST_FILTER` 覆盖 smoke filter

- `run_push_gate.sh`
  - 手动执行的全量门禁
  - 默认并行跑三条 pipeline：
    - `Debug + ASan` 快速功能覆盖 `gpu_model_tests`
      - 默认排除 `RequestedShapes/*`
      - 默认排除 `RequestedThreadScales/*`
      - 可用环境变量 `GPU_MODEL_GATE_DEBUG_ASAN_GTEST_FILTER` 覆盖
    - `Release` 全量 `gpu_model_tests`
    - `Release` 全部 `examples/01-11/run.sh`
  - 三条 pipeline 各自使用独立 build 目录，避免互相影响
  - 完整日志默认落到 `results/push-gate/`
  - 其中 release/debug 的 `gpu_model_tests` 还会额外生成
    `*.slowest.txt`，列出最慢测试 Top 20
  - 同时会保存：
    - `release.gpu_model_tests.xml`
    - `debug_asan.gpu_model_tests.xml`

- `run_exec_checks.sh`
  - 最小基础执行检查
- `run_shared_heavy_regression.sh`
  - shared-heavy 四锚点回归
  - 包括 focused gtests、HIP CTS/feature CTS 和 examples `03/05/09/10`
- `run_real_hip_kernel_regression.sh`
  - 更上层真实 HIP kernel 回归
  - 先跑 shared-heavy ring，再补 atomic focused ring 和 example `04`
- `run_abi_regression.sh`
  - ABI-heavy 回归
  - 覆盖 by-value aggregate、`3D hidden args / builtin ids` 和 llvm-mc ABI fixture
- `run_scaling_regression.sh`
  - 形状/线程规模回归
- `run_disable_trace_smoke.sh`
  - 关闭 trace 的非 trace 冒烟回归
  - 统一使用：
    - `GPU_MODEL_DISABLE_TRACE=1`
  - 适合在快速推进模型语义时验证：
    - runtime / execution / cycle stats 不依赖 trace 仍然正确
  - 默认使用 `build-ninja`
  - 可用环境变量 `GPU_MODEL_DISABLE_TRACE_GTEST_FILTER` 覆盖测试集合

生成类脚本：

- `gen_gcn_isa_db.py`
- `gen_gcn_full_opcode_table.py`
- `report_isa_coverage.py`
