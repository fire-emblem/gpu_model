# scripts

这里放仓库级辅助脚本。

当前比较关键的入口：

- `install_git_hooks.sh`
  - 安装仓库内 git hooks
  - 会把 `core.hooksPath` 指到仓库里的 `.githooks`

- `run_push_gate.sh`
  - `git push` 前的统一门禁
  - 默认并行跑三条 pipeline：
    - `Debug + ASan` 全量 `gpu_model_tests`
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

生成类脚本：

- `gen_gcn_isa_db.py`
- `gen_gcn_full_opcode_table.py`
- `report_isa_coverage.py`
