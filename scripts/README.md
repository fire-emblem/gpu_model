# scripts

这里放仓库级辅助脚本。

当前比较关键的入口：

- `run_exec_checks.sh`
  - 最小基础执行检查
- `run_shared_heavy_regression.sh`
  - shared-heavy 四锚点回归
  - 包括 focused gtests、HIP CTS/feature CTS 和 examples `03/05/09/10`
- `run_real_hip_kernel_regression.sh`
  - 更上层真实 HIP kernel 回归
  - 先跑 shared-heavy ring，再补 atomic focused ring 和 example `04`
- `run_scaling_regression.sh`
  - 形状/线程规模回归

生成类脚本：

- `gen_gcn_isa_db.py`
- `gen_gcn_full_opcode_table.py`
- `report_isa_coverage.py`
