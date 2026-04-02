# Conditional Multibarrier Example Design

## 目标

新增一个真实、可运行、可校验的 example，展示：

- 多 block
- 多 barrier
- 条件控制额外计算
- barrier 对所有线程一致执行
- host 侧有明确预期值检查
- 可对比 `st / mt / cycle`

## 放置位置

新增目录：

- `examples/08-conditional-multibarrier/`

包含文件：

- `conditional_multibarrier.hip`
- `run.sh`
- `README.md`

## 示例程序形状

- `grid_dim_x > 1`
- `block_dim_x = 128`
- block 内使用 `__shared__ int tile[128]`
- 至少 3 次 `__syncthreads()`
- 条件只影响计算和写回，不影响 barrier 执行路径

## host 校验

host 侧做两层检查：

1. 逐元素精确比对
2. 逐 block 辅助摘要
   - `sum`
   - `first`
   - `last`

## run.sh 行为

脚本负责：

- 编译 HIP 程序
- 分别用 `st / mt / cycle` 运行
- 打印每种模式的：
  - pass/fail
  - 输出摘要
  - barrier / shared / global / wave_exits 统计
  - `program_cycle_stats` 存在性和 `total_cycles` 摘要

## README 内容

README 至少说明：

- 这个例子验证什么
- 为什么条件分支是合法的 barrier 用法
- 运行命令
- 预期输出大致长什么样

## 验收标准

1. example 可独立运行
2. `st / mt / cycle` 输出结果一致
3. host 逐元素校验通过
4. README 清楚解释示例目的
