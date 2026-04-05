# 02 FMA Loop

## 例子作用

这个例子是在 `01-vecadd-basic` 之后的第一步复杂化样例，用来验证：

- 有循环的 kernel 是否能正确执行
- 浮点累积路径在三种执行模式下是否一致
- trace 中是否能看到比最小样例更长的计算序列

它仍然是基础验证例子，但已经不再是“单条直线控制流”。

## 本例子的关注点

- 循环体是否按预期重复执行
- 浮点乘加风格计算是否在 `st` / `mt` / `cycle` 下保持一致
- trace 中是否能看到更密集的 ALU 指令序列
- `cycle` 模式下总周期是否明显高于 `01`

## 程序行为

kernel 的核心逻辑是：

- 读取输入 `a[i]` 和 `b[i]`
- 在循环中做 `acc = acc * x + y`
- 最终把累积结果写回 `c[i]`
- host 侧按相同公式在 CPU 上构造 `expect[i]`，逐元素比较

这里的重点不是数学复杂度，而是“循环 + 浮点累积 + 控制流”是否稳定。

## 运行方式

```bash
./examples/02-fma-loop/run.sh
```

脚本会：

1. 编译 [fma_loop.hip](/data/gpu_model/examples/02-fma-loop/fma_loop.hip)
2. 分别以 `st` / `mt` / `cycle` 运行
3. 检查每个模式的 `stdout.txt` 是否包含 `fma_loop host path ok`

## 关键产物

运行成功后通常会看到：

- `results/fma_loop.out`
- `results/stdout.txt`
- `results/st`
- `results/mt`
- `results/cycle`

每个模式目录下应有：

- `stdout.txt`
- `trace.txt`
- `trace.jsonl`
- `timeline.perfetto.json`
- `launch_summary.txt`

重新按当前主线运行后，也可能出现：

- `timeline.perfetto.pb`

## 预期结果

预期成功标准：

- 三个模式的 `stdout.txt` 都包含 `fma_loop host path ok`
- 三个模式的 `launch_summary.txt` 中都应有 `ok=1`
- `trace.txt` 中应能看到明显长于 `vecadd` 的计算事件序列
- `cycle` 模式的 `total_cycles` 应为正整数，且通常会比 `01` 更大

## 建议观察项

这个例子建议观察：

- `trace.txt` 中同一 wave 的连续计算序列
- `launch_summary.txt` 的 `total_cycles`
- `timeline.perfetto.json` 或 `.pb` 中更密集的计算切片

它适合作为“最小控制流样例”看循环成本，但不适合看 barrier、atomic 或复杂调度。

## 调试入口

出现问题时，建议按顺序排查：

1. `results/<mode>/stdout.txt`
   确认是 host 校验失败还是运行异常
2. `results/<mode>/trace.txt`
   看循环体对应的计算序列是否被截断或顺序异常
3. `results/<mode>/launch_summary.txt`
   对比不同模式的 `total_cycles`
4. `timeline.perfetto.*`
   只在需要看时间线密度和片段分布时使用

## 结果解读

- 这个例子通过，说明“基础控制流 + 浮点累积”路径大致可靠
- 它仍然不覆盖 shared memory、barrier、atomic
- 但它比 `01` 更适合看 trace 是否已经能承载稍长的指令序列

## 备注

- 仓库中现有 `results/` 主要作为快照参考
- 默认重新运行 `run.sh` 会把结果写到 `.cache/example-results/02-fma-loop/`
- 若需要刷新仓库内快照，显式设置 `GPU_MODEL_EXAMPLE_RESULTS_MODE=repo`
- 如果旧结果没有 `.pb`，以重跑后的结果目录为准
