# 07 VecAdd Cycle Splitting

这个例子用于承载“不同 HIP 程序写法的 vecadd cycle 对比”目标。

当前状态：

- 已提供 3 个不同写法的真实 HIP 程序
- 每个程序都可以独立用 `hipcc` 生成 `.out`
- 当前脚本会对同一个 `.out` 分别跑 `st` / `mt` / `cycle`
- 每个模式默认落盘 `stdout.txt`、`trace.txt`、`trace.jsonl`、`timeline.perfetto.pb`、`timeline.perfetto.json`、`launch_summary.txt`
- 其中 `timeline.perfetto.pb` 更适合直接在 Perfetto 里看原生层级；`timeline.perfetto.json` 更适合文本检查和 diff
- 当前统一 slot 语义下，`cycle` 是 `resident_fixed`，`st/mt` 是 `logical_unbounded`
- `cycle` 模式会对三个程序做真实 cycle 对比并生成 `cycle_comparison.txt`

程序列表：

- `vecadd_direct.hip`
  一元素一线程
- `vecadd_grid_stride.hip`
  grid-stride 循环
- `vecadd_chunk2.hip`
  一线程处理两个元素

运行：

```bash
./examples/07-vecadd-cycle-splitting/run.sh
```
