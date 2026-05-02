# Chip Design Sweep

这个例子用同一个 cycle kernel 对比不同芯片配置下的性能，用来评估硬件设计参数对模型时间的影响。

## 对比内容

默认扫五组配置：

| 配置 | 目的 |
|------|------|
| `baseline` | mac500 默认拓扑和默认 DRAM latency |
| `dram_fast` | 降低 DRAM latency，观察访存延迟收益 |
| `ap_128` | 把 AP 从 104 提升到 128，观察 block 排队下降的收益 |
| `smem_128` | 把 shared memory 从 64K 提升到 128K，观察 resident block 数上升的收益 |
| `smem_192` | 把 shared memory 从 64K 提升到 192K，观察更高驻留并发的收益 |

kernel 同时包含 global load、`s_waitcnt`、shared memory 读写和 barrier，因此会受访存延迟、AP 并发和 shared memory resident capacity 共同影响。这个例子把单 block shared memory 压力固定在 48K，并把 AP resident block 上限放到 4，所以 64K / 128K / 192K 会分别对应 1 / 2 / 4 个可驻留 block。

## 运行方式

```bash
./examples/14-chip-design-sweep/run.sh
```

输出：

- `results/stdout.txt`
- `results/cycle_comparison.txt`
- `results/cycle_report.md`
- `results/cycle_report.json`
- `results/timeline_summary.txt`
- `results/<variant>/trace.txt`
- `results/<variant>/trace.jsonl`
- `results/<variant>/timeline.perfetto.json`
- `results/<variant>/launch_summary.txt`

`cycle_comparison.txt` 包含每个配置的 `total_cycles`、`active_cycles`、`ipc`、`ap_count`、`smem_per_mp` 和 `dram_latency`。
`cycle_report.md` 是面向硬件设计的汇总报告，包含基线、最佳配置、AP / shared memory / DRAM 的对比结论，以及 stall breakdown。
`cycle_report.json` 是同一份结果的机器可读版本，适合批量扫参、二次处理和自动汇总。
`timeline_summary.txt` 则给出每个 variant 的 timeline 入口、主导 stall、stall 构成和基础运行统计，方便直接定位对应 Perfetto 产物。

当前仓库配置下的一组参考结果：

| 配置 | total_cycles | active_cycles |
|------|--------------|---------------|
| `baseline` | 792 | 720 |
| `dram_fast` | 624 | 608 |
| `ap_128` | 612 | 540 |
| `smem_128` | 630 | 558 |
| `smem_192` | 508 | 465 |
