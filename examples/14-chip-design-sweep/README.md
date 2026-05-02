# Chip Design Sweep

这个例子用同一个 cycle kernel 对比不同芯片配置下的性能，用来评估硬件设计参数对模型时间的影响。

## 对比内容

默认扫四组配置：

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

`cycle_comparison.txt` 包含每个配置的 `total_cycles`、`active_cycles`、`ipc`、`ap_count`、`smem_per_mp` 和 `dram_latency`。
