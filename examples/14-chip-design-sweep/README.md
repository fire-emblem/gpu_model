# Chip Design Sweep

这个例子用同一个 cycle kernel 对比不同芯片配置下的性能，用来评估硬件设计参数对模型时间的影响。

## 对比内容

默认扫四组配置：

| 配置 | 目的 |
|------|------|
| `baseline` | mac500 默认拓扑和默认 DRAM latency |
| `low_latency` | 降低 DRAM latency，观察访存延迟收益 |
| `ap_sparse` | 减少 AP 数量，观察块排队和并发降低的影响 |
| `smem_tight` | 收紧 AP 级 shared memory 容量，观察 resident block 数下降的影响 |

kernel 同时包含 global load、`s_waitcnt`、shared memory 读写和 barrier，因此会受访存延迟、AP 并发和 shared memory resident capacity 共同影响。

## 运行方式

```bash
./examples/14-chip-design-sweep/run.sh
```

输出：

- `results/stdout.txt`
- `results/cycle_comparison.txt`

`cycle_comparison.txt` 包含每个配置的 `total_cycles`、`active_cycles`、`ipc`、`ap_count`、`smem_per_mp` 和 `dram_latency`。
