# Doc

Command executed by this usage bundle:

```bash
./build/fma_loop_cycle_trace_main \
  --mode cycle \
  --grid 2 \
  --block 65 \
  --n 8 \
  --iterations 2 \
  --mul0 2 \
  --add0 1 \
  --mul1 3 \
  --add1 2 \
  --latency 9 \
  --timeline-columns 40 \
  --group-by block \
  --out-dir usages/cycle-fma-trace/results
```

Expected artifacts:

- `results/stdout.txt`
- `results/fma_loop_cycle_trace.txt`
- `results/fma_loop_cycle_trace.jsonl`
- `results/fma_loop_cycle_timeline.txt`

Expected behavior:

- returns a non-zero `total_cycles`,
- prints per-element output values,
- writes both raw traces and an aggregated block-level timeline.
