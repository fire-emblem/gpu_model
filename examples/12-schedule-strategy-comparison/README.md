# Schedule Strategy Comparison

Demonstrates how grid/block dimensions affect execution cycles in the GPU cycle model.

## What This Example Shows

Three configurations of the same vector addition kernel, each processing 4096 elements using grid-stride loops. The only difference is how many threads participate and how many iterations each thread performs.

## Configurations

| Variant | Grid | Block | Total Threads | Iterations/Thread |
|---------|------|-------|---------------|-------------------|
| low_parallelism | 1x1x1 | 64x1x1 | 64 | 64 |
| moderate_parallelism | 8x1x1 | 128x1x1 | 1024 | 4 |
| optimal_parallelism | 16x1x1 | 256x1x1 | 4096 | 1 |

## Cycle Model Results

| Configuration | Total Cycles |
|---------------|-------------|
| low_parallelism (64 threads) | 5608 |
| moderate_parallelism (1024 threads) | 808 |
| optimal_parallelism (4096 threads) | 524 |

**Speedup from low to optimal: 10.7x**

## Analysis

- **Low parallelism**: Only 64 threads across 1 block. The GPU computes are severely underutilized — most SIMD lanes sit idle while a single wavefront loops 64 times.
- **Moderate parallelism**: 1024 threads across 8 blocks. Better occupancy, but some threads still iterate 4 times.
- **Optimal parallelism**: 4096 threads across 16 blocks, 1 iteration each. Full utilization of the GPU's parallel compute resources.

The cycle model correctly captures the relationship between parallelism configuration and execution efficiency, demonstrating why choosing appropriate `gridDim` and `blockDim` is critical for GPU performance.

## How to Run

```bash
./run.sh
```

Results are written to `results/cycle_comparison.txt`.
