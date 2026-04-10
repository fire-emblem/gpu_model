# Algorithm Comparison: Matrix Transpose

Compares three matrix transpose algorithms on a single-precision matrix, demonstrating how memory access patterns affect GPU performance.

## What This Example Shows

All three algorithms compute the same result (`out[x][y] = in[y][x]`), but use different strategies for moving data through the GPU memory hierarchy. The cycle model captures the impact of these algorithmic differences.

## Algorithms

### 1. Naive Transpose (`transpose_naive`)
Direct global memory transpose. Each thread reads one element coalesced and writes one element strided.

- **Read pattern**: Coalesced (good)
- **Write pattern**: Strided across rows (poor — non-coalesced global writes)

### 2. Shared Memory Transpose (`transpose_shared`)
Uses dynamic shared memory (`extern __shared__`) as a tile buffer. Threads read a tile coalesced into shared memory, then write the transposed tile coalesced to global memory.

- **Read pattern**: Coalesced global load into shared memory
- **Write pattern**: Coalesced global store from shared memory
- **Requires**: `__syncthreads()` barrier between shared memory read/write phases

### 3. Padded Shared Memory Transpose (`transpose_diagonal`)
Uses statically-allocated shared memory with padding (`tile[TILE][TILE+1]`). The padding avoids shared memory bank conflicts, which can further improve throughput.

- **Read pattern**: Coalesced global load
- **Write pattern**: Coalesced global store
- **Optimization**: Bank-conflict-free shared memory access via padding

默认运行使用 `128x128` 矩阵，便于把 example 执行时间控制在可接受范围内。
可用环境变量恢复更大规模，例如：

```bash
GPU_MODEL_TRANSPOSE_WIDTH=256 GPU_MODEL_TRANSPOSE_HEIGHT=256 ./run.sh
```

## Cycle Model Results

| Algorithm | Memory Strategy | Total Cycles |
|-----------|----------------|-------------|
| transpose_naive | Direct (strided writes) | 1197 |
| transpose_shared | Dynamic shared memory | 1353 |
| transpose_diagonal | Padded static shared memory | 1365 |

## Analysis

In this cycle model configuration, the naive transpose actually shows fewer total cycles than the shared memory variants. This is because:

1. **The naive kernel has fewer instructions** — no shared memory loads/stores, no barrier synchronization. The cycle model counts each instruction's issue cycles.
2. **The shared memory kernels pay overhead** — `ds_write_b32`, `ds_read_b32`, and `s_barrier` instructions add measurable cycle cost.
3. **On real hardware**, shared memory wins due to DRAM coalescing benefits and cache effects that the simple cycle model does not fully capture.

This demonstrates an important lesson: a **naive cycle model** captures instruction-level costs but may not reflect real hardware memory system behavior. For accurate performance prediction of memory-bound kernels, more sophisticated memory modeling (cache simulation, DRAM scheduling) would be needed.

## How to Run

```bash
./run.sh
```

Results are written to `results/cycle_comparison.txt`.
