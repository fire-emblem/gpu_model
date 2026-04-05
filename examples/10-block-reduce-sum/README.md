# 10 Block Reduce Sum

## 例子作用

这个例子验证“多 block + block 内归约”这条典型结构。

它和 `09-dynamic-shared-sum` 的区别在于：

- `09` 更强调动态 shared
- `10` 更强调每个 block 独立做 reduction，并把结果写回各自输出槽位

## 本例子的关注点

- 多 block reduction 是否正确
- block 内 shared memory + barrier 是否稳定
- 不同 block 的归约结果是否都正确
- grid 级并行下每个 block 是否保持独立状态

## 程序行为

kernel 的逻辑是：

1. 每个线程以 grid-stride 方式累积部分和
2. 把部分和写到 block 内 shared memory
3. 做 block 内树形 reduction
4. `tid == 0` 把该 block 的总和写到输出

当前输入全为 1，参数是：

- `n = 1024`
- `grid_dim = 4`
- `block_dim = 256`

因此每个 block 的理论输出都应是：

- `1024 / 4 = 256`

## 运行方式

```bash
./examples/10-block-reduce-sum/run.sh
```

脚本会：

1. 编译 [block_reduce_sum.hip](/data/gpu_model/examples/10-block-reduce-sum/block_reduce_sum.hip)
2. 分别以 `st` / `mt` / `cycle` 运行
3. 检查 `stdout.txt` 中是否包含 `block_reduce_sum mismatches=0`

## 关键产物

运行后应有：

- `results/block_reduce_sum.out`
- `results/st`
- `results/mt`
- `results/cycle`

每个模式目录下应有：

- `stdout.txt`
- `trace.txt`
- `trace.jsonl`
- `timeline.perfetto.json`
- `launch_summary.txt`

按当前主线重跑后，还可能看到：


## 预期结果

预期成功标准：

- 三个模式的 `stdout.txt` 都包含 `block_reduce_sum mismatches=0`
- 三个模式的 `launch_summary.txt` 中都应有 `ok=1`
- host 侧每个 block 的输出都应接近 `256`
- trace 中应能看到 shared memory 访问、归约阶段和最终写回

## 建议观察项

这个例子建议重点观察：

- 每个 block 是否独立完成 reduction
- block 内树形 reduction 的阶段是否完整
- 多 block 并行时是否存在错误共享状态

如果你要看“单 block reduction”，这个例子比 `05` 更简单；如果你要看“动态 shared”，则应看 `09`。

## 调试入口

建议按顺序排查：

1. `results/<mode>/stdout.txt`
   看 `mismatches` 是否为 0
2. `results/<mode>/trace.txt`
   看 reduction 阶段和最后写回
3. `results/<mode>/launch_summary.txt`
   确认是否正常结束
4. `timeline.perfetto.json`
   需要按时间轴看 reduction 阶段时再使用

## 结果解读

- 这个例子通过，说明“多 block 下每个 block 做独立 reduction”这条结构基本可用
- 如果失败，需要先分清是 block 内 reduction 出错，还是 block 间状态串扰
- 它适合作为 reduction 类例子的通用基线

## 备注

- 仓库中的旧 `results/` 主要作为快照参考
