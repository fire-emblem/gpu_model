# 10 Block Reduce Sum

这个例子展示带 shared memory、barrier 和多 block 归约的真实 HIP 程序。

它会：

- 用 `hipcc` 编译 `block_reduce_sum.hip`
- 生成真实 `.out`
- 通过 `LD_PRELOAD` 分别以 `st` / `mt` / `cycle` 运行
- 在 host 侧检查每个 block 的归约结果是否正确

运行：

```bash
./examples/10-block-reduce-sum/run.sh
```
