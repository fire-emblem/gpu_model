# 04 Atomic Reduction

这个例子展示带 global atomic 的真实 HIP 程序。

它会：

- 用 `hipcc` 编译 `atomic_reduction.hip`
- 生成真实 `.out`
- 通过 `LD_PRELOAD` 运行 host `main()`
- 在 host 侧检查 atomic 归约结果

运行：

```bash
./examples/04-atomic-reduction/run.sh
```
