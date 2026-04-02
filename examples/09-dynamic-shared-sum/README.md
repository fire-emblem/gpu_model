# 09 Dynamic Shared Sum

这个例子展示带 `extern __shared__` 和 barrier 的真实 HIP 程序。

它会：

- 用 `hipcc` 编译 `dynamic_shared_sum.hip`
- 生成真实 `.out`
- 通过 `LD_PRELOAD` 运行 host `main()`
- 在 host 侧检查每个 block 的动态 shared reduction 结果是否正确

运行：

```bash
./examples/09-dynamic-shared-sum/run.sh
```
