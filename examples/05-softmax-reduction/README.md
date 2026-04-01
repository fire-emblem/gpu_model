# 05 Softmax Reduction

这个例子展示带 shared memory 和多次归约的真实 HIP 程序。

它会：

- 用 `hipcc` 编译 `softmax_reduction.hip`
- 生成真实 `.out`
- 通过 `LD_PRELOAD` 运行 host `main()`
- 在 host 侧检查 softmax 结果是否接近预期

运行：

```bash
./examples/05-softmax-reduction/run.sh
```
