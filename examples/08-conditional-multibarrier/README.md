# 08 Conditional Multibarrier

这个例子展示一个合法的多 block、多 barrier、带条件计算的真实 HIP 程序。

它会：

- 用 `hipcc` 编译 `conditional_multibarrier.hip`
- 生成真实 `.out`
- 通过 `LD_PRELOAD` 分别以 `st` / `mt` / `cycle` 运行
- 在 host 侧做逐元素精确校验

合法性关键点：

- 条件分支只改变每个线程的计算路径，不会让任何线程跳过 `__syncthreads()`
- 所有线程都会按相同顺序到达 3 次 barrier，因此这个 barrier 用法是合法的

运行：

```bash
./examples/08-conditional-multibarrier/run.sh
```
