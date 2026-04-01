# 03 Shared Reverse

这个例子展示带 shared memory 和 barrier 的真实 HIP 程序。

它会：

- 用 `hipcc` 编译 `shared_reverse.hip`
- 生成真实 `.out`
- 通过 `LD_PRELOAD` 运行 host `main()`
- 在 host 侧检查 block 内逆序结果是否正确

运行：

```bash
./examples/03-shared-reverse/run.sh
```
