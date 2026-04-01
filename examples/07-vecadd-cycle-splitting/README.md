# 07 VecAdd Cycle Splitting

这个例子用于承载“不同 HIP 程序写法的 vecadd cycle 对比”目标。

当前状态：

- 已提供 3 个不同写法的真实 HIP 程序
- 每个程序都可以独立用 `hipcc` 生成 `.out`
- 当前脚本先完成编译与 host 校验路径骨架
- cycle 对比接口后续接入时，这个目录就是统一入口

程序列表：

- `vecadd_direct.hip`
  一元素一线程
- `vecadd_grid_stride.hip`
  grid-stride 循环
- `vecadd_chunk2.hip`
  一线程处理两个元素

运行：

```bash
./examples/07-vecadd-cycle-splitting/run.sh
```
