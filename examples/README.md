# Examples

这里放面向客户的可执行例子。

约束：

- 每个例子一个目录
- 每个例子都是真实 HIP 源码
- 每个例子都能先通过 `hipcc` 生成 `.out`
- 每个目录都有独立 `run.sh`
- 目录按由简到难编号
- 当前特性尚未完全接通的部分，在目录 `README.md` 里明确标记

当前目标例子：

1. [01-vecadd-basic](./01-vecadd-basic)
2. [02-fma-loop](./02-fma-loop)
3. [03-shared-reverse](./03-shared-reverse)
4. [04-atomic-reduction](./04-atomic-reduction)
5. [05-softmax-reduction](./05-softmax-reduction)
6. [06-mma-gemm](./06-mma-gemm)
7. [07-vecadd-cycle-splitting](./07-vecadd-cycle-splitting)
