# Examples

这里统一放对用户可见、可直接运行的例子。

规则：

- 每个 case 一个目录
- 每个 case 一个独立 `run.sh`
- 目录按“由简到难”编号
- 内部 API 级示例不再放这里，改由单测和工具代码维护

当前保留的高价值例子：

1. [01-hip-command-line-interposer](./01-hip-command-line-interposer)
   最直接的端到端例子。编译真实 HIP vecadd，可通过 `LD_PRELOAD` 把 kernel 执行重定向到模型。
2. [02-hip-fma-loop-interposer](./02-hip-fma-loop-interposer)
   真实 HIP host+device 路径，包含循环和浮点运算。
3. [03-cycle-fma-trace](./03-cycle-fma-trace)
   展示 cycle trace、jsonl trace、timeline 的可观测输出。
4. [04-cycle-comparison-block-count](./04-cycle-comparison-block-count)
   展示不同 block 数量下的 cycle 计数差异。
5. [05-cycle-comparison-kernel-shape](./05-cycle-comparison-kernel-shape)
   展示不同 kernel 写法下的 cycle 计数差异。

常用命令：

```bash
./examples/01-hip-command-line-interposer/run.sh
./examples/03-cycle-fma-trace/run.sh
./examples/04-cycle-comparison-block-count/run.sh
./examples/05-cycle-comparison-kernel-shape/run.sh
```
