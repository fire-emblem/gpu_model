# 06 MMA GEMM

这个例子用于承载 MMA / MFMA 路径的目标示例。

当前状态：

- 目录、源码、编译脚本已搭好
- 使用真实 HIP 源码生成 `.out`
- 若环境支持 `gfx90a` / `mfma` 编译，则尝试运行最小 GEMM probe
- 若环境不支持，则脚本会明确打印 `STATUS: unsupported_yet`

运行：

```bash
./examples/06-mma-gemm/run.sh
```
