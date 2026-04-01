# 01 HIP 命令行 Interposer

这是最直接的端到端展示例子：

- 使用 `hipcc` 编译真实 HIP `vecadd`
- 正常执行 host `main()`
- 通过 `LD_PRELOAD` 把 HIP runtime 调用重定向到模型

运行：

```bash
./examples/01-hip-command-line-interposer/run.sh
```
