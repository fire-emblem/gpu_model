# 02 HIP FMA Loop Interposer

这个例子和 `01` 一样走真实 HIP host+device 路径，但 kernel 包含循环与浮点运算。

适合用来观察：

- host 路径是否正常
- loop / branch / FMA 的模型执行是否正确

运行：

```bash
./examples/02-hip-fma-loop-interposer/run.sh
```
