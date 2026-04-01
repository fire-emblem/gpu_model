# 02 FMA Loop

这个例子和 `01` 一样走真实 HIP `.out` 路径，但 kernel 包含循环与浮点运算。

适合用来观察：

- host 校验是否通过
- loop / branch / FMA 的执行路径是否正常

运行：

```bash
./examples/02-fma-loop/run.sh
```
