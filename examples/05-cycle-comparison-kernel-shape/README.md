# 05 Cycle 计数对比：不同 Kernel 写法

这个例子固定 launch 形状，对比：

- 纯 ALU
- global waitcnt
- shared barrier

观察不同 kernel 结构的 cycle 总数差异。

运行：

```bash
./examples/05-cycle-comparison-kernel-shape/run.sh
```
