# 01 VecAdd Basic

这是最简单的端到端例子：

- 使用 `hipcc` 编译真实 HIP `vecadd`
- 生成真实 `.out`
- 通过 `LD_PRELOAD` 运行 host `main()`
- 由 host 侧完成结果校验

运行：

```bash
./examples/01-vecadd-basic/run.sh
```
