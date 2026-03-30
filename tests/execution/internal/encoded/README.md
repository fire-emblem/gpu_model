# Encoded Exec Tests

这里保留 encoded 执行内部测试的参考说明。

当前对应关系已经迁到：

- `tests/execution/internal/encoded/*`
- `src/instruction/encoded/internal/*`
- `src/execution/*`

当前测试目录分为：

- `descriptor/`
  - 静态 descriptor 分类与 fallback 测试
- `binding/`
  - decoded instruction 到 executable object 的绑定决策测试
- `object/`
  - executable object 外壳、parser、factory 测试
- `semantics/`
  - raw/encoded 语义执行与 registry 测试

原则：

- 测试目录结构尽量跟随源码层级
- 优先在所属层直接验证该层职责
- 减少只通过更高层间接覆盖低层行为
