# `instruction/encoded/internal`

这里存放 `instruction/encoded/*` 的内部支撑件。

约束：

- 这些文件属于长期内部层，不是过渡性兼容目录
- 对外公开接口应优先通过 `instruction/encoded/*` 暴露
- 这里主要承载：
  - encoded GCN 格式与 operand 元数据
  - generated opcode table / instruction db
  - encoded descriptor / binding 等实现细节

判断标准：

- 如果一个类型主要服务于 encoded 指令解析或绑定实现，而不应成为上层稳定 API，就应留在这里
