# Encoded Execution Layout

`exec/encoded/` 保存与真实 machine encoding 强绑定的执行主线。

当前按层次分为：

- `descriptor/`
  - 静态 instruction descriptor 解析与分类
  - 负责把 decoded instruction 映射到更稳定的静态类别
- `object/`
  - 绑定 operand 后的 executable instruction object
  - 负责把 decoded instruction 绑定成可执行对象
- `semantics/`
  - raw/encoded 语义 handler
  - 负责具体指令效果
- `executor/`
  - program-object execution 的顶层驱动

设计原则：

- `descriptor` 只描述“这是什么指令”
- `object` 只描述“这条具体实例如何绑定”
- `semantics` 只描述“执行效果是什么”
- `executor` 只描述“如何驱动执行”
