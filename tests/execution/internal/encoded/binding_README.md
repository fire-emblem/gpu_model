# Binding Tests

本目录测试：

- `DecodedInstruction`
- `EncodedInstructionDescriptor`
- `BindEncodedInstructionObject()`

之间的绑定关系是否正确。

关注点：

- concrete object 是否绑定正确
- unsupported / unknown 指令对象是否按约定落到兜底绑定
- binding 不应依赖 parser/object 层间接验证
