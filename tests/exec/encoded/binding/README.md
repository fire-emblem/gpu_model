# Binding Tests

本目录测试：

- `DecodedInstruction`
- `RawGcnInstructionDescriptor`
- `BindRawGcnInstructionObject()`

之间的绑定关系是否正确。

关注点：

- concrete object 是否绑定正确
- placeholder / unknown fallback 是否正确
- binding 不应依赖 parser/object 层间接验证
