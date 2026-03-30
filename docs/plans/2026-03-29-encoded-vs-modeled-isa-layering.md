# Encoded vs Modeled ISA Layering

> [!NOTE]
> 历史计划文档。用于保留当时的拆解和决策上下文，不作为当前代码结构的权威描述。当前主线以 `docs/my_design.md` 和 `docs/module-development-status.md` 为准。


## 目标

回答下面几个问题：

1. 现在的 `raw / canonical` 命名是否清晰
2. 是否应该把“encoding 解析”和“执行指令对象”再拆层
3. `encoded ISA` 和 `modeled ISA` 的职责边界是什么
4. 哪些部分应该长期稳定，哪些部分可以保持实现弹性

---

## 结论先说

### 结论 1

当前的 `raw / canonical` 命名不够清晰，长期看应替换。

更合适的命名是：

- `encoded ISA`
- `modeled ISA`

原因：

- `raw` 容易被误解成“原始 bytes”或“临时粗糙实现”
- `canonical` 容易被误解成“官方标准 ISA”

但项目里实际表达的是：

- 一层与真实 encoding 强绑定
- 一层为项目 functional/cycle 建模服务

所以 `encoded / modeled` 更准确。

### 结论 2

应该把当前 `raw` 路径至少拆成两层。

推荐最少分层：

1. `EncodingDescriptor` / 静态描述层
2. `DecodedInstruction` / 已绑定 operand 的实例层
3. `ExecutableInstruction` / 可执行绑定层

如果只愿意先拆两层，也应至少区分：

- 静态 descriptor
- 绑定 operand 后的可执行指令实例

### 结论 3

`modeled ISA` 不应该依附于 `encoded ISA` 的表示层，但两者应该共享“语义/effect 层”。

一句话：

- 不共享“机器表示”
- 共享“状态效果落地”

### 结论 4

这已经属于长期维护框架问题，值得长期化。

但应该长期化的是：

- 分层边界
- 数据流方向
- coverage taxonomy

而不是现在就把每个类的最终实现形式彻底定死。

---

## 为什么 `raw / canonical` 不够好

## `raw`

问题：

- 容易让人想到 `raw bytes`
- 容易让人想到“未经处理”
- 容易让人想到“过渡态”

但项目当前这层实际已经包含：

- decoded operands
- instruction object
- semantic handler binding
- direct execute

所以它早就不是“raw bytes”意义上的 raw。

## `canonical`

问题：

- 容易被理解成“官方标准表示”
- 容易被理解成“比 encoded 更接近真实 AMD ISA”

但项目里的这层本质上是：

- internal modeled ISA
- 为 functional / cycle 建模做的项目内收敛表示

所以不应继续用 `canonical` 来表达。

---

## 推荐长期命名

推荐统一为：

- `encoded`
- `modeled`

对应关系：

- `raw GCN instruction`
  -> `encoded GCN instruction`
- `canonical/internal ISA`
  -> `modeled ISA`

示例：

- `EncodedGcnInstruction`
- `EncodedGcnInstructionObject`
- `EncodedGcnSemanticHandler`
- `ModeledInstruction`
- `ModeledSemantics`
- `EncodedToModeledLowering`

---

## 为什么要拆成多层

当前逻辑里混杂了三种不同性质的信息：

1. 静态 encoding 描述
2. 某条具体指令实例的 operand 绑定结果
3. 执行行为绑定

这三者不应长期耦合在一起。

原因：

- 生命周期不同
- 复用粒度不同
- coverage 语义不同
- unsupported 诊断阶段不同

---

## 推荐分层

## Layer 1: Encoding Descriptor

这是静态描述层。

作用：

- 描述一类指令
- 不描述某条具体指令实例

应包含：

- `encoding_id`
- `format_class`
- `opcode`
- `size_bytes`
- `bitfield layout`
- `operand schema`
- `semantic_family`
- `issue_family`
- `flags`
- `implicit registers`

不应包含：

- 当前 PC
- 当前指令实例的 immediate 值
- 当前寄存器编号的具体绑定结果
- 当前执行上下文

可接受命名：

- `GcnEncodingDescriptor`
- `InstructionDescriptor`
- `StaticInstructionInfo`

## Layer 2: Decoded Instruction

这是“绑定了 operand 的具体指令实例”，但还不一定带执行能力。

作用：

- 表示某条具体机器指令
- 记录 decode 结果
- 提供 disasm / diagnostics / lowering / execute 的共同输入

应包含：

- 指向 descriptor 的引用或 key
- `pc`
- `words`
- `size_bytes`
- `operands`
- layout/decode flags

可接受命名：

- `DecodedGcnInstruction`
- `BoundEncodedInstruction`

## Layer 3: Executable Instruction

这是“可执行绑定层”。

作用：

- 将 decode 结果绑定到具体执行逻辑
- 可能是 direct execute
- 也可能是 plan emission / lowering bridge

应包含：

- decoded instruction
- semantic handler / op emitter
- 可选的 direct `Execute()`
- 可选的 `EmitPlan()`

可接受命名：

- `ExecutableGcnInstruction`
- `EncodedInstructionObject`

---

## 推荐数据流

建议长期固定为：

1. 输入层
   - asm text
   - raw bytes
   - code object section
2. decode 层
   - `bytes -> descriptor lookup -> decoded instruction`
3. binding 层
   - `decoded instruction -> executable instruction`
4. execution / lowering 层
   - direct execute
   - or lower to modeled ISA
5. runtime / functional / cycle 层

简写：

- `bytes -> descriptor -> decoded -> executable -> execute/lower`

---

## `encoded ISA` 和 `modeled ISA` 的边界

## Encoded ISA

目标：

- 与真实机器 encoding 保持紧密映射
- 支撑 code object / `.text` / `.out` 主路径
- 支撑 decode / operand extraction / disasm / unsupported diagnostics

更关心：

- opcode
- format
- operand bitfield
- special regs
- machine-oriented semantics

## Modeled ISA

目标：

- 为 functional / cycle 执行提供稳定内部表示
- 为性能建模、what-if 分析、相对收益比较服务

更关心：

- effect / plan
- memory domain
- issue category
- dependency / timing category
- 更稳定的内部抽象

## 最关键的边界原则

不要让 `modeled ISA` 依附于 `encoded ISA` 的表示层。

但要让两者共享：

- 状态构建
- memory / sync helper
- trace helper
- effect apply / plan commit helper

一句话：

- 不共享“机器表示”
- 共享“状态和效果落地”

---

## 为什么 `modeled` 不能直接建立在 `encoded` 表示层上

不是完全不能复用，而是不能依附。

原因：

1. `encoded` 的 operand 形态太机器化
2. `modeled` 需要保留抽象自由度
3. `cycle` / `functional` 需要更稳定的 effect 语义
4. 某些 modeled op 可能代表一条以上机器指令的收敛表达

所以正确关系不是：

- `modeled -> encoded`

而是：

- `encoded -> shared effect layer`
- `modeled -> shared effect layer`

---

## 最值得长期固定的部分

这些部分应尽快稳定为长期框架：

1. `descriptor / decoded / executable` 三层边界
2. `encoded` 与 `modeled` 的命名与职责边界
3. coverage taxonomy
4. unsupported diagnostics taxonomy

推荐长期固定的 coverage 维度：

- descriptor coverage
- decode coverage
- executable binding coverage
- direct execute coverage
- modeled lowering coverage
- tests coverage

---

## 不要过早固定的部分

下面这些可以继续保持实现弹性：

- executable object 是类层次还是函数表
- semantic handler 是虚函数还是表驱动回调
- `encoded` 到 `modeled` 的 lowering 粒度
- cycle 是否直接消费 decoded instruction 或 modeled instruction

也就是说：

- 分层边界要长期化
- 具体实现策略先不要过度僵化

---

## 对当前项目的直接建议

## Step 1

先在文档和术语上完成切换：

- `raw` -> `encoded`
- `canonical/internal` -> `modeled`

不需要第一时间全仓重命名代码。

## Step 2

把当前 coverage 脚本扩展为按层统计：

- tracked subset coverage
- descriptor coverage
- decode coverage
- execute binding coverage
- tests coverage

## Step 3

把当前 `raw_gcn_instruction_object` 之前的静态描述层再抽干净：

- descriptor lookup
- operand schema
- flags
- semantic family

## Step 4

把当前 `DecodedGcnInstruction` 明确定位为：

- 已绑定 operand 的实例层

## Step 5

再决定 `ExecutableInstruction` 的最终实现风格：

- class hierarchy
- table-driven function pointers
- hybrid

## Step 6

当指令解析覆盖已经接近饱和后，推进重点从“继续补 decode”切换到“补行为覆盖与测试”：

- 为已能稳定解析的指令批量补 direct execute tests
- 为 `llvm-mc` 可组装的专题汇编批量补 module-level behavior tests
- 优先补齐控制流、标量 ALU、vector ALU、flat/global、lds/ds 的行为断言
- tensor / MFMA 提高优先级，至少先覆盖：
  - fp32
  - fp16
  - int8
- 对暂时不支持的 tensor fixtures，允许先进入自动发现并显式 `skip`
- 一旦 tensor decode 稳定，再把 skip fixture 逐个转成真实 decode + behavior case

---

## 最终判断

最终判断如下：

1. 应该拆层
2. 应该把这件事当成长期维护框架来设计
3. 但要长期化的是“层次与边界”，不是立刻冻结所有实现细节
4. 这套分层会直接降低后续：
   - ISA 扩展成本
   - coverage 统计歧义
   - unsupported 诊断混乱
   - `encoded / modeled / functional / cycle` 之间的耦合

一句话：

- 这不是过度设计
- 这是为后续规模化扩展提前修路
