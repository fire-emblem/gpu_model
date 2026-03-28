# Instruction Exec Layering

## 目标

把指令相关实现稳定分成三层：

1. `encoding / bitfield`
2. `inst definition / decode`
3. `exec`

同时在 `exec` 层再分一层大类：

- 计算
- 访存
- 控制
- 同步

这样后续补 GCN ISA、替换手写 decode、补 raw exec 和 cycle exec 时，结构不会失控。

## 第一层：Encoding / Bitfield

这一层负责：

- raw instruction binary
- format classify
- bitfield extract
- field meaning

推荐结构：

- `FormatBase`
- `SOP1 / SOP2 / SOPC / SOPK / SOPP / SMRD / VOP1 / VOP2 / VOP3A / DS / FLAT ...`

这一层不负责：

- operand 语义
- instruction 语义
- 执行

这层更适合：

- bitfield struct / union
- field extractor
- generated field metadata

## 第二层：InstDef / Decode

这一层负责：

- opcode
- mnemonic
- operand schema
- flags
- implicit read / write
- semantic family
- issue family

这一层建议以 **表驱动** 为主，不要用大量手写派生类。

适合的数据：

- `profiles.yaml`
- `operand_kinds.yaml`
- `semantic_families.yaml`
- `format_classes.yaml`
- `instructions.yaml`

当前项目已经开始走这条路：

- [src/spec/gcn_db/](/data/gpu_model/src/spec/gcn_db)
- [scripts/gen_gcn_isa_db.py](/data/gpu_model/scripts/gen_gcn_isa_db.py)
- [generated_gcn_inst_db.h](/data/gpu_model/include/gpu_model/decode/generated_gcn_inst_db.h)

## 第三层：Exec

这一层负责：

- 读操作数
- 应用 mask
- memory request
- 写回寄存器
- branch / barrier / waitcnt / atomic / matrix 语义

但 `exec` 层不要直接做成“每条指令一个派生类”的大爆炸继承树。

正确分层是：

### 3.1 ExecHandlerBase

统一入口，负责：

- dispatch 到大类 handler
- 公共 trace/stats 钩子

### 3.2 大类基类

第一层大类派生：

- `ComputeHandlerBase`
- `MemoryHandlerBase`
- `ControlHandlerBase`
- `SyncHandlerBase`

### 3.3 Family 层

在大类之下再分 family：

- `ScalarAluHandler`
- `VectorAluHandler`
- `MfmaHandler`
- `ScalarMemoryHandler`
- `VectorMemoryHandler`
- `LdsMemoryHandler`
- `AtomicMemoryHandler`
- `BranchHandler`
- `MaskHandler`
- `BuiltinStateHandler`
- `WaitcntHandler`
- `BarrierHandler`

### 3.4 少量具体 opcode 特例

只有当 family 通用逻辑覆盖不了时，才给具体 opcode 做 override。

原则：

- 绝大多数指令不需要一个单独的 C++ 子类
- 绝大多数指令应该走：
  - family handler
  - + per-op metadata
  - + 少量 op-specific hook

### 3.5 具体计算操作使用 Functor 注册

对于“计算类”指令，不建议继续扩大：

- `switch(opcode)`
- `if (mnemonic == ...)`
- 每条指令一个手写派生类

更合适的是：

- `family handler`
- `functor registry`

例如：

- `ScalarAluHandler` 负责：
  - 通用读操作数
  - 通用写回
  - 通用 trace/stats
- 具体计算逻辑通过 functor 注册：
  - `s_add_u32`
  - `s_mul_i32`
  - `v_add_u32_e32`
  - `v_sub_f32_e32`
  - `v_fma_f32`

也就是说：

- 大类 / family 用类层次
- 具体计算操作用 functor 注册

这样后续新增指令时，很多情况只需要：

1. 在 `gcn_db` 里补 definition
2. 在对应 family registry 里注册一个 functor

不需要新增一个完整 C++ 子类

## 为什么这样分

### Encoding 层适合类层次

因为 format class 天然就是一组结构固定的字段。

### InstDef / Decode 层适合表驱动

因为 opcode 数量很大，手写类会失控。

### Exec 层适合“大类 + family + 特例”

因为执行逻辑有明显公共部分：

- 计算指令有共同的操作数读取和写回模式
- 访存指令有共同的地址 / space / request / wait domain 模式
- 控制指令有共同的 PC / mask / condition 模式
- 同步指令有共同的 barrier / waitcnt 模式

## 推荐目录

建议新增目录：

- `include/gpu_model/exec/handlers/`
- `src/exec/handlers/`

建议文件结构：

```text
include/gpu_model/exec/handlers/
  exec_handler_base.h
  compute_handler_base.h
  memory_handler_base.h
  control_handler_base.h
  sync_handler_base.h
  scalar_alu_handler.h
  vector_alu_handler.h
  mfma_handler.h
  scalar_memory_handler.h
  vector_memory_handler.h
  lds_memory_handler.h
  atomic_memory_handler.h
  branch_handler.h
  mask_handler.h
  builtin_state_handler.h
  waitcnt_handler.h
  barrier_handler.h

src/exec/handlers/
  exec_handler_base.cpp
  compute_handler_base.cpp
  memory_handler_base.cpp
  control_handler_base.cpp
  sync_handler_base.cpp
  ...
```

## 当前项目的迁移顺序

### Step 1

先创建目录和基类接口，不迁移现有实现。

### Step 2

把当前 `semantic_handlers.cpp` 中最明显的大类关系映射到：

- 计算
- 访存
- 控制
- 同步

### Step 3

先迁移最稳定的大类：

- scalar ALU
- vector ALU
- scalar memory
- vector memory
- branch
- barrier / waitcnt

其中：

- `scalar/vector compute`
  - 优先做 `family handler + functor registry`
- `memory/control/sync`
  - 先做 family handler
  - 再看是否需要细分 functor

### Step 4

最后迁移复杂特例：

- LDS atomic
- matrix / MFMA
- 特殊 builtins

## 结论

这套分层最稳：

- `encoding` 用类/format 层次
- `decode/definition` 用表驱动
- `exec` 用：
  - `ExecHandlerBase`
  - `大类基类`
  - `family handler`
  - `compute functor registry`
  - `少量具体指令特例`

这能同时满足：

- 扩 ISA
- 控制复杂度
- function / cycle / raw 三条执行路径后续共享同一套 family 语义
