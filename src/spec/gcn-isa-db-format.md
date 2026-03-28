# GCN ISA DB Format

## Goal

把 GCN ISA 从“说明文档 + 手写 C++ switch/表”推进到：

1. 可人工维护
2. 可程序读取
3. 可生成 decode/disasm/semantic lookup 代码
4. 可按架构代际扩展

## 结论

建议采用：

- **YAML 作为源码格式**
- **按层拆分多个 YAML 文件**
- **生成 C++ 静态表，不在 runtime 直接解析 YAML**

也就是三层：

1. `src/spec/gcn_db/*.yaml`
   - 人工维护的结构化 ISA 数据
2. `tools/gen_gcn_isa_db.py`
   - 读取 YAML，生成 C++ 头/源文件
3. `src/decode/generated/*`
   - 编译期静态表

## 为什么选 YAML

原因：

- 比 JSON 更适合人工维护
- 允许注释
- 结构层次清楚
- 对 opcode/field/operand 这种表结构很友好
- 后续如果要生成 td-like 文件或 C++ 表，脚本处理很直接

不建议直接用：

- 纯 Markdown
  - 人能看，代码不好读
- 纯 JSON
  - 机器友好，但维护体验差，不能写注释
- 直接手写 C++ constexpr 大表
  - 可编译，但资料和实现耦合太死

## 总体目录建议

建议新增目录：

- `src/spec/gcn_db/`

目录结构建议：

```text
src/spec/gcn_db/
  profiles.yaml
  format_classes.yaml
  operand_kinds.yaml
  special_registers.yaml
  semantic_families.yaml
  opcodes/
    sop1.yaml
    sop2.yaml
    sopc.yaml
    sopk.yaml
    sopp.yaml
    smrd.yaml
    vop1.yaml
    vop2.yaml
    vopc.yaml
    vop3a.yaml
    vop3p.yaml
    ds.yaml
    flat.yaml
    global.yaml
    mubuf.yaml
    mtbuf.yaml
    mimg.yaml
    exp.yaml
    vintrp.yaml
    mfma.yaml
```

## 分层原则

不要把所有信息塞进一个大 YAML。

建议拆成四层：

### 1. 架构 profile 层

描述：

- `gfx6-gfx8`
- `gfx9`
- `cdna1`
- `cdna2`

用途：

- 指定格式字段差异
- 指定 opcode 是否存在
- 指定 waitcnt / matrix / 特殊寄存器差异

### 2. format class 层

描述：

- `SOP1`
- `SOP2`
- `VOP2`
- `VOP3A`
- `DS`
- `FLAT`

用途：

- 定义 instruction size
- 定义 opcode field
- 定义每个 operand field 的 bit range
- 定义 literal constant 规则

### 3. opcode definition 层

描述单条指令：

- mnemonic
- format class
- opcode 值
- 支持的 profile
- operand schema
- implicit read/write
- semantic family

### 4. semantic family 层

描述：

- `ScalarAlu`
- `ScalarMemory`
- `VectorAluInt`
- `VectorMemory`
- `LocalDataShare`
- `Branch`
- `Sync`
- `Matrix`

用途：

- 给 execution handler lookup 用
- 给 issue family / trace / flags 用

## YAML 建议字段

### `profiles.yaml`

建议字段：

```yaml
profiles:
  - id: gfx6_gfx8
    display_name: "GCN gfx6-gfx8"
    wave_size: 64
    has_accvgpr: false
    waitcnt_layout: legacy

  - id: gfx9_cdna
    display_name: "gfx9+/CDNA"
    wave_size: 64
    has_accvgpr: true
    waitcnt_layout: gfx9
```

### `format_classes.yaml`

建议字段：

```yaml
format_classes:
  - id: sop2
    size_bytes: 4
    opcode_field:
      word: 0
      lsb: 23
      width: 7
    fields:
      - { name: sdst, word: 0, lsb: 16, width: 7 }
      - { name: ssrc0, word: 0, lsb: 0, width: 8 }
      - { name: ssrc1, word: 0, lsb: 8, width: 8 }
    supports_literal: true
```

### `operand_kinds.yaml`

建议字段：

```yaml
operand_kinds:
  - id: scalar_reg
  - id: scalar_reg_range
  - id: vector_reg
  - id: vector_reg_range
  - id: special_reg
  - id: inline_const
  - id: literal_const
  - id: branch_target
  - id: waitcnt_fields
```

### opcode 文件

建议一条记录长这样：

```yaml
instructions:
  - mnemonic: s_add_u32
    format: sop2
    opcode: 0
    profiles: [gfx6_gfx8, gfx9_cdna]
    semantic_family: scalar_alu
    issue_family: scalar_alu_or_memory
    operands:
      - { name: dst, field: sdst, kind: scalar_reg, role: def }
      - { name: src0, field: ssrc0, kind: scalar_src, role: use }
      - { name: src1, field: ssrc1, kind: scalar_src, role: use }
    implicit_reads: []
    implicit_writes: [scc]
    flags:
      is_branch: false
      is_memory: false
      is_atomic: false
      is_barrier: false
      writes_exec: false
      writes_vcc: false
      writes_scc: true
```

## 必须单独结构化的几类特殊字段

### 1. `s_waitcnt`

不要把它只记成普通 immediate。

建议：

```yaml
special_decode:
  type: waitcnt
  fields:
    - { name: vmcnt, lsb: 0, width: 4 }
    - { name: expcnt, lsb: 4, width: 3 }
    - { name: lgkmcnt, lsb: 8, width: 4 }
```

### 2. `VOP3` 修饰位

要结构化：

- neg
- abs
- clamp
- omod

### 3. memory flags

要结构化：

- glc
- slc
- dlc
- offen
- idxen
- addr64

### 4. matrix/mfma selector

要结构化：

- instruction shape
- input packing
- accumulator type

## 代码生成目标

YAML 不应该在 runtime 直接读。

生成脚本应该输出：

### 1. format layout 表

例如：

- `generated_gcn_format_layouts.h`

### 2. opcode definition 表

例如：

- `generated_gcn_inst_defs.h`
- `generated_gcn_inst_defs.cpp`

### 3. mnemonic -> def lookup

给 disassembler / asm parser 用

### 4. `(format, opcode) -> def` lookup

给 binary decoder 用

### 5. operand decode recipe

给 decoder 用

### 6. semantic family lookup

给 execution handler dispatch 用

## 最小生成后 C++ 数据结构

建议目标结构：

```cpp
struct BitFieldRef {
  uint8_t word_index;
  uint8_t lsb;
  uint8_t width;
  bool sign_extend;
};

struct OperandSpec {
  const char* name;
  OperandKind kind;
  OperandRole role;
  uint16_t field_index;
};

struct GcnInstDef {
  uint32_t id;
  GcnProfileId profile;
  GcnInstFormatClass format_class;
  uint32_t opcode;
  uint8_t size_bytes;
  const char* mnemonic;
  SemanticFamily semantic_family;
  IssueFamily issue_family;
  uint64_t flags;
  uint16_t operand_begin;
  uint16_t operand_count;
};
```

## 后续实现建议

实现顺序建议：

1. 先把 `format_classes.yaml` 固化
2. 再把 `operand_kinds.yaml` 和 `semantic_families.yaml` 固化
3. 先迁移当前已支持 opcode 到 YAML
4. 写生成脚本输出 C++ 静态表
5. 用生成表替换 `gcn_inst_encoding_def.cpp` 的手写定义
6. 再补全剩余 ISA

## 结论

最合适的呈现方式是：

- **YAML 作为可维护源码**
- **按 format/opcode/profile 分层**
- **生成 C++ 静态表**

这样可以同时满足：

- 人工补 ISA
- 代码直接读取
- 后续生成 td / code file
- 最小化手工维护成本
