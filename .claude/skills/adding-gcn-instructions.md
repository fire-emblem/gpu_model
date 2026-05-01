# Adding GCN ISA Instructions

This skill guides the process of adding support for new GCN ISA instructions to the gpu_model project.

## Overview

Adding a new GCN instruction requires modifications in three areas:
1. **Encoding Definition** - How the instruction is decoded from binary
2. **Semantic Handler** - How the instruction is executed
3. **Registration** - Connecting the handler to the dispatcher

## Step-by-Step Process

### Step 1: Identify the Instruction

1. Find the instruction in the opcode table:
   ```bash
   grep -n "instruction_name" src/instruction/decode/encoded/internal/generated_encoded_gcn_full_opcode_table.cpp
   ```

2. Note the `GcnIsaOpType` (e.g., `Vop3a`, `Vop2`, `Sop1`) and opcode number.

### Step 2: Add Encoding Definition

Edit `src/instruction/decode/encoded/internal/encoded_gcn_encoding_def.cpp`:

1. **Add to `kEncodedGcnEncodingDefs` array:**
   ```cpp
   EncodedGcnEncodingDef{.id = <next_id>,
                         .format_class = EncodedGcnInstFormatClass::<format>,
                         .op = <opcode>,
                         .size_bytes = <4_or_8>,
                         .mnemonic = "instruction_name"},
   ```

2. **Add to `kDecoderOverrides` array (if needed):**
   ```cpp
   {"instruction_name", EncodedOperandDecoderKind::<decoder_kind>},
   ```

   Common decoder kinds:
   - `Vop3aGeneric` - Standard VOP3a with 4 operands (dst, src0, src1, src2)
   - `Vop2Generic` - Standard VOP2 with 3 operands
   - `Sop1`, `Sop2`, etc. - Scalar operation variants

### Step 3: Implement Semantic Handler

Edit `src/instruction/semantics/vector/vector_handlers.cpp` (for vector instructions)
or `src/instruction/semantics/scalar/scalar_handlers.cpp` (for scalar instructions):

1. **Add handler class:**
   ```cpp
   // instruction_name: dst = operation description
   class InstructionNameHandler final : public VectorLaneHandler<InstructionNameHandler> {
    public:
     void ExecuteLane(const DecodedInstruction& instruction,
                      EncodedWaveContext& context, uint32_t lane) const {
       const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
       const uint32_t src0 = static_cast<uint32_t>(
           ResolveVectorLane(instruction.operands.at(1), context, lane));
       // ... resolve other operands
       context.wave.vgpr.Write(vdst, lane, <computed_result>);
     }
   };
   ```

2. **Add static instance:**
   ```cpp
   static const InstructionNameHandler kInstructionNameHandler;
   ```

3. **Register the handler:**
   ```cpp
   registry.Register("instruction_name", &kInstructionNameHandler);
   ```

### Step 4: Build and Test

```bash
source scripts/setup_env.sh
cmake --build --preset dev-fast
./build-ninja/tests/gpu_model_tests --gtest_filter=*related_test*
```

## Example: Adding v_add_lshl_u32

This instruction computes: `dst = (src0 + src1) << src2`

### 1. Opcode table already had entry:
```cpp
GcnIsaOpcodeDescriptor{ GcnIsaOpType::Vop3a, 0x1fe, "v_add_lshl_u32", ... }
```

### 2. Added encoding definition:
```cpp
EncodedGcnEncodingDef{.id = 132,
                      .format_class = EncodedGcnInstFormatClass::Vop3a,
                      .op = 0x1fe,
                      .size_bytes = 8,
                      .mnemonic = "v_add_lshl_u32"},
```

### 3. Added decoder override:
```cpp
{"v_add_lshl_u32", EncodedOperandDecoderKind::Vop3aGeneric},
```

### 4. Added semantic handler:
```cpp
class VAddLshlU32Handler final : public VectorLaneHandler<VAddLshlU32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t src0 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const uint32_t src1 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    const uint32_t src2 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(3), context, lane));
    context.wave.vgpr.Write(vdst, lane, (src0 + src1) << (src2 & 31u));
  }
};

static const VAddLshlU32Handler kVAddLshlU32Handler;
// In RegisterVectorHandlers():
registry.Register("v_add_lshl_u32", &kVAddLshlU32Handler);
```

## Common Patterns

### VOP3a Instructions (4 operands)
- Template: `dst, src0, src1, src2`
- Use `Vop3aGeneric` decoder
- Access operands: `operands.at(0)` through `operands.at(3)`

### VOP2 Instructions (3 operands)
- Template: `dst, src0, src1`
- Use `Vop2Generic` decoder
- Often have `_e32` and `_e64` variants

### Scalar Instructions
- Use `ScalarHandler` base class
- Write to SGPR instead of VGPR
- No lane parameter needed

## Debugging

If instruction is not recognized:
1. Check opcode table entry exists
2. Check encoding definition matches format class and opcode
3. Check decoder override is correct
4. Verify handler is registered with exact mnemonic match

If operands are empty:
- The decoder override may be missing or wrong
- Check `EncodedOperandDecoderKind` matches instruction format
