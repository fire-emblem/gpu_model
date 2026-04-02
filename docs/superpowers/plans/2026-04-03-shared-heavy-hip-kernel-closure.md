# Shared-Heavy HIP Kernel Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the most valuable remaining shared-heavy encoded/raw-GCN support gaps by making both `dynamic_shared_sum` and `shared_reverse` real HIP `.out` paths stable, while tightening the associated decode/binding regressions.

**Architecture:** Drive implementation from real HIP kernels first. Use `dynamic_shared_sum` and `shared_reverse` as the two required acceptance anchors, then patch the minimum missing shared-heavy instruction chain across encoding defs, instruction binding, and semantic handling. Keep placeholder fallback only for genuinely unsupported families that are not on the execution-critical path of these kernels.

**Tech Stack:** C++20, gtest, existing encoded/raw-GCN path, `hipcc`, object reader / encoded instruction binding / semantic handler stack

---

## File Map

- Modify: `src/instruction/encoded/internal/encoded_gcn_encoding_def.cpp`
  - Own missing encoding defs / match coverage for shared-heavy opcodes encountered in real HIP kernels.
- Modify: `src/instruction/encoded/internal/encoded_instruction_binding.cpp`
  - Own concrete object binding or placeholder fallback behavior.
- Modify: `src/execution/encoded_semantic_handler.cpp`
  - Own minimal execution semantics for newly required shared-heavy instructions.
- Modify: `tests/loader/amdgpu_code_object_decoder_test.cpp`
  - Own decode-level regressions when needed.
- Modify: `tests/instruction/instruction_array_parser_test.cpp`
  - Own parser/object placeholder regressions when needed.
- Modify: `tests/instruction/encoded_instruction_binding_test.cpp`
  - Own binding/placeholder behavior regressions.
- Modify: `tests/runtime/hip_runtime_test.cpp`
  - Own the real HIP `dynamic_shared_sum` raw-GCN acceptance case.

## Task 1: Lock the current shared-heavy acceptance anchor set

**Files:**
- Modify: `tests/runtime/hip_runtime_test.cpp`
- Modify: `tests/instruction/instruction_array_parser_test.cpp`
- Modify: `tests/instruction/encoded_instruction_binding_test.cpp`

- [ ] **Step 1: Add or refresh a narrow regression around the dynamic shared path**

In `tests/runtime/hip_runtime_test.cpp`, keep the current `HipRuntimeTest.LaunchesHipDynamicSharedExecutableInRawGcnPath` as the primary acceptance anchor and, if needed, add a nearby assertion that makes the path dependency explicit:

```cpp
EXPECT_TRUE(image.metadata.values.contains("hidden_arg_layout"));
EXPECT_NE(image.metadata.values.at("hidden_arg_layout").find("hidden_dynamic_lds_size"),
          std::string::npos);
```

If not already present, add a check that the output remains the expected reduction:

```cpp
EXPECT_EQ(output, static_cast<int32_t>(block_dim * (block_dim + 1) / 2));
```

- [ ] **Step 2: Add a failing decode/binding regression for the currently missing shared-heavy chain**

In `tests/instruction/instruction_array_parser_test.cpp` or `tests/instruction/encoded_instruction_binding_test.cpp`, add one minimal regression that locks the current failing shared-heavy path component once identified. Example shape:

```cpp
TEST(EncodedInstructionBindingTest, BindsConcreteObjectForSharedHeavyOpcodeChain) {
  auto object = BindEncodedInstructionObject(
      MakeDecoded({/* real words */}, EncodedGcnInstFormatClass::Ds, "ds_read2_b32"));
  ASSERT_NE(object, nullptr);
  EXPECT_EQ(object->op_type_name(), "ds");
}
```

Use the concrete opcode discovered from the current failing real HIP case, not a guessed placeholder.

- [ ] **Step 3: Run the minimal failing acceptance set**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.LaunchesHipDynamicSharedExecutableInRawGcnPath:InstructionArrayParserTest.*:EncodedInstructionBindingTest.*'
```

Expected:

- At least one test fails on the currently missing shared-heavy instruction chain

- [ ] **Step 4: Confirm `shared_reverse` still stays green as the second anchor**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.LaunchesHipSharedReverseExecutableInRawGcnPath:HipRuntimeTest.LaunchesHipSharedReverseExecutableAndValidatesOutput'
```

Expected:

- PASS

- [ ] **Step 5: Commit the regression-only slice**

```bash
git add tests/runtime/hip_runtime_test.cpp tests/instruction/instruction_array_parser_test.cpp tests/instruction/encoded_instruction_binding_test.cpp
git commit -m "test: lock shared-heavy hip kernel acceptance anchors"
```

## Task 2: Patch the missing encoding matches for the shared-heavy chain

**Files:**
- Modify: `src/instruction/encoded/internal/encoded_gcn_encoding_def.cpp`
- Test: `tests/instruction/instruction_array_parser_test.cpp`
- Test: `tests/instruction/encoded_instruction_binding_test.cpp`

- [ ] **Step 1: Add the failing match-coverage test for the exact opcode chain**

Use the exact opcode(s) discovered from the `dynamic_shared_sum` disassembly and lock them in a decode/binding test before patching. Example:

```cpp
TEST(InstructionArrayParserTest, CreatesConcreteObjectForDynamicSharedOpcodeChain) {
  std::vector<DecodedInstruction> decoded;
  decoded.push_back(MakeDecoded({0x...u}, EncodedGcnInstFormatClass::Ds, "ds_read2_b32"));
  auto objects = InstructionArrayParser::Parse(decoded);
  ASSERT_EQ(objects.size(), 1u);
  EXPECT_EQ(objects[0]->op_type_name(), "ds");
}
```

- [ ] **Step 2: Run the focused decode/binding tests and confirm failure**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='InstructionArrayParserTest.*:EncodedInstructionBindingTest.*'
```

Expected:

- FAIL due to missing match record or missing concrete binding for the exact shared-heavy opcode

- [ ] **Step 3: Add the minimum encoding defs / decoder overrides**

In `src/instruction/encoded/internal/encoded_gcn_encoding_def.cpp`, add only the missing defs actually required by the real HIP path. Follow the existing manual-definition style:

```cpp
EncodedGcnEncodingDef{.id = <new_id>,
                      .format_class = EncodedGcnInstFormatClass::<...>,
                      .op = <opcode>,
                      .size_bytes = <4 or 8>,
                      .mnemonic = "<mnemonic>"},
```

If the mnemonic needs a specific operand decoder, extend the override table:

```cpp
{"<mnemonic>", EncodedOperandDecoderKind::<Kind>},
```

- [ ] **Step 4: Re-run the focused decode/binding tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='InstructionArrayParserTest.*:EncodedInstructionBindingTest.*'
```

Expected:

- PASS, or the failure shifts from “missing match record” to “unsupported raw GCN opcode”, which is the intended next layer

- [ ] **Step 5: Commit the match-coverage slice**

```bash
git add src/instruction/encoded/internal/encoded_gcn_encoding_def.cpp tests/instruction/instruction_array_parser_test.cpp tests/instruction/encoded_instruction_binding_test.cpp
git commit -m "feat: add shared-heavy encoded match coverage"
```

## Task 3: Patch binding and semantic handling for execution-critical shared-heavy opcodes

**Files:**
- Modify: `src/instruction/encoded/internal/encoded_instruction_binding.cpp`
- Modify: `src/execution/encoded_semantic_handler.cpp`
- Test: `tests/runtime/hip_runtime_test.cpp`

- [ ] **Step 1: Add a failing real-kernel acceptance rerun**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.LaunchesHipDynamicSharedExecutableInRawGcnPath'
```

Expected:

- FAIL with the next concrete blocker, such as:
  - missing instruction factory
  - unsupported raw GCN opcode

- [ ] **Step 2: Add the minimal binding support**

In `src/instruction/encoded/internal/encoded_instruction_binding.cpp`, add only the concrete classes/factory entries needed by the current blocker:

```cpp
DEFINE_RAW_GCN_OPCODE_CLASS(<ClassName>, <BaseClass>, "<mnemonic>");
...
{"<mnemonic>", &MakeInstruction<<ClassName>>},
```

If the opcode is execution-critical in the real HIP path, do not route it to placeholder fallback.

- [ ] **Step 3: Add the minimal semantic handling**

In `src/execution/encoded_semantic_handler.cpp`, add the smallest semantics necessary for correct execution of the new opcode in the real kernel path.

Rules:

- if the instruction genuinely affects correctness in the dynamic shared path, implement real semantics
- if it is only decode-visible and not correctness-critical in this path, a temporary no-op is acceptable only if the real kernel behavior remains correct and tests prove it

- [ ] **Step 4: Re-run the two real shared-heavy anchors**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.LaunchesHipDynamicSharedExecutableInRawGcnPath:HipRuntimeTest.LaunchesHipSharedReverseExecutableInRawGcnPath:HipRuntimeTest.LaunchesHipSharedReverseExecutableAndValidatesOutput'
```

Expected:

- All three PASS

- [ ] **Step 5: Commit the binding/semantic slice**

```bash
git add src/instruction/encoded/internal/encoded_instruction_binding.cpp src/execution/encoded_semantic_handler.cpp tests/runtime/hip_runtime_test.cpp
git commit -m "feat: support shared-heavy hip kernels in encoded path"
```

## Task 4: Expand shared-heavy runtime/decode verification just enough to avoid one-off fixes

**Files:**
- Modify: `tests/loader/amdgpu_code_object_decoder_test.cpp`
- Modify: `tests/instruction/instruction_array_parser_test.cpp`
- Modify: `tests/instruction/encoded_instruction_binding_test.cpp`

- [ ] **Step 1: Add one more targeted regression for the discovered shared-heavy chain**

Do not add a broad suite. Add one regression that locks the newly supported chain under decode or parser coverage. Example:

```cpp
TEST(AmdgpuCodeObjectDecoderTest, DecodesDynamicSharedExecutableWithoutUnknownInstructions) {
  // Build the dynamic shared HIP artifact and assert unknown_count == 0
}
```

Only add this if it materially protects against regressing the just-fixed path.

- [ ] **Step 2: Run the focused decode/runtime ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='AmdgpuCodeObjectDecoderTest.*:InstructionArrayParserTest.*:EncodedInstructionBindingTest.*:HipRuntimeTest.LaunchesHipDynamicSharedExecutableInRawGcnPath'
```

Expected:

- PASS

- [ ] **Step 3: Check that placeholder behavior for unsupported families still holds**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='InstructionArrayParserTest.CreatesConcreteAndPlaceholderInstructionObjects:EncodedInstructionBindingTest.BindsPlaceholderForRecognizedButUnsupportedFamilies:EncodedInstructionBindingTest.BindsUnknownPlaceholderForUnrecognizedWords'
```

Expected:

- PASS

- [ ] **Step 4: Commit the expanded verification slice**

```bash
git add tests/loader/amdgpu_code_object_decoder_test.cpp tests/instruction/instruction_array_parser_test.cpp tests/instruction/encoded_instruction_binding_test.cpp
git commit -m "test: expand shared-heavy encoded verification"
```

- [ ] **Step 5: Record the two-anchor acceptance result in handoff notes**

Use this exact text in the handoff:

```text
Shared-heavy anchors:
- dynamic_shared_sum raw-GCN path PASS
- shared_reverse raw-GCN path PASS
```

## Task 5: Final regression ring and status sync

**Files:**
- Modify: `docs/module-development-status.md`

- [ ] **Step 1: Run the shared-heavy affected ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.LaunchesHipDynamicSharedExecutableInRawGcnPath:HipRuntimeTest.LaunchesHipSharedReverseExecutableInRawGcnPath:HipRuntimeTest.LaunchesHipSharedReverseExecutableAndValidatesOutput:AmdgpuCodeObjectDecoderTest.*:InstructionArrayParserTest.*:EncodedInstructionBindingTest.*'
```

Expected:

- PASS

- [ ] **Step 2: Run the next-larger runtime/decode ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.*RawGcnPath*:AmdgpuCodeObjectDecoderTest.*:InstructionArrayParserTest.*:EncodedInstructionBindingTest.*'
```

Expected:

- PASS

- [ ] **Step 3: Update the status board**

In `docs/module-development-status.md`, update `M3/M4`-adjacent wording to reflect the newly closed shared-heavy path. Example direction:

```md
| `M3` | ... | ...；shared-heavy真实 HIP case 的 decode/binding 缺口已继续收口到 `dynamic_shared_sum + shared_reverse` 双锚点 | ... |
| `M4` | ... | ...；shared-heavy真实 HIP执行主线不再只依赖单一 case，raw-GCN 路径对 `dynamic shared + ds_read2 + barrier` 已有真实程序回归 | ... |
```

- [ ] **Step 4: Run full project regression**

Run:

```bash
./build-ninja/tests/gpu_model_tests
```

Expected:

- PASS

- [ ] **Step 5: Commit the status sync**

```bash
git add docs/module-development-status.md
git commit -m "docs: record shared-heavy hip kernel closure progress"
```

## Self-Review

- Spec coverage:
  - `dynamic_shared_sum` and `shared_reverse` both used as acceptance anchors: Tasks 1, 3, 5
  - real HIP kernels drive the work, not broad family abstraction: Tasks 1-4
  - placeholder vs concrete object boundary preserved: Tasks 2-4
  - no weakening of existing real HIP output assertions: Tasks 1, 3, 5
- Placeholder scan:
  - No `TODO` / `TBD` placeholders remain
  - Every step has exact files, commands, and acceptance outcomes
- Type consistency:
  - Plan consistently uses `dynamic_shared_sum`, `shared_reverse`, encoded match/binding/semantic layers, and the existing runtime/decode tests
