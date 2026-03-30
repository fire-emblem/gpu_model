# Runtime/Program Phase 2 Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the legacy runtime/program framework surface and leave only the new `HipRuntime / ModelRuntime / RuntimeEngine / ProgramObject / ExecutableKernel / EncodedProgramObject / ObjectReader / ExecutionRoute` architecture.

**Architecture:** This cleanup is a hard cut, not another compatibility phase. First make the new public headers the only declaration source, then replace all runtime/program includes and type names, then rename implementation files and remove legacy headers and sources, and finally clean tests and docs so no legacy public names remain.

**Tech Stack:** C++20, CMake, gtest, existing `gpu_model_tests`, repository docs under `docs/`

---

## File Structure

### New source-of-truth files that must remain
- `include/gpu_model/runtime/hip_runtime.h`
- `include/gpu_model/runtime/model_runtime.h`
- `include/gpu_model/runtime/runtime_engine.h`
- `include/gpu_model/program/program_object.h`
- `include/gpu_model/program/executable_kernel.h`
- `include/gpu_model/program/encoded_program_object.h`
- `include/gpu_model/program/object_reader.h`
- `include/gpu_model/program/program_execution_route.h`
- `include/gpu_model/program/execution_route.h`
- `src/runtime/hip_runtime.cpp`
- `src/runtime/runtime_engine.cpp`
- `src/program/object_reader.cpp`
- `src/program/execution_route.cpp`

### Legacy files that must be deleted by the end
- `include/gpu_model/runtime/model_runtime_api.h`
- `include/gpu_model/runtime/runtime_hooks.h`
- `include/gpu_model/runtime/host_runtime.h`
- `include/gpu_model/runtime/program_execution.h`
- `include/gpu_model/isa/program_image.h`
- `include/gpu_model/isa/kernel_program.h`
- `src/runtime/runtime_hooks.cpp`
- `src/runtime/host_runtime.cpp`
- `src/runtime/program_execution.cpp`
- `include/gpu_model/loader/program_file_loader.h`
- `include/gpu_model/loader/amdgpu_obj_loader.h`
- `include/gpu_model/loader/amdgpu_code_object_decoder.h`
- `src/loader/program_file_loader.cpp`
- `src/loader/amdgpu_obj_loader.cpp`
- `src/loader/amdgpu_code_object_decoder.cpp`

---

### Task 1: Make New Runtime/Program Headers the Only Declaration Source

**Files:**
- Modify: `include/gpu_model/runtime/hip_runtime.h`
- Modify: `include/gpu_model/runtime/model_runtime.h`
- Modify: `include/gpu_model/runtime/runtime_engine.h`
- Modify: `include/gpu_model/program/program_object.h`
- Modify: `include/gpu_model/program/executable_kernel.h`
- Modify: `include/gpu_model/program/encoded_program_object.h`
- Modify: `include/gpu_model/program/object_reader.h`
- Modify: `include/gpu_model/program/execution_route.h`
- Modify: `include/gpu_model/program/program_execution_route.h`
- Modify: `tests/runtime/compatibility_alias_test.cpp`
- Test: `tests/runtime/runtime_naming_test.cpp`
- Test: `tests/program/program_object_types_test.cpp`

- [ ] **Step 1: Write failing tests that stop referencing legacy public names**

```cpp
// tests/runtime/runtime_naming_test.cpp
#include "gpu_model/runtime/hip_runtime.h"
#include "gpu_model/runtime/model_runtime.h"
#include "gpu_model/runtime/runtime_engine.h"

static_assert(std::is_class_v<ModelRuntime>);
static_assert(std::is_class_v<HipRuntime>);
static_assert(std::is_class_v<RuntimeEngine>);
```

```cpp
// tests/program/program_object_types_test.cpp
#include "gpu_model/program/program_object.h"
#include "gpu_model/program/executable_kernel.h"
#include "gpu_model/program/encoded_program_object.h"

static_assert(std::is_class_v<ProgramObject>);
static_assert(std::is_class_v<ExecutableKernel>);
static_assert(std::is_class_v<EncodedProgramObject>);
```

- [ ] **Step 2: Run tests to verify they fail if new headers still depend on deleted legacy declarations**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='*RuntimeNamingTest*:*ProgramNamingTest*'`
Expected: FAIL once the new tests stop accepting legacy aliases.

- [ ] **Step 3: Move actual declarations into new headers**

```cpp
// include/gpu_model/program/program_object.h
namespace gpu_model {

class ProgramObject {
 public:
  ProgramObject() = default;
  ProgramObject(std::string kernel_name,
                std::string assembly_text,
                MetadataBlob metadata = {},
                ConstSegment const_segment = {},
                RawDataSegment raw_data_segment = {});

  const std::string& kernel_name() const;
  const std::string& assembly_text() const;
  const MetadataBlob& metadata() const;
  const ConstSegment& const_segment() const;
  const RawDataSegment& raw_data_segment() const;

 private:
  std::string kernel_name_;
  std::string assembly_text_;
  MetadataBlob metadata_;
  ConstSegment const_segment_;
  RawDataSegment raw_data_segment_;
};

}  // namespace gpu_model
```

```cpp
// include/gpu_model/program/executable_kernel.h
namespace gpu_model {

class ExecutableKernel {
 public:
  ExecutableKernel() = default;
  ExecutableKernel(std::string name,
                   std::vector<Instruction> instructions,
                   std::unordered_map<std::string, uint64_t> labels,
                   MetadataBlob metadata = {},
                   ConstSegment const_segment = {});

  const std::string& name() const;
  const std::vector<Instruction>& instructions() const;
  const MetadataBlob& metadata() const;
  const ConstSegment& const_segment() const;
  uint64_t ResolveLabel(std::string_view label) const;
};

}  // namespace gpu_model
```

- [ ] **Step 4: Remove alias-based expectations from compatibility tests**

```cpp
// tests/runtime/compatibility_alias_test.cpp
// delete old/new alias equivalence asserts for runtime/program names in this slice
// keep only compatibility checks for slices not yet in Phase 2 scope
```

- [ ] **Step 5: Run targeted tests**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='*RuntimeNamingTest*:*ProgramNamingTest*:*CompatibilityAliasTest*'`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add include/gpu_model/runtime include/gpu_model/program tests/runtime tests/program
git commit -m "refactor: promote runtime and program headers to source of truth"
```

### Task 2: Replace Runtime/Program Public Includes and Type Names Everywhere in Scope

**Files:**
- Modify: `src/runtime/hip_interposer_state.cpp`
- Modify: `src/runtime/runtime_engine.cpp` (from renamed file in Task 3 if already done, otherwise current runtime file)
- Modify: `src/runtime/hip_runtime.cpp`
- Modify: `src/program/execution_route.cpp`
- Modify: runtime/program related tests under `tests/runtime/*` and `tests/program/*`

- [ ] **Step 1: Write a failing grep gate for legacy names in runtime/program slice**

Run: `rg -n "ModelRuntimeApi|RuntimeHooks|HostRuntime|ProgramImage|KernelProgram|AmdgpuCodeObjectImage" include/gpu_model/runtime include/gpu_model/program src/runtime src/program tests/runtime tests/program`
Expected: matches found before cleanup.

- [ ] **Step 2: Replace includes and type names with new names**

```cpp
// before
#include "gpu_model/runtime/runtime_hooks.h"
RuntimeHooks hooks;
ProgramImage image;
KernelProgram kernel;

// after
#include "gpu_model/runtime/hip_runtime.h"
HipRuntime hooks;
ProgramObject object;
ExecutableKernel kernel;
```

- [ ] **Step 3: Run the grep gate again**

Run: `rg -n "ModelRuntimeApi|RuntimeHooks|HostRuntime|ProgramImage|KernelProgram|AmdgpuCodeObjectImage" include/gpu_model/runtime include/gpu_model/program src/runtime src/program tests/runtime tests/program`
Expected: no matches in this slice.

- [ ] **Step 4: Run runtime/program targeted tests**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='*Runtime*:*Program*'`
Expected: PASS or only skips caused by environment, with no compile failures.

- [ ] **Step 5: Commit**

```bash
git add src/runtime src/program tests/runtime tests/program
git commit -m "refactor: replace legacy runtime-program names in code and tests"
```

### Task 3: Rename Runtime and Program Implementation Files

**Files:**
- Move: `src/runtime/runtime_hooks.cpp` -> `src/runtime/hip_runtime.cpp`
- Move: `src/runtime/host_runtime.cpp` -> `src/runtime/runtime_engine.cpp`
- Move: `src/runtime/program_execution.cpp` -> `src/program/execution_route.cpp`
- Move: `src/loader/program_file_loader.cpp` -> `src/program/object_reader.cpp`
- Modify: top-level `CMakeLists.txt`

- [ ] **Step 1: Move source files to new paths**

```bash
mv src/runtime/runtime_hooks.cpp src/runtime/hip_runtime.cpp
mv src/runtime/host_runtime.cpp src/runtime/runtime_engine.cpp
mv src/runtime/program_execution.cpp src/program/execution_route.cpp
mv src/loader/program_file_loader.cpp src/program/object_reader.cpp
```

- [ ] **Step 2: Update CMake source lists**

```cmake
# CMakeLists.txt
src/runtime/hip_runtime.cpp
src/runtime/runtime_engine.cpp
src/program/execution_route.cpp
src/program/object_reader.cpp
```

- [ ] **Step 3: Build to verify no stale source paths remain**

Run: `cmake --build build-ninja --target gpu_model_tests`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt src/runtime src/program
git commit -m "refactor: rename runtime-program implementation files"
```

### Task 4: Re-home Loader-Owned Runtime/Program Entry Points

**Files:**
- Move/replace: `include/gpu_model/loader/program_file_loader.h`
- Move/replace: `include/gpu_model/loader/amdgpu_obj_loader.h`
- Move/replace: `include/gpu_model/loader/amdgpu_code_object_decoder.h`
- Move/replace corresponding `src/loader/*.cpp`
- Modify: `include/gpu_model/program/object_reader.h`
- Modify: `include/gpu_model/program/encoded_program_object.h`

- [ ] **Step 1: Promote object reader declarations into `program/`**

```cpp
// include/gpu_model/program/object_reader.h
namespace gpu_model {

class ObjectReader {
 public:
  ProgramObject LoadFromStem(const std::filesystem::path& stem) const;
  ProgramObject LoadFromObject(const std::filesystem::path& path,
                               std::optional<std::string> kernel_name = std::nullopt) const;
  EncodedProgramObject DecodeEncodedObject(const std::filesystem::path& path,
                                           std::optional<std::string> kernel_name = std::nullopt) const;
};

}  // namespace gpu_model
```

- [ ] **Step 2: Update implementation to match `ObjectReader` ownership**

```cpp
// src/program/object_reader.cpp
ProgramObject ObjectReader::LoadFromStem(const std::filesystem::path& stem) const { ... }
ProgramObject ObjectReader::LoadFromObject(const std::filesystem::path& path,
                                           std::optional<std::string> kernel_name) const { ... }
EncodedProgramObject ObjectReader::DecodeEncodedObject(const std::filesystem::path& path,
                                                       std::optional<std::string> kernel_name) const { ... }
```

- [ ] **Step 3: Delete loader headers/sources in this slice**

Run: `rm -f include/gpu_model/loader/program_file_loader.h include/gpu_model/loader/amdgpu_obj_loader.h include/gpu_model/loader/amdgpu_code_object_decoder.h src/loader/program_file_loader.cpp src/loader/amdgpu_obj_loader.cpp src/loader/amdgpu_code_object_decoder.cpp`

- [ ] **Step 4: Run object/program focused tests**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='*Program*:*ObjectReader*:*EncodedProgramObject*:*LoadModule*'`
Expected: PASS or environment skips only.

- [ ] **Step 5: Commit**

```bash
git add include/gpu_model/program src/program include/gpu_model/loader src/loader tests/program tests/runtime
 git commit -m "refactor: rehome loader runtime-program entry points"
```

### Task 5: Delete Legacy Public Headers and Runtime/Program Compatibility Surface

**Files:**
- Delete: legacy headers listed in the spec
- Modify: all remaining includes that still reference them

- [ ] **Step 1: Delete legacy headers**

```bash
rm -f include/gpu_model/runtime/model_runtime_api.h \
      include/gpu_model/runtime/runtime_hooks.h \
      include/gpu_model/runtime/host_runtime.h \
      include/gpu_model/runtime/program_execution.h \
      include/gpu_model/isa/program_image.h \
      include/gpu_model/isa/kernel_program.h
```

- [ ] **Step 2: Verify no includes remain**

Run: `rg -n "model_runtime_api.h|runtime_hooks.h|host_runtime.h|program_execution.h|isa/program_image.h|isa/kernel_program.h" include src tests`
Expected: no matches.

- [ ] **Step 3: Run targeted build and tests**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='*Runtime*:*Program*'`
Expected: PASS or environment skips only.

- [ ] **Step 4: Commit**

```bash
git add include/gpu_model/runtime include/gpu_model/program include/gpu_model/isa tests src
 git commit -m "refactor: delete legacy runtime-program headers"
```

### Task 6: Remove Runtime/Program Legacy Names from Docs and Tests

**Files:**
- Modify: `README.md`
- Modify: `docs/runtime-layering.md`
- Modify: `docs/module-development-status.md`
- Modify: runtime/program test names if any legacy names remain

- [ ] **Step 1: Scan for forbidden names**

Run: `rg -n "ModelRuntimeApi|RuntimeHooks|HostRuntime|ProgramImage|KernelProgram|AmdgpuCodeObjectImage" README.md docs/runtime-layering.md docs/module-development-status.md tests/runtime tests/program`
Expected: matches found before cleanup.

- [ ] **Step 2: Rewrite docs and test names to remove legacy references**

```md
- remove all “legacy/compatibility alias” notes for removed names
- describe only `HipRuntime / ModelRuntime / RuntimeEngine`
- describe only `ProgramObject / ExecutableKernel / EncodedProgramObject`
```

- [ ] **Step 3: Re-run forbidden-name scan**

Run: `rg -n "ModelRuntimeApi|RuntimeHooks|HostRuntime|ProgramImage|KernelProgram|AmdgpuCodeObjectImage" README.md docs/runtime-layering.md docs/module-development-status.md tests/runtime tests/program`
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/runtime-layering.md docs/module-development-status.md tests/runtime tests/program
 git commit -m "docs: remove legacy runtime-program terminology"
```

### Task 7: Clear Deletion Markers and Final Verification

**Files:**
- Any file carrying phase2 runtime-program delete markers

- [ ] **Step 1: Scan deletion markers**

Run: `rg -n "PHASE2-DELETE\(runtime-program\)" include src tests docs`
Expected: zero matches by the end of this task.

- [ ] **Step 2: Run final runtime/program verification set**

Run: `cmake --build build-ninja --target gpu_model_tests`
Expected: PASS

Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='*Runtime*:*Program*:*CompatibilityAliasTest*'`
Expected: compatibility alias test should be removed or updated for the slices that remain in Phase 1; runtime/program tests PASS or skip only for environment.

Run: `git status --short`
Expected: no unstaged changes.

- [ ] **Step 3: Commit**

```bash
git add -A
 git commit -m "refactor: complete runtime-program phase2 cleanup"
```

## Self-Review

### Spec coverage
This plan covers public API cleanup, implementation-file renames, loader-to-program re-homing, test cleanup, doc cleanup, deletion markers, and final verification.

### Placeholder scan
No `TODO`, `TBD`, or “similar to previous task” placeholders are used.

### Type consistency
The plan consistently treats the final public names as:
- `HipRuntime`
- `ModelRuntime`
- `RuntimeEngine`
- `ProgramObject`
- `ExecutableKernel`
- `EncodedProgramObject`
- `ObjectReader`
- `ExecutionRoute`
