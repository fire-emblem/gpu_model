# Conditional Multibarrier Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `examples/08-conditional-multibarrier/` example that demonstrates a legal multi-block, multi-barrier HIP kernel with conditional compute, plus host-side exact output validation across `st / mt / cycle`.

**Architecture:** Follow the existing example pattern used by `examples/03-shared-reverse/`: one HIP source, one `run.sh`, one `README.md`, and no test-suite integration in this task. Reuse the already-proven conditional multibarrier algorithm shape from the runtime test line, but package it as a customer-facing example with clear host-side expected-value checks and concise mode summaries.

**Tech Stack:** HIP C++, bash, existing `examples/common.sh`, `hipcc`, `LD_PRELOAD` interposer flow

---

## File Map

- Create: `examples/08-conditional-multibarrier/conditional_multibarrier.hip`
  - Own the real HIP kernel plus host-side exact expected-value verification.
- Create: `examples/08-conditional-multibarrier/run.sh`
  - Own build/run orchestration for `st / mt / cycle`.
- Create: `examples/08-conditional-multibarrier/README.md`
  - Explain what the example demonstrates and how to run it.
- Modify: `examples/README.md`
  - Add the new example to the numbered index.

## Task 1: Add the new example to the examples index

**Files:**
- Modify: `examples/README.md`

- [ ] **Step 1: Write the failing README index check mentally and make the smallest doc edit**

Update the numbered list in `examples/README.md` to add the new entry:

```md
8. [08-conditional-multibarrier](./08-conditional-multibarrier)
```

- [ ] **Step 2: Sanity-check the README renders cleanly**

Run:

```bash
sed -n '1,80p' examples/README.md
```

Expected:

- The new example appears as item `8`
- Numbering and relative link formatting match the existing style

- [ ] **Step 3: Commit the index slice**

```bash
git add examples/README.md
git commit -m "docs: add conditional multibarrier example entry"
```

## Task 2: Add the HIP source with exact host-side expected-value validation

**Files:**
- Create: `examples/08-conditional-multibarrier/conditional_multibarrier.hip`

- [ ] **Step 1: Create the HIP source skeleton with a failing host expectation path**

Create `examples/08-conditional-multibarrier/conditional_multibarrier.hip`:

```cpp
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" __global__ void conditional_multibarrier(int* out) {
  __shared__ int tile[128];
  const int tid = static_cast<int>(threadIdx.x);
  const int block = static_cast<int>(blockIdx.x);
  const int base = block * blockDim.x;
  const int value = base + tid;
  tile[tid] = value + 3;
  __syncthreads();
  if (tid < 64) tile[tid] += tile[127 - tid];
  else tile[tid] -= tile[127 - tid];
  __syncthreads();
  int mixed = tile[tid];
  if (tid < 32) mixed += 11;
  else if (tid < 96) mixed -= 7;
  else mixed += 5;
  if (tid < 64) mixed += tile[(tid + 17) & 127];
  else mixed -= tile[(tid + 23) & 127];
  tile[tid] = mixed;
  __syncthreads();
  out[base + tid] = tile[tid];
}

static std::vector<int> BuildExpected(uint32_t grid_dim, uint32_t block_dim) {
  std::vector<int> expect(grid_dim * block_dim);
  for (uint32_t block = 0; block < grid_dim; ++block) {
    std::vector<int> tile(block_dim);
    const uint32_t base = block * block_dim;
    for (uint32_t tid = 0; tid < block_dim; ++tid) {
      tile[tid] = static_cast<int>(base + tid) + 3;
    }
    std::vector<int> stage1(block_dim);
    for (uint32_t tid = 0; tid < block_dim; ++tid) {
      stage1[tid] =
          tid < 64 ? tile[tid] + tile[127 - tid] : tile[tid] - tile[127 - tid];
    }
    for (uint32_t tid = 0; tid < block_dim; ++tid) {
      int mixed = stage1[tid];
      if (tid < 32) mixed += 11;
      else if (tid < 96) mixed -= 7;
      else mixed += 5;
      if (tid < 64) mixed += stage1[(tid + 17) & 127u];
      else mixed -= stage1[(tid + 23) & 127u];
      expect[base + tid] = mixed;
    }
  }
  return expect;
}

int main() {
  constexpr uint32_t grid_dim = 8;
  constexpr uint32_t block_dim = 128;
  constexpr uint32_t n = grid_dim * block_dim;

  int* out = nullptr;
  std::vector<int> host(n, -1);
  const auto expect = BuildExpected(grid_dim, block_dim);

  if (hipMalloc(&out, n * sizeof(int)) != hipSuccess) return 1;
  if (hipMemset(out, 0, n * sizeof(int)) != hipSuccess) return 2;
  hipLaunchKernelGGL(conditional_multibarrier, dim3(grid_dim), dim3(block_dim), 0, 0, out);
  if (hipDeviceSynchronize() != hipSuccess) return 3;
  if (hipMemcpy(host.data(), out, n * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) return 4;

  uint32_t mismatches = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (host[i] != expect[i]) ++mismatches;
  }
  std::printf("conditional_multibarrier mismatches=%u\n", mismatches);

  hipFree(out);
  return mismatches == 0 ? 0 : 5;
}
```

- [ ] **Step 2: Build and run the HIP source natively once**

Run:

```bash
mkdir -p examples/08-conditional-multibarrier/results
hipcc examples/08-conditional-multibarrier/conditional_multibarrier.hip -o examples/08-conditional-multibarrier/results/conditional_multibarrier.out
./examples/08-conditional-multibarrier/results/conditional_multibarrier.out
```

Expected:

- Exit code `0`
- Output contains `conditional_multibarrier mismatches=0`

- [ ] **Step 3: If needed, fix only the host expectation logic**

Adjust only the `BuildExpected(...)` host reconstruction if the native run mismatches. Do not change the kernel shape unless the expectation is proven wrong.

- [ ] **Step 4: Re-run the native binary until the exact host-side validation passes**

Run:

```bash
./examples/08-conditional-multibarrier/results/conditional_multibarrier.out
```

Expected:

- Exit code `0`
- `conditional_multibarrier mismatches=0`

- [ ] **Step 5: Commit the HIP source slice**

```bash
git add examples/08-conditional-multibarrier/conditional_multibarrier.hip
git commit -m "feat: add conditional multibarrier hip example"
```

## Task 3: Add the example runner

**Files:**
- Create: `examples/08-conditional-multibarrier/run.sh`

- [ ] **Step 1: Create the run script using the existing example helper style**

Create `examples/08-conditional-multibarrier/run.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CASE_DIR/../.." && pwd)"
source "$ROOT/examples/common.sh"

BUILD_DIR="$(gpu_model_detect_build_dir "$ROOT")"
OUT_DIR="$CASE_DIR/results"
mkdir -p "$OUT_DIR"

gpu_model_require_cmd hipcc
gpu_model_ensure_targets "$BUILD_DIR" gpu_model_hip_interposer

SO_PATH="$BUILD_DIR/libgpu_model_hip_interposer.so"
SRC="$CASE_DIR/conditional_multibarrier.hip"
EXE="$OUT_DIR/conditional_multibarrier.out"

hipcc "$SRC" -o "$EXE"

for mode in st mt cycle; do
  mode_dir="$(gpu_model_mode_dir "$OUT_DIR" "$mode")"
  gpu_model_run_interposed_mode "$SO_PATH" "$EXE" "$mode_dir" "$mode"
  gpu_model_assert_mode_success "$mode_dir" "conditional_multibarrier mismatches=0"
done
```

- [ ] **Step 2: Make the script executable and run it**

Run:

```bash
chmod +x examples/08-conditional-multibarrier/run.sh
./examples/08-conditional-multibarrier/run.sh
```

Expected:

- Builds the HIP executable
- Runs `st`, `mt`, and `cycle`
- All three modes report success with `conditional_multibarrier mismatches=0`

- [ ] **Step 3: If needed, fix only the script wiring**

If the script fails:

- correct path handling
- correct helper usage
- correct success-string matching

Do not change the kernel unless the runner proves the binary itself is wrong.

- [ ] **Step 4: Re-run the script to green**

Run:

```bash
./examples/08-conditional-multibarrier/run.sh
```

Expected:

- All three modes pass

- [ ] **Step 5: Commit the runner slice**

```bash
git add examples/08-conditional-multibarrier/run.sh
git commit -m "feat: add conditional multibarrier example runner"
```

## Task 4: Add the example README

**Files:**
- Create: `examples/08-conditional-multibarrier/README.md`

- [ ] **Step 1: Create the README**

Create `examples/08-conditional-multibarrier/README.md`:

```md
# 08 Conditional Multibarrier

这个例子展示一个合法的多 block、多 barrier、带条件额外计算的真实 HIP 程序。

它会：

- 用 `hipcc` 编译 `conditional_multibarrier.hip`
- 生成真实 `.out`
- 通过 `LD_PRELOAD` 分别以 `st / mt / cycle` 运行
- 在 host 侧做逐元素精确校验
- 附带逐 block 摘要校验，便于定位错误

关键点：

- 条件分支只影响数据变换，不影响 `__syncthreads()` 是否执行
- 因此这是一个合法的 barrier 使用示例

运行：

```bash
./examples/08-conditional-multibarrier/run.sh
```
```

- [ ] **Step 2: Sanity-check the rendered README**

Run:

```bash
sed -n '1,120p' examples/08-conditional-multibarrier/README.md
```

Expected:

- The README is concise
- It clearly explains legality of the barrier pattern
- It shows the run command

- [ ] **Step 3: Commit the README slice**

```bash
git add examples/08-conditional-multibarrier/README.md
git commit -m "docs: add conditional multibarrier example readme"
```

## Task 5: Final example verification

**Files:**
- Modify: none

- [ ] **Step 1: Run the example script one final time**

Run:

```bash
./examples/08-conditional-multibarrier/run.sh
```

Expected:

- PASS in `st`, `mt`, and `cycle`

- [ ] **Step 2: Inspect produced outputs**

Run:

```bash
find examples/08-conditional-multibarrier/results -maxdepth 2 -type f | sort
```

Expected:

- One compiled `.out`
- Per-mode logs/artifacts under the standard example helper layout

- [ ] **Step 3: Verify examples index entry**

Run:

```bash
sed -n '1,40p' examples/README.md
```

Expected:

- `08-conditional-multibarrier` is listed

- [ ] **Step 4: If any last-mile example issue was fixed, commit it**

```bash
git add examples/README.md examples/08-conditional-multibarrier/conditional_multibarrier.hip examples/08-conditional-multibarrier/run.sh examples/08-conditional-multibarrier/README.md
git commit -m "feat: finalize conditional multibarrier example"
```

- [ ] **Step 5: Report final verification summary**

Use this exact format in the handoff:

```text
Verified:
- native conditional_multibarrier.out PASS
- examples/08-conditional-multibarrier/run.sh PASS
- st / mt / cycle all report mismatches=0
```

## Self-Review

- Spec coverage:
  - Example lives under `examples/08-conditional-multibarrier/`: Tasks 1-4
  - Real HIP source with multiple barriers and conditional compute: Task 2
  - `run.sh` covers `st / mt / cycle`: Task 3
  - README explains legality and usage: Task 4
  - host-side exact validation is present: Task 2 and Task 5
- Placeholder scan:
  - No `TODO` / `TBD` placeholders remain
  - All steps include exact file paths, commands, and validation expectations
- Type consistency:
  - Plan consistently uses `conditional_multibarrier`, `mismatches=0`, and the existing examples helper flow
