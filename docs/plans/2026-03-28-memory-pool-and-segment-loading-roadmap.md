# Memory Pool And Segment Loading Roadmap

> [!NOTE]
> 历史计划文档。用于保留当时的拆解和决策上下文，不作为当前代码结构的权威描述。当前主线以 `docs/my_design.md` 和 `docs/module-development-status.md` 为准。


## Why this matters now

The project has already crossed the point where “instruction decode only” is the
main blocker. Real HIP host artifacts now run through:

- runtime hook direct launch
- interposer launch
- raw code-object decode
- raw code-section execution

That makes memory residency and segment loading the next structural bottleneck.

Today the implementation still spreads memory concerns across multiple places:

- `MemorySystem` holds one flat global byte vector
- shared memory is created ad hoc per block
- private memory is embedded inside wave state
- constant data lives in `ConstSegment`
- kernarg images are built inline in executors
- code bytes are loaded, but not represented through a common segment model

## Recommendation for the current stage

Yes: the current stage should already introduce the design for memory pools and
segment-oriented loading.

The goal is not to implement a full allocator immediately. The goal is to lock
down stable interfaces before:

- managed memory support
- richer HIP runtime coverage
- constant/data/raw-data ELF loading
- code/data separation at module load time
- host/device copy versus map policy

## Minimal pool taxonomy

Use explicit pool kinds even if some are still backed by host vectors.

Recommended pool split:

- `Global`
- `Constant`
- `Shared`
- `Private`
- `Managed`
- `Kernarg`
- `Code`
- `RawData`

Recommended allocation metadata:

- pool kind
- base address
- size
- alignment
- logical tag
- mapping kind: `Copy`, `Map`, `ZeroFill`

## Segment-oriented load model

All loaders should be able to emit a `DeviceLoadPlan`.

Recommended segment kinds:

- `Code`
- `ConstantData`
- `RawData`
- `KernargTemplate`

Each segment should describe:

- destination pool
- copy or map policy
- required byte size
- alignment
- payload bytes

This is enough to unify:

- raw ELF `.text`
- const segments
- future ELF data sections
- future raw binary blobs
- launch-time kernarg materialization

## Host/device transfer and map policy

At this stage, use a simple policy table:

- host buffers copied into `Global`
- results copied back from `Global`
- static `__shared__` allocated from `Shared`
- per-lane scratch allocated from `Private`
- launch-time arg image allocated from `Kernarg`
- code bytes copied into `Code`
- const segment copied into `Constant`
- managed memory reserved in the interface now, migration behavior later

## ELF and code object loading direction

Short term:

- keep decoding instructions from code bytes
- also build a segment plan from code object metadata

Medium term:

- parse constant/data/raw-data ELF sections into segment images
- split module load from kernel launch
- make runtime hook and interposer share one segment-loading path

Long term:

- keep module code/data resident across multiple launches
- support map semantics for selected host-visible allocations
- add explicit managed memory residency and migration policy

## What was added now

This step adds minimal scaffolding only:

- `MemoryPoolKind`
- `MemoryMappingKind`
- `DeviceMemoryAllocation`
- `DeviceSegmentImage`
- `DeviceLoadPlan`
- helper builders for `ProgramImage` and `AmdgpuCodeObjectImage`

That is enough to give the codebase one stable vocabulary before replacing the
current flat memory implementation.
