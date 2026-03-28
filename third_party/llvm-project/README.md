# LLVM 14 Source Snapshot

This directory is a local LLVM 14 source snapshot assembled for AMDGPU
TableGen work in this repository.

Contents:

- `llvm/lib/Target/AMDGPU/*.td`
  - fetched from `llvm-project` `release/14.x`
- `llvm/include/llvm/*.td`
  - copied from the system package installed with the current `llvm-tblgen`

Version intent:

- runtime tool: `llvm-tblgen 14.0.0`
- source target: LLVM 14 AMDGPU backend TD set

Validated command:

```bash
llvm-tblgen \
  -I /usr/include/llvm-14 \
  -I /data/gpu_model/third_party/llvm-project/llvm/lib/Target/AMDGPU \
  --null-backend \
  /data/gpu_model/third_party/llvm-project/llvm/lib/Target/AMDGPU/AMDGPU.td
```

Notes:

- The system include root `/usr/include/llvm-14` is used for TableGen public
  headers.
- This is not a full git clone of `llvm-project`; it is a working source tree
  assembled to satisfy AMDGPU TableGen parsing in the current environment.
