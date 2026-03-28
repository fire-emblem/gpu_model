# Exec Handlers

This directory is the skeleton for the future execution-layer hierarchy:

1. `ExecHandlerBase`
2. domain-level bases
   - compute
   - memory
   - control
   - sync
3. family handlers
   - scalar/vector/mfma
   - scalar/vector/lds/atomic memory
   - branch/mask/builtin
   - waitcnt/barrier

At this stage the files are interface scaffolding only.
The current implementation still lives in the existing semantic/exec pipeline.
