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

For compute-family handlers, the preferred per-op extension style is:

- `family handler + functor registry`

not:

- one subclass per arithmetic opcode

At this stage the files are interface scaffolding only.
The current implementation still lives in the existing semantic/exec pipeline.
