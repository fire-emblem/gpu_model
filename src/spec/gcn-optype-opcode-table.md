# GCN OpType And Opcode Table

## Purpose

This file records the current generated `optype` encoding enums and per-format opcode enums that are already in the codebase. The numeric values are aligned with the current ISA encoding definitions used by the project.

## OpType Encoding

| OpType | Encoding |
|---|---:|
| `VOP2` | `0x0` |
| `SOP2` | `0x2` |
| `SOPK` | `0xb` |
| `SMRD` | `0x18` |
| `VOP3A` | `0x34` |
| `DS` | `0x36` |
| `FLAT` | `0x37` |
| `VOPC` | `0x3e` |
| `VOP1` | `0x3f` |
| `SOP1` | `0x17d` |
| `SOPC` | `0x17e` |
| `SOPP` | `0x17f` |

## GcnSoppOpcode

| Enum | Value |
|---|---:|
| `S_ENDPGM` | `1` |
| `S_CBRANCH_EXECZ` | `8` |
| `S_WAITCNT` | `12` |
| `S_CBRANCH_SCC1` | `5` |
| `S_CBRANCH_SCC0` | `4` |
| `S_BRANCH` | `2` |
| `S_BARRIER` | `10` |
| `S_CBRANCH_VCCZ` | `6` |
| `S_NOP` | `0` |
| `S_CBRANCH_EXECNZ` | `9` |

## GcnSmrdOpcode

| Enum | Value |
|---|---:|
| `S_LOAD_DWORD` | `0` |
| `S_LOAD_DWORDX2` | `32` |
| `S_LOAD_DWORDX4` | `64` |

## GcnSop2Opcode

| Enum | Value |
|---|---:|
| `S_AND_B32` | `12` |
| `S_MUL_I32` | `36` |
| `S_ADD_I32` | `2` |
| `S_OR_B64` | `15` |
| `S_CSELECT_B64` | `11` |
| `S_ANDN2_B64` | `19` |
| `S_LSHR_B32` | `30` |
| `S_ADD_U32` | `0` |
| `S_ADDC_U32` | `4` |
| `S_ASHR_I32` | `32` |
| `S_LSHL_B64` | `29` |
| `S_AND_B64` | `13` |

## GcnVop2Opcode

| Enum | Value |
|---|---:|
| `V_ADD_U32_E32` | `52` |
| `V_ADD_F32_E32` | `1` |
| `V_ASHRREV_I32_E32` | `17` |
| `V_ADD_CO_U32_E32` | `25` |
| `V_ADDC_CO_U32_E32` | `28` |
| `V_LSHLREV_B32_E32` | `18` |
| `V_SUB_F32_E32` | `2` |
| `V_MUL_F32_E32` | `5` |
| `V_MAX_F32_E32` | `11` |
| `V_FMAC_F32_E32` | `59` |
| `V_CNDMASK_B32_E32` | `0` |

## GcnVopcOpcode

| Enum | Value |
|---|---:|
| `V_CMP_GT_I32_E32` | `196` |
| `V_CMP_GT_U32_E32` | `204` |
| `V_CMP_NGT_F32_E32` | `75` |
| `V_CMP_NLT_F32_E32` | `78` |
| `V_CMP_EQ_U32_E32` | `202` |
| `V_CMP_LE_I32_E32` | `195` |
| `V_CMP_LT_I32_E32` | `193` |

## GcnSop1Opcode

| Enum | Value |
|---|---:|
| `S_AND_SAVEEXEC_B64` | `32` |
| `S_MOV_B32` | `0` |
| `S_MOV_B64` | `1` |
| `S_BCNT1_I32_B64` | `13` |

## GcnVop1Opcode

| Enum | Value |
|---|---:|
| `V_MOV_B32_E32` | `1` |
| `V_NOT_B32_E32` | `43` |
| `V_CVT_I32_F32_E32` | `8` |
| `V_RNDNE_F32_E32` | `30` |
| `V_EXP_F32_E32` | `32` |
| `V_RCP_F32_E32` | `34` |
| `V_CVT_F32_I32_E32` | `5` |

## GcnVop3aOpcode

| Enum | Value |
|---|---:|
| `V_LSHLREV_B64` | `327` |
| `V_FMA_F32` | `229` |
| `V_LSHL_ADD_U32` | `254` |
| `V_ADD_CO_U32_E64` | `140` |
| `V_ADDC_CO_U32_E64` | `142` |
| `V_CMP_GT_I32_E64` | `98` |
| `V_CNDMASK_B32_E64` | `128` |
| `V_DIV_FIXUP_F32` | `239` |
| `V_DIV_SCALE_F32` | `240` |
| `V_DIV_FMAS_F32` | `241` |
| `V_LDEXP_F32` | `324` |
| `V_MFMA_F32_16X16X4F32` | `482` |
| `V_MAD_U64_U32` | `244` |
| `V_MBCNT_LO_U32_B32` | `326` |
| `V_MBCNT_HI_U32_B32` | `326` |

## GcnFlatOpcode

| Enum | Value |
|---|---:|
| `GLOBAL_LOAD_DWORD` | `20` |
| `GLOBAL_STORE_DWORD` | `28` |
| `GLOBAL_ATOMIC_ADD` | `66` |

## GcnSopcOpcode

| Enum | Value |
|---|---:|
| `S_CMP_LT_I32` | `4` |
| `S_CMP_EQ_U32` | `6` |
| `S_CMP_GT_U32` | `8` |
| `S_CMP_LT_U32` | `10` |

## GcnDsOpcode

| Enum | Value |
|---|---:|
| `DS_WRITE_B32` | `13` |
| `DS_READ_B32` | `54` |

## GcnSopkOpcode

| Enum | Value |
|---|---:|
| `S_MOVK_I32` | `0` |
