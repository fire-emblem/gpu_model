# ISA Coverage Report

Generated: `2026-03-29T14:01:29`

Coverage scope:
- tracked instruction entries from `src/spec/gcn_db/instructions.yaml`: `84`
- tracked unique mnemonics from `src/spec/gcn_db/instructions.yaml`: `79`
- full opcode rows extracted from `generated_gcn_full_opcode_table.cpp`: `1575`
- full unique mnemonics extracted from `generated_gcn_full_opcode_table.cpp`: `1559`

Tracked-subset coverage:
- raw object support: `79` / `79` (100.0%)
- decode unit tests: `35` / `79` (44.3%)
- exec unit tests: `30` / `79` (38.0%)
- loader integration tests: `15` / `79` (19.0%)
- any tests: `45` / `79` (57.0%)

## Per Semantic Family

| Family | Unique | Raw Object | Decode Tests | Exec Tests | Loader Tests | Any Tests |
|---|---:|---:|---:|---:|---:|---:|
| `branch_or_sync` | `10` | `10` (100.0%) | `6` (60.0%) | `9` (90.0%) | `3` (30.0%) | `10` (100.0%) |
| `lds` | `2` | `2` (100.0%) | `2` (100.0%) | `2` (100.0%) | `2` (100.0%) | `2` (100.0%) |
| `scalar_alu` | `17` | `17` (100.0%) | `3` (17.6%) | `1` (5.9%) | `0` (0.0%) | `4` (23.5%) |
| `scalar_compare` | `4` | `4` (100.0%) | `3` (75.0%) | `3` (75.0%) | `1` (25.0%) | `4` (100.0%) |
| `scalar_memory` | `3` | `3` (100.0%) | `3` (100.0%) | `3` (100.0%) | `3` (100.0%) | `3` (100.0%) |
| `vector_alu` | `33` | `33` (100.0%) | `11` (33.3%) | `5` (15.2%) | `5` (15.2%) | `14` (42.4%) |
| `vector_compare` | `7` | `7` (100.0%) | `5` (71.4%) | `5` (71.4%) | `0` (0.0%) | `5` (71.4%) |
| `vector_memory` | `3` | `3` (100.0%) | `2` (66.7%) | `2` (66.7%) | `1` (33.3%) | `3` (100.0%) |

## Per Encoding / Format Class

| Format | Unique | Raw Object | Decode Tests | Exec Tests | Loader Tests | Any Tests |
|---|---:|---:|---:|---:|---:|---:|
| `ds` | `2` | `2` (100.0%) | `2` (100.0%) | `2` (100.0%) | `2` (100.0%) | `2` (100.0%) |
| `flat` | `3` | `3` (100.0%) | `2` (66.7%) | `2` (66.7%) | `1` (33.3%) | `3` (100.0%) |
| `smrd` | `3` | `3` (100.0%) | `3` (100.0%) | `3` (100.0%) | `3` (100.0%) | `3` (100.0%) |
| `sop1` | `4` | `4` (100.0%) | `1` (25.0%) | `1` (25.0%) | `0` (0.0%) | `2` (50.0%) |
| `sop2` | `12` | `12` (100.0%) | `1` (8.3%) | `0` (0.0%) | `0` (0.0%) | `1` (8.3%) |
| `sopc` | `4` | `4` (100.0%) | `3` (75.0%) | `3` (75.0%) | `1` (25.0%) | `4` (100.0%) |
| `sopk` | `1` | `1` (100.0%) | `1` (100.0%) | `0` (0.0%) | `0` (0.0%) | `1` (100.0%) |
| `sopp` | `10` | `10` (100.0%) | `6` (60.0%) | `9` (90.0%) | `3` (30.0%) | `10` (100.0%) |
| `vop1` | `7` | `7` (100.0%) | `2` (28.6%) | `1` (14.3%) | `0` (0.0%) | `2` (28.6%) |
| `vop2` | `11` | `11` (100.0%) | `4` (36.4%) | `3` (27.3%) | `2` (18.2%) | `6` (54.5%) |
| `vop3a` | `15` | `15` (100.0%) | `5` (33.3%) | `1` (6.7%) | `3` (20.0%) | `6` (40.0%) |
| `vopc` | `7` | `7` (100.0%) | `5` (71.4%) | `5` (71.4%) | `0` (0.0%) | `5` (71.4%) |

## Supported But Untested

Count: `34`

- `s_add_i32`
- `s_add_u32`
- `s_addc_u32`
- `s_and_b32`
- `s_and_b64`
- `s_andn2_b64`
- `s_ashr_i32`
- `s_bcnt1_i32_b64`
- `s_cselect_b64`
- `s_lshr_b32`
- `s_mov_b64`
- `s_mul_i32`
- `s_or_b64`
- `v_addc_co_u32_e64`
- `v_ashrrev_i32_e32`
- `v_cmp_gt_i32_e64`
- `v_cmp_ngt_f32_e32`
- `v_cmp_nlt_f32_e32`
- `v_cndmask_b32_e32`
- `v_cvt_i32_f32_e32`
- `v_div_fixup_f32`
- `v_div_fmas_f32`
- `v_div_scale_f32`
- `v_exp_f32_e32`
- `v_fmac_f32_e32`
- `v_ldexp_f32`
- `v_lshl_add_u32`
- `v_lshlrev_b32_e32`
- `v_mbcnt_hi_u32_b32`
- `v_mbcnt_lo_u32_b32`
- `v_not_b32_e32`
- `v_rcp_f32_e32`
- `v_rndne_f32_e32`
- `v_sub_f32_e32`

## Missing Raw Object Support

Count: `0`

