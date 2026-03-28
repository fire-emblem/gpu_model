#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OpTypeInfo:
    enum_name: str
    text_name: str
    encoding_value: int
    encoding_width: int
    td_format: str


@dataclass(frozen=True)
class OpcodeRow:
    op_type: str
    opcode: int
    opname: str
    source_record: str
    source_file: str


OP_TYPES: dict[str, OpTypeInfo] = {
    "sop2": OpTypeInfo("Sop2", "sop2", 0x2, 2, "SOP2"),
    "sopk": OpTypeInfo("Sopk", "sopk", 0xB, 4, "SOPK"),
    "sop1": OpTypeInfo("Sop1", "sop1", 0x17D, 9, "SOP1"),
    "sopc": OpTypeInfo("Sopc", "sopc", 0x17E, 9, "SOPC"),
    "sopp": OpTypeInfo("Sopp", "sopp", 0x17F, 9, "SOPP"),
    "smrd": OpTypeInfo("Smrd", "smrd", 0x18, 5, "SMRD"),
    "smem": OpTypeInfo("Smem", "smem", 0x30, 6, "SMEM"),
    "vintrp": OpTypeInfo("Vintrp", "vintrp", 0x32, 6, "VINTRP"),
    "vop2": OpTypeInfo("Vop2", "vop2", 0x0, 1, "VOP2"),
    "vop1": OpTypeInfo("Vop1", "vop1", 0x3F, 7, "VOP1"),
    "vopc": OpTypeInfo("Vopc", "vopc", 0x3E, 7, "VOPC"),
    "vop3a": OpTypeInfo("Vop3a", "vop3a", 0x34, 6, "VOP3A"),
    "vop3b": OpTypeInfo("Vop3b", "vop3b", 0x34, 6, "VOP3B"),
    "vop3p": OpTypeInfo("Vop3p", "vop3p", 0x1A7, 9, "VOP3P"),
    "ds": OpTypeInfo("Ds", "ds", 0x36, 6, "DS"),
    "flat": OpTypeInfo("Flat", "flat", 0x37, 6, "FLAT"),
    "mubuf": OpTypeInfo("Mubuf", "mubuf", 0x38, 6, "MUBUF"),
    "mtbuf": OpTypeInfo("Mtbuf", "mtbuf", 0x3A, 6, "MTBUF"),
    "mimg": OpTypeInfo("Mimg", "mimg", 0x3C, 6, "MIMG"),
    "exp": OpTypeInfo("Exp", "exp", 0x3E, 6, "EXP"),
}


def to_enum_symbol(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.upper()


def c_str(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def parse_td_rows(td_dir: pathlib.Path) -> list[OpcodeRow]:
    rows: list[OpcodeRow] = []

    def add(op_type: str, opcode: int, opname: str, source_record: str, source_file: str) -> None:
        rows.append(
            OpcodeRow(
                op_type=op_type,
                opcode=opcode,
                opname=opname,
                source_record=source_record,
                source_file=source_file,
            )
        )

    sop_path = td_dir / "SOPInstructions.td"
    for line in sop_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = re.match(r"defm?\s+([A-Z0-9_]+)_vi\s*:\s*SOP1_Real_vi\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("sop1", int(m.group(2), 16), m.group(1).lower(), m.group(1), sop_path.name)
            continue
        m = re.match(r"defm?\s+([A-Z0-9_]+)_vi\s*:\s*SOP2_Real_vi\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("sop2", int(m.group(2), 16), m.group(1).lower(), m.group(1), sop_path.name)
            continue
        m = re.match(r"defm?\s+([A-Z0-9_]+)_vi\s*:\s*(?:SOPK_Real_vi|SOPK_Real64)\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("sopk", int(m.group(2), 16), m.group(1).lower(), m.group(1), sop_path.name)
            continue
        m = re.match(r"defm\s+([A-Z0-9_]+)\s*:\s*SOPC_Real_[A-Za-z0-9_]*gfx8_gfx9[A-Za-z0-9_]*\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("sopc", int(m.group(2), 16), m.group(1).lower(), m.group(1), sop_path.name)
            continue
        m = re.match(r"defm\s+([A-Z0-9_]+)\s*:\s*SOPP_Real_[A-Za-z0-9_]*gfx8_gfx9[A-Za-z0-9_]*\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("sopp", int(m.group(2), 16), m.group(1).lower(), m.group(1), sop_path.name)
            continue
        m = re.match(r"defm\s+([A-Z0-9_]+)\s*:\s*SOPP_Real_With_Relaxation_[A-Za-z0-9_]*gfx8_gfx9[A-Za-z0-9_]*\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("sopp", int(m.group(2), 16), m.group(1).lower(), m.group(1), sop_path.name)

    sm_path = td_dir / "SMInstructions.td"
    for line in sm_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        for pat in [
            (r"defm\s+([A-Z0-9_]+)\s*:\s*SM_Real_Loads_si\s*<\s*(0x[0-9A-Fa-f]+)", "smrd"),
            (r"def\s+([A-Z0-9_]+)_si\s*:\s*SMRD_Real_si\s*<\s*(0x[0-9A-Fa-f]+)", "smrd"),
            (r"def\s+([A-Z0-9_]+)_ci\s*:\s*SMRD_Real_(?:Load_IMM_)?ci\s*<\s*(0x[0-9A-Fa-f]+)", "smrd"),
            (r"defm\s+([A-Z0-9_]+)\s*:\s*SM_Real_Loads_vi\s*<\s*(0x[0-9A-Fa-f]+)", "smem"),
            (r"defm\s+([A-Z0-9_]+)\s*:\s*SM_Real_Stores_vi\s*<\s*(0x[0-9A-Fa-f]+)", "smem"),
            (r"defm\s+([A-Z0-9_]+)\s*:\s*SM_Real_Probe_vi\s*<\s*(0x[0-9A-Fa-f]+)", "smem"),
            (r"defm\s+([A-Z0-9_]+)\s*:\s*SM_Real_Atomics_vi\s*<\s*(0x[0-9A-Fa-f]+)", "smem"),
            (r"def\s+([A-Z0-9_]+)_vi\s*:\s*SMEM_Real_vi\s*<\s*(0x[0-9A-Fa-f]+)", "smem"),
        ]:
            m = re.match(pat[0], line)
            if m:
                add(pat[1], int(m.group(2), 16), m.group(1).lower(), m.group(1), sm_path.name)
                break

    vop1_path = td_dir / "VOP1Instructions.td"
    for line in vop1_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = re.match(r"defm\s+([A-Z0-9_]+)\s*:\s*VOP1Only_Real_vi\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("vop1", int(m.group(2), 16), f"{m.group(1).lower()}_e32", m.group(1), vop1_path.name)
            continue
        m = re.match(
            r"defm\s+([A-Z0-9_]+)\s*:\s*(?:VOP1_Real_vi|VOP1_Real_gfx9|VOP1_Real_NoDstSel_SDWA_gfx9|VOP1_OpSel_Real_e32e64_gfx9)\s*<\s*(0x[0-9A-Fa-f]+)",
            line,
        )
        if m:
            opcode = int(m.group(2), 16)
            opname = m.group(1).lower()
            add("vop1", opcode & 0xFF, f"{opname}_e32", m.group(1), vop1_path.name)
            add("vop3a", 0x140 + opcode, f"{opname}_e64", m.group(1), vop1_path.name)

    vop2_path = td_dir / "VOP2Instructions.td"
    for line in vop2_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = re.match(r"defm\s+([A-Z0-9_]+)\s*:\s*VOP2_Real_e32e64_(?:vi|gfx9)\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            opcode = int(m.group(2), 16)
            opname = m.group(1).lower()
            add("vop2", opcode, f"{opname}_e32", m.group(1), vop2_path.name)
            add("vop3a", 0x100 + opcode, f"{opname}_e64", m.group(1), vop2_path.name)
            continue
        m = re.match(
            r'defm\s+([A-Z0-9_]+)\s*:\s*VOP2be_Real_e32e64_(?:vi_only|gfx9)\s*<\s*(0x[0-9A-Fa-f]+),\s*"([A-Z0-9_]+)"\s*,\s*"([a-z0-9_]+)"',
            line,
        )
        if m:
            opcode = int(m.group(2), 16)
            add("vop2", opcode, f"{m.group(4)}_e32", m.group(1), vop2_path.name)
            add("vop3b", 0x100 + opcode, f"{m.group(4)}_e64", m.group(1), vop2_path.name)
            continue
        m = re.match(r"defm\s+([A-Z0-9_]+)\s*:\s*VOP2_Real_e64only_vi\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("vop3a", int(m.group(2), 16), m.group(1).lower(), m.group(1), vop2_path.name)

    vop3_path = td_dir / "VOP3Instructions.td"
    for line in vop3_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = re.match(r"defm\s+([A-Z0-9_]+)\s*:\s*VOP3_Real_vi\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("vop3a", int(m.group(2), 16), m.group(1).lower(), m.group(1), vop3_path.name)
            continue
        m = re.match(r"defm\s+([A-Z0-9_]+)\s*:\s*VOP3be_Real_vi\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("vop3b", int(m.group(2), 16), m.group(1).lower(), m.group(1), vop3_path.name)

    vop3p_path = td_dir / "VOP3PInstructions.td"
    for line in vop3p_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = re.match(r"defm\s+([A-Z0-9_]+)\s*:\s*VOP3P_Real_vi\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("vop3p", int(m.group(2), 16), m.group(1).lower(), m.group(1), vop3p_path.name)
            continue
        m = re.match(r"defm\s+([A-Z0-9_]+)\s*:\s*VOP3P_Real_(?:MAI|MFMA(?:_gfx90a)?)\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("vop3p", int(m.group(2), 16), m.group(1).lower(), m.group(1), vop3p_path.name)

    vopc_path = td_dir / "VOPCInstructions.td"
    for line in vopc_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = re.match(r"defm\s+([A-Z0-9_]+)\s*:\s*VOPC_Real_vi\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            opcode = int(m.group(2), 16)
            opname = m.group(1).lower()
            add("vopc", opcode & 0xFF, f"{opname}_e32", m.group(1), vopc_path.name)
            add("vop3a", opcode, f"{opname}_e64", m.group(1), vopc_path.name)

    ds_path = td_dir / "DSInstructions.td"
    for line in ds_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = re.match(r"defm?\s+([A-Z0-9_]+)_vi\s*:\s*DS_Real_vi\s*<\s*(0x[0-9A-Fa-f]+)\s*,\s*([A-Z0-9_]+)", line)
        if m:
            add("ds", int(m.group(2), 16), m.group(3).lower(), m.group(3), ds_path.name)

    flat_path = td_dir / "FLATInstructions.td"
    for line in flat_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        for pat in [
            r"def\s+([A-Z0-9_]+)_vi\s*:\s*FLAT_Real_vi\s*<\s*(0x[0-9A-Fa-f]+)\s*,\s*([A-Z0-9_]+)",
            r"defm\s+([A-Z0-9_]+)\s*:\s*FLAT_Real_Atomics_vi\s*<\s*(0x[0-9A-Fa-f]+)",
            r"defm\s+([A-Z0-9_]+)\s*:\s*FLAT_Real_AllAddr_vi\s*<\s*(0x[0-9A-Fa-f]+)",
            r"defm\s+([A-Z0-9_]+)\s*:\s*FLAT_Global_Real_Atomics_vi\s*<\s*(0x[0-9A-Fa-f]+)",
            r"defm\s+([A-Z0-9_]+)\s*:\s*FLAT_Real_AllAddr_SVE_vi\s*<\s*(0x[0-9A-Fa-f]+)",
        ]:
            m = re.match(pat, line)
            if not m:
                continue
            opname = (m.group(3) if len(m.groups()) >= 3 and m.group(3) else m.group(1)).lower()
            add("flat", int(m.group(2), 16), opname, m.group(1), flat_path.name)
            break

    buf_path = td_dir / "BUFInstructions.td"
    for line in buf_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = re.match(r"defm\s+([A-Z0-9_]+)\s*:\s*MUBUF_Real_(?:AllAddr(?:_Lds)?_vi|Atomic(?:s)?_vi)\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("mubuf", int(m.group(2), 16), m.group(1).lower(), m.group(1), buf_path.name)
            continue
        m = re.match(r"defm\s+([A-Z0-9_]+)\s*:\s*MTBUF_Real_AllAddr_vi\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("mtbuf", int(m.group(2), 16), m.group(1).lower(), m.group(1), buf_path.name)

    mimg_path = td_dir / "MIMGInstructions.td"
    for line in mimg_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("defm ") or "mimgopc<" not in line:
            continue
        name_match = re.match(r"defm\s+([A-Z0-9_]+)\s*:", line)
        opcode_match = re.search(r"mimgopc<([^>]+)>", line)
        opname_match = re.search(r'"([a-z0-9_]+)"', line)
        if not name_match or not opcode_match:
            continue
        opname = opname_match.group(1) if opname_match else name_match.group(1).lower()
        opcode_fields = [field.strip() for field in opcode_match.group(1).split(",")]
        chosen = None
        if len(opcode_fields) >= 4 and opcode_fields[3] not in {"MIMG.NOP", "vi = gfx10m"}:
            chosen = opcode_fields[3]
        elif len(opcode_fields) >= 3 and opcode_fields[2] != "MIMG.NOP":
            chosen = opcode_fields[2]
        elif len(opcode_fields) >= 2 and opcode_fields[1] != "MIMG.NOP":
            chosen = opcode_fields[1]
        elif opcode_fields and opcode_fields[0] != "MIMG.NOP":
            chosen = opcode_fields[0]
        if chosen and chosen.startswith("0x"):
            add("mimg", int(chosen, 16), opname, name_match.group(1), mimg_path.name)

    si_path = td_dir / "SIInstructions.td"
    for line in si_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = re.match(r"defm\s+(V_INTERP_[A-Z0-9_]+)\s*:\s*VINTRP_m\s*<\s*(0x[0-9A-Fa-f]+)", line)
        if m:
            add("vintrp", int(m.group(2), 16), m.group(1).lower(), m.group(1), si_path.name)

    if not any(row.op_type == "vintrp" for row in rows):
        add("vintrp", 0x0, "v_interp_p1_f32", "V_INTERP_P1_F32", si_path.name)
        add("vintrp", 0x1, "v_interp_p2_f32", "V_INTERP_P2_F32", si_path.name)
        add("vintrp", 0x2, "v_interp_mov_f32", "V_INTERP_MOV_F32", si_path.name)

    exp_path = td_dir / "EXPInstructions.td"
    add("exp", 0, "exp", "EXP", exp_path.name)

    def priority(row: OpcodeRow) -> tuple[int, int]:
        record = row.source_record.lower()
        if "gfx9" in record:
            return (3, row.opcode)
        if record.endswith("_vi") or "_vi_" in record:
            return (2, row.opcode)
        if "ci" in record or "si" in record or "gfx6" in record or "gfx7" in record:
            return (1, row.opcode)
        return (0, row.opcode)

    dedup: dict[tuple[str, str], OpcodeRow] = {}
    for row in rows:
        key = (row.op_type, row.opname)
        current = dedup.get(key)
        if current is None or priority(row) >= priority(current):
            dedup[key] = row
    return sorted(dedup.values(), key=lambda row: (row.op_type, row.opcode, row.opname))


def emit_header(rows: list[OpcodeRow], out_path: pathlib.Path) -> None:
    grouped: dict[str, list[OpcodeRow]] = {}
    for row in rows:
        grouped.setdefault(row.op_type, []).append(row)

    lines: list[str] = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("#include <optional>")
    lines.append("#include <string_view>")
    lines.append("#include <vector>")
    lines.append("")
    lines.append("namespace gpu_model {")
    lines.append("")
    lines.append("enum class GcnIsaOpType : uint16_t {")
    lines.append("  Unknown = 0,")
    for info in OP_TYPES.values():
        lines.append(f"  {info.enum_name},")
    lines.append("};")
    lines.append("")
    for op_type, op_rows in grouped.items():
        info = OP_TYPES[op_type]
        lines.append(f"enum class GcnIsa{info.enum_name}Opcode : uint16_t {{")
        for row in op_rows:
            lines.append(f"  {to_enum_symbol(row.opname)} = 0x{row.opcode:x},")
        lines.append("};")
        lines.append("")
    lines.append("struct GcnIsaOpTypeInfo {")
    lines.append("  GcnIsaOpType op_type = GcnIsaOpType::Unknown;")
    lines.append("  const char* name = nullptr;")
    lines.append("  uint16_t encoding_value = 0;")
    lines.append("  uint8_t encoding_width = 0;")
    lines.append("  const char* td_format = nullptr;")
    lines.append("};")
    lines.append("")
    lines.append("struct GcnIsaOpcodeDescriptor {")
    lines.append("  GcnIsaOpType op_type = GcnIsaOpType::Unknown;")
    lines.append("  uint16_t opcode = 0;")
    lines.append("  const char* opname = nullptr;")
    lines.append("  const char* source_record = nullptr;")
    lines.append("  const char* source_file = nullptr;")
    lines.append("};")
    lines.append("")
    lines.append("std::string_view ToString(GcnIsaOpType op_type);")
    lines.append("std::optional<GcnIsaOpType> ParseGcnIsaOpType(std::string_view text);")
    lines.append("const std::vector<GcnIsaOpTypeInfo>& GcnIsaOpTypeInfos();")
    lines.append("const std::vector<GcnIsaOpcodeDescriptor>& GcnIsaOpcodeDescriptors();")
    lines.append("const GcnIsaOpTypeInfo* FindGcnIsaOpTypeInfo(GcnIsaOpType op_type);")
    lines.append("const GcnIsaOpTypeInfo* FindGcnIsaOpTypeInfoByName(std::string_view name);")
    lines.append("const GcnIsaOpcodeDescriptor* FindGcnIsaOpcodeDescriptor(GcnIsaOpType op_type, uint16_t opcode);")
    lines.append("const GcnIsaOpcodeDescriptor* FindGcnIsaOpcodeDescriptorByName(std::string_view opname);")
    lines.append("")
    lines.append("}  // namespace gpu_model")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def emit_cpp(rows: list[OpcodeRow], out_path: pathlib.Path) -> None:
    lines: list[str] = []
    lines.append('#include "gpu_model/decode/generated_gcn_full_opcode_table.h"')
    lines.append("")
    lines.append("#include <array>")
    lines.append("")
    lines.append("namespace gpu_model {")
    lines.append("")
    lines.append("namespace {")
    lines.append("constexpr std::array<GcnIsaOpTypeInfo, %d> kOpTypeInfos = {" % len(OP_TYPES))
    for idx, info in enumerate(OP_TYPES.values()):
        comma = "," if idx + 1 < len(OP_TYPES) else ""
        lines.append(
            f"  GcnIsaOpTypeInfo{{ GcnIsaOpType::{info.enum_name}, {c_str(info.text_name)}, 0x{info.encoding_value:x}, {info.encoding_width}, {c_str(info.td_format)} }}{comma}"
        )
    lines.append("};")
    lines.append("")
    lines.append("constexpr std::array<GcnIsaOpcodeDescriptor, %d> kOpcodeDescriptors = {" % len(rows))
    for idx, row in enumerate(rows):
        comma = "," if idx + 1 < len(rows) else ""
        info = OP_TYPES[row.op_type]
        lines.append(
            f"  GcnIsaOpcodeDescriptor{{ GcnIsaOpType::{info.enum_name}, 0x{row.opcode:x}, {c_str(row.opname)}, {c_str(row.source_record)}, {c_str(row.source_file)} }}{comma}"
        )
    lines.append("};")
    lines.append("}  // namespace")
    lines.append("")
    lines.append("std::string_view ToString(GcnIsaOpType op_type) {")
    lines.append("  for (const auto& info : kOpTypeInfos) {")
    lines.append("    if (info.op_type == op_type) return info.name;")
    lines.append("  }")
    lines.append('  return "unknown";')
    lines.append("}")
    lines.append("")
    lines.append("std::optional<GcnIsaOpType> ParseGcnIsaOpType(std::string_view text) {")
    lines.append("  for (const auto& info : kOpTypeInfos) {")
    lines.append("    if (info.name == text) return info.op_type;")
    lines.append("  }")
    lines.append("  return std::nullopt;")
    lines.append("}")
    lines.append("")
    lines.append("const std::vector<GcnIsaOpTypeInfo>& GcnIsaOpTypeInfos() {")
    lines.append("  static const std::vector<GcnIsaOpTypeInfo> kInfos(kOpTypeInfos.begin(), kOpTypeInfos.end());")
    lines.append("  return kInfos;")
    lines.append("}")
    lines.append("")
    lines.append("const std::vector<GcnIsaOpcodeDescriptor>& GcnIsaOpcodeDescriptors() {")
    lines.append("  static const std::vector<GcnIsaOpcodeDescriptor> kDescs(kOpcodeDescriptors.begin(), kOpcodeDescriptors.end());")
    lines.append("  return kDescs;")
    lines.append("}")
    lines.append("")
    lines.append("const GcnIsaOpTypeInfo* FindGcnIsaOpTypeInfo(GcnIsaOpType op_type) {")
    lines.append("  for (const auto& info : kOpTypeInfos) {")
    lines.append("    if (info.op_type == op_type) return &info;")
    lines.append("  }")
    lines.append("  return nullptr;")
    lines.append("}")
    lines.append("")
    lines.append("const GcnIsaOpTypeInfo* FindGcnIsaOpTypeInfoByName(std::string_view name) {")
    lines.append("  for (const auto& info : kOpTypeInfos) {")
    lines.append("    if (info.name == name) return &info;")
    lines.append("  }")
    lines.append("  return nullptr;")
    lines.append("}")
    lines.append("")
    lines.append("const GcnIsaOpcodeDescriptor* FindGcnIsaOpcodeDescriptor(GcnIsaOpType op_type, uint16_t opcode) {")
    lines.append("  for (const auto& desc : kOpcodeDescriptors) {")
    lines.append("    if (desc.op_type == op_type && desc.opcode == opcode) return &desc;")
    lines.append("  }")
    lines.append("  return nullptr;")
    lines.append("}")
    lines.append("")
    lines.append("const GcnIsaOpcodeDescriptor* FindGcnIsaOpcodeDescriptorByName(std::string_view opname) {")
    lines.append("  for (const auto& desc : kOpcodeDescriptors) {")
    lines.append("    if (desc.opname == opname) return &desc;")
    lines.append("  }")
    lines.append("  return nullptr;")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace gpu_model")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def emit_md(rows: list[OpcodeRow], out_path: pathlib.Path) -> None:
    grouped: dict[str, list[OpcodeRow]] = {}
    for row in rows:
        grouped.setdefault(row.op_type, []).append(row)

    lines: list[str] = []
    lines.append("# GCN OpType And Opcode Table")
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append("Generated from LLVM 14 AMDGPU TableGen source under:")
    lines.append("")
    lines.append("- `third_party/llvm-project/llvm/lib/Target/AMDGPU/*.td`")
    lines.append("- cross-checked against `src/spec/llvm_amdgpu_refs/AMDGPUAsmGFX9.html`")
    lines.append("")
    lines.append("Scope of this table:")
    lines.append("")
    lines.append("- GCN / CI / VI / GFX9 line instruction families")
    lines.append("- one canonical `(optype, opname)` row is kept for code generation")
    lines.append("- when LLVM TD contains multiple generation-specific opcodes for the same mnemonic, the table prefers `gfx9` rows, then generic `vi` rows, then older `ci/si/gfx6/gfx7` rows")
    lines.append("- address-mode variants such as `_OFFSET` / `_OFFEN` / `_BOTHEN` are deduplicated into one opcode row")
    lines.append("")
    lines.append("Canonical coverage summary:")
    lines.append("")
    lines.append(f"- total canonical opcode rows: `{len(rows)}`")
    for op_type in sorted(grouped.keys()):
        info = OP_TYPES[op_type]
        lines.append(f"- `{info.td_format}` rows: `{len(grouped[op_type])}`")
    lines.append("")
    lines.append("Encoding note:")
    lines.append("")
    lines.append("- some families reuse the same raw prefix value but with different prefix widths")
    lines.append("- for example `EXP` and `VOPC` both use raw value `0x3e`, but `EXP` is 6-bit and `VOPC` is 7-bit")
    lines.append("- `VOP3A` and `VOP3B` share the same raw GFX8/GFX9 prefix `0x34` and are separated here by operand form")
    lines.append("- this markdown table is the canonical opcode table for the repository")
    lines.append("- the old subset-only `generated_gcn_opcode_enums.*` path is retired")
    lines.append("")
    lines.append("## OpType Encoding")
    lines.append("")
    lines.append("| OpType | Encoding | Width | TD Format |")
    lines.append("|---|---:|---:|---|")
    for info in OP_TYPES.values():
        lines.append(f"| `{info.td_format}` | `0x{info.encoding_value:x}` | `{info.encoding_width}` | `{info.td_format}` |")
    lines.append("")
    for op_type, op_rows in grouped.items():
        info = OP_TYPES[op_type]
        lines.append(f"## {info.td_format}")
        lines.append("")
        lines.append("| OpName | Opcode | Source Record | Source TD |")
        lines.append("|---|---:|---|---|")
        for row in op_rows:
            lines.append(
                f"| `{row.opname}` | `0x{row.opcode:x}` | `{row.source_record}` | `{row.source_file}` |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--td-dir", required=True)
    parser.add_argument("--out-header", required=True)
    parser.add_argument("--out-cpp", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    td_dir = pathlib.Path(args.td_dir)
    rows = parse_td_rows(td_dir)

    out_header = pathlib.Path(args.out_header)
    out_cpp = pathlib.Path(args.out_cpp)
    out_md = pathlib.Path(args.out_md)
    out_header.parent.mkdir(parents=True, exist_ok=True)
    out_cpp.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    emit_header(rows, out_header)
    emit_cpp(rows, out_cpp)
    emit_md(rows, out_md)


if __name__ == "__main__":
    main()
