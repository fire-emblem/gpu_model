#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Any

import yaml


FORMAT_ENUM = {
    "unknown": "Unknown",
    "sop2": "Sop2",
    "sopk": "Sopk",
    "sop1": "Sop1",
    "sopc": "Sopc",
    "sopp": "Sopp",
    "smrd": "Smrd",
    "vop2": "Vop2",
    "vop1": "Vop1",
    "vopc": "Vopc",
    "vop3a": "Vop3a",
    "vop3b": "Vop3b",
    "vintrp": "Vintrp",
    "ds": "Ds",
    "flat": "Flat",
    "mubuf": "Mubuf",
    "mtbuf": "Mtbuf",
    "mimg": "Mimg",
    "exp": "Exp",
}

FORMAT_ENCODING_VALUES = {
    "vop2": 0x0,
    "sop2": 0x2,
    "sopk": 0xB,
    "smrd": 0x18,
    "vop3a": 0x34,
    "ds": 0x36,
    "flat": 0x37,
    "vopc": 0x3E,
    "vop1": 0x3F,
    "sop1": 0x17D,
    "sopc": 0x17E,
    "sopp": 0x17F,
}

FORMAT_OPCODE_ENUM = {
    "sop2": "GcnSop2Opcode",
    "sopk": "GcnSopkOpcode",
    "sop1": "GcnSop1Opcode",
    "sopc": "GcnSopcOpcode",
    "sopp": "GcnSoppOpcode",
    "smrd": "GcnSmrdOpcode",
    "vop2": "GcnVop2Opcode",
    "vop1": "GcnVop1Opcode",
    "vopc": "GcnVopcOpcode",
    "vop3a": "GcnVop3aOpcode",
    "ds": "GcnDsOpcode",
    "flat": "GcnFlatOpcode",
}

FLAG_ENUM = {
    "is_branch": "kGcnInstFlagIsBranch",
    "is_memory": "kGcnInstFlagIsMemory",
    "is_atomic": "kGcnInstFlagIsAtomic",
    "is_barrier": "kGcnInstFlagIsBarrier",
    "writes_exec": "kGcnInstFlagWritesExec",
    "writes_vcc": "kGcnInstFlagWritesVcc",
    "writes_scc": "kGcnInstFlagWritesScc",
    "is_waitcnt": "kGcnInstFlagIsWaitcnt",
    "is_matrix": "kGcnInstFlagIsMatrix",
}


def c_str(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def to_enum_symbol(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.upper()


def load_yaml(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def emit_header(out_path: pathlib.Path) -> None:
    text = """#pragma once

#include <cstdint>
#include <vector>

#include "gpu_model/decode/gcn_inst_encoding_def.h"
#include "gpu_model/decode/gcn_inst_format.h"

namespace gpu_model {

struct GcnGeneratedFieldRef {
  const char* name;
  uint8_t word_index;
  uint8_t lsb;
  uint8_t width;
  bool sign_extend;
  const char* meaning;
};

struct GcnGeneratedProfileDef {
  const char* id;
  const char* display_name;
  uint32_t wave_size;
  bool has_accvgpr;
  const char* waitcnt_layout;
};

struct GcnGeneratedOperandKindDef {
  const char* id;
  const char* category;
  const char* description;
};

struct GcnGeneratedSemanticFamilyDef {
  const char* id;
  const char* exec_domain;
  const char* description;
};

enum GcnGeneratedInstFlags : uint64_t {
  kGcnInstFlagNone = 0,
  kGcnInstFlagIsBranch = 1ull << 0,
  kGcnInstFlagIsMemory = 1ull << 1,
  kGcnInstFlagIsAtomic = 1ull << 2,
  kGcnInstFlagIsBarrier = 1ull << 3,
  kGcnInstFlagWritesExec = 1ull << 4,
  kGcnInstFlagWritesVcc = 1ull << 5,
  kGcnInstFlagWritesScc = 1ull << 6,
  kGcnInstFlagIsWaitcnt = 1ull << 7,
  kGcnInstFlagIsMatrix = 1ull << 8,
};

struct GcnGeneratedImplicitRegRef {
  const char* name;
  bool is_write;
};

struct GcnGeneratedOperandSpec {
  const char* name;
  const char* kind;
  const char* role;
  const char* field;
  uint8_t reg_count;
  int8_t scale;
  const char* special_reg;
};

struct GcnGeneratedFormatDef {
  const char* id;
  GcnInstFormatClass format_class;
  uint8_t size_bytes;
  GcnGeneratedFieldRef opcode_field;
  uint16_t field_begin;
  uint16_t field_count;
};

struct GcnGeneratedInstDef {
  uint32_t id;
  const char* profile;
  GcnInstFormatClass format_class;
  uint32_t opcode;
  uint8_t size_bytes;
  const char* mnemonic;
  const char* exec_domain;
  const char* semantic_family;
  const char* issue_family;
  uint64_t flags;
  uint16_t implicit_begin;
  uint16_t implicit_count;
  uint16_t operand_begin;
  uint16_t operand_count;
};

const std::vector<GcnGeneratedProfileDef>& GeneratedGcnProfileDefs();
const std::vector<GcnGeneratedOperandKindDef>& GeneratedGcnOperandKindDefs();
const std::vector<GcnGeneratedSemanticFamilyDef>& GeneratedGcnSemanticFamilyDefs();
const std::vector<GcnGeneratedImplicitRegRef>& GeneratedGcnImplicitRegRefs();
const std::vector<GcnGeneratedOperandSpec>& GeneratedGcnOperandSpecs();
const std::vector<GcnGeneratedFieldRef>& GeneratedGcnFieldRefs();
const std::vector<GcnGeneratedFormatDef>& GeneratedGcnFormatDefs();
const std::vector<GcnGeneratedInstDef>& GeneratedGcnInstDefs();
const std::vector<GcnInstEncodingDef>& GeneratedGcnEncodingDefs();

}  // namespace gpu_model
"""
    out_path.write_text(text, encoding="utf-8")


def emit_opcode_enum_header(inst_rows: list[dict[str, Any]], out_path: pathlib.Path) -> None:
    grouped: dict[str, list[tuple[int, str]]] = {}
    for inst in inst_rows:
        fmt = inst["format"]
        grouped.setdefault(fmt, [])
        key = (inst["opcode"], inst["mnemonic"])
        if key not in grouped[fmt]:
          grouped[fmt].append(key)

    lines = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("// Compatibility subset only.")
    lines.append("//")
    lines.append("// Source-of-truth opcode coverage now lives in:")
    lines.append('//   "gpu_model/decode/generated_gcn_full_opcode_table.h"')
    lines.append("//")
    lines.append("// This header is kept to avoid large immediate churn in older tests")
    lines.append("// and generated subset paths.")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("#include <optional>")
    lines.append("#include <string_view>")
    lines.append("")
    lines.append("namespace gpu_model {")
    lines.append("")
    lines.append("enum class GcnOpTypeEncoding : uint16_t {")
    lines.append("  Unknown = 0xffff,")
    for fmt, value in sorted(FORMAT_ENCODING_VALUES.items(), key=lambda x: x[1]):
        lines.append(f"  {to_enum_symbol(fmt)} = 0x{value:x},")
    lines.append("};")
    lines.append("")
    for fmt, entries in grouped.items():
        if fmt not in FORMAT_OPCODE_ENUM:
            continue
        lines.append(f"enum class {FORMAT_OPCODE_ENUM[fmt]} : uint16_t {{")
        for opcode, mnemonic in entries:
            lines.append(f"  {to_enum_symbol(mnemonic)} = {opcode},")
        lines.append("};")
        lines.append("")
    lines.append("std::string_view ToString(GcnOpTypeEncoding op_type);")
    lines.append("std::optional<GcnOpTypeEncoding> ParseGcnOpTypeEncoding(std::string_view text);")
    lines.append("")
    lines.append("struct GcnOpcodeDescriptor {")
    lines.append("  GcnOpTypeEncoding op_type = GcnOpTypeEncoding::Unknown;")
    lines.append("  uint16_t opcode = 0;")
    lines.append("  const char* name = nullptr;")
    lines.append("};")
    lines.append("")
    lines.append(
        "const GcnOpcodeDescriptor* FindGcnOpcodeDescriptor(GcnOpTypeEncoding op_type, uint16_t opcode);"
    )
    lines.append(
        "const GcnOpcodeDescriptor* FindGcnOpcodeDescriptorByName(std::string_view name);"
    )
    lines.append("}  // namespace gpu_model")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def emit_opcode_enum_cpp(inst_rows: list[dict[str, Any]], out_path: pathlib.Path) -> None:
    grouped: dict[str, list[tuple[int, str]]] = {}
    descriptors: list[tuple[str, int, str]] = []
    seen_desc = set()
    for inst in inst_rows:
        fmt = inst["format"]
        grouped.setdefault(fmt, [])
        key = (inst["opcode"], inst["mnemonic"])
        if key not in grouped[fmt]:
            grouped[fmt].append(key)
        desc_key = (fmt, inst["opcode"], inst["mnemonic"])
        if desc_key not in seen_desc:
            descriptors.append((fmt, inst["opcode"], inst["mnemonic"]))
            seen_desc.add(desc_key)

    lines = []
    lines.append('#include "gpu_model/decode/generated_gcn_opcode_enums.h"')
    lines.append("")
    lines.append("// Compatibility subset implementation.")
    lines.append("// Full canonical opcode coverage is provided by")
    lines.append('// "gpu_model/decode/generated_gcn_full_opcode_table.cpp".')
    lines.append("")
    lines.append("#include <vector>")
    lines.append("")
    lines.append("namespace gpu_model {")
    lines.append("")
    lines.append("namespace {")
    lines.append("const std::vector<GcnOpcodeDescriptor> kOpcodeDescriptors = {")
    for i, (fmt, opcode, mnemonic) in enumerate(descriptors):
        comma = "," if i + 1 < len(descriptors) else ""
        lines.append(
            f"  {{ GcnOpTypeEncoding::{to_enum_symbol(fmt)}, static_cast<uint16_t>({FORMAT_OPCODE_ENUM[fmt]}::{to_enum_symbol(mnemonic)}), {c_str(mnemonic)} }}{comma}"
        )
    lines.append("};")
    lines.append("}  // namespace")
    lines.append("")
    lines.append("std::string_view ToString(GcnOpTypeEncoding op_type) {")
    lines.append("  switch (op_type) {")
    lines.append("    case GcnOpTypeEncoding::Unknown: return \"unknown\";")
    for fmt in FORMAT_ENCODING_VALUES:
        lines.append(f"    case GcnOpTypeEncoding::{to_enum_symbol(fmt)}: return \"{fmt}\";")
    lines.append("  }")
    lines.append("  return \"unknown\";")
    lines.append("}")
    lines.append("")
    lines.append("std::optional<GcnOpTypeEncoding> ParseGcnOpTypeEncoding(std::string_view text) {")
    for fmt in FORMAT_ENCODING_VALUES:
        lines.append(f"  if (text == \"{fmt}\") return GcnOpTypeEncoding::{to_enum_symbol(fmt)};")
    lines.append("  return std::nullopt;")
    lines.append("}")
    lines.append("")
    lines.append("const GcnOpcodeDescriptor* FindGcnOpcodeDescriptor(GcnOpTypeEncoding op_type, uint16_t opcode) {")
    lines.append("  for (const auto& desc : kOpcodeDescriptors) {")
    lines.append("    if (desc.op_type == op_type && desc.opcode == opcode) return &desc;")
    lines.append("  }")
    lines.append("  return nullptr;")
    lines.append("}")
    lines.append("")
    lines.append("const GcnOpcodeDescriptor* FindGcnOpcodeDescriptorByName(std::string_view name) {")
    lines.append("  for (const auto& desc : kOpcodeDescriptors) {")
    lines.append("    if (desc.name == name) return &desc;")
    lines.append("  }")
    lines.append("  return nullptr;")
    lines.append("}")
    lines.append("}  // namespace gpu_model")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def emit_cpp(db_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    profiles_doc = load_yaml(db_dir / "profiles.yaml")
    operand_kinds_doc = load_yaml(db_dir / "operand_kinds.yaml")
    semantic_families_doc = load_yaml(db_dir / "semantic_families.yaml")
    formats_doc = load_yaml(db_dir / "format_classes.yaml")
    insts_doc = load_yaml(db_dir / "instructions.yaml")

    profile_rows = profiles_doc["profiles"]
    operand_kind_rows = operand_kinds_doc["operand_kinds"]
    semantic_family_rows = semantic_families_doc["semantic_families"]
    format_rows = formats_doc["format_classes"]
    inst_rows = insts_doc["instructions"]

    field_rows: list[dict[str, Any]] = []
    format_entries: list[str] = []

    field_index = 0
    for fmt in format_rows:
        fields = fmt.get("fields", [])
        for field in fields:
            field_rows.append(
                {
                    "name": field["name"],
                    "word_index": field["word"],
                    "lsb": field["lsb"],
                    "width": field["width"],
                    "sign_extend": bool(field.get("sign_extend", False)),
                    "meaning": field.get("meaning", ""),
                }
            )
        opcode_field = fmt["opcode_field"]
        format_entries.append(
            "  {{ {id}, GcnInstFormatClass::{format_class}, {size_bytes}, "
            "{{ {opcode_name}, {opcode_word}, {opcode_lsb}, {opcode_width}, false, {opcode_meaning} }}, "
            "{field_begin}, {field_count} }}".format(
                id=c_str(fmt["id"]),
                format_class=FORMAT_ENUM[fmt["id"]],
                size_bytes=fmt["size_bytes"],
                opcode_name=c_str(opcode_field.get("name", "op")),
                opcode_word=opcode_field["word"],
                opcode_lsb=opcode_field["lsb"],
                opcode_width=opcode_field["width"],
                opcode_meaning=c_str(opcode_field.get("meaning", "opcode field")),
                field_begin=field_index,
                field_count=len(fields),
            )
        )
        field_index += len(fields)

    field_entries = [
        "  {{ {name}, {word_index}, {lsb}, {width}, {sign_extend}, {meaning} }}".format(
            name=c_str(row["name"]),
            word_index=row["word_index"],
            lsb=row["lsb"],
            width=row["width"],
            sign_extend="true" if row["sign_extend"] else "false",
            meaning=c_str(row["meaning"]),
        )
        for row in field_rows
    ]

    profile_entries = [
        "  {{ {id}, {display_name}, {wave_size}, {has_accvgpr}, {waitcnt_layout} }}".format(
            id=c_str(row["id"]),
            display_name=c_str(row["display_name"]),
            wave_size=row["wave_size"],
            has_accvgpr="true" if row.get("has_accvgpr", False) else "false",
            waitcnt_layout=c_str(row.get("waitcnt_layout", "")),
        )
        for row in profile_rows
    ]

    operand_kind_entries = [
        "  {{ {id}, {category}, {description} }}".format(
            id=c_str(row["id"]),
            category=c_str(row["category"]),
            description=c_str(row["description"]),
        )
        for row in operand_kind_rows
    ]

    semantic_family_entries = [
        "  {{ {id}, {exec_domain}, {description} }}".format(
            id=c_str(row["id"]),
            exec_domain=c_str(row["exec_domain"]),
            description=c_str(row["description"]),
        )
        for row in semantic_family_rows
    ]

    semantic_domain_by_id = {
        row["id"]: row["exec_domain"] for row in semantic_family_rows
    }

    generated_inst_entries = []
    encoding_entries = []
    implicit_ref_entries = []
    operand_spec_entries = []
    implicit_index = 0
    operand_index = 0
    for inst in inst_rows:
        fmt_enum = FORMAT_ENUM[inst["format"]]
        flags = inst.get("flags", {})
        flag_terms = [FLAG_ENUM[name] for name, value in flags.items() if value]
        flag_expr = " | ".join(flag_terms) if flag_terms else "kGcnInstFlagNone"
        implicit_reads = inst.get("implicit_reads", [])
        implicit_writes = inst.get("implicit_writes", [])
        operands = inst.get("operands", [])
        for ref in implicit_reads:
            implicit_ref_entries.append(
                "  {{ {name}, false }}".format(name=c_str(ref))
            )
        for ref in implicit_writes:
            implicit_ref_entries.append(
                "  {{ {name}, true }}".format(name=c_str(ref))
            )
        for operand in operands:
            operand_spec_entries.append(
                "  {{ {name}, {kind}, {role}, {field}, {reg_count}, {scale}, {special_reg} }}".format(
                    name=c_str(operand["name"]),
                    kind=c_str(operand["kind"]),
                    role=c_str(operand["role"]),
                    field=c_str(operand.get("field", "")),
                    reg_count=operand.get("reg_count", 1),
                    scale=operand.get("scale", 1),
                    special_reg=c_str(operand.get("special_reg", "")),
                )
            )
        generated_inst_entries.append(
            "  {{ {id}, {profile}, GcnInstFormatClass::{fmt}, {opcode}, {size_bytes}, {mnemonic}, {exec_domain}, {semantic_family}, {issue_family}, {flags}, {implicit_begin}, {implicit_count}, {operand_begin}, {operand_count} }}".format(
                id=inst["id"],
                profile=c_str(inst.get("profile", "gfx6_gfx8")),
                fmt=fmt_enum,
                opcode=inst["opcode"],
                size_bytes=inst["size_bytes"],
                mnemonic=c_str(inst["mnemonic"]),
                exec_domain=c_str(semantic_domain_by_id.get(inst.get("semantic_family", "unknown"), "special")),
                semantic_family=c_str(inst.get("semantic_family", "unknown")),
                issue_family=c_str(inst.get("issue_family", "unknown")),
                flags=flag_expr,
                implicit_begin=implicit_index,
                implicit_count=len(implicit_reads) + len(implicit_writes),
                operand_begin=operand_index,
                operand_count=len(operands),
            )
        )
        implicit_index += len(implicit_reads) + len(implicit_writes)
        operand_index += len(operands)
        encoding_entries.append(
            "  GcnInstEncodingDef{{ .id = {id}, .format_class = GcnInstFormatClass::{fmt}, .op = {opcode}, .size_bytes = {size_bytes}, .mnemonic = {mnemonic} }}".format(
                id=inst["id"],
                fmt=fmt_enum,
                opcode=inst["opcode"],
                size_bytes=inst["size_bytes"],
                mnemonic=c_str(inst["mnemonic"]),
            )
        )

    lines: list[str] = []
    lines.append('#include "gpu_model/decode/generated_gcn_inst_db.h"')
    lines.append("")
    lines.append("#include <vector>")
    lines.append("")
    lines.append("namespace gpu_model {")
    lines.append("")
    lines.append("const std::vector<GcnGeneratedProfileDef>& GeneratedGcnProfileDefs() {")
    lines.append("  static const std::vector<GcnGeneratedProfileDef> kProfileDefs = {")
    lines.extend([entry + "," for entry in profile_entries[:-1]])
    if profile_entries:
        lines.append(profile_entries[-1])
    lines.append("  };")
    lines.append("  return kProfileDefs;")
    lines.append("}")
    lines.append("")
    lines.append("const std::vector<GcnGeneratedOperandKindDef>& GeneratedGcnOperandKindDefs() {")
    lines.append("  static const std::vector<GcnGeneratedOperandKindDef> kOperandKindDefs = {")
    lines.extend([entry + "," for entry in operand_kind_entries[:-1]])
    if operand_kind_entries:
        lines.append(operand_kind_entries[-1])
    lines.append("  };")
    lines.append("  return kOperandKindDefs;")
    lines.append("}")
    lines.append("")
    lines.append("const std::vector<GcnGeneratedSemanticFamilyDef>& GeneratedGcnSemanticFamilyDefs() {")
    lines.append("  static const std::vector<GcnGeneratedSemanticFamilyDef> kSemanticFamilyDefs = {")
    lines.extend([entry + "," for entry in semantic_family_entries[:-1]])
    if semantic_family_entries:
        lines.append(semantic_family_entries[-1])
    lines.append("  };")
    lines.append("  return kSemanticFamilyDefs;")
    lines.append("}")
    lines.append("")
    lines.append("const std::vector<GcnGeneratedImplicitRegRef>& GeneratedGcnImplicitRegRefs() {")
    lines.append("  static const std::vector<GcnGeneratedImplicitRegRef> kImplicitRegRefs = {")
    lines.extend([entry + "," for entry in implicit_ref_entries[:-1]])
    if implicit_ref_entries:
        lines.append(implicit_ref_entries[-1])
    lines.append("  };")
    lines.append("  return kImplicitRegRefs;")
    lines.append("}")
    lines.append("")
    lines.append("const std::vector<GcnGeneratedOperandSpec>& GeneratedGcnOperandSpecs() {")
    lines.append("  static const std::vector<GcnGeneratedOperandSpec> kOperandSpecs = {")
    lines.extend([entry + "," for entry in operand_spec_entries[:-1]])
    if operand_spec_entries:
        lines.append(operand_spec_entries[-1])
    lines.append("  };")
    lines.append("  return kOperandSpecs;")
    lines.append("}")
    lines.append("")
    lines.append("const std::vector<GcnGeneratedFieldRef>& GeneratedGcnFieldRefs() {")
    lines.append("  static const std::vector<GcnGeneratedFieldRef> kFieldRefs = {")
    lines.extend([entry + "," for entry in field_entries[:-1]])
    if field_entries:
        lines.append(field_entries[-1])
    lines.append("  };")
    lines.append("  return kFieldRefs;")
    lines.append("}")
    lines.append("")
    lines.append("const std::vector<GcnGeneratedFormatDef>& GeneratedGcnFormatDefs() {")
    lines.append("  static const std::vector<GcnGeneratedFormatDef> kFormatDefs = {")
    lines.extend([entry + "," for entry in format_entries[:-1]])
    if format_entries:
        lines.append(format_entries[-1])
    lines.append("  };")
    lines.append("  return kFormatDefs;")
    lines.append("}")
    lines.append("")
    lines.append("const std::vector<GcnGeneratedInstDef>& GeneratedGcnInstDefs() {")
    lines.append("  static const std::vector<GcnGeneratedInstDef> kInstDefs = {")
    lines.extend([entry + "," for entry in generated_inst_entries[:-1]])
    if generated_inst_entries:
        lines.append(generated_inst_entries[-1])
    lines.append("  };")
    lines.append("  return kInstDefs;")
    lines.append("}")
    lines.append("")
    lines.append("const std::vector<GcnInstEncodingDef>& GeneratedGcnEncodingDefs() {")
    lines.append("  static const std::vector<GcnInstEncodingDef> kEncodingDefs = {")
    lines.extend([entry + "," for entry in encoding_entries[:-1]])
    if encoding_entries:
        lines.append(encoding_entries[-1])
    lines.append("  };")
    lines.append("  return kEncodingDefs;")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace gpu_model")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--out-header", required=True)
    parser.add_argument("--out-cpp", required=True)
    parser.add_argument("--out-opcode-header")
    parser.add_argument("--out-opcode-cpp")
    args = parser.parse_args()

    db_dir = pathlib.Path(args.db_dir)
    out_header = pathlib.Path(args.out_header)
    out_cpp = pathlib.Path(args.out_cpp)
    out_header.parent.mkdir(parents=True, exist_ok=True)
    out_cpp.parent.mkdir(parents=True, exist_ok=True)

    emit_header(out_header)
    emit_cpp(db_dir, out_cpp)
    if args.out_opcode_header:
        out_opcode_header = pathlib.Path(args.out_opcode_header)
        out_opcode_header.parent.mkdir(parents=True, exist_ok=True)
        emit_opcode_enum_header(inst_rows=load_yaml(db_dir / "instructions.yaml")["instructions"],
                                out_path=out_opcode_header)
    if args.out_opcode_cpp:
        out_opcode_cpp = pathlib.Path(args.out_opcode_cpp)
        out_opcode_cpp.parent.mkdir(parents=True, exist_ok=True)
        emit_opcode_enum_cpp(inst_rows=load_yaml(db_dir / "instructions.yaml")["instructions"],
                             out_path=out_opcode_cpp)


if __name__ == "__main__":
    main()
