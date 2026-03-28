#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
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


def c_str(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


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
  const char* semantic_family;
  const char* issue_family;
};

const std::vector<GcnGeneratedFieldRef>& GeneratedGcnFieldRefs();
const std::vector<GcnGeneratedFormatDef>& GeneratedGcnFormatDefs();
const std::vector<GcnGeneratedInstDef>& GeneratedGcnInstDefs();
const std::vector<GcnInstEncodingDef>& GeneratedGcnEncodingDefs();

}  // namespace gpu_model
"""
    out_path.write_text(text, encoding="utf-8")


def emit_cpp(db_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    formats_doc = load_yaml(db_dir / "format_classes.yaml")
    insts_doc = load_yaml(db_dir / "instructions.yaml")

    format_rows = formats_doc["format_classes"]
    inst_rows = insts_doc["instructions"]

    field_rows: list[dict[str, Any]] = []
    format_entries: list[str] = []
    for fmt in format_rows:
      pass

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

    generated_inst_entries = []
    encoding_entries = []
    for inst in inst_rows:
        fmt_enum = FORMAT_ENUM[inst["format"]]
        generated_inst_entries.append(
            "  {{ {id}, {profile}, GcnInstFormatClass::{fmt}, {opcode}, {size_bytes}, {mnemonic}, {semantic_family}, {issue_family} }}".format(
                id=inst["id"],
                profile=c_str(inst.get("profile", "gfx6_gfx8")),
                fmt=fmt_enum,
                opcode=inst["opcode"],
                size_bytes=inst["size_bytes"],
                mnemonic=c_str(inst["mnemonic"]),
                semantic_family=c_str(inst.get("semantic_family", "unknown")),
                issue_family=c_str(inst.get("issue_family", "unknown")),
            )
        )
        encoding_entries.append(
            "  {{ {id}, GcnInstFormatClass::{fmt}, {opcode}, {size_bytes}, {mnemonic} }}".format(
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
    args = parser.parse_args()

    db_dir = pathlib.Path(args.db_dir)
    out_header = pathlib.Path(args.out_header)
    out_cpp = pathlib.Path(args.out_cpp)
    out_header.parent.mkdir(parents=True, exist_ok=True)
    out_cpp.parent.mkdir(parents=True, exist_ok=True)

    emit_header(out_header)
    emit_cpp(db_dir, out_cpp)


if __name__ == "__main__":
    main()
