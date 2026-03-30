#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
from collections import defaultdict

import yaml


def load_instructions(repo_root: pathlib.Path) -> list[dict]:
    data = yaml.safe_load((repo_root / "src/spec/gcn_db/instructions.yaml").read_text(encoding="utf-8"))
    return data["instructions"]


def extract_quoted_mnemonics(text: str, known: set[str]) -> set[str]:
    quoted = set(re.findall(r'"([A-Za-z0-9_]+(?:_e32|_e64)?)"', text))
    covered: set[str] = set()
    for mnemonic in quoted:
        if mnemonic in known:
            covered.add(mnemonic)
            continue
        if (mnemonic + "_e32") in known:
            covered.add(mnemonic + "_e32")
            continue
        if (mnemonic + "_e64") in known:
            covered.add(mnemonic + "_e64")
    return covered


def percent(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(100.0 * part / total):.1f}%"


def summarize_family(name: str, items: list[dict], coverage: dict[str, set[str]]) -> dict[str, object]:
    mnemonics = [item["mnemonic"] for item in items]
    total = len(mnemonics)
    unique = set(mnemonics)
    row: dict[str, object] = {
        "family": name,
        "total": total,
        "unique_total": len(unique),
    }
    for key, covered in coverage.items():
        hit = len(unique & covered)
        row[key] = {"covered": hit, "percent": percent(hit, len(unique))}
    return row


def summarize_format(name: str, items: list[dict], coverage: dict[str, set[str]]) -> dict[str, object]:
    mnemonics = [item["mnemonic"] for item in items]
    unique = set(mnemonics)
    row: dict[str, object] = {
        "format": name,
        "total": len(mnemonics),
        "unique_total": len(unique),
    }
    for key, covered in coverage.items():
        hit = len(unique & covered)
        row[key] = {"covered": hit, "percent": percent(hit, len(unique))}
    return row


def build_report(repo_root: pathlib.Path) -> tuple[str, dict]:
    instructions = load_instructions(repo_root)
    known = {item["mnemonic"] for item in instructions}
    full_opcode_text = (repo_root / "src/instruction/encoded/internal/generated_encoded_gcn_full_opcode_table.cpp").read_text(
        encoding="utf-8"
    )
    full_opcode_rows = re.findall(
        r'GcnIsaOpcodeDescriptor\{\s*GcnIsaOpType::([A-Za-z0-9]+),\s*0x[0-9a-f]+,\s*"([^"]+)"',
        full_opcode_text,
    )
    full_opcode_unique = {mnemonic for _, mnemonic in full_opcode_rows}

    source_files = {
        "raw_object_support": repo_root / "src/exec/encoded/binding/raw_gcn_instruction_binding.cpp",
        "decode_unit_tests": repo_root / "tests/decode/gcn_inst_decoder_test.cpp",
        "binding_unit_tests": repo_root / "tests/exec/encoded/binding/raw_gcn_instruction_binding_test.cpp",
        "object_unit_tests": repo_root / "tests/exec/encoded/object/raw_gcn_instruction_object_execute_test.cpp",
        "semantic_unit_tests": repo_root / "tests/exec/encoded/semantics/raw_gcn_semantic_execute_test.cpp",
        "registry_unit_tests": repo_root / "tests/exec/encoded/semantics/raw_gcn_semantic_handler_registry_test.cpp",
        "loader_integration_tests": repo_root / "tests/loader/amdgpu_code_object_decoder_test.cpp",
    }

    coverage_sets: dict[str, set[str]] = {}
    for key, path in source_files.items():
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        coverage_sets[key] = extract_quoted_mnemonics(text, known)

    coverage_sets["exec_test_union"] = (
        coverage_sets["binding_unit_tests"]
        | coverage_sets["object_unit_tests"]
        | coverage_sets["semantic_unit_tests"]
        | coverage_sets["registry_unit_tests"]
    )
    coverage_sets["all_test_union"] = (
        coverage_sets["decode_unit_tests"]
        | coverage_sets["exec_test_union"]
        | coverage_sets["loader_integration_tests"]
    )

    by_family: dict[str, list[dict]] = defaultdict(list)
    by_format: dict[str, list[dict]] = defaultdict(list)
    for item in instructions:
        by_family[item["semantic_family"]].append(item)
        by_format[item["format"]].append(item)

    family_rows = [
        summarize_family(name, items, {
            "raw_object_support": coverage_sets["raw_object_support"],
            "decode_unit_tests": coverage_sets["decode_unit_tests"],
            "exec_test_union": coverage_sets["exec_test_union"],
            "loader_integration_tests": coverage_sets["loader_integration_tests"],
            "all_test_union": coverage_sets["all_test_union"],
        })
        for name, items in sorted(by_family.items())
    ]
    format_rows = [
        summarize_format(name, items, {
            "raw_object_support": coverage_sets["raw_object_support"],
            "decode_unit_tests": coverage_sets["decode_unit_tests"],
            "exec_test_union": coverage_sets["exec_test_union"],
            "loader_integration_tests": coverage_sets["loader_integration_tests"],
            "all_test_union": coverage_sets["all_test_union"],
        })
        for name, items in sorted(by_format.items())
    ]

    unsupported = sorted(known - coverage_sets["raw_object_support"])
    supported_without_tests = sorted(
        coverage_sets["raw_object_support"] - coverage_sets["all_test_union"]
    )

    report = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "tracked_instruction_entries": len(instructions),
        "tracked_instruction_unique_total": len(known),
        "full_opcode_rows": len(full_opcode_rows),
        "full_opcode_unique_total": len(full_opcode_unique),
        "coverage": {
            key: {
                "covered": len(value),
                "percent": percent(len(value), len(known)),
                "mnemonics": sorted(value),
            }
            for key, value in coverage_sets.items()
        },
        "by_semantic_family": family_rows,
        "by_format_class": format_rows,
        "unsupported_raw_object_mnemonics": unsupported,
        "supported_without_any_tests": supported_without_tests,
    }

    lines = [
        "# ISA Coverage Report",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "Coverage scope:",
        f"- tracked instruction entries from `src/spec/gcn_db/instructions.yaml`: `{report['tracked_instruction_entries']}`",
        f"- tracked unique mnemonics from `src/spec/gcn_db/instructions.yaml`: `{report['tracked_instruction_unique_total']}`",
        f"- full opcode rows extracted from `generated_gcn_full_opcode_table.cpp`: `{report['full_opcode_rows']}`",
        f"- full unique mnemonics extracted from `generated_gcn_full_opcode_table.cpp`: `{report['full_opcode_unique_total']}`",
        "",
        "Tracked-subset coverage:",
        f"- raw object support: `{report['coverage']['raw_object_support']['covered']}` / `{report['tracked_instruction_unique_total']}` ({report['coverage']['raw_object_support']['percent']})",
        f"- decode unit tests: `{report['coverage']['decode_unit_tests']['covered']}` / `{report['tracked_instruction_unique_total']}` ({report['coverage']['decode_unit_tests']['percent']})",
        f"- exec unit tests: `{report['coverage']['exec_test_union']['covered']}` / `{report['tracked_instruction_unique_total']}` ({report['coverage']['exec_test_union']['percent']})",
        f"- loader integration tests: `{report['coverage']['loader_integration_tests']['covered']}` / `{report['tracked_instruction_unique_total']}` ({report['coverage']['loader_integration_tests']['percent']})",
        f"- any tests: `{report['coverage']['all_test_union']['covered']}` / `{report['tracked_instruction_unique_total']}` ({report['coverage']['all_test_union']['percent']})",
        "",
        "## Per Semantic Family",
        "",
        "| Family | Unique | Raw Object | Decode Tests | Exec Tests | Loader Tests | Any Tests |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in family_rows:
        lines.append(
            f"| `{row['family']}` | `{row['unique_total']}` | "
            f"`{row['raw_object_support']['covered']}` ({row['raw_object_support']['percent']}) | "
            f"`{row['decode_unit_tests']['covered']}` ({row['decode_unit_tests']['percent']}) | "
            f"`{row['exec_test_union']['covered']}` ({row['exec_test_union']['percent']}) | "
            f"`{row['loader_integration_tests']['covered']}` ({row['loader_integration_tests']['percent']}) | "
            f"`{row['all_test_union']['covered']}` ({row['all_test_union']['percent']}) |"
        )

    lines.extend(
        [
            "",
            "## Per Encoding / Format Class",
            "",
            "| Format | Unique | Raw Object | Decode Tests | Exec Tests | Loader Tests | Any Tests |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in format_rows:
        lines.append(
            f"| `{row['format']}` | `{row['unique_total']}` | "
            f"`{row['raw_object_support']['covered']}` ({row['raw_object_support']['percent']}) | "
            f"`{row['decode_unit_tests']['covered']}` ({row['decode_unit_tests']['percent']}) | "
            f"`{row['exec_test_union']['covered']}` ({row['exec_test_union']['percent']}) | "
            f"`{row['loader_integration_tests']['covered']}` ({row['loader_integration_tests']['percent']}) | "
            f"`{row['all_test_union']['covered']}` ({row['all_test_union']['percent']}) |"
        )

    lines.extend(
        [
            "",
            "## Supported But Untested",
            "",
            f"Count: `{len(supported_without_tests)}`",
            "",
        ]
    )
    for mnemonic in supported_without_tests[:40]:
        lines.append(f"- `{mnemonic}`")

    lines.extend(
        [
            "",
            "## Missing Raw Object Support",
            "",
            f"Count: `{len(unsupported)}`",
            "",
        ]
    )
    for mnemonic in unsupported[:40]:
        lines.append(f"- `{mnemonic}`")

    return "\n".join(lines) + "\n", report


def main() -> int:
    parser = argparse.ArgumentParser(description="Report ISA support and test coverage.")
    parser.add_argument(
        "--repo-root",
        default=pathlib.Path(__file__).resolve().parents[1],
        type=pathlib.Path,
        help="Repository root",
    )
    parser.add_argument(
        "--markdown-out",
        default=None,
        type=pathlib.Path,
        help="Optional markdown output path",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        type=pathlib.Path,
        help="Optional JSON output path",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    markdown, report = build_report(repo_root)

    markdown_out = args.markdown_out or (repo_root / "docs/isa_coverage_report.md")
    json_out = args.json_out or (repo_root / "docs/isa_coverage_report.json")

    markdown_out.write_text(markdown, encoding="utf-8")
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(markdown)
    print(f"wrote {markdown_out}")
    print(f"wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
