from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from zipfile import ZipFile

import xml.etree.ElementTree as ET


NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract XLSX workbooks into CSV plus a lightweight index.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Workbook paths.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("references/paper_source_data/s43588-026-00977"),
        help="Directory where extracted CSVs and indexes are written.",
    )
    return parser.parse_args()


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "sheet"


def _col_to_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return max(value - 1, 0)


def _load_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root.findall("main:si", NS):
        parts = [node.text or "" for node in si.iterfind(".//main:t", NS)]
        strings.append("".join(parts))
    return strings


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iterfind(".//main:t", NS))

    value_node = cell.find("main:v", NS)
    if value_node is None or value_node.text is None:
        return ""

    raw = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (IndexError, ValueError):
            return raw
    if cell_type == "b":
        return "TRUE" if raw == "1" else "FALSE"
    return raw


def _sheet_rows(zf: ZipFile, target: str, shared_strings: list[str]) -> list[list[str]]:
    root = ET.fromstring(zf.read(target))
    sheet_data = root.find("main:sheetData", NS)
    if sheet_data is None:
        return []

    rows: list[list[str]] = []
    for row in sheet_data.findall("main:row", NS):
        cells = row.findall("main:c", NS)
        if not cells:
            rows.append([])
            continue

        width = max(_col_to_index(cell.attrib.get("r", "A1")) for cell in cells) + 1
        values = [""] * width
        for cell in cells:
            idx = _col_to_index(cell.attrib.get("r", "A1"))
            values[idx] = _cell_value(cell, shared_strings)
        rows.append(values)
    return rows


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)


def _write_workbook_index(workbook_dir: Path, workbook_name: str, sheet_meta: list[dict[str, object]]) -> None:
    manifest_path = workbook_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"workbook": workbook_name, "sheets": sheet_meta}, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [f"# {workbook_name}", ""]
    for meta in sheet_meta:
        lines.append(f"- `{meta['sheet_name']}` -> `{meta['csv_path']}` ({meta['rows']} rows × {meta['cols']} cols)")
    lines.append("")
    (workbook_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def extract_workbook(workbook_path: Path, output_dir: Path) -> dict[str, object]:
    workbook_name = workbook_path.name
    workbook_dir = output_dir / _sanitize(workbook_path.stem)
    workbook_dir.mkdir(parents=True, exist_ok=True)

    sheet_meta: list[dict[str, object]] = []
    with ZipFile(workbook_path) as zf:
        shared_strings = _load_shared_strings(zf)
        workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
        rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_root.findall("pkg:Relationship", NS)}

        sheets = workbook_root.find("main:sheets", NS)
        if sheets is None:
            raise ValueError(f"No sheets found in {workbook_path}")

        for sheet in sheets.findall("main:sheet", NS):
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = "xl/" + rel_map[rid]
            rows = _sheet_rows(zf, target, shared_strings)
            csv_name = f"{_sanitize(sheet.attrib['name'])}.csv"
            csv_path = workbook_dir / csv_name
            _write_csv(csv_path, rows)

            sheet_meta.append(
                {
                    "sheet_name": sheet.attrib["name"],
                    "csv_path": str(csv_path.relative_to(output_dir.parent.parent.parent)),
                    "rows": len(rows),
                    "cols": max((len(row) for row in rows), default=0),
                }
            )

    _write_workbook_index(workbook_dir, workbook_name, sheet_meta)
    return {"workbook": workbook_name, "dir": str(workbook_dir), "sheets": sheet_meta}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"workbooks": []}
    for workbook in args.inputs:
        manifest["workbooks"].append(extract_workbook(workbook.resolve(), output_dir))

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = ["# Source Data Index", ""]
    for workbook in manifest["workbooks"]:
        lines.append(f"## {workbook['workbook']}")
        lines.append("")
        for sheet in workbook["sheets"]:
            lines.append(
                f"- `{sheet['sheet_name']}` -> `{Path(sheet['csv_path']).as_posix()}` ({sheet['rows']} rows × {sheet['cols']} cols)"
            )
        lines.append("")
    (output_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote extracted source data to {output_dir}")


if __name__ == "__main__":
    main()
