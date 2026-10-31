#!/usr/bin/env python3
"""Verify publication notebook structure and code-cell source hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "notebook_source_integrity.json"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []

    for item in manifest["notebooks"]:
        path = ROOT / "research" / "notebooks" / item["name"]
        if not path.is_file():
            failures.append(f"Missing notebook: {path}")
            continue

        raw = path.read_bytes()
        actual_file_hash = hashlib.sha256(raw).hexdigest()
        if actual_file_hash != item["publication_sha256"]:
            failures.append(f"Notebook file hash mismatch: {item['name']}")

        try:
            notebook = nbformat.reads(raw.decode("utf-8"), as_version=4)
            nbformat.validate(notebook)
        except Exception as exc:  # pragma: no cover - exact error depends on nbformat
            failures.append(f"Invalid notebook {item['name']}: {exc}")
            continue

        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        actual_source_hashes = [sha256_text(cell.source) for cell in code_cells]
        if actual_source_hashes != item["code_cell_source_hashes"]:
            failures.append(f"Code-cell source hash mismatch: {item['name']}")

        if any(cell.outputs or cell.execution_count is not None for cell in code_cells):
            failures.append(f"Unexpected saved outputs/execution counts: {item['name']}")

        print(
            f"PASS {item['name']}: "
            f"{len(code_cells)} code cells, source hashes match, outputs cleared"
        )

    if failures:
        print("\nNOTEBOOK INTEGRITY: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nNOTEBOOK INTEGRITY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
