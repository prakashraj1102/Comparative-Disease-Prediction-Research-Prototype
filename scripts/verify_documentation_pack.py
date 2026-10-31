#!/usr/bin/env python3
"""Verify documentation files against DOCUMENTATION_MANIFEST.json.

Run this script from the repository root after extracting the documentation pack.
It is read-only and does not modify any files.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "DOCUMENTATION_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        print(f"ERROR: manifest not found: {MANIFEST}")
        return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = manifest.get("files", [])
    if not entries:
        print("ERROR: manifest contains no file entries")
        return 2

    failures = []
    for entry in entries:
        relative = Path(entry["path"])
        path = (ROOT / relative).resolve()
        if ROOT.resolve() not in path.parents:
            failures.append((entry["path"], "unsafe manifest path"))
            continue
        if not path.is_file():
            failures.append((entry["path"], "missing"))
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != entry["size_bytes"]:
            failures.append((entry["path"], f"size mismatch ({actual_size})"))
        elif actual_hash != entry["sha256"]:
            failures.append((entry["path"], "SHA-256 mismatch"))
        else:
            print(f"PASS {entry['path']}")

    if failures:
        print("\nDOCUMENTATION INTEGRITY: FAIL")
        for name, reason in failures:
            print(f"FAIL {name}: {reason}")
        return 1

    print(f"\nDOCUMENTATION INTEGRITY: PASS ({len(entries)} files SHA-256 verified)")
    print("No documentation files were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
