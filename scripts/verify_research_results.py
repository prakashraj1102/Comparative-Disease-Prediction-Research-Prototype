#!/usr/bin/env python3
"""Verify the archived research-result files against their SHA-256 manifest.

Run from the repository root:
    python scripts/verify_research_results.py

This script reads files only. It does not alter or decompress the archived files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (defaults to the current working directory).",
    )
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    manifest_path = root / "research" / "results" / "RESULTS_FILE_MANIFEST.json"

    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"ERROR: invalid manifest: {exc}", file=sys.stderr)
        return 2

    failures = []
    verified = 0
    for entry in entries:
        rel = Path(entry["path"])
        # Disallow absolute paths and path traversal in the manifest.
        if rel.is_absolute() or ".." in rel.parts:
            failures.append((str(rel), "unsafe path in manifest"))
            continue
        path = root / rel
        if not path.is_file():
            failures.append((str(rel), "missing file"))
            continue
        actual = sha256_file(path)
        expected = entry.get("sha256")
        if actual != expected:
            failures.append((str(rel), "SHA-256 mismatch"))
            continue
        verified += 1

    if failures:
        print("RESEARCH RESULTS INTEGRITY: FAIL")
        for rel, reason in failures:
            print(f"FAIL {rel}: {reason}")
        print(f"Verified: {verified}/{len(entries)}")
        return 1

    print(f"RESEARCH RESULTS INTEGRITY: PASS ({verified} files SHA-256 verified)")
    print("No research result files were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
