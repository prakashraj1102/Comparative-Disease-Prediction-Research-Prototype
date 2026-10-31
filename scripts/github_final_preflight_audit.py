#!/usr/bin/env python3
"""Read-only final pre-push audit for the Comparative Disease Prediction repository.

Uses only Python's standard library. Writes a concise report to ~/Downloads.
Does not modify project files.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
REPORT = Path.home() / "Downloads" / "GitHub_Final_Preflight_Audit.txt"

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".ipynb", ".csv", ".toml",
    ".yaml", ".yml", ".cff", ".ini", ".cfg", ".gitignore"
}
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache"}

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
LOCAL_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|/mnt/data|C:\\\\Users\\\\[^\\\\\s]+)",
    re.IGNORECASE,
)
SECRET_PATTERNS = [
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk_live_[A-Za-z0-9]{12,}\b"),
]
SENSITIVE_COLUMN_RE = re.compile(
    r"^(?:subject_id|hadm_id|stay_id|patient_id|mrn|patient_name|full_name|"
    r"email|phone|telephone|dob|date_of_birth|address|ssn|passport)$",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_project_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        yield p


def scan_text_blob(text: str, rel: str, findings: list[str]):
    if EMAIL_RE.search(text):
        findings.append(f"EMAIL-LIKE STRING: {rel}")
    if LOCAL_PATH_RE.search(text):
        findings.append(f"PERSONAL/LOCAL PATH: {rel}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(f"POSSIBLE SECRET TOKEN/KEY: {rel}")
            break


def main() -> int:
    lines: list[str] = []
    warnings: list[str] = []
    findings: list[str] = []
    files = list(iter_project_files())
    total_bytes = sum(p.stat().st_size for p in files)

    lines.append("GITHUB FINAL PRE-PUSH AUDIT (READ-ONLY)")
    lines.append(f"Project: {ROOT}")
    lines.append(f"Files scanned: {len(files)}")
    lines.append(f"Project file size: {total_bytes:,} bytes ({total_bytes / 1024 / 1024:.1f} MiB)")

    # Required root files and folders.
    lines.append("\n[REQUIRED STRUCTURE]")
    required = [
        "README.md", ".gitignore", "requirements.txt",
        "requirements-research.txt", "requirements-dev.txt",
        "app.py", "artifacts", "configurations", "core", "frontend",
        "research/notebooks/01_stable_fci_causal_feature_selection.ipynb",
        "research/notebooks/02_star_ci_selected_feature_screening.ipynb",
        "research/results/stable_fci", "research/results/star_ci",
        "docs", "screenshots", "scripts", "tests",
    ]
    for item in required:
        exists = (ROOT / item).exists()
        lines.append(f"{'PASS' if exists else 'MISSING'}: {item}")
        if not exists:
            warnings.append(f"Missing required path: {item}")

    # .gitignore review.
    lines.append("\n[.GITIGNORE CHECK]")
    gi_path = ROOT / ".gitignore"
    gi = gi_path.read_text(errors="replace") if gi_path.exists() else ""
    checks = {
        "ignores Python caches": "__pycache__/" in gi,
        "ignores virtual environments": ".venv/" in gi,
        "ignores environment secrets": ".env" in gi,
        "does not globally ignore notebooks": "*.ipynb" not in gi,
        "does not globally ignore joblib models": "*.joblib" not in gi,
        "excludes original train dataset": "**/train_encoded.csv" in gi,
        "excludes original STAR test dataset": "**/Test_Only_Symptoms.csv" in gi,
    }
    for label, ok in checks.items():
        lines.append(f"{'PASS' if ok else 'CHECK'}: {label}")
        if not ok:
            warnings.append(f".gitignore: {label}")

    # Exact model inventory and manifest hashes.
    lines.append("\n[MODEL ARTIFACTS]")
    manifest_path = ROOT / "artifact_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            models = manifest.get("models", [])
            for model in models:
                rel = model.get("path", "")
                p = ROOT / rel
                if not p.is_file():
                    lines.append(f"MISSING: {rel}")
                    warnings.append(f"Missing model artifact: {rel}")
                    continue
                actual_hash = sha256(p)
                size_ok = p.stat().st_size == model.get("size_bytes")
                hash_ok = actual_hash == model.get("sha256")
                lines.append(
                    f"{'PASS' if size_ok and hash_ok else 'FAIL'}: {rel} "
                    f"({p.stat().st_size:,} bytes; size={size_ok}; sha256={hash_ok})"
                )
                if not (size_ok and hash_ok):
                    warnings.append(f"Model manifest mismatch: {rel}")
        except Exception as exc:
            lines.append(f"ERROR reading artifact_manifest.json: {exc}")
            warnings.append("Could not verify artifact manifest")
    else:
        lines.append("MISSING: artifact_manifest.json")
        warnings.append("Missing artifact_manifest.json")

    # README image links.
    lines.append("\n[README IMAGE LINKS]")
    readme = ROOT / "README.md"
    if readme.exists():
        image_refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme.read_text(errors="replace"))
        for ref in image_refs:
            if re.match(r"^[a-z]+://", ref, re.IGNORECASE):
                continue
            target = (ROOT / ref).resolve()
            ok = target.is_file() and (ROOT == target or ROOT in target.parents)
            lines.append(f"{'PASS' if ok else 'FAIL'}: {ref}")
            if not ok:
                warnings.append(f"README image missing/outside repository: {ref}")
    else:
        lines.append("README.md missing")

    # Notebook outputs and source-level privacy scan.
    lines.append("\n[NOTEBOOK CHECKS]")
    for name in [
        "research/notebooks/01_stable_fci_causal_feature_selection.ipynb",
        "research/notebooks/02_star_ci_selected_feature_screening.ipynb",
    ]:
        p = ROOT / name
        if not p.exists():
            continue
        try:
            nb = json.loads(p.read_text(errors="replace"))
            cells = nb.get("cells", [])
            outputs = sum(len(c.get("outputs", [])) for c in cells if c.get("cell_type") == "code")
            exec_counts = sum(c.get("execution_count") is not None for c in cells if c.get("cell_type") == "code")
            lines.append(f"{name}: outputs={outputs}; execution_counts_set={exec_counts}")
            if outputs or exec_counts:
                warnings.append(f"Notebook contains saved execution outputs/counters: {name}")
        except Exception as exc:
            lines.append(f"Could not parse {name}: {exc}")
            warnings.append(f"Notebook could not be parsed: {name}")

    # Scan text-based project files for common accidental disclosures.
    lines.append("\n[PRIVACY / SECRET PATTERN SCAN]")
    scanned = 0
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if p.suffix.lower() in TEXT_SUFFIXES or p.name == ".gitignore":
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                scan_text_blob(text, rel, findings)
                scanned += 1
            except Exception:
                pass
        elif p.name.endswith(".csv.gz"):
            # Scan the compressed CSV as text without writing an extracted copy.
            try:
                with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        scan_text_blob(line, rel, findings)
                scanned += 1
            except Exception as exc:
                warnings.append(f"Could not scan compressed CSV {rel}: {exc}")
    lines.append(f"Text-based files scanned: {scanned}")
    if findings:
        for item in sorted(set(findings)):
            lines.append(f"REVIEW: {item}")
            warnings.append(item)
    else:
        lines.append("No common email, personal-path, or high-confidence secret patterns found.")

    # CSV headers for obvious direct identifiers.
    lines.append("\n[CSV IDENTIFIER COLUMN CHECK]")
    csv_count = 0
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if p.suffix.lower() == ".csv" or p.name.endswith(".csv.gz"):
            try:
                opener = gzip.open if p.name.endswith(".gz") else open
                with opener(p, "rt", encoding="utf-8", errors="replace", newline="") as f:
                    header = next(csv.reader(f), [])
                flagged = [col for col in header if SENSITIVE_COLUMN_RE.match(col.strip())]
                csv_count += 1
                if flagged:
                    lines.append(f"REVIEW: {rel} columns={flagged}")
                    warnings.append(f"Possible direct-identifier columns in {rel}: {flagged}")
            except Exception as exc:
                lines.append(f"CHECK: {rel} header read failed: {exc}")
    lines.append(f"CSV/CSV.GZ files checked: {csv_count}")

    # File-size audit.
    lines.append("\n[FILE SIZE CHECK]")
    over50 = []
    over100 = []
    for p in files:
        size = p.stat().st_size
        if size > 100 * 1024 * 1024:
            over100.append((size, p.relative_to(ROOT).as_posix()))
        elif size > 50 * 1024 * 1024:
            over50.append((size, p.relative_to(ROOT).as_posix()))
    lines.append(f"Files >50 MiB: {len(over50)}")
    for size, rel in sorted(over50, reverse=True):
        lines.append(f"WARNING: {rel}: {size:,} bytes ({size / 1024 / 1024:.2f} MiB)")
        warnings.append(f"GitHub >50 MiB warning expected: {rel}")
    lines.append(f"Files >100 MiB: {len(over100)}")
    for size, rel in sorted(over100, reverse=True):
        lines.append(f"BLOCKER: {rel}: {size:,} bytes")
        warnings.append(f"GitHub regular Git file limit exceeded: {rel}")

    # Generated cache directories.
    lines.append("\n[CACHE CHECK]")
    caches = [p for p in ROOT.rglob("__pycache__") if p.is_dir()]
    pyc = [p for p in ROOT.rglob("*.pyc") if p.is_file()]
    lines.append(f"__pycache__ directories: {len(caches)}")
    lines.append(f".pyc files: {len(pyc)}")
    if caches or pyc:
        warnings.append("Python cache files remain; .gitignore excludes them but remove before push")

    # License and rights reminder.
    lines.append("\n[PUBLICATION RIGHTS CHECK]")
    license_files = [p.name for p in ROOT.glob("LICENSE*") if p.is_file()]
    lines.append(f"License files: {license_files if license_files else 'NONE'}")
    if not license_files:
        warnings.append("No repository license file; decide license and confirm institutional/data/model rights before public release")
    lines.append("Manual confirmation still required: institutional/supervisor permission and dataset/model redistribution terms.")

    lines.append("\n[SUMMARY]")
    lines.append(f"Warnings/reviews: {len(set(warnings))}")
    for item in sorted(set(warnings)):
        lines.append(f"- {item}")
    lines.append("\nThis audit is a screening tool, not a guarantee that all sensitive content or licensing issues have been identified.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("GitHub preflight audit completed (read-only).")
    print(f"Report: {REPORT}")
    print(f"Report size: {REPORT.stat().st_size:,} bytes")
    print(f"Review items: {len(set(warnings))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
