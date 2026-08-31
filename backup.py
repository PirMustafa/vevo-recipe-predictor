"""Snapshot the project's code and trained models, with integrity verification.

    python backup.py "label"                # code + models
    python backup.py "label" --code-only    # code only (fast, ~350 KB)
    python backup.py "label" --with-data    # also copy the source .cxf files
    python backup.py --list                 # show existing snapshots
    python backup.py --restore <dir-name>   # roll back (takes a safety snapshot first)

This repository is not under version control, so these snapshots are the only
way back from a mistake. Two habits this is meant to fix:

  1. Backing up only BEFORE risky operations protects what already existed and
     leaves new work exposed. Five source modules once sat unbacked for two days
     because the most recent snapshot predated them. Run this AFTER finishing a
     piece of work, not just before starting a dangerous one.
  2. Trained models are expensive rather than irreplaceable -- regenerating the
     full set costs GPU hours -- so they are included by default, and every
     copied file is SHA256-verified rather than assumed good.

Snapshots follow the convention already in the repo:
    _backup_pre_stage12_experiments/src_backup_<timestamp>_<label>/
    _backup_pre_stage12_experiments/models_backup_<timestamp>_<label>/
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).parent
BACKUP_ROOT = ROOT / "_backup_pre_stage12_experiments"

# Directories copied whole (code + front end). .venv and caches are never included.
CODE_DIRS = ("src", "static", "templates")
CODE_GLOBS = ("*.py", "requirements.txt")
DATA_GLOBS = ("*.cxf",)
SKIP_DIRS = {"__pycache__", ".venv", ".git"}

MANIFEST = "manifest.json"


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_verified(src: pathlib.Path, dst: pathlib.Path) -> tuple[str, int]:
    """Copy one file and confirm the copy is byte-identical."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    a, b = _sha256(src), _sha256(dst)
    if a != b:
        raise RuntimeError(f"copy verification FAILED for {src} -> {dst}")
    return a, src.stat().st_size


def _slug(label: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_").lower()
    return s or "snapshot"


def create(label: str, code_only: bool, with_data: bool) -> None:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug(label)
    src_dir = BACKUP_ROOT / f"src_backup_{ts}_{slug}"
    models_dir = BACKUP_ROOT / f"models_backup_{ts}_{slug}"

    entries: list[dict] = []
    total = 0

    def take(path: pathlib.Path, base: pathlib.Path, dest_root: pathlib.Path):
        nonlocal total
        rel = path.relative_to(base)
        digest, size = _copy_verified(path, dest_root / rel)
        entries.append({"file": str(rel).replace("\\", "/"),
                        "dest": dest_root.name, "sha256": digest, "bytes": size})
        total += size

    print(f"Snapshot: {label}   ({ts})")

    # ---- code ----
    for pattern in CODE_GLOBS:
        for f in sorted(ROOT.glob(pattern)):
            if f.is_file():
                take(f, ROOT, src_dir)
    for d in CODE_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if f.is_file() and not any(p in SKIP_DIRS for p in f.parts):
                take(f, ROOT, src_dir)
    n_code = len(entries)
    print(f"  code:   {n_code} files -> {src_dir.name}")

    # ---- raw source data (optional; large and never modified) ----
    if with_data:
        for pattern in DATA_GLOBS:
            for f in sorted(ROOT.glob(pattern)):
                if f.is_file():
                    take(f, ROOT, src_dir)
        print(f"  data:   {len(entries) - n_code} .cxf files")

    # ---- trained models ----
    if not code_only:
        mbase = ROOT / "models"
        n_before = len(entries)
        if mbase.is_dir():
            for f in sorted(mbase.rglob("*.joblib")):
                take(f, mbase, models_dir)
        print(f"  models: {len(entries) - n_before} files -> {models_dir.name}")
    else:
        print("  models: skipped (--code-only)")

    manifest = {
        "label": label,
        "timestamp": ts,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "code_only": code_only,
        "with_data": with_data,
        "total_bytes": total,
        "files": entries,
    }
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))

    print(f"  verified: {len(entries)} files, all SHA256-matched")
    print(f"  size: {total / 1e6:.1f} MB")
    print(f"  manifest: {src_dir / MANIFEST}")


def listing() -> None:
    if not BACKUP_ROOT.is_dir():
        print("No snapshots yet.")
        return
    rows = []
    for d in sorted(BACKUP_ROOT.iterdir()):
        if not d.is_dir():
            continue
        man = d / MANIFEST
        if man.is_file():
            m = json.loads(man.read_text())
            rows.append((d.name, m.get("created", "?"), m.get("label", ""),
                         m.get("total_bytes", 0), len(m.get("files", []))))
        else:
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            rows.append((d.name, "(pre-dates manifests)", "", size,
                         sum(1 for f in d.rglob("*") if f.is_file())))
    print(f"{'snapshot':<52} {'created':<20} {'files':>6} {'size':>10}")
    for name, created, label, size, n in rows:
        print(f"{name:<52} {created:<20} {n:>6} {size / 1e6:>9.1f}M")
    print(f"\n{len(rows)} snapshot directories in {BACKUP_ROOT.name}")


def restore(name: str) -> None:
    """Roll back from a snapshot, taking a safety snapshot of the current state first."""
    target = BACKUP_ROOT / name
    if not target.is_dir():
        print(f"No such snapshot: {name}\nRun --list to see what exists.")
        sys.exit(1)

    print(f"About to restore from {name}.")
    print("Taking a safety snapshot of the CURRENT state first, so this is reversible.")
    create(f"pre_restore_of_{name}", code_only=False, with_data=False)

    restored = 0
    if name.startswith("src_backup"):
        for f in target.rglob("*"):
            if f.is_file() and f.name != MANIFEST:
                dest = ROOT / f.relative_to(target)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                restored += 1
    elif name.startswith("models_backup"):
        for f in target.rglob("*.joblib"):
            dest = ROOT / "models" / f.relative_to(target)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            restored += 1
    else:
        print("Unrecognised snapshot type; expected src_backup_* or models_backup_*")
        sys.exit(1)

    print(f"\nRestored {restored} files from {name}.")
    print("Note: this OVERWRITES matching files but does not delete files added since")
    print("the snapshot was taken. Check for strays if you expected an exact rollback.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Snapshot project code and trained models, with verification")
    ap.add_argument("label", nargs="?", help="short description, e.g. \"backing modes wired in\"")
    ap.add_argument("--code-only", action="store_true", help="skip the trained models")
    ap.add_argument("--with-data", action="store_true", help="also copy the source .cxf files")
    ap.add_argument("--list", action="store_true", help="list existing snapshots")
    ap.add_argument("--restore", metavar="DIR", help="restore from a snapshot directory name")
    args = ap.parse_args()

    if args.list:
        listing()
    elif args.restore:
        restore(args.restore)
    elif args.label:
        create(args.label, args.code_only, args.with_data)
    else:
        ap.print_help()
        print("\nTip: run this after finishing a piece of work, not only before risky ones.")


if __name__ == "__main__":
    main()
