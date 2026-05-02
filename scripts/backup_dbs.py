#!/usr/bin/env python3
"""
Dragon Trader — Nightly DB backup with retention.

Snapshots all SQLite DBs (data/*.db + trade_journal.db) using sqlite3's online
.backup API (lockless, atomic) and gzip-compresses them into:

    ~/backups/dragon/YYYY-MM-DD_HH-MM-SS/
        ├── beast.db.gz
        ├── rl_learner.db.gz
        ├── trade_journal.db.gz
        └── manifest.json   (timestamp, sizes, sha256 of original + .gz)

Retention policy (applied after each successful backup):
  - Keep ALL backups from the last 7 days  (hourly granularity OK)
  - Keep ONE per day        for days  8 .. 30 (daily)
  - Keep ONE per week       for weeks 5 .. 52 (weekly, Monday)
  - Drop everything older than 12 months

Idempotent: re-running within the same second skips creating a duplicate.

Usage:
    python3 -B scripts/backup_dbs.py [--target-dir DIR]

Run via launchd: launchd/com.dragon.backup.plist (03:00 daily).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_TARGET = Path.home() / "backups" / "dragon"
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Regex must match folder names produced by _stamp_dir() so we don't pick up
# foreign directories (manual copies, partial runs, etc).
STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
STAMP_FMT = "%Y-%m-%d_%H-%M-%S"


def _setup_logging() -> logging.Logger:
    log = logging.getLogger("dragon.backup")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(LOG_DIR / "backup.log", mode="a")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


log = _setup_logging()


# ═══════════════════════════════════════════════════════════════
#  DB DISCOVERY
# ═══════════════════════════════════════════════════════════════

def _discover_dbs() -> List[Path]:
    """All SQLite DBs we care about: data/*.db plus the canonical trade_journal.

    Note: a 0-byte trade_journal.db at repo root is stale (see issue #21);
    we skip any 0-byte file regardless of location.
    """
    candidates: List[Path] = []
    if DATA_DIR.is_dir():
        for p in sorted(DATA_DIR.glob("*.db")):
            candidates.append(p)
    # Some legacy code paths reference root-level trade_journal.db, but the
    # canonical path is data/trade_journal.db. Include any non-zero-byte one
    # we find at the root only as a safety net (and warn loudly).
    legacy_journal = REPO_ROOT / "trade_journal.db"
    if legacy_journal.exists() and legacy_journal.stat().st_size > 0:
        log.warning("Root-level trade_journal.db is non-empty (%d bytes); including in backup. "
                    "Canonical path is data/trade_journal.db — investigate.",
                    legacy_journal.stat().st_size)
        candidates.append(legacy_journal)

    # De-dup by resolved path
    seen, unique = set(), []
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if p.stat().st_size == 0:
            log.info("Skipping 0-byte DB: %s", p)
            continue
        unique.append(p)
    return unique


# ═══════════════════════════════════════════════════════════════
#  BACKUP CORE
# ═══════════════════════════════════════════════════════════════

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _online_backup(src: Path, dst_uncompressed: Path) -> None:
    """Use sqlite3's .backup() API — does NOT lock the source DB."""
    src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30.0)
    try:
        # Backup to a temporary file first, then move. Atomic on same filesystem.
        with tempfile.NamedTemporaryFile(
            dir=str(dst_uncompressed.parent),
            prefix=f".{dst_uncompressed.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        dst_conn = sqlite3.connect(str(tmp_path), timeout=30.0)
        try:
            src_conn.backup(dst_conn, pages=0)  # 0 = whole DB in one call
        finally:
            dst_conn.close()
        tmp_path.replace(dst_uncompressed)
    finally:
        src_conn.close()


def _gzip_file(src: Path, dst_gz: Path) -> None:
    """gzip the source file. Atomic via temp + rename."""
    tmp = dst_gz.with_suffix(dst_gz.suffix + ".part")
    with src.open("rb") as fin, gzip.open(str(tmp), "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, length=1 << 20)
    tmp.replace(dst_gz)


def _stamp_dir(target_root: Path, ts: datetime) -> Path:
    return target_root / ts.strftime(STAMP_FMT)


def run_backup(target_root: Path, ts: datetime | None = None) -> Path:
    """Snapshot all DBs into a single timestamped folder. Returns folder path."""
    target_root.mkdir(parents=True, exist_ok=True)
    ts = ts or datetime.now()
    out_dir = _stamp_dir(target_root, ts)

    # Idempotency: if folder exists with a manifest, skip.
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        log.info("Backup already exists for %s — skipping (idempotent).", out_dir.name)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    dbs = _discover_dbs()
    if not dbs:
        log.warning("No DBs discovered to back up.")

    entries: List[Dict] = []
    for src in dbs:
        size_src = src.stat().st_size
        sha_src = _sha256(src)
        # Snapshot to a plain .db inside the temp area, then gzip it.
        with tempfile.TemporaryDirectory(prefix="dragon_bkp_", dir=str(out_dir)) as td:
            staged = Path(td) / src.name
            log.info("Snapshotting %s (%d bytes) -> %s.gz", src.name, size_src, src.name)
            _online_backup(src, staged)
            gz_path = out_dir / (src.name + ".gz")
            _gzip_file(staged, gz_path)

        size_gz = gz_path.stat().st_size
        sha_gz = _sha256(gz_path)
        entries.append({
            "name": src.name,
            "source_path": str(src),
            "source_size_bytes": size_src,
            "source_sha256": sha_src,
            "backup_filename": gz_path.name,
            "backup_size_bytes": size_gz,
            "backup_sha256": sha_gz,
            "compression_ratio": round(size_gz / size_src, 4) if size_src else 0,
        })

    manifest = {
        "timestamp": ts.strftime(STAMP_FMT),
        "timestamp_iso": ts.replace(tzinfo=timezone.utc if ts.tzinfo is None else ts.tzinfo).isoformat(),
        "host": os.uname().nodename,
        "repo_root": str(REPO_ROOT),
        "tool": "scripts/backup_dbs.py",
        "version": 1,
        "entries": entries,
    }
    # Atomic manifest write
    tmp_manifest = manifest_path.with_suffix(".json.part")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp_manifest.replace(manifest_path)

    total_src = sum(e["source_size_bytes"] for e in entries)
    total_gz = sum(e["backup_size_bytes"] for e in entries)
    log.info("Backup complete: %s (%d DBs, %.1f KB -> %.1f KB gz)",
             out_dir.name, len(entries), total_src / 1024, total_gz / 1024)
    return out_dir


# ═══════════════════════════════════════════════════════════════
#  RETENTION
# ═══════════════════════════════════════════════════════════════

def _list_backups(target_root: Path) -> List[Tuple[datetime, Path]]:
    """All valid timestamped backup folders, oldest-first."""
    if not target_root.exists():
        return []
    out: List[Tuple[datetime, Path]] = []
    for p in target_root.iterdir():
        if not p.is_dir() or not STAMP_RE.match(p.name):
            continue
        if not (p / "manifest.json").exists():
            # Incomplete or in-progress — skip from retention selection.
            continue
        try:
            ts = datetime.strptime(p.name, STAMP_FMT)
        except ValueError:
            continue
        out.append((ts, p))
    out.sort(key=lambda x: x[0])
    return out


def _select_keep(backups: List[Tuple[datetime, Path]],
                 now: datetime) -> Tuple[set, Dict[str, int]]:
    """Apply retention windows; return (paths-to-keep, stats).

    Windows (using `now` as the anchor):
      - last 7 days        : keep ALL (hourly resolution)
      - days 8..30         : keep latest of each calendar day
      - weeks 5..52        : keep latest of each ISO week
      - older than 12 mo   : drop
    """
    keep: set = set()
    stats: Dict[str, int] = {"hourly": 0, "daily": 0, "weekly": 0, "drop_old": 0}

    cutoff_hourly = now - timedelta(days=7)
    cutoff_daily = now - timedelta(days=30)
    cutoff_weekly = now - timedelta(days=365)

    by_day: Dict[str, List[Tuple[datetime, Path]]] = defaultdict(list)
    by_week: Dict[str, List[Tuple[datetime, Path]]] = defaultdict(list)

    for ts, p in backups:
        if ts >= cutoff_hourly:
            keep.add(p)
            stats["hourly"] += 1
        elif ts >= cutoff_daily:
            by_day[ts.strftime("%Y-%m-%d")].append((ts, p))
        elif ts >= cutoff_weekly:
            iso = ts.isocalendar()
            by_week[f"{iso.year}-W{iso.week:02d}"].append((ts, p))
        else:
            stats["drop_old"] += 1

    # Latest per day for the daily window
    for _, items in by_day.items():
        items.sort(key=lambda x: x[0])
        keep.add(items[-1][1])
        stats["daily"] += 1

    # Latest per ISO week for the weekly window
    for _, items in by_week.items():
        items.sort(key=lambda x: x[0])
        keep.add(items[-1][1])
        stats["weekly"] += 1

    return keep, stats


def _apply_retention(target_root: Path, now: datetime | None = None,
                     dry_run: bool = False) -> Dict[str, int]:
    """Delete backups outside retention windows. Returns counts."""
    now = now or datetime.now()
    backups = _list_backups(target_root)
    if not backups:
        return {"kept": 0, "deleted": 0, "hourly": 0, "daily": 0, "weekly": 0, "drop_old": 0}

    keep, stats = _select_keep(backups, now)
    deleted = 0
    for _, p in backups:
        if p in keep:
            continue
        log.info("Retention: deleting %s%s", p.name, " (dry-run)" if dry_run else "")
        if not dry_run:
            shutil.rmtree(p, ignore_errors=True)
        deleted += 1

    result = {"kept": len(keep), "deleted": deleted, **stats}
    log.info("Retention summary: %s", result)
    return result


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dragon Trader DB backup + retention.")
    ap.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET,
                    help=f"Backup root directory (default: {DEFAULT_TARGET}).")
    ap.add_argument("--no-retention", action="store_true",
                    help="Skip retention pruning step.")
    ap.add_argument("--retention-dry-run", action="store_true",
                    help="Show what retention would delete, don't delete.")
    ap.add_argument("--skip-backup", action="store_true",
                    help="Run only retention; do not create a new snapshot.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    target = args.target_dir.expanduser()
    log.info("Target root: %s", target)

    try:
        if not args.skip_backup:
            run_backup(target)
        if not args.no_retention:
            _apply_retention(target, dry_run=args.retention_dry_run)
    except Exception as e:
        log.exception("Backup failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
