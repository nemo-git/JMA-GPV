#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wait for ENS1M raw delivery files, then run the daily batch.

Default behavior:
- target date: today
- poll every 60 seconds
- stop waiting at 22:00 local time

Readiness rules for the surface pipeline:
- previous-day 12UTC Lsurf weekly files: 1w >= 2 files and 2w >= 1 file
- on Thursdays, monthly 1m Lsurf files must also exist for Tuesday and Wednesday 00UTC (EPSC)
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
import re
import subprocess
import sys
import time as time_mod

from ens1m_paths import DEFAULT_JMADATA_ROOT, DEFAULT_OUT_ROOT


DEFAULT_VARS = "TMP,RH,UGRD,VGRD,APCP,PRMSL,TCDC"


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")


def _today_yyyymmdd() -> str:
    return date.today().strftime("%Y%m%d")


def _fd_key(path: Path) -> tuple[int, int]:
    m = re.search(r"FD(\d{4})-(\d{4})", path.name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (999999, 999999)


def _collect_hits(roots: list[Path], patterns: list[str]) -> list[Path]:
    hits: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            hits.extend(sorted(root.glob(pat)))
    return sorted(set(hits), key=_fd_key)


def _weekly_hits(base_dir: Path, run_date_yyyymmdd: str) -> tuple[list[Path], list[Path], str]:
    target = (_parse_date(run_date_yyyymmdd) - timedelta(days=1)).strftime("%Y%m%d")
    yyyy_prev = target[:4]
    patterns = [
        f"Z__C_*_{target}12*_Lsurf_*_grib2.bin",
        f"*{target}12*Lsurf*grib2.bin",
    ]
    hits_1w = _collect_hits(
        [base_dir / "jmadata" / "ens_1w" / yyyy_prev, base_dir / "ens_1w" / yyyy_prev,
         base_dir / "jmadata" / "1wEGPV" / yyyy_prev, base_dir / "1wEGPV" / yyyy_prev],
        patterns,
    )
    hits_2w = _collect_hits(
        [base_dir / "jmadata" / "ens_2w1m" / yyyy_prev, base_dir / "ens_2w1m" / yyyy_prev,
         base_dir / "jmadata" / "2wEGPV" / yyyy_prev, base_dir / "2wEGPV" / yyyy_prev],
        patterns,
    )
    if len(hits_1w) < 2 or len(hits_2w) < 1:
        fallback_1w: list[Path] = []
        fallback_2w: list[Path] = []
        for pat in patterns:
            for path in base_dir.rglob(pat):
                sp = str(path)
                if "ens_1w" in sp or "1wEGPV" in sp:
                    fallback_1w.append(path)
                elif "ens_2w1m" in sp or "2wEGPV" in sp:
                    fallback_2w.append(path)
        hits_1w = sorted(set(hits_1w) | set(fallback_1w), key=_fd_key)
        hits_2w = sorted(set(hits_2w) | set(fallback_2w), key=_fd_key)
    return hits_1w, hits_2w, target


def _monthly_hits(base_dir: Path, target_yyyymmdd: str) -> list[Path]:
    yyyy = target_yyyymmdd[:4]
    patterns = [
        f"Z__C_*_{target_yyyymmdd}000000_EPSC_*_Lsurf_*_grib2.bin",
        f"*{target_yyyymmdd}000000*EPSC*Lsurf*grib2.bin",
    ]
    hits = _collect_hits(
        [base_dir / "jmadata" / "ens_2w1m" / yyyy, base_dir / "jmadata" / "ens_2w1m",
         base_dir / "ens_2w1m" / yyyy, base_dir / "ens_2w1m",
         base_dir / "jmadata" / "1mEGPV" / yyyy, base_dir / "1mEGPV" / yyyy],
        patterns,
    )
    if not hits:
        fallback: list[Path] = []
        for pat in patterns:
            for path in base_dir.rglob(pat):
                sp = str(path)
                if "ens_2w1m" in sp or "1mEGPV" in sp:
                    fallback.append(path)
        hits = sorted(set(fallback), key=_fd_key)
    return hits


def _build_readiness_report(run_date_yyyymmdd: str, base_dir: Path) -> tuple[bool, list[str]]:
    proc_date = _parse_date(run_date_yyyymmdd)
    hits_1w, hits_2w, prev_day = _weekly_hits(base_dir, run_date_yyyymmdd)
    ready = len(hits_1w) >= 2 and len(hits_2w) >= 1

    lines = [
        f"weekly 1w: {len(hits_1w)}/2 for {prev_day}12UTC Lsurf",
        f"weekly 2w: {len(hits_2w)}/1 for {prev_day}12UTC Lsurf",
    ]

    if proc_date.weekday() == 3:
        tue = (proc_date - timedelta(days=2)).strftime("%Y%m%d")
        wed = (proc_date - timedelta(days=1)).strftime("%Y%m%d")
        hits_tue = _monthly_hits(base_dir, tue)
        hits_wed = _monthly_hits(base_dir, wed)
        ready = ready and bool(hits_tue) and bool(hits_wed)
        lines.extend(
            [
                f"monthly 1m: {len(hits_tue)} file(s) for {tue}00UTC Lsurf (EPSC)",
                f"monthly 1m: {len(hits_wed)} file(s) for {wed}00UTC Lsurf (EPSC)",
            ]
        )

    return ready, lines


def _deadline_time(now: datetime, deadline_hhmm: str) -> datetime:
    hh, mm = deadline_hhmm.split(":")
    return now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)


def _run_batch(args: argparse.Namespace, target_date: str) -> int:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    cmd = [
        sys.executable,
        str(scripts_dir / "00_run_ens1m_batch.py"),
        "--date",
        target_date,
        "--vars",
        args.vars,
        "--jmadata",
        str(Path(args.jmadata).expanduser().resolve()),
        "--out-root",
        str(Path(args.out_root).expanduser().resolve()),
        "--jobs",
        str(args.jobs),
    ]
    if args.stop_on_error:
        cmd.append("--stop-on-error")
    print("[RUN]", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd)
    return completed.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Wait for ENS1M delivery files, then run the batch.")
    ap.add_argument("--date", default=_today_yyyymmdd(), help="Target date YYYYMMDD (default: today)")
    ap.add_argument("--jmadata", default=str(DEFAULT_JMADATA_ROOT), help=f"Input jmadata root (default: {DEFAULT_JMADATA_ROOT})")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help=f"Output root (default: {DEFAULT_OUT_ROOT})")
    ap.add_argument("--vars", default=DEFAULT_VARS, help=f"Variables to pass to 00_run_ens1m_batch.py (default: {DEFAULT_VARS})")
    ap.add_argument("--jobs", type=int, default=2, help="Parallel jobs per stage for the batch runner")
    ap.add_argument("--stop-on-error", action="store_true", help="Pass --stop-on-error to the batch runner")
    ap.add_argument("--sleep-seconds", type=int, default=60, help="Polling interval in seconds (default: 60)")
    ap.add_argument("--deadline", default="22:00", help="Local deadline HH:MM (default: 22:00)")
    args = ap.parse_args(argv)

    try:
        _parse_date(args.date)
    except ValueError:
        print("[ERROR] --date must be YYYYMMDD", file=sys.stderr)
        return 2

    if args.sleep_seconds < 1:
        print("[ERROR] --sleep-seconds must be >= 1", file=sys.stderr)
        return 2

    now = datetime.now().astimezone()
    deadline = _deadline_time(now, args.deadline)
    if now > deadline:
        print(f"[ERROR] current time is already past deadline {deadline.strftime('%Y-%m-%d %H:%M %Z')}", file=sys.stderr)
        return 2

    base_dir = Path(args.jmadata).expanduser().resolve()
    print(f"[INFO] target date: {args.date}")
    print(f"[INFO] raw data root: {base_dir}")
    print(f"[INFO] wait deadline: {deadline.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    while True:
        now = datetime.now().astimezone()
        if now > deadline:
            print(f"[TIMEOUT] delivery files were not ready by {deadline.strftime('%Y-%m-%d %H:%M:%S %Z')}", file=sys.stderr)
            return 1

        ready, lines = _build_readiness_report(args.date, base_dir)
        print(f"[CHECK] {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        for line in lines:
            print(f"  {line}")

        if ready:
            print("[READY] required delivery files are present")
            return _run_batch(args, args.date)

        sleep_until = min(now + timedelta(seconds=args.sleep_seconds), deadline)
        print(f"[WAIT] sleep until {sleep_until.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        time_mod.sleep(args.sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
