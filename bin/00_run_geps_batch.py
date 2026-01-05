#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run GEPS pipeline scripts over a date range for selected variables."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_JMADATA = "/Volumes/Mesh_01/jmadata/JMA_month"
DEFAULT_OUT_ROOT = "/Volumes/Mesh_01/NC_GEPS"
# DEFAULT_VARS = ["TMP", "UGRD", "VGRD", "APCP"]
DEFAULT_VARS = ["RH"]


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")


def _date_range(start: datetime, end: datetime):
    cur = start
    while cur <= end:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def _run(cmd: list[str], stop_on_error: bool) -> bool:
    print("[RUN]", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] exit={e.returncode}: {' '.join(cmd)}", file=sys.stderr, flush=True)
        if stop_on_error:
            raise
        return False


def _run_parallel(cmds: list[list[str]], jobs: int, stop_on_error: bool) -> bool:
    if not cmds:
        return True
    ok = True
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        future_map = {ex.submit(_run, cmd, stop_on_error): cmd for cmd in cmds}
        for fut in as_completed(future_map):
            try:
                if not fut.result():
                    ok = False
            except Exception:
                ok = False
                if stop_on_error:
                    raise
    return ok


def _hourly_paths(out_root: Path, yyyymmdd: str):
    yyyy = yyyymmdd[:4]
    base = out_root / "GEPS_NC" / yyyy
    return (
        base / "UGRD" / f"GEPS_hourly_{yyyymmdd}_UGRD.nc",
        base / "VGRD" / f"GEPS_hourly_{yyyymmdd}_VGRD.nc",
    )


def _daily_paths(out_root: Path, yyyymmdd: str):
    yyyy = yyyymmdd[:4]
    base = out_root / "GEPS_NC" / yyyy
    return (
        base / "UGRD" / f"GEPS_daily_{yyyymmdd}_UGRD.nc",
        base / "VGRD" / f"GEPS_daily_{yyyymmdd}_VGRD.nc",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Batch runner for GEPS 01-05 scripts.")
    ap.add_argument("--start", required=True, help="Start date YYYYMMDD")
    ap.add_argument("--end", required=True, help="End date YYYYMMDD (inclusive)")
    ap.add_argument("--vars", default=",".join(DEFAULT_VARS), help="Comma-separated vars (default: TMP,UGRD,VGRD,APCP)")
    ap.add_argument("--jmadata", default=DEFAULT_JMADATA, help=f"Input jmadata root (default: {DEFAULT_JMADATA})")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help=f"Output root (default: {DEFAULT_OUT_ROOT})")
    ap.add_argument("--jobs", type=int, default=2, help="Parallel jobs per stage (default: 2)")
    ap.add_argument("--stop-on-error", action="store_true", help="Stop on first error (default: continue)")
    args = ap.parse_args(argv)

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        print("[ERROR] end must be >= start", file=sys.stderr)
        return 2

    vars_list = [v.strip().upper() for v in args.vars.split(",") if v.strip()]
    if not vars_list:
        print("[ERROR] --vars is empty", file=sys.stderr)
        return 2

    scripts_dir = Path(__file__).resolve().parent
    py = sys.executable
    out_root = Path(args.out_root).expanduser().resolve()
    jmadata = Path(args.jmadata).expanduser().resolve()

    s01 = scripts_dir / "01_make_geps_netcdf_convert.py"
    s02 = scripts_dir / "02_make_geps_hourly.py"
    s03 = scripts_dir / "03_make_geps_hourly_wind.py"
    s04 = scripts_dir / "04_make_geps_daily.py"
    s05 = scripts_dir / "05_make_geps_daily_wind.py"

    for d in _date_range(start, end):
        print(f"[DATE] {d}", flush=True)

        # 01: convert grib -> netcdf (1w2w/1m)
        cmds = []
        for v in vars_list:
            cmds.append(
                [
                    py,
                    str(s01),
                    "--date",
                    d,
                    "--var",
                    v,
                    "--dir",
                    str(jmadata),
                    "--out-root",
                    str(out_root),
                ]
            )
        _run_parallel(cmds, args.jobs, args.stop_on_error)

        # 02: hourly
        cmds = []
        for v in vars_list:
            cmds.append(
                [
                    py,
                    str(s02),
                    "--date",
                    d,
                    "--var",
                    v,
                    "--base",
                    str(out_root),
                ]
            )
        _run_parallel(cmds, args.jobs, args.stop_on_error)

        # 03: hourly wind (WS/WD)
        u_path, v_path = _hourly_paths(out_root, d)
        if u_path.exists() and v_path.exists():
            _run(
                [py, str(s03), "--date", d, "--base", str(out_root)],
                args.stop_on_error,
            )
        else:
            print(f"[SKIP] hourly wind missing U/V for {d}", flush=True)

        # 04: daily
        cmds = []
        for v in vars_list:
            cmds.append(
                [
                    py,
                    str(s04),
                    "--date",
                    d,
                    "--var",
                    v,
                    "--base",
                    str(out_root),
                ]
            )
        _run_parallel(cmds, args.jobs, args.stop_on_error)

        # 05: daily wind (WS/WD)
        u_path, v_path = _daily_paths(out_root, d)
        if u_path.exists() and v_path.exists():
            _run(
                [py, str(s05), "--date", d, "--base", str(out_root)],
                args.stop_on_error,
            )
        else:
            print(f"[SKIP] daily wind missing U/V for {d}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
