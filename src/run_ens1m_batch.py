#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run ENS1M pipeline scripts over a date range for selected variables."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xarray as xr

from ens1m_paths import DEFAULT_JMADATA_ROOT, DEFAULT_OUT_ROOT, file_candidates, output_dir

# Surface elements handled end-to-end by scripts 01-05.
# WS/WD are derived automatically from UGRD/VGRD in steps 03 and 05.
DEFAULT_VARS = ["TMP", "RH", "UGRD", "VGRD", "APCP", "PRMSL", "TCDC"]


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def _date_log_context(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as fh:
        sys.stdout = _TeeStream(orig_stdout, fh)
        sys.stderr = _TeeStream(orig_stderr, fh)
        try:
            yield
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _today_yyyymmdd() -> str:
    return date.today().strftime("%Y%m%d")


def _date_range(start: datetime, end: datetime):
    cur = start
    while cur <= end:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def _cmd_label(cmd: list[str]) -> str:
    script = Path(cmd[1]).name if len(cmd) > 1 else cmd[0]
    parts = [script]
    if "--date" in cmd:
        try:
            parts.append(f"date={cmd[cmd.index('--date') + 1]}")
        except Exception:
            pass
    if "--var" in cmd:
        try:
            parts.append(f"var={cmd[cmd.index('--var') + 1]}")
        except Exception:
            pass
    return " ".join(parts)


def _run(cmd: list[str], stop_on_error: bool) -> bool:
    label = _cmd_label(cmd)
    started_at = time.monotonic()
    print(f"[SUBPROC START] {_now_str()} {label}", flush=True)
    print("[RUN]", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.monotonic() - started_at
        print(f"[SUBPROC END] {_now_str()} {label} status=ok elapsed_sec={elapsed:.1f}", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.monotonic() - started_at
        print(f"[SUBPROC END] {_now_str()} {label} status=error exit={e.returncode} elapsed_sec={elapsed:.1f}", file=sys.stderr, flush=True)
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


def _next_log_path(log_dir: Path, yyyymmdd: str) -> Path:
    base = log_dir / f"log_ens1m_{yyyymmdd}.log"
    if not base.exists():
        return base
    n = 2
    while True:
        candidate = log_dir / f"log_ens1m_{yyyymmdd}-{n:02d}.log"
        if not candidate.exists():
            return candidate
        n += 1


def _prev_thursday_yyyymmdd(yyyymmdd: str) -> str | None:
    target = _parse_date(yyyymmdd)
    if target.weekday() == 3:
        return None
    days_back = (target.weekday() - 3) % 7
    if days_back == 0:
        days_back = 7
    return (target - timedelta(days=days_back)).strftime("%Y%m%d")


def _data_file_candidates(out_root: Path, yyyymmdd: str, var: str, kind: str) -> list[Path]:
    yyyy = yyyymmdd[:4]
    return file_candidates(out_root, yyyy, var, f"ENS1M_{kind}_{yyyymmdd}_{var}", f"GEPS_{kind}_{yyyymmdd}_{var}")


def _find_existing_data_file(out_root: Path, yyyymmdd: str, var: str, kind: str) -> Path | None:
    for path in _data_file_candidates(out_root, yyyymmdd, var, kind):
        if path.exists():
            return path
    return None


def _append_from_previous_thursday(current_path: Path, previous_path: Path, target_date: str, previous_date: str, kind: str) -> bool:
    with xr.open_dataset(current_path, decode_times=True, mask_and_scale=True) as ds_cur, xr.open_dataset(previous_path, decode_times=True, mask_and_scale=True) as ds_prev:
        cur_times = ds_cur.indexes.get("time")
        prev_times = ds_prev.indexes.get("time")
        if cur_times is None or prev_times is None or len(cur_times) == 0 or len(prev_times) == 0:
            return False

        last_cur = cur_times[-1]
        ds_prev_tail = ds_prev.sel(time=ds_prev["time"] > np.datetime64(last_cur))
        prev_tail_times = ds_prev_tail.indexes.get("time")
        if prev_tail_times is None or len(prev_tail_times) == 0:
            return False

        merged = xr.concat([ds_cur.load(), ds_prev_tail.load()], dim="time")
        merged = merged.sortby("time")
        _, uniq_idx = np.unique(merged["time"].values, return_index=True)
        if len(uniq_idx) != merged.sizes["time"]:
            merged = merged.isel(time=np.sort(uniq_idx))

        appended_count = len(prev_tail_times)
        attrs = dict(ds_cur.attrs or {})
        hist = attrs.get("history", "")
        if hist:
            hist += "\n"
        hist += (
            f"Extended {kind} coverage for {target_date} using trailing values from previous Thursday "
            f"{previous_date}: {previous_path.name} (appended {appended_count} time steps)."
        )
        attrs["history"] = hist
        attrs["extension_source_run_date"] = previous_date
        attrs["extension_source_file"] = str(previous_path)
        attrs["extension_target_run_date"] = target_date
        attrs["extension_kind"] = kind
        attrs["extension_appended_time_steps"] = str(appended_count)
        merged.attrs = attrs

        tmp_path = current_path.with_suffix(current_path.suffix + ".tmp")
        merged.to_netcdf(tmp_path)
        merged.close()
        tmp_path.replace(current_path)
        return True


def _extend_outputs_with_previous_thursday(out_root: Path, target_date: str, vars_list: list[str]) -> None:
    previous_date = _prev_thursday_yyyymmdd(target_date)
    if previous_date is None:
        print(f"[INFO] {target_date} is Thursday; skip previous-Thursday extension", flush=True)
        return

    extend_vars = list(dict.fromkeys(vars_list + ["WS", "WD"]))
    for kind in ("hourly", "daily"):
        for var in extend_vars:
            current_path = _find_existing_data_file(out_root, target_date, var, kind)
            previous_path = _find_existing_data_file(out_root, previous_date, var, kind)
            if current_path is None or previous_path is None:
                print(f"[SKIP] extend {kind} {var}: missing current or previous-Thursday file", flush=True)
                continue
            if _append_from_previous_thursday(current_path, previous_path, target_date, previous_date, kind):
                print(f"[DONE] extended {kind} {var} for {target_date} using {previous_date}", flush=True)
            else:
                print(f"[SKIP] extend {kind} {var}: no trailing times after current coverage", flush=True)


def _hourly_paths(out_root: Path, yyyymmdd: str):
    yyyy = yyyymmdd[:4]
    candidates = [
        (
            output_dir(out_root, yyyy, "UGRD") / f"ENS1M_hourly_{yyyymmdd}_UGRD.nc",
            output_dir(out_root, yyyy, "VGRD") / f"ENS1M_hourly_{yyyymmdd}_VGRD.nc",
        ),
        (
            _data_file_candidates(out_root, yyyymmdd, "UGRD", "hourly")[-2],
            _data_file_candidates(out_root, yyyymmdd, "VGRD", "hourly")[-2],
        ),
        (
            _data_file_candidates(out_root, yyyymmdd, "UGRD", "hourly")[-1],
            _data_file_candidates(out_root, yyyymmdd, "VGRD", "hourly")[-1],
        ),
    ]
    for u_path, v_path in candidates:
        if u_path.exists() and v_path.exists():
            return u_path, v_path
    return candidates[0]


def _daily_paths(out_root: Path, yyyymmdd: str):
    yyyy = yyyymmdd[:4]
    candidates = [
        (
            output_dir(out_root, yyyy, "UGRD") / f"ENS1M_daily_{yyyymmdd}_UGRD.nc",
            output_dir(out_root, yyyy, "VGRD") / f"ENS1M_daily_{yyyymmdd}_VGRD.nc",
        ),
        (
            _data_file_candidates(out_root, yyyymmdd, "UGRD", "daily")[-2],
            _data_file_candidates(out_root, yyyymmdd, "VGRD", "daily")[-2],
        ),
        (
            _data_file_candidates(out_root, yyyymmdd, "UGRD", "daily")[-1],
            _data_file_candidates(out_root, yyyymmdd, "VGRD", "daily")[-1],
        ),
    ]
    for u_path, v_path in candidates:
        if u_path.exists() and v_path.exists():
            return u_path, v_path
    return candidates[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Batch runner for ENS1M 01-05 scripts.")
    ap.add_argument("--date", help="Target date YYYYMMDD (default: today)")
    ap.add_argument("--start", help="Start date YYYYMMDD")
    ap.add_argument("--end", help="End date YYYYMMDD (inclusive)")
    ap.add_argument(
        "--vars",
        default=",".join(DEFAULT_VARS),
        help=(
            "Comma-separated surface vars "
            f"(default: {','.join(DEFAULT_VARS)}; WS/WD are derived automatically from UGRD/VGRD)"
        ),
    )
    ap.add_argument(
        "--jmadata",
        default=str(DEFAULT_JMADATA_ROOT),
        help=f"Input jmadata root (default: {DEFAULT_JMADATA_ROOT})",
    )
    ap.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help=f"Output root (default: {DEFAULT_OUT_ROOT})",
    )
    ap.add_argument("--log-dir", default="log", help="Directory for per-date batch logs (default: ./log)")
    ap.add_argument("--jobs", type=int, default=2, help="Parallel jobs per stage (default: 2)")
    ap.add_argument("--stop-on-error", action="store_true", help="Stop on first error (default: continue)")
    args = ap.parse_args(argv)

    if args.date and (args.start or args.end):
        print("[ERROR] --date cannot be combined with --start/--end", file=sys.stderr)
        return 2

    if args.date:
        start_s = end_s = args.date
    elif args.start or args.end:
        if not (args.start and args.end):
            print("[ERROR] --start and --end must be specified together", file=sys.stderr)
            return 2
        start_s = args.start
        end_s = args.end
    else:
        start_s = end_s = _today_yyyymmdd()

    start = _parse_date(start_s)
    end = _parse_date(end_s)
    if end < start:
        print("[ERROR] end must be >= start", file=sys.stderr)
        return 2

    vars_list = [v.strip().upper() for v in args.vars.split(",") if v.strip()]
    if not vars_list:
        print("[ERROR] --vars is empty", file=sys.stderr)
        return 2

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    py = sys.executable
    out_root = Path(args.out_root).expanduser().resolve()
    jmadata = Path(args.jmadata).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()

    s01 = scripts_dir / "01_make_ens1m_netcdf_convert.py"
    s02 = scripts_dir / "02_make_ens1m_hourly.py"
    s03 = scripts_dir / "03_make_ens1m_hourly_wind.py"
    s04 = scripts_dir / "04_make_ens1m_daily.py"
    s05 = scripts_dir / "05_make_ens1m_daily_wind.py"

    for d in _date_range(start, end):
        log_path = _next_log_path(log_dir, d)
        with _date_log_context(log_path):
            batch_started_at = time.monotonic()
            print(f"[INFO] log file: {log_path}", flush=True)
            print(f"[BATCH START] {_now_str()}", flush=True)
            print(f"[RANGE] start={start_s} end={end_s}", flush=True)
            print(f"[DATE] {d}", flush=True)

            # 01: convert grib -> netcdf (1w2w/1m)
            print(f"[STAGE START] {_now_str()} stage=01_make_ens1m_netcdf_convert", flush=True)
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
            print(f"[STAGE END] {_now_str()} stage=01_make_ens1m_netcdf_convert", flush=True)

            # 02: hourly
            print(f"[STAGE START] {_now_str()} stage=02_make_ens1m_hourly", flush=True)
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
            print(f"[STAGE END] {_now_str()} stage=02_make_ens1m_hourly", flush=True)

            # 03: hourly wind (WS/WD)
            print(f"[STAGE START] {_now_str()} stage=03_make_ens1m_hourly_wind", flush=True)
            u_path, v_path = _hourly_paths(out_root, d)
            if u_path.exists() and v_path.exists():
                _run(
                    [py, str(s03), "--date", d, "--base", str(out_root)],
                    args.stop_on_error,
                )
            else:
                print(f"[SKIP] hourly wind missing U/V for {d}", flush=True)
            print(f"[STAGE END] {_now_str()} stage=03_make_ens1m_hourly_wind", flush=True)

            # 04: daily
            print(f"[STAGE START] {_now_str()} stage=04_make_ens1m_daily", flush=True)
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
            print(f"[STAGE END] {_now_str()} stage=04_make_ens1m_daily", flush=True)

            # 05: daily wind (WS/WD)
            print(f"[STAGE START] {_now_str()} stage=05_make_ens1m_daily_wind", flush=True)
            u_path, v_path = _daily_paths(out_root, d)
            if u_path.exists() and v_path.exists():
                _run(
                    [py, str(s05), "--date", d, "--base", str(out_root)],
                    args.stop_on_error,
                )
            else:
                print(f"[SKIP] daily wind missing U/V for {d}", flush=True)
            print(f"[STAGE END] {_now_str()} stage=05_make_ens1m_daily_wind", flush=True)

            print(f"[STAGE START] {_now_str()} stage=extend_with_previous_thursday", flush=True)
            _extend_outputs_with_previous_thursday(out_root, d, vars_list)
            print(f"[STAGE END] {_now_str()} stage=extend_with_previous_thursday", flush=True)
            print(f"[BATCH END] {_now_str()} date={d} elapsed_sec={time.monotonic() - batch_started_at:.1f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
