#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run ENS1M pipeline scripts over a date range for selected variables."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xarray as xr

from ens1m_paths import DEFAULT_JMADATA_ROOT, DEFAULT_OUT_ROOT, file_candidates, output_dir
from status_manager import update_status, read_status

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


def _mem_available_gib() -> float | None:
    """Return available memory in GiB on Linux, if detectable."""
    try:
        meminfo = Path("/proc/meminfo")
        if not meminfo.exists():
            return None
        values: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            parts = rest.strip().split()
            if not parts:
                continue
            try:
                values[key] = int(parts[0])  # kB
            except Exception:
                continue
        kb = values.get("MemAvailable") or values.get("MemFree") or values.get("MemTotal")
        if kb is None:
            return None
        return kb / 1024.0 / 1024.0
    except Exception:
        return None


def _heavy_stage_jobs(requested_jobs: int, mem_gib: float | None, disable_auto_jobs: bool) -> int:
    """Choose safe parallel jobs for memory-heavy stages."""
    jobs = max(1, int(requested_jobs))
    if disable_auto_jobs or mem_gib is None:
        return jobs
    # Low-memory environments (e.g., 3.8 GiB) are prone to OOM with parallel workers.
    if mem_gib < 5.0:
        return 1
    return jobs


def _auto_jobs(mem_gib: float | None, vars_count: int) -> int:
    """Estimate a practical parallelism level for heavy stages."""
    cpu = os.cpu_count() or 2
    max_tasks = max(1, vars_count)

    if mem_gib is None:
        return min(max_tasks, max(2, min(cpu, 6)))
    if mem_gib >= 128.0:
        return min(max_tasks, max(8, min(cpu, 16)))
    if mem_gib >= 64.0:
        return min(max_tasks, max(6, min(cpu, 12)))
    if mem_gib >= 32.0:
        return min(max_tasks, max(4, min(cpu, 8)))
    if mem_gib >= 8.0:
        return min(max_tasks, max(2, min(cpu, 4)))
    return 1


def _resolve_jobs(requested_jobs: int | None, mem_gib: float | None, vars_count: int) -> tuple[int, bool]:
    """Return (base_jobs, auto_selected)."""
    if requested_jobs is not None:
        return max(1, int(requested_jobs)), False
    return _auto_jobs(mem_gib, vars_count), True


def _use_high_memory_mode(mem_gib: float | None, disable_auto_high_memory_mode: bool) -> bool:
    """Enable fast high-memory path for stage-01 when memory is sufficient."""
    if disable_auto_high_memory_mode:
        return False
    if mem_gib is None:
        return False
    return mem_gib >= 8.0


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


def _wait_for_data(out_root: Path, yyyymmdd: str, vars_list: list[str], max_wait_minutes: int = 60) -> bool:
    """Wait for required data files to be available.
    
    Check if all required data files exist, wait if not (max 60 minutes).
    Checks every 5 minutes.
    
    Returns:
        True if all data files found, False if timeout
    """
    required_files = []
    for var in vars_list:
        required_files.extend([
            ("hourly", var),
            ("daily", var),
        ])
    
    start_time = time.monotonic()
    max_wait_seconds = max_wait_minutes * 60
    check_interval = 300  # 5 minutes
    check_count = 0
    
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > max_wait_seconds:
            print(f"[TIMEOUT] Waited {max_wait_minutes} minutes for data, giving up", flush=True)
            return False
        
        missing_count = 0
        for kind, var in required_files:
            if _find_existing_data_file(out_root, yyyymmdd, var, kind) is None:
                missing_count += 1
        
        if missing_count == 0:
            print(f"[DATA READY] All required data files found after {elapsed:.0f} seconds", flush=True)
            return True
        
        check_count += 1
        if check_count == 1:
            print(f"[WAITING] Required data not ready. Waiting for other processes (missing: {missing_count}/{len(required_files)})", flush=True)
            update_status(1, f"Waiting for {missing_count} data files")
        
        print(f"[WAITING] Check #{check_count}: still waiting for {missing_count} files... (elapsed: {elapsed:.0f}s)", flush=True)
        time.sleep(min(check_interval, max_wait_seconds - elapsed))


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
    
    # Setup default log directory in JMA-GPV root
    project_root = Path(__file__).resolve().parents[1]
    default_log_dir = str(project_root / "log")
    
    ap.add_argument("--log-dir", default=default_log_dir, help=f"Directory for per-date batch logs (default: {default_log_dir})")
    ap.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel jobs per stage (default: auto; scales up on high-memory hosts).",
    )
    ap.add_argument(
        "--disable-auto-jobs",
        action="store_true",
        help="Disable memory-aware auto downshift (heavy stages run with --jobs as-is).",
    )
    ap.add_argument(
        "--disable-auto-high-memory-mode",
        action="store_true",
        help="Disable automatic --high-memory-mode for stage 01 on high-memory hosts.",
    )
    ap.add_argument("--stop-on-error", action="store_true", help="Stop on first error (default: continue)")
    ap.add_argument("--plot-only", action="store_true", help="Run only plotting scripts (16, 17)")
    ap.add_argument("--no-status", action="store_true", help="Disable status file updates")
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
    mem_gib = _mem_available_gib()
    base_jobs, auto_jobs_selected = _resolve_jobs(args.jobs, mem_gib, len(vars_list))
    heavy_jobs = _heavy_stage_jobs(base_jobs, mem_gib, args.disable_auto_jobs)
    use_high_memory_mode = _use_high_memory_mode(mem_gib, args.disable_auto_high_memory_mode)

    s01 = scripts_dir / "01_make_ens1m_netcdf_convert.py"
    s02 = scripts_dir / "02_make_ens1m_hourly.py"
    s03 = scripts_dir / "03_make_ens1m_hourly_wind.py"
    s04 = scripts_dir / "04_make_ens1m_daily.py"
    s05 = scripts_dir / "05_make_ens1m_daily_wind.py"
    s16 = scripts_dir / "16_plot_ens1m_daily_check_points.py"
    s17 = scripts_dir / "17_plot_ens1m_hourly_check_points.py"

    # Determine if processing today (for status management)
    is_today_processing = (start_s == end_s == _today_yyyymmdd())
    enable_status = is_today_processing and not args.no_status

    if enable_status:
        print(f"[INFO] Status updates enabled for today's processing", flush=True)
        update_status(1, "Batch processing started")

    # If --plot-only mode, run only plotting scripts
    if args.plot_only:
        for d in _date_range(start, end):
            log_path = _next_log_path(log_dir, d)
            with _date_log_context(log_path):
                batch_started_at = time.monotonic()
                print(f"[INFO] log file: {log_path}", flush=True)
                print(f"[BATCH START] {_now_str()}", flush=True)
                print(f"[DATE] {d}", flush=True)

                # 16: daily check plot
                print(f"[STAGE START] {_now_str()} stage=16_plot_ens1m_daily_check_points", flush=True)
                _run(
                    [py, str(s16), "--date", d, "--base", str(out_root)],
                    args.stop_on_error,
                )
                print(f"[STAGE END] {_now_str()} stage=16_plot_ens1m_daily_check_points", flush=True)

                # 17: hourly check plot
                print(f"[STAGE START] {_now_str()} stage=17_plot_ens1m_hourly_check_points", flush=True)
                _run(
                    [py, str(s17), "--date", d, "--base", str(out_root)],
                    args.stop_on_error,
                )
                print(f"[STAGE END] {_now_str()} stage=17_plot_ens1m_hourly_check_points", flush=True)

                print(f"[BATCH END] {_now_str()} date={d} elapsed_sec={time.monotonic() - batch_started_at:.1f}", flush=True)

        return 0

    for d in _date_range(start, end):
        log_path = _next_log_path(log_dir, d)
        with _date_log_context(log_path):
            batch_started_at = time.monotonic()
            print(f"[INFO] log file: {log_path}", flush=True)
            print(f"[BATCH START] {_now_str()}", flush=True)
            print(f"[RANGE] start={start_s} end={end_s}", flush=True)
            print(f"[DATE] {d}", flush=True)
            if mem_gib is None:
                print(
                    f"[INFO] memory-aware jobs: heavy_stages={heavy_jobs} general={base_jobs} (memory probe unavailable)",
                    flush=True,
                )
            else:
                print(
                    f"[INFO] memory available={mem_gib:.2f} GiB, memory-aware jobs: heavy_stages={heavy_jobs} general={base_jobs}",
                    flush=True,
                )
            if auto_jobs_selected:
                print("[INFO] --jobs not specified; auto parallelism is active", flush=True)
            if args.disable_auto_jobs:
                print("[INFO] auto job downshift is disabled by --disable-auto-jobs", flush=True)
            print(
                f"[INFO] stage01 high-memory mode: {'enabled' if use_high_memory_mode else 'disabled'}",
                flush=True,
            )

            # For today's processing, check if required data is available
            if enable_status:
                print(f"[CHECK] Waiting for required data files...", flush=True)
                if not _wait_for_data(out_root, d, vars_list, max_wait_minutes=60):
                    print(f"[ERROR] Required data files not available after timeout", flush=True)
                    if args.stop_on_error:
                        return 1
                    continue

            # 01: convert grib -> netcdf (1w2w/1m)
            if enable_status:
                update_status(2, "Converting data format")
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
                if use_high_memory_mode:
                    cmds[-1].append("--high-memory-mode")
            _run_parallel(cmds, heavy_jobs, args.stop_on_error)
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
            _run_parallel(cmds, heavy_jobs, args.stop_on_error)
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
            _run_parallel(cmds, heavy_jobs, args.stop_on_error)
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

            # Update status before plotting
            if enable_status:
                update_status(3, "Data processing completed, generating check plots")

            # 16: daily check plot
            print(f"[STAGE START] {_now_str()} stage=16_plot_ens1m_daily_check_points", flush=True)
            _run(
                [py, str(s16), "--date", d, "--base", str(out_root)],
                args.stop_on_error,
            )
            print(f"[STAGE END] {_now_str()} stage=16_plot_ens1m_daily_check_points", flush=True)

            # 17: hourly check plot
            print(f"[STAGE START] {_now_str()} stage=17_plot_ens1m_hourly_check_points", flush=True)
            _run(
                [py, str(s17), "--date", d, "--base", str(out_root)],
                args.stop_on_error,
            )
            print(f"[STAGE END] {_now_str()} stage=17_plot_ens1m_hourly_check_points", flush=True)

            # Update status - all processing completed
            if enable_status:
                update_status(4, "All processing completed successfully")

            print(f"[BATCH END] {_now_str()} date={d} elapsed_sec={time.monotonic() - batch_started_at:.1f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
