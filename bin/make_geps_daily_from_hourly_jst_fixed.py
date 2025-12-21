#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create daily (JST-day) statistics from GEPS hourly netCDF.

Input:
  GEPS_hourly_YYYYMMDD_VAR.nc  (created previously)

Output:
  GEPS_daily_YYYYMMDD_VAR.nc

Daily definition (as requested):
- JST day boundary
- Each day uses 24 hourly samples: JST 01:00 .. 24:00
  (i.e., includes 00:00 of the next JST day as '24:00' of the current day)
- Incomplete days at the start/end are dropped (no output for that day).

Computed stats (for variables that have 'time' dimension):
- <name>_daymean : mean over the 24 hourly samples
- <name>_daymin  : min  over the 24 hourly samples
- <name>_daymax  : max  over the 24 hourly samples

Notes:
- For accumulative variables (e.g., APCP), daily sum should be handled separately.
  This script is intended for TMP first (mean/min/max).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr


SUPPORTED_VARS = {"TMP", "RH", "UGRD", "VGRD", "PRMSL", "TCDC", "APCP", "HGT", "VVEL"}


def _normalize_var(v: str) -> str:
    return v.strip().replace("　", "").upper()


def _find_hourly_file(base_dir: Path, yyyymmdd: str, var: str) -> Path:
    year = yyyymmdd[:4]
    cand_dir = base_dir / "GEPS_NC" / year / var
    f = cand_dir / f"GEPS_hourly_{yyyymmdd}_{var}.nc"
    if not f.exists():
        raise FileNotFoundError(f"hourly file not found: {f}")
    return f


def _open_dataset(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=True, mask_and_scale=True)


def _build_day_labels_jst(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return day labels for requested JST-day definition.

    Steps:
    - Convert naive UTC-like times to JST-naive by +9h
    - Shift by -1h so that JST 00:00 is attributed to the previous day (as 24:00)
    - Take normalized date (00:00) as the day label
    """
    t_jst = times + pd.Timedelta(hours=9)
    labels = (t_jst - pd.Timedelta(hours=1)).normalize()
    return labels


def _select_full_days(times: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Find full days (24 samples) under the requested JST-day rule.

    Returns:
      full_day_labels: DatetimeIndex of day labels to keep (sorted)
      keep_mask: boolean array selecting times that belong to those full days
    """
    labels = _build_day_labels_jst(times)
    s = pd.Series(np.arange(len(times)), index=times)  # positions

    # Group indices by day label
    groups = {}
    for pos, lab in enumerate(labels):
        groups.setdefault(lab, []).append(pos)

    full_days = []
    for lab, idxs in groups.items():
        if len(idxs) != 24:
            continue
        # Check uniqueness and hourly continuity in JST space
        t_jst = (times[idxs] + pd.Timedelta(hours=9)).sort_values()
        # Under our definition, the 24 samples should span 23 hours
        if (t_jst.max() - t_jst.min()) != pd.Timedelta(hours=23):
            continue
        # Check consecutive 1-hour steps
        diffs = np.diff(t_jst.values.astype('datetime64[h]').astype('int64'))
        if not np.all(diffs == 1):
            continue
        full_days.append(lab)

    full_days = pd.DatetimeIndex(sorted(full_days))
    keep_mask = np.asarray(labels.isin(full_days), dtype=bool)
    return full_days, keep_mask


def _daily_reduce(ds: xr.Dataset, full_days: pd.DatetimeIndex) -> xr.Dataset:
    """Compute daymean/daymin/daymax for all variables with 'time' dimension."""
    times = pd.DatetimeIndex(ds["time"].values)
    labels = _build_day_labels_jst(times)

    # Build a new coordinate for grouping
    day_coord = xr.DataArray(labels.values, dims=("time",), name="day")

    # Group and reduce
    out_vars = {}
    for name, da in ds.data_vars.items():
        if "time" not in da.dims:
            continue

        g = da.groupby(day_coord)

        mean_da = g.mean("time", skipna=False)
        min_da = g.min("time", skipna=False)
        max_da = g.max("time", skipna=False)

        # Keep only full days (exactly those in full_days)
        # The groupby coordinate is 'day' which becomes a dimension in outputs.
        mean_da = mean_da.sel(day=full_days.values)
        min_da = min_da.sel(day=full_days.values)
        max_da = max_da.sel(day=full_days.values)

        # Rename 'day' dimension to 'time' for the output file
        mean_da = mean_da.rename({"day": "time"})
        min_da = min_da.rename({"day": "time"})
        max_da = max_da.rename({"day": "time"})

        out_vars[f"{name}_daymean"] = mean_da
        out_vars[f"{name}_daymin"] = min_da
        out_vars[f"{name}_daymax"] = max_da

    out = xr.Dataset(out_vars)

    # Set output time coordinate as the day labels (JST day, naive)
    out = out.assign_coords(time=("time", full_days.values))

    # Carry over non-time coords (ensemble/lat/lon) if present
    for c in ds.coords:
        if c == "time":
            continue
        if c in out.coords:
            continue
        # Copy coordinate if used by any variable
        if c in ds.variables:
            out = out.assign_coords({c: ds[c]})

    # Global attrs
    out.attrs = dict(ds.attrs or {})
    hist = out.attrs.get("history", "")
    if hist:
        hist += "\n"
    hist += "Daily stats created from hourly file with JST-day definition: 01:00..24:00 (24 samples)."
    out.attrs["history"] = hist
    out.attrs["daily_timezone"] = "JST (+09:00), stored as naive datetime64"
    out.attrs["daily_day_definition"] = "Each day uses 24 samples from JST 01:00..24:00 (00:00 of next day counts as 24:00)."
    out.attrs["daily_incomplete_days"] = "Dropped (no output for incomplete day at start/end)."
    return out


def _default_encoding(ds: xr.Dataset) -> dict:
    enc: dict = {}
    for v in ds.data_vars:
        enc[v] = {"zlib": True, "complevel": 4}
        if np.issubdtype(ds[v].dtype, np.floating):
            enc[v]["dtype"] = "float32"
    if "time" in ds.coords:
        enc["time"] = {"units": "days since 1900-01-01 00:00:00", "calendar": "gregorian"}
    return enc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Create daily mean/min/max from GEPS hourly netCDF (JST day).")
    p.add_argument("--date", required=True, help="Target date YYYYMMDD (same as hourly output date)")
    p.add_argument("--var", required=True, help="Variable name (e.g., TMP)")
    p.add_argument(
        "--base",
        default=str(Path.home() / "Dropbox" / "linux_work" / "JMA_GPV" / "data"),
        help="Base dir containing GEPS_NC/YYYY/VAR (default: ~/Dropbox/linux_work/JMA_GPV/data)",
    )
    p.add_argument("--out-dir", default=None, help="Output directory (default: same as hourly input dir)")
    args = p.parse_args(argv)

    yyyymmdd = args.date.strip().replace("　", "")
    var = _normalize_var(args.var)

    if var not in SUPPORTED_VARS:
        print(f"[ERROR] --var {args.var} は未対応です。サポート: {', '.join(sorted(SUPPORTED_VARS))}", file=sys.stderr)
        return 2

    base_dir = Path(args.base).expanduser().resolve()
    try:
        f_hourly = _find_hourly_file(base_dir, yyyymmdd, var)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    print("[INPUT]")
    print(f"  hourly: {f_hourly}")

    ds = _open_dataset(f_hourly)
    try:
        times = pd.DatetimeIndex(ds["time"].values)
        full_days, keep_mask = _select_full_days(times)

        if len(full_days) == 0:
            print("[WARN] No full days (24 samples) found under the requested JST-day definition.", file=sys.stderr)
            return 1

        ds_sel = ds.isel(time=keep_mask)

        out = _daily_reduce(ds_sel, full_days)

        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else f_hourly.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"GEPS_daily_{yyyymmdd}_{var}.nc"

        enc = _default_encoding(out)
        out.to_netcdf(out_path, encoding=enc)
        out.close()

        print(f"[INFO] full days kept: {len(full_days)}  ({full_days[0].date()} .. {full_days[-1].date()})")
        print(f"[DONE] {out_path}")
        return 0
    finally:
        ds.close()


if __name__ == "__main__":
    raise SystemExit(main())
