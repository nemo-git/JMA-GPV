#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GEPS hourly interpolation with boundary gap-bridging (TMP first).

Problem addressed:
- When hourly data are created by interpolating 1w2w and 1m separately then concatenating,
  there can be an *internal gap* at the boundary (e.g., 1w2w ends at T, 1m starts at T+?),
  which makes one JST-day incomplete (24 samples missing) and the daily file drops that day.

Solution:
1) Interpolate 1w2w and 1m *independently* to hourly (same as segment-wise).
2) Merge them onto a *full* hourly axis spanning min..max.
   - Fill with 1w2w first, then overwrite with 1m where available (or vice versa).
3) Detect missing timestamps inside the interior range.
4) Bridge short internal gaps by time interpolation over NaNs:
   - Use linear interpolation for gap filling (stable).
   - Default bridges gaps up to --max-gap-hours (e.g., 12 or 24).
   - For TMP this is usually acceptable as an approximation.

Inputs:
  GEPS_1w2w_YYYYMMDD_VAR.nc
  GEPS_1m_YYYYMMDD_VAR.nc   (optional)

Output:
  GEPS_hourly_YYYYMMDD_VAR.nc
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

SUPPORTED_VARS = {"TMP", "RH", "UGRD", "VGRD", "PRMSL", "TCDC", "APCP", "HGT", "VVEL"}


def _normalize_var(v: str) -> str:
    return v.strip().replace("　", "").upper()


def _find_input_files(base_dir: Path, yyyymmdd: str, var: str) -> tuple[Path, Path | None]:
    year = yyyymmdd[:4]
    cand_dir = base_dir / "GEPS_NC" / year / var
    f_1w2w = cand_dir / f"GEPS_1w2w_{yyyymmdd}_{var}.nc"
    f_1m = cand_dir / f"GEPS_1m_{yyyymmdd}_{var}.nc"
    if not f_1w2w.exists():
        raise FileNotFoundError(f"1w2w file not found: {f_1w2w}")
    if not f_1m.exists():
        f_1m = None
    return f_1w2w, f_1m


def _open_dataset(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=True, mask_and_scale=True)


def _pick_interp_method(preferred: str) -> str:
    preferred = preferred.lower()
    if preferred not in {"cubic", "linear"}:
        return "linear"
    if preferred == "cubic":
        try:
            import scipy  # noqa: F401
            return "cubic"
        except Exception:
            warnings.warn("scipy not available; falling back to linear interpolation", RuntimeWarning)
            return "linear"
    return "linear"


def _hourly_axis_from_times(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.date_range(times.min(), times.max(), freq="1h")


def _interp_to_hourly(ds: xr.Dataset, method: str) -> xr.Dataset:
    method = _pick_interp_method(method)
    t = pd.DatetimeIndex(ds["time"].values)
    hourly = _hourly_axis_from_times(t)
    out = ds.interp(time=hourly, method=method)

    out.attrs = dict(ds.attrs or {})
    hist = out.attrs.get("history", "")
    if hist:
        hist += "\n"
    hist += f"Interpolated to hourly using xarray.interp(method='{method}')"
    out.attrs["history"] = hist
    out.attrs["interpolation_method"] = method
    return out


def _full_hourly_axis(ds_list: list[xr.Dataset]) -> pd.DatetimeIndex:
    times_all = pd.DatetimeIndex(np.concatenate([d["time"].values for d in ds_list]))
    return pd.date_range(times_all.min(), times_all.max(), freq="1h")




def _infer_step_hours(times: pd.DatetimeIndex) -> int:
    """Infer GEPS original step length (hours) from time coordinate.

    Expected: 3 (Lsurf) or 6 (L-pall). We use median step to be robust.
    """
    if len(times) < 2:
        raise ValueError("Need at least 2 timesteps to infer step hours.")
    dt_hours = np.median(np.diff(times.values).astype("timedelta64[h]").astype(int))
    if int(dt_hours) not in (3, 6):
        raise ValueError(f"Unexpected time step (hours): {dt_hours}. Expected 3 or 6.")
    return int(dt_hours)


def _apcp_equal_disagg_to_hourly(ds: xr.Dataset, varname: str = "APCP") -> xr.Dataset:
    """Disaggregate APCP (previous 3h/6h accumulation) into hourly values by equal split.

    Interpretation:
      - ds[varname] at time t represents the accumulation over (t-step, t] (step=3 or 6 hours).
      - We distribute value/step to each hour that ends at t, t-1h, ..., t-(step-1)h.

    This guarantees mass conservation: sum of the produced hourly values over the interval equals
    the original accumulated amount.

    Note:
      - This is a *modeling assumption* (no within-interval timing information exists).
      - Missing hours created by merging segments can be filled later (we will fill to 0 for APCP).
    """
    if varname not in ds:
        raise KeyError(f"{varname} not in dataset.")

    t = pd.DatetimeIndex(ds["time"].values)
    step = _infer_step_hours(t)

    hourly = _hourly_axis_from_times(t)

    da = ds[varname]

    parts = []
    for k in range(step):
        shifted = da.assign_coords(time=da["time"] - np.timedelta64(k, "h")).reindex(time=hourly)
        parts.append(shifted)

    out_da = xr.concat(parts, dim="__k").sum("__k") / step

    out = out_da.to_dataset(name=varname)

    # Keep attrs/encoding minimal & add history
    out.attrs = dict(ds.attrs or {})
    hist = out.attrs.get("history", "")
    if hist:
        hist += "\n"
    out.attrs["history"] = hist + f"APCP disaggregated to hourly by equal split (step={step}h, value/step to each hour)."
    out.attrs["apcp_disaggregation"] = "equal_split"
    out.attrs["apcp_original_step_hours"] = str(step)
    return out


def _add_ens_summary_vars(ds: xr.Dataset, varname: str) -> xr.Dataset:
    """Add ensemble summary variables (mean, spread, percentiles) for varname."""
    if varname not in ds:
        return ds
    da_ens = ds[varname]
    if "ensemble" not in da_ens.dims:
        return ds

    da_mean = da_ens.mean(dim="ensemble")
    da_std = da_ens.std(dim="ensemble", ddof=0)

    units = str(da_ens.attrs.get("units", ""))

    mean_name = f"{varname}_mean"
    spr_name = f"{varname}_spread"

    da_mean.name = mean_name
    da_std.name = spr_name
    if units:
        da_mean.attrs["units"] = units
        da_std.attrs["units"] = units
    da_mean.attrs["description"] = "Ensemble mean"
    da_std.attrs["description"] = "Ensemble spread (standard deviation, ddof=0)"

    perc_list = [1, 5, 10, 20, 50, 80, 90, 95, 99]
    qs = [p / 100.0 for p in perc_list]
    try:
        q_da = da_ens.quantile(qs, dim="ensemble", method="linear")
    except TypeError:
        q_da = da_ens.quantile(qs, dim="ensemble", interpolation="linear")

    for p, q in zip(perc_list, qs):
        vname = f"{varname}_p{p:02d}"
        one = q_da.sel(quantile=q, method="nearest").drop_vars("quantile")
        one.name = vname
        if units:
            one.attrs["units"] = units
        one.attrs["description"] = f"Ensemble percentile {p}%"
        ds[vname] = one

    ds[mean_name] = da_mean
    ds[spr_name] = da_std
    return ds


def _to_hourly_segment(ds: xr.Dataset, var: str, method: str) -> xr.Dataset:
    """Convert one segment to hourly.

    - For APCP: equal-split disaggregation (ignores interp method).
    - Others: xarray.interp along time.
    """
    if var.upper() == "APCP":
        return _apcp_equal_disagg_to_hourly(ds, varname="APCP")
    return _interp_to_hourly(ds, method=method)

def _merge_on_full_axis(ds_list: list[xr.Dataset], full_hourly: pd.DatetimeIndex, prefer: str) -> xr.Dataset:
    """Merge multiple hourly datasets onto full axis.
    prefer: '1m' means later dataset (1m) overwrites, '1w2w' means 1w2w overwrites.
    """
    # Align all datasets to the same hourly axis
    aligned = [d.reindex(time=full_hourly) for d in ds_list]

    if len(aligned) == 1:
        return aligned[0]

    if prefer == "1m":
        base = aligned[0]
        for d in aligned[1:]:
            base = base.combine_first(d)  # fill NaNs in base from d
        # combine_first keeps existing base values; we want 1m overwrite, so reverse:
        # if order is [1w2w, 1m], do 1m.combine_first(1w2w)
        if len(aligned) == 2:
            base = aligned[1].combine_first(aligned[0])
        return base
    else:
        # prefer 1w2w: keep first dataset values where present
        base = aligned[0]
        for d in aligned[1:]:
            base = base.combine_first(d)
        return base


def _gap_report(full_hourly: pd.DatetimeIndex, ds: xr.Dataset) -> list[pd.Timestamp]:
    present = pd.DatetimeIndex(ds["time"].values)
    # After reindex, ds.time is full_hourly already. Missing are where key vars are NaN.
    return []


def _bridge_gaps(ds: xr.Dataset, max_gap_hours: int) -> xr.Dataset:
    """Fill internal NaN runs along time up to max_gap_hours using linear interpolation.

    We only fill NaN *inside* the time range (not extrapolate ends):
    limit_area='inside' ensures no edge extrapolation.
    """
    out = ds
    has_limit_area = "limit_area" in inspect.signature(xr.DataArray.interpolate_na).parameters

    # Identify variables that depend on time
    for name, da in ds.data_vars.items():
        if "time" not in da.dims:
            continue

        # If the series already has no NaN, skip
        if not np.isnan(da.isel(time=slice(None)).values).any():
            continue

        # interpolate_na works along time and respects other dims (ensemble/lat/lon).
        # limit=max_gap_hours fills runs up to that length (in number of steps, 1h steps).
        interp_kwargs = {
            "dim": "time",
            "method": "linear",
            "limit": max_gap_hours,
        }
        if has_limit_area:
            interp_kwargs["limit_area"] = "inside"

        out[name] = da.interpolate_na(**interp_kwargs)

    # attrs
    out.attrs = dict(ds.attrs or {})
    hist = out.attrs.get("history", "")
    if hist:
        hist += "\n"
    if has_limit_area:
        hist += f"Bridged internal hourly gaps (<= {max_gap_hours}h) using interpolate_na(method='linear', limit_area='inside')."
    else:
        hist += f"Bridged internal hourly gaps (<= {max_gap_hours}h) using interpolate_na(method='linear')."
    out.attrs["history"] = hist
    out.attrs["gap_bridge_max_hours"] = str(max_gap_hours)
    return out


def _default_encoding(ds: xr.Dataset) -> dict:
    enc: dict = {}
    for v in ds.data_vars:
        enc[v] = {"zlib": True, "complevel": 4}
        if np.issubdtype(ds[v].dtype, np.floating):
            enc[v]["dtype"] = "float32"
    if "time" in ds.coords:
        enc["time"] = {"units": "hours since 1900-01-01 00:00:00", "calendar": "gregorian"}
    return enc


def _count_missing_hours_for_var(ds: xr.Dataset, varname: str) -> int:
    if varname not in ds:
        return -1
    da = ds[varname]
    if "time" not in da.dims:
        return 0
    # Reduce over non-time dims: missing if all points are NaN at that time
    other_dims = [d for d in da.dims if d != "time"]
    allnan = da.isnull()
    if other_dims:
        allnan = allnan.all(dim=other_dims)
    return int(allnan.sum().values)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Create hourly GEPS netCDF (segment-wise + gap bridging).")
    p.add_argument("--date", required=True, help="Target date YYYYMMDD")
    p.add_argument("--var", required=True, help="Variable name (e.g., TMP)")
    p.add_argument(
        "--base",
        default=str(Path.home() / "Dropbox" / "linux_work" / "JMA_GPV" / "data"),
        help="Base dir containing GEPS_NC/YYYY/VAR (default: ~/Dropbox/linux_work/JMA_GPV/data)",
    )
    p.add_argument("--out-dir", default=None, help="Output directory (default: same as input dir)")
    p.add_argument("--method", default="cubic", choices=["cubic", "linear"], help="Interpolation method for each segment (default: cubic; APCP ignores this)")
    p.add_argument("--max-gap-hours", type=int, default=12, help="Bridge internal gaps up to this many hours (default: 12)")
    p.add_argument(
        "--prefer",
        default="1m",
        choices=["1m", "1w2w"],
        help="When both segments have values at the same hour, which to keep (default: 1m overwrites 1w2w)",
    )
    args = p.parse_args(argv)

    yyyymmdd = args.date.strip().replace("　", "")
    var = _normalize_var(args.var)

    if var not in SUPPORTED_VARS:
        print(f"[ERROR] --var {args.var} は未対応です。サポート: {', '.join(sorted(SUPPORTED_VARS))}", file=sys.stderr)
        return 2

    base_dir = Path(args.base).expanduser().resolve()
    try:
        f_1w2w, f_1m = _find_input_files(base_dir, yyyymmdd, var)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    print("[INPUT]")
    print(f"  1w2w: {f_1w2w}")
    print(f"  1m  : {f_1m if f_1m else '(not found)'}")

    ds_list: list[xr.Dataset] = []

    ds_1w2w = _open_dataset(f_1w2w)
    try:
        ds_list.append(_to_hourly_segment(ds_1w2w, var=var, method=args.method))
    finally:
        ds_1w2w.close()

    if f_1m is not None:
        ds_1m = _open_dataset(f_1m)
        try:
            ds_list.append(_to_hourly_segment(ds_1m, var=var, method=args.method))
        finally:
            ds_1m.close()

    full_hourly = _full_hourly_axis(ds_list)
    print(f"[INFO] full hourly axis: {full_hourly[0]} .. {full_hourly[-1]}  (n={len(full_hourly)})")

    # Merge on full axis (without creating NaN-caused cubic failure)
    merged = _merge_on_full_axis(ds_list, full_hourly, prefer=args.prefer)

    # Count missing hours before bridging (use primary variable if present)
    miss_before = _count_missing_hours_for_var(merged, var)
    if miss_before >= 0:
        print(f"[INFO] missing hours (all-NaN) for '{var}' before bridging: {miss_before}")

    if var == "APCP":
        # For precipitation, we do NOT interpolate; fill missing hours with 0 to enable daily aggregation.
        bridged = merged.fillna(0.0)
        bridged = _add_ens_summary_vars(bridged, "APCP")
    else:
        bridged = _bridge_gaps(merged, max_gap_hours=args.max_gap_hours)

    miss_after = _count_missing_hours_for_var(bridged, var)
    if miss_after >= 0:
        print(f"[INFO] missing hours (all-NaN) for '{var}' after bridging: {miss_after}")

    bridged.attrs["source_files"] = str(f_1w2w) + (f" ; {f_1m}" if f_1m else "")
    bridged.attrs["interpolation_mode"] = "segment-wise + merge-on-full-axis + internal gap bridging"
    bridged.attrs["interpolation_note"] = (
        "Spline/linear interpolation is an approximation of model output. "
        "Avoid using it for accumulative variables like APCP."
    )

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else f_1w2w.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"GEPS_hourly_{yyyymmdd}_{var}.nc"

    enc = _default_encoding(bridged)
    bridged.to_netcdf(out_path, encoding=enc)
    bridged.close()

    # Close intermediate datasets
    for d in ds_list:
        try:
            d.close()
        except Exception:
            pass
    try:
        merged.close()
    except Exception:
        pass

    print(f"[DONE] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
