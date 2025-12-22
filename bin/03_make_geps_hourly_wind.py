#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create hourly WIND (speed/direction) from hourly UGRD/VGRD netCDF.

Input:
  GEPS_hourly_YYYYMMDD_UGRD.nc
  GEPS_hourly_YYYYMMDD_VGRD.nc

Output:
  GEPS_hourly_YYYYMMDD_WS.nc
  GEPS_hourly_YYYYMMDD_WD.nc

Variables in output:
  - WS: wind speed (ensemble members)
  - WD: wind direction (ensemble members)
  - Summary vars for both: mean, spread, percentiles (1,5,10,20,50,80,90,95,99)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import xarray as xr


def _find_hourly_file(base_dir: Path, yyyymmdd: str, var: str) -> Path:
    year = yyyymmdd[:4]
    cand_dir = base_dir / "GEPS_NC" / year / var
    f = cand_dir / f"GEPS_hourly_{yyyymmdd}_{var}.nc"
    if not f.exists():
        raise FileNotFoundError(f"hourly file not found: {f}")
    return f


def _open_dataset(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=True, mask_and_scale=True)


def _add_ens_summary_vars(ds: xr.Dataset, da_ens: xr.DataArray, base_name: str) -> xr.Dataset:
    """Add ensemble summary variables (mean, spread, percentiles) into ds."""
    if "ensemble" not in da_ens.dims:
        return ds

    da_mean = da_ens.mean(dim="ensemble")
    da_std = da_ens.std(dim="ensemble", ddof=0)

    units = str(da_ens.attrs.get("units", ""))

    mean_name = f"{base_name}_mean"
    spr_name = f"{base_name}_spread"

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
        vname = f"{base_name}_p{p:02d}"
        one = q_da.sel(quantile=q, method="nearest").drop_vars("quantile")
        one.name = vname
        if units:
            one.attrs["units"] = units
        one.attrs["description"] = f"Ensemble percentile {p}%"
        ds[vname] = one

    ds[mean_name] = da_mean
    ds[spr_name] = da_std
    return ds


def _wind_speed_dir(u: xr.DataArray, v: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute wind speed and meteorological wind direction (deg)."""
    ws = np.sqrt(u ** 2 + v ** 2)
    ws.name = "WS"
    ws.attrs = dict(u.attrs or {})
    ws.attrs["description"] = "Wind speed"

    wd = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    wd.name = "WD"
    wd.attrs = dict(u.attrs or {})
    wd.attrs["units"] = "deg"
    wd.attrs["description"] = "Wind direction (meteorological, degrees from which wind blows)"
    return ws, wd


def _default_encoding(ds: xr.Dataset) -> dict:
    enc: dict = {}
    for v in ds.data_vars:
        enc[v] = {"zlib": True, "complevel": 4}
        if np.issubdtype(ds[v].dtype, np.floating):
            enc[v]["dtype"] = "float32"
    if "time" in ds.coords:
        enc["time"] = {"units": "hours since 1900-01-01 00:00:00", "calendar": "gregorian"}
    return enc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Create hourly WS/WD from UGRD/VGRD netCDF.")
    p.add_argument("--date", required=True, help="Target date YYYYMMDD (same as hourly output date)")
    p.add_argument(
        "--base",
        default=str(Path.home() / "Dropbox" / "linux_work" / "JMA_GPV" / "data"),
        help="Base dir containing GEPS_NC/YYYY/VAR (default: ~/Dropbox/linux_work/JMA_GPV/data)",
    )
    p.add_argument("--out-dir", default=None, help="Output directory (default: same as hourly input dir)")
    args = p.parse_args(argv)

    yyyymmdd = args.date.strip().replace("　", "")
    base_dir = Path(args.base).expanduser().resolve()

    try:
        f_u = _find_hourly_file(base_dir, yyyymmdd, "UGRD")
        f_v = _find_hourly_file(base_dir, yyyymmdd, "VGRD")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    print("[INPUT]")
    print(f"  UGRD: {f_u}")
    print(f"  VGRD: {f_v}")

    ds_u = _open_dataset(f_u)
    ds_v = _open_dataset(f_v)
    try:
        if "UGRD" not in ds_u or "VGRD" not in ds_v:
            print("[ERROR] Missing UGRD or VGRD variable in input files.", file=sys.stderr)
            return 2

        u = ds_u["UGRD"]
        v = ds_v["VGRD"]

        u_aligned, v_aligned = xr.align(u, v, join="inner")
        if u_aligned.sizes != u.sizes or v_aligned.sizes != v.sizes:
            warnings.warn("UGRD/VGRD coords not identical; using intersection for WIND.", RuntimeWarning)

        ws, wd = _wind_speed_dir(u_aligned, v_aligned)

        out_ws = xr.Dataset({"WS": ws})
        out_ws = _add_ens_summary_vars(out_ws, ws, "WS")

        out_wd = xr.Dataset({"WD": wd})
        out_wd = _add_ens_summary_vars(out_wd, wd, "WD")

        # Carry over coords
        for c in u_aligned.coords:
            if c not in out_ws.coords:
                out_ws = out_ws.assign_coords({c: u_aligned[c]})
            if c not in out_wd.coords:
                out_wd = out_wd.assign_coords({c: u_aligned[c]})

        for out_ds in (out_ws, out_wd):
            out_ds.attrs = dict(ds_u.attrs or {})
            hist = out_ds.attrs.get("history", "")
            if hist:
                hist += "\n"
            hist += "WS/WD computed from hourly UGRD/VGRD."
            out_ds.attrs["history"] = hist
            out_ds.attrs["source_files"] = f"{f_u} ; {f_v}"

        base_out = Path(args.out_dir).expanduser().resolve() if args.out_dir else f_u.parent
        out_dir_ws = base_out.parent / "WS"
        out_dir_wd = base_out.parent / "WD"
        out_dir_ws.mkdir(parents=True, exist_ok=True)
        out_dir_wd.mkdir(parents=True, exist_ok=True)
        out_path_ws = out_dir_ws / f"GEPS_hourly_{yyyymmdd}_WS.nc"
        out_path_wd = out_dir_wd / f"GEPS_hourly_{yyyymmdd}_WD.nc"

        enc_ws = _default_encoding(out_ws)
        out_ws.to_netcdf(out_path_ws, encoding=enc_ws)
        out_ws.close()

        enc_wd = _default_encoding(out_wd)
        out_wd.to_netcdf(out_path_wd, encoding=enc_wd)
        out_wd.close()

        print(f"[DONE] {out_path_ws}")
        print(f"[DONE] {out_path_wd}")
        return 0
    finally:
        ds_u.close()
        ds_v.close()


if __name__ == "__main__":
    raise SystemExit(main())
