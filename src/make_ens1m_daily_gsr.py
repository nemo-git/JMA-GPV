#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create daily global solar radiation (GSR) from daily total cloud cover.

The input is ``ENS1M_daily_YYYYMMDD_TCDC.nc``.  For every forecast day and
grid latitude, daily extraterrestrial radiation (Ra) is calculated with the
FAO-56 equations (Allen et al., 1998).  Surface radiation is then estimated as

    GSR = Ra * (0.75 - 0.50 * C)

where C is daily-mean total cloud cover expressed as a fraction (0..1).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from ens1m_paths import DEFAULT_OUT_ROOT, OUTPUT_SUBDIR, file_candidates, first_existing, output_dir


PERCENTILES = (1, 5, 10, 20, 50, 80, 90, 95, 99)


def extraterrestrial_radiation(latitude_deg: xr.DataArray, day_of_year: xr.DataArray) -> xr.DataArray:
    """Return FAO-56 daily extraterrestrial radiation [MJ m-2 day-1].

    ``latitude_deg`` and ``day_of_year`` may be xarray arrays with different
    dimensions; xarray broadcasting creates the required time/latitude grid.
    Clipping the sunset-angle argument also gives the physically appropriate
    limits for polar night (0) and polar day (pi).
    """
    phi = np.deg2rad(latitude_deg)
    doy = day_of_year.astype(np.float64)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    acos_arg = (-np.tan(phi) * np.tan(delta)).clip(min=-1.0, max=1.0)
    sunset_hour_angle = np.arccos(acos_arg)

    solar_constant = 0.0820  # MJ m-2 min-1
    ra = (
        (24.0 * 60.0 / np.pi)
        * solar_constant
        * dr
        * (
            sunset_hour_angle * np.sin(phi) * np.sin(delta)
            + np.cos(phi) * np.cos(delta) * np.sin(sunset_hour_angle)
        )
    )
    ra = ra.clip(min=0.0)
    ra.name = "Ra"
    ra.attrs = {
        "long_name": "daily extraterrestrial radiation",
        "units": "MJ m-2 day-1",
        "standard_name": "toa_incoming_shortwave_flux",
        "calculation": "FAO-56 equations 21-25 (Allen et al., 1998)",
    }
    return ra


def _find_daily_tcdc_file(base_dir: Path, yyyymmdd: str) -> Path:
    year = yyyymmdd[:4]
    candidates = file_candidates(
        base_dir,
        year,
        "TCDC",
        f"ENS1M_daily_{yyyymmdd}_TCDC",
        f"GEPS_daily_{yyyymmdd}_TCDC",
    )
    path = first_existing(candidates) or candidates[0]
    if not path.exists():
        raise FileNotFoundError(f"daily TCDC file not found: {path}")
    return path


def _pick_cloud_variable(ds: xr.Dataset) -> xr.DataArray:
    """Select ensemble-member daily mean cloud cover, never a summary field."""
    for name in ("TCDC_daymean", "TCDC"):
        if name in ds:
            return ds[name]
    raise KeyError("TCDC_daymean (or TCDC) is not present in the daily TCDC file")


def _cloud_fraction(cloud: xr.DataArray, scale: str) -> tuple[xr.DataArray, str]:
    """Normalize cloud cover to 0..1 and return it with the detected scale."""
    units = str(cloud.attrs.get("units", "")).strip().lower()
    selected = scale
    if selected == "auto":
        if "%" in units or "percent" in units or units in {"pct", "percentage"}:
            selected = "percent"
        elif units in {"1", "fraction", "proportion", "0-1", "0..1"}:
            selected = "fraction"
        elif units in {"", "unknown"}:
            # This pipeline preserves JMA TCDC in its native percentage scale,
            # but pygrib reports the unit as "unknown" for current ENS1M files.
            # Do not infer fraction merely because an unusually clear dataset
            # happens to contain only values <= 1.
            selected = "percent"
            warnings.warn(
                f"TCDC units are {units!r}; treating ENS1M TCDC as percent.",
                RuntimeWarning,
            )
        else:
            # GRIB encoders do not always retain a useful unit.  A maximum over
            # this one input field is inexpensive compared with writing output.
            finite_max = float(cloud.max(skipna=True).load())
            if not np.isfinite(finite_max):
                raise ValueError("TCDC contains no finite values")
            selected = "fraction" if finite_max <= 1.5 else "percent"
            warnings.warn(
                f"TCDC units are ambiguous ({units!r}); inferred {selected} from data range.",
                RuntimeWarning,
            )

    fraction = cloud / 100.0 if selected == "percent" else cloud.astype(np.float64)
    finite_min = float(fraction.min(skipna=True).load())
    finite_max = float(fraction.max(skipna=True).load())
    if not np.isfinite(finite_min) or not np.isfinite(finite_max):
        raise ValueError("TCDC contains no finite values")
    if finite_min < -0.05 or finite_max > 1.05:
        raise ValueError(
            "TCDC is outside the plausible cloud-cover range after conversion "
            f"(min={finite_min:.4g}, max={finite_max:.4g}, scale={selected})"
        )
    if finite_min < 0.0 or finite_max > 1.0:
        warnings.warn(
            "TCDC slightly exceeds 0..1, probably due to interpolation; values will be clipped "
            f"(min={finite_min:.4g}, max={finite_max:.4g}).",
            RuntimeWarning,
        )
    fraction = fraction.clip(min=0.0, max=1.0)
    return fraction, selected


def _latitude_coord(ds: xr.Dataset, cloud: xr.DataArray) -> xr.DataArray:
    for name in ("latitude", "lat"):
        if name in cloud.coords:
            latitude = cloud.coords[name]
            break
        if name in ds.coords:
            latitude = ds.coords[name]
            break
    else:
        raise KeyError("latitude/lat coordinate is not present")
    if latitude.ndim not in (1, 2):
        raise ValueError(f"latitude coordinate must be 1-D or 2-D, got {latitude.ndim}-D")
    lat_min = float(latitude.min().load())
    lat_max = float(latitude.max().load())
    if lat_min < -90.0 or lat_max > 90.0:
        raise ValueError(f"invalid latitude range: {lat_min} .. {lat_max}")
    return latitude


def _add_ensemble_summaries(ds: xr.Dataset, gsr: xr.DataArray) -> xr.Dataset:
    if "ensemble" not in gsr.dims:
        return ds
    attrs = {"units": "MJ m-2 day-1"}
    mean = gsr.mean("ensemble")
    mean.attrs = {**attrs, "description": "Ensemble mean"}
    spread = gsr.std("ensemble", ddof=0)
    spread.attrs = {**attrs, "description": "Ensemble spread (standard deviation, ddof=0)"}
    ds["GSR_mean"] = mean
    ds["GSR_spread"] = spread
    for percentile in PERCENTILES:
        try:
            value = gsr.quantile(percentile / 100.0, dim="ensemble", method="linear")
        except TypeError:  # older xarray
            value = gsr.quantile(percentile / 100.0, dim="ensemble", interpolation="linear")
        if "quantile" in value.coords:
            value = value.drop_vars("quantile")
        if "quantile" in value.dims:
            value = value.squeeze("quantile", drop=True)
        value.attrs = {**attrs, "description": f"Ensemble percentile {percentile}%"}
        ds[f"GSR_p{percentile:02d}"] = value
    return ds


def build_gsr_dataset(ds: xr.Dataset, cloud_scale: str = "auto") -> xr.Dataset:
    """Build a GSR dataset from an opened daily TCDC dataset."""
    cloud = _pick_cloud_variable(ds)
    if "time" not in cloud.dims or "time" not in ds.coords:
        raise ValueError("daily TCDC must have a time dimension and coordinate")
    latitude = _latitude_coord(ds, cloud)
    times = pd.DatetimeIndex(ds["time"].values)
    if times.hasnans:
        raise ValueError("time coordinate contains NaT")
    doy = xr.DataArray(
        times.dayofyear.astype(np.int16),
        dims=("time",),
        coords={"time": ds["time"]},
    )

    fraction, selected_scale = _cloud_fraction(cloud, cloud_scale)
    ra = extraterrestrial_radiation(latitude, doy)
    gsr = ra * (0.75 - 0.50 * fraction)
    # Keep the source variable's dimension ordering (normally time, ensemble,
    # latitude, longitude) to match all other ENS1M daily products.
    gsr = gsr.transpose(*cloud.dims)
    gsr.name = "GSR"
    gsr.attrs = {
        "long_name": "estimated daily global solar radiation at the surface",
        "units": "MJ m-2 day-1",
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "cell_methods": "time: sum (interval: 1 day)",
        "calculation": "GSR = Ra * (0.75 - 0.50 * C)",
        "cloud_cover_source": str(cloud.name),
        "cloud_cover_scale": selected_scale,
    }

    out = xr.Dataset({"GSR": gsr, "Ra": ra})
    out = _add_ensemble_summaries(out, gsr)
    out.attrs = dict(ds.attrs or {})
    history = out.attrs.get("history", "")
    if history:
        history += "\n"
    out.attrs["history"] = history + "Daily GSR estimated from daily-mean TCDC using FAO-56 Ra."
    out.attrs["gsr_formula"] = "GSR = Ra * (0.75 - 0.50 * C)"
    out.attrs["gsr_units"] = "MJ m-2 day-1"
    out.attrs["ra_method"] = "FAO-56 equations 21-25 (Allen et al., 1998)"
    out.attrs["cloud_cover_variable"] = str(cloud.name)
    out.attrs["cloud_cover_input_scale"] = selected_scale
    return out


def _encoding(ds: xr.Dataset) -> dict:
    encoding: dict = {}
    for name in ds.data_vars:
        encoding[name] = {"zlib": True, "complevel": 4}
        if np.issubdtype(ds[name].dtype, np.floating):
            encoding[name]["dtype"] = "float32"
    if "time" in ds.coords:
        encoding["time"] = {
            "units": "days since 1900-01-01 00:00:00",
            "calendar": "gregorian",
        }
    return encoding


def create_one(base_dir: Path, yyyymmdd: str, out_dir: Path | None, cloud_scale: str) -> Path:
    input_path = _find_daily_tcdc_file(base_dir, yyyymmdd)
    destination_dir = out_dir or output_dir(base_dir, yyyymmdd[:4], "GSR")
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / f"ENS1M_daily_{yyyymmdd}_GSR.nc"
    temporary_path = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}")

    print(f"[INPUT] TCDC: {input_path}")
    try:
        with xr.open_dataset(input_path, decode_times=True, mask_and_scale=True) as source:
            result = build_gsr_dataset(source, cloud_scale=cloud_scale)
            result.attrs["source_file"] = str(input_path)
            try:
                result.to_netcdf(temporary_path, encoding=_encoding(result))
            finally:
                result.close()
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"[DONE] {output_path}")
    return output_path


def _valid_date(value: str) -> str:
    value = value.strip().replace("　", "")
    datetime.strptime(value, "%Y%m%d")
    return value


def _date_range(start: str, end: str):
    current = datetime.strptime(start, "%Y%m%d")
    last = datetime.strptime(end, "%Y%m%d")
    if current > last:
        raise ValueError("--start must be on or before --end")
    while current <= last:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Create daily ENS1M GSR from daily TCDC without changing the main batch."
    )
    dates = parser.add_mutually_exclusive_group(required=True)
    dates.add_argument("--date", type=_valid_date, help="One run date (YYYYMMDD)")
    dates.add_argument("--start", type=_valid_date, help="First run date for backfill (YYYYMMDD)")
    parser.add_argument("--end", type=_valid_date, help="Last run date for backfill (requires --start)")
    parser.add_argument(
        "--base",
        default=str(DEFAULT_OUT_ROOT),
        help=f"Base dir containing {OUTPUT_SUBDIR}/YYYY/TCDC (default: {DEFAULT_OUT_ROOT})",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override GSR output directory (normally BASE/ENS1M/YYYY/GSR; single-year use recommended)",
    )
    parser.add_argument(
        "--cloud-scale",
        choices=("auto", "percent", "fraction"),
        default="auto",
        help="TCDC scale (default: infer from units, then values)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="For a date range, warn and continue when a daily TCDC file is absent",
    )
    args = parser.parse_args(argv)

    if args.start and not args.end:
        parser.error("--end is required with --start")
    if args.end and not args.start:
        parser.error("--end requires --start")

    try:
        run_dates = [args.date] if args.date else list(_date_range(args.start, args.end))
    except ValueError as exc:
        parser.error(str(exc))
    base_dir = Path(args.base).expanduser().resolve()
    chosen_out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    failed = False
    for yyyymmdd in run_dates:
        try:
            create_one(base_dir, yyyymmdd, chosen_out_dir, args.cloud_scale)
        except FileNotFoundError as exc:
            if args.skip_missing and len(run_dates) > 1:
                warnings.warn(str(exc), RuntimeWarning)
                continue
            print(f"[ERROR] {exc}", file=sys.stderr)
            failed = True
            break
        except (KeyError, ValueError, OSError) as exc:
            print(f"[ERROR] {yyyymmdd}: {exc}", file=sys.stderr)
            failed = True
            break
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
