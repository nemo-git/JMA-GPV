#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find AMD frost events and compute GEPS percentile at event dates.

Outputs a CSV with event_date, GEPS init date, lead time, and percentile.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from math import erf, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


THRESHOLDS = [-1.0, 3.0]


@dataclass
class Site:
    name: str
    lat: float
    lon: float


def decode_time_to_datetimeindex(da_time: xr.DataArray) -> pd.DatetimeIndex:
    if np.issubdtype(da_time.dtype, np.datetime64):
        return pd.DatetimeIndex(da_time.values)
    units = da_time.attrs.get("units", None)
    calendar = da_time.attrs.get("calendar", "standard")
    try:
        from xarray.coding.times import decode_cf_datetime
        dt = decode_cf_datetime(da_time.values, units=units, calendar=calendar)
        return pd.DatetimeIndex(pd.to_datetime([str(x) for x in dt]))
    except Exception:
        if isinstance(units, str) and units.startswith("days since 1900-01-01"):
            base = pd.Timestamp("1900-01-01")
            vals = np.asarray(da_time.values).astype("int64")
            return pd.DatetimeIndex(base + pd.to_timedelta(vals, unit="D"))
        raise RuntimeError(
            f"Failed to decode time. dtype={da_time.dtype}, units={units}, calendar={calendar}"
        )


def maybe_k_to_c(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float64")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    med = float(np.nanmedian(finite))
    if med > 100:
        return arr - 273.15
    return arr


def get_coord_name(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(f"None of coord candidates found: {list(candidates)}")


def get_nearest_index(lat_vals: np.ndarray, lon_vals: np.ndarray, lat: float, lon: float) -> tuple:
    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        lat_idx = int(np.nanargmin(np.abs(lat_vals - lat)))
        lon_idx = int(np.nanargmin(np.abs(lon_vals - lon)))
        return (lat_idx, lon_idx)
    if lat_vals.ndim == 2 and lon_vals.ndim == 2:
        dist2 = (lat_vals - lat) ** 2 + (lon_vals - lon) ** 2
        flat_idx = int(np.nanargmin(dist2))
        return np.unravel_index(flat_idx, lat_vals.shape)
    raise ValueError(f"Unsupported lat/lon shapes: {lat_vals.shape}, {lon_vals.shape}")


def build_indexer(ds: xr.Dataset, lat_name: str, lon_name: str, lat: float, lon: float) -> tuple[dict, float, float]:
    lat_da = ds[lat_name]
    lon_da = ds[lon_name]
    lat_vals = np.asarray(lat_da.values)
    lon_vals = np.asarray(lon_da.values)
    idx = get_nearest_index(lat_vals, lon_vals, lat, lon)
    if lat_vals.ndim == 1 and lon_vals.ndim == 1 and lat_da.dims == (lat_name,) and lon_da.dims == (lon_name,):
        chosen_lat = float(lat_vals[idx[0]])
        chosen_lon = float(lon_vals[idx[1]])
        return {lat_name: idx[0], lon_name: idx[1]}, chosen_lat, chosen_lon
    dims = lat_da.dims
    if len(dims) != len(idx):
        raise ValueError(f"Dim mismatch: dims={dims}, idx={idx}")
    chosen_lat = float(lat_vals[idx])
    chosen_lon = float(lon_vals[idx])
    indexer = {dims[i]: idx[i] for i in range(len(idx))}
    return indexer, chosen_lat, chosen_lon


def normal_cdf_percentile(x: float, mean: float, sigma: float) -> float:
    z = (x - mean) / sigma
    return 100.0 * 0.5 * (1.0 + erf(z / sqrt(2.0)))


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def load_sites(csv_path: Path) -> list[Site]:
    if csv_path.stat().st_size == 0:
        raise ValueError(f"Station CSV is empty: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Station CSV has no columns: {csv_path}") from e
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name") or cols.get("station")
    lat_col = cols.get("latitude") or cols.get("lat")
    lon_col = cols.get("longitude") or cols.get("lon")
    if not (name_col and lat_col and lon_col):
        raise ValueError("CSV must contain headers Name,latitude,longitude")
    sites = []
    for _, row in df.iterrows():
        sites.append(Site(str(row[name_col]), float(row[lat_col]), float(row[lon_col])))
    return sites


def get_geps_file(root: Path, var: str, geps_init_date: datetime) -> Path:
    yyyymmdd = geps_init_date.strftime("%Y%m%d")
    yyyy = geps_init_date.strftime("%Y")
    return root / yyyy / var / f"GEPS_daily_{yyyymmdd}_{var}.nc"


def load_amd_tmp_min(
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    amd_url: str | None,
) -> pd.Series:
    try:
        import AMD_Tools4 as AMD
    except Exception as e:
        raise ImportError(
            "Failed to import AMD_Tools4.py. Put AMD_Tools4.py in the same directory as this script, "
            "or ensure it is on PYTHONPATH."
        ) from e

    timedomain = [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]
    lalodomain = [float(lat), float(lat), float(lon), float(lon)]
    if amd_url is None:
        data, tim, _lat, _lon = AMD.GetMetData_Area("TMP_min", timedomain, lalodomain, cli=False)
    else:
        data, tim, _lat, _lon = AMD.GetMetData_Area("TMP_min", timedomain, lalodomain, cli=False, url=amd_url)

    arr = np.asarray(data)
    if arr.ndim == 3:
        arr = arr[:, 0, 0]
    elif arr.ndim != 1:
        raise RuntimeError(f"Unexpected AMD data shape: {arr.shape}")

    t = pd.to_datetime(np.asarray(tim))
    t = pd.DatetimeIndex(t).normalize()

    arr = maybe_k_to_c(arr)
    s = pd.Series(arr, index=t, name="AMD_TMP_min_C")
    s = s[(s.index >= start) & (s.index <= end)].sort_index()
    return s


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(s: str) -> str:
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    for ch in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]:
        s = s.replace(ch, "-")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def plot_debug_timeseries(
    site: Site,
    geps_init_date: datetime,
    amd_series: pd.Series,
    geps_mean_bc: pd.Series,
    geps_spread: pd.Series,
    geps_members_bc: pd.DataFrame | None,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    if geps_members_bc is not None:
        for col in geps_members_bc.columns:
            ax.plot(
                geps_members_bc.index,
                geps_members_bc[col].values,
                color="gray",
                alpha=0.2,
                linewidth=0.6,
                zorder=1,
            )
    ax.plot(amd_series.index, amd_series.values, label="AMD TMP_min", color="black", linewidth=1.2, zorder=3)
    ax.plot(
        geps_mean_bc.index,
        geps_mean_bc.values,
        label="GEPS mean (bias-corrected)",
        color="#1f77b4",
        linewidth=1.2,
        zorder=4,
    )

    z_lookup = {1: 2.326347874, 5: 1.644853627, 10: 1.281551566}
    for p in [1, 5, 10]:
        z = z_lookup[p]
        low = geps_mean_bc - z * geps_spread
        high = geps_mean_bc + z * geps_spread
        ax.fill_between(
            geps_mean_bc.index,
            low.values,
            high.values,
            alpha=0.15,
            label=f"P{p:02d}-P{100 - p:02d}",
        )
    ax.set_title(f"{site.name} GEPS init {geps_init_date.strftime('%Y-%m-%d')}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily minimum temperature (℃)")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    stem = sanitize_filename(f"{site.name}_{geps_init_date.strftime('%Y%m%d')}")
    out_path = out_dir / f"debug_geps_amd_{stem}.png"
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--var", required=True, help="Variable name (TMP)")
    ap.add_argument("--stndata", required=True, help="Station list CSV (Name,latitude,longitude)")
    ap.add_argument("--gepsncdata", required=True, help="GEPS_NC root directory")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--amd-url", default=None, help="AMD source URL or local path")
    ap.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING")
    ap.add_argument("--debug", action="store_true", help="Output AMD/GEPS timeseries plots to ../plots")
    args = ap.parse_args(argv)

    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")
    logger = logging.getLogger("geps_frost")

    start_date = parse_date(args.start)
    end_date = parse_date(args.end)
    if end_date < start_date:
        raise ValueError("end must be >= start")

    stn_path = Path(args.stndata)
    if not stn_path.exists():
        raise FileNotFoundError(stn_path)
    geps_root = Path(args.gepsncdata)
    if not geps_root.exists():
        raise FileNotFoundError(geps_root)
    out_dir = Path(args.out_dir)
    ensure_output_dir(out_dir)
    debug_plot_dir = (Path(__file__).resolve().parent / ".." / "plots").resolve()

    sites = load_sites(stn_path)
    logger.info("Start: %s End: %s, sites=%d", args.start, args.end, len(sites))

    out_csv = out_dir / f"FROST_th_find_{args.start}_{args.end}.csv"
    header_needed = not out_csv.exists() or out_csv.stat().st_size == 0

    rows_written = 0
    skipped_missing = 0

    current = start_date
    while current <= end_date:
        logger.info("Processing GEPS init date=%s", current.strftime("%Y-%m-%d"))
        geps_file = get_geps_file(geps_root, args.var, current)
        if not geps_file.exists():
            logger.warning("GEPS file not found: %s (skip)", geps_file)
            skipped_missing += 1
            current += timedelta(days=1)
            continue

        ds = xr.open_dataset(geps_file, decode_times=False, mask_and_scale=True)
        try:
            time_index = decode_time_to_datetimeindex(ds["time"]).normalize()
        except Exception as e:
            logger.warning("Failed to decode time for %s: %s", geps_file, e)
            skipped_missing += 1
            ds.close()
            current += timedelta(days=1)
            continue

        lat_name = get_coord_name(ds, ["latitude", "lat"])
        lon_name = get_coord_name(ds, ["longitude", "lon"])

        mean_name = f"{args.var}_mean_daymin"
        spread_name = f"{args.var}_spread_daymin"
        if mean_name not in ds or spread_name not in ds:
            logger.warning("Missing variables in %s: %s or %s", geps_file, mean_name, spread_name)
            skipped_missing += len(sites)
            ds.close()
            current += timedelta(days=1)
            continue

        with open(out_csv, "a", newline="") as f:
            if header_needed:
                f.write(
                    "name,latitude,longitude,threshold,event_date,geps_init_date,lead_time_days,"
                    "event_flag,amd_tmp_min,percentile\n"
                )
                header_needed = False

            for site in sites:
                try:
                    indexer, chosen_lat, chosen_lon = build_indexer(ds, lat_name, lon_name, site.lat, site.lon)
                except Exception as e:
                    logger.warning("Failed to pick nearest grid for %s: %s", site.name, e)
                    skipped_missing += 1
                    continue

                logger.debug(
                    "Nearest grid for %s: lat=%.4f lon=%.4f", site.name, chosen_lat, chosen_lon
                )

                mean_da = ds[mean_name].isel(indexer)
                spread_da = ds[spread_name].isel(indexer)

                geps_mean = maybe_k_to_c(np.asarray(mean_da.values).astype("float64"))
                geps_spread = np.asarray(spread_da.values).astype("float64")

                geps_mean_s = pd.Series(geps_mean, index=time_index, name="GEPS_mean_C")
                geps_spread_s = pd.Series(geps_spread, index=time_index, name="GEPS_spread")

                member_name = f"{args.var}_daymin"
                geps_members = None
                if member_name in ds and "ensemble" in ds[member_name].dims:
                    member_da = ds[member_name].isel(indexer)
                    member_vals = maybe_k_to_c(np.asarray(member_da.values).astype("float64"))
                    geps_members = pd.DataFrame(
                        member_vals,
                        index=time_index,
                        columns=[f"m{int(x)}" for x in member_da["ensemble"].values],
                    )

                try:
                    amd_series = load_amd_tmp_min(site.lat, site.lon, time_index[0], time_index[-1], args.amd_url)
                except Exception as e:
                    logger.warning("AMD load failed for %s: %s", site.name, e)
                    skipped_missing += 1
                    continue

                if time_index[0] not in amd_series.index:
                    logger.warning("AMD missing first day for bias correction: %s", site.name)
                    skipped_missing += 1
                    continue

                amd_aligned = amd_series.reindex(time_index)
                if not np.isfinite(amd_aligned.iloc[0]):
                    logger.warning("AMD NaN at first day for %s", site.name)
                    skipped_missing += 1
                    continue
                if not np.isfinite(geps_mean_s.iloc[0]):
                    logger.warning("GEPS mean NaN at first day for %s", site.name)
                    skipped_missing += 1
                    continue

                delta = float(amd_aligned.iloc[0] - geps_mean_s.iloc[0])
                geps_mean_bc = geps_mean_s + delta
                geps_members_bc = geps_members + delta if geps_members is not None else None
                logger.debug("Bias delta for %s: %.3f", site.name, delta)
                if args.debug:
                    try:
                        plot_debug_timeseries(
                            site,
                            current,
                            amd_aligned,
                            geps_mean_bc,
                            geps_spread_s,
                            geps_members_bc,
                            debug_plot_dir,
                        )
                    except Exception as e:
                        logger.warning("Debug plot failed for %s: %s", site.name, e)

                for threshold in THRESHOLDS:
                    for event_date in time_index:
                        if event_date not in geps_mean_bc.index:
                            continue

                        amd_val = float(amd_aligned.loc[event_date])
                        mean_val = float(geps_mean_bc.loc[event_date])
                        sigma_val = float(geps_spread_s.loc[event_date])

                        if not np.isfinite(amd_val) or not np.isfinite(mean_val) or not np.isfinite(sigma_val):
                            logger.warning("NaN in inputs for %s on %s", site.name, event_date.date())
                            skipped_missing += 1
                            continue
                        if sigma_val <= 0:
                            logger.warning("Sigma <= 0 for %s on %s", site.name, event_date.date())
                            skipped_missing += 1
                            continue

                        event_flag = 1 if amd_val <= threshold else 0
                        pct_input = amd_val if event_flag == 1 else float(threshold)
                        pct = normal_cdf_percentile(pct_input, mean_val, sigma_val)
                        logger.debug(
                            "Percentile %s %s th=%.1f amd=%.3f input=%.3f mean=%.3f sigma=%.3f pct=%.2f",
                            site.name,
                            event_date.date(),
                            threshold,
                            amd_val,
                            pct_input,
                            mean_val,
                            sigma_val,
                            pct,
                        )

                        lead_days = (event_date - pd.Timestamp(current)).days
                        row = (
                            f"{site.name},{site.lat:.6f},{site.lon:.6f},"
                            f"{threshold:.1f},{event_date.strftime('%Y-%m-%d')},"
                            f"{current.strftime('%Y-%m-%d')},{lead_days},"
                            f"{event_flag},{amd_val:.2f},{pct:.1f}\n"
                        )
                        f.write(row)
                        rows_written += 1

        ds.close()
        current += timedelta(days=1)

    logger.info("Done. rows_written=%d, skipped_missing=%d", rows_written, skipped_missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
