#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find AMD frost events and compute GEPS percentile at event dates with lead-based bias.

Bias modes:
- anchor: use D0 AMD-GEPS delta for all leads (legacy)
- lead_mean: lead-based mean bias from past window
- lead_trim: lead-based trimmed mean bias from past window
- lead_qm: lead-based quantile mapping (fallback to lead_trim/anchor)

Example:
python 13_GEPS_FROST_th_find_biaslead.py \
  --start 2025-04-10 --end 2025-05-20 --var TMP \
  --stndata stations.csv --gepsncdata ../data/GEPS_NC \
  --out-dir out --amd-url /Volumes/Mesh_01/mesh_work/AMD \
  --bias-mode lead_trim --bias-target p10 \
  --bias-window-days 90 --bias-min-samples 30 \
  --lead-max 16 --ignore-lead0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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


@dataclass
class AmdCacheEntry:
    start: pd.Timestamp
    end: pd.Timestamp
    series: pd.Series


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


def build_indexer(
    ds: xr.Dataset, lat_name: str, lon_name: str, lat: float, lon: float
) -> tuple[dict, float, float]:
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
    ax.set_ylabel("Daily minimum temperature (C)")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    stem = sanitize_filename(f"{site.name}_{geps_init_date.strftime('%Y%m%d')}")
    out_path = out_dir / f"debug_geps_amd_{stem}.png"
    fig.savefig(out_path)
    plt.close(fig)


def trimmed_mean(values: np.ndarray, trim_rate: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    if trim_rate <= 0:
        return float(np.nanmean(finite))
    n = int(finite.size)
    k = int(n * trim_rate)
    if n - 2 * k <= 0:
        return float(np.nanmean(finite))
    trimmed = np.sort(finite)[k : n - k]
    return float(np.nanmean(trimmed))


def quantile_map_value(x: float, geps_vals: np.ndarray, amd_vals: np.ndarray) -> float | None:
    if not np.isfinite(x):
        return None
    geps = geps_vals[np.isfinite(geps_vals)]
    amd = amd_vals[np.isfinite(amd_vals)]
    if geps.size == 0 or amd.size == 0:
        return None
    geps_sorted = np.sort(geps)
    amd_sorted = np.sort(amd)
    if geps_sorted.size == 1:
        p = 0.5
    else:
        p = float(np.interp(x, geps_sorted, np.linspace(0.0, 1.0, geps_sorted.size)))
    mapped = float(np.interp(p, np.linspace(0.0, 1.0, amd_sorted.size), amd_sorted))
    return mapped


class AmdCache:
    def __init__(self, amd_url: str | None, logger: logging.Logger) -> None:
        self._amd_url = amd_url
        self._logger = logger
        self._cache: dict[str, AmdCacheEntry] = {}

    def get_series(self, site: Site, start: datetime, end: datetime) -> pd.Series:
        key = site.name
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        entry = self._cache.get(key)
        if entry is None or start_ts < entry.start or end_ts > entry.end:
            new_start = start_ts if entry is None else min(start_ts, entry.start)
            new_end = end_ts if entry is None else max(end_ts, entry.end)
            self._logger.debug(
                "Loading AMD for %s: %s to %s",
                site.name,
                new_start.strftime("%Y-%m-%d"),
                new_end.strftime("%Y-%m-%d"),
            )
            series = load_amd_tmp_min(site.lat, site.lon, new_start, new_end, self._amd_url)
            entry = AmdCacheEntry(start=new_start, end=new_end, series=series)
            self._cache[key] = entry
        return entry.series[(entry.series.index >= start_ts) & (entry.series.index <= end_ts)]


def collect_bias_samples(
    sites: list[Site],
    past_init_dates: list[datetime],
    geps_root: Path,
    var: str,
    amd_cache: AmdCache,
    bias_target: str,
    bias_lead_max: int | None,
    ignore_lead0: bool,
    indexer_cache: dict[str, dict[str, dict]],
    logger: logging.Logger,
) -> tuple[
    dict[str, dict[int, list[float]]],
    dict[str, dict[int, list[float]]],
    dict[str, dict[int, list[float]]],
]:
    bias_errs: dict[str, dict[int, list[float]]] = {site.name: defaultdict(list) for site in sites}
    bias_geps: dict[str, dict[int, list[float]]] = {site.name: defaultdict(list) for site in sites}
    bias_amd: dict[str, dict[int, list[float]]] = {site.name: defaultdict(list) for site in sites}

    for init_date in past_init_dates:
        geps_file = get_geps_file(geps_root, var, init_date)
        if not geps_file.exists():
            logger.debug("Bias training GEPS missing: %s", geps_file)
            continue

        ds = xr.open_dataset(geps_file, decode_times=False, mask_and_scale=True)
        try:
            time_index = decode_time_to_datetimeindex(ds["time"]).normalize()
        except Exception as e:
            logger.debug("Bias training time decode failed: %s (%s)", geps_file, e)
            ds.close()
            continue

        lat_name = get_coord_name(ds, ["latitude", "lat"])
        lon_name = get_coord_name(ds, ["longitude", "lon"])
        mean_name = f"{var}_mean_daymin"
        p10_name = f"{var}_p10_daymin"
        member_name = f"{var}_daymin"
        if mean_name not in ds:
            logger.debug("Bias training missing %s: %s", mean_name, geps_file)
            ds.close()
            continue
        target_local = bias_target
        if target_local == "p10" and p10_name not in ds:
            if member_name not in ds or "ensemble" not in ds[member_name].dims:
                logger.warning("Bias target p10 unavailable in %s; fallback to mean", geps_file)
                target_local = "mean"

        file_cache = indexer_cache.setdefault(str(geps_file), {})
        lead_max_local = (len(time_index) - 1) if bias_lead_max is None else min(bias_lead_max, len(time_index) - 1)

        for site in sites:
            try:
                indexer = file_cache.get(site.name)
                if indexer is None:
                    indexer, _, _ = build_indexer(ds, lat_name, lon_name, site.lat, site.lon)
                    file_cache[site.name] = indexer
            except Exception as e:
                logger.debug("Bias training indexer failed for %s: %s", site.name, e)
                continue

            mean_da = ds[mean_name].isel(indexer)
            geps_mean = maybe_k_to_c(np.asarray(mean_da.values).astype("float64"))
            geps_mean_s = pd.Series(geps_mean, index=time_index)
            if target_local == "p10":
                if p10_name in ds:
                    p10_da = ds[p10_name].isel(indexer)
                    geps_train = maybe_k_to_c(np.asarray(p10_da.values).astype("float64"))
                    geps_train_s = pd.Series(geps_train, index=time_index)
                else:
                    try:
                        member_da = ds[member_name].isel(indexer)
                        member_vals = maybe_k_to_c(np.asarray(member_da.values).astype("float64"))
                        geps_train = np.nanpercentile(member_vals, 10, axis=member_da.get_axis_num("ensemble"))
                        geps_train_s = pd.Series(geps_train, index=time_index)
                    except Exception as e:
                        logger.warning("Bias target p10 failed in %s: %s; fallback to mean", geps_file, e)
                        geps_train_s = geps_mean_s
            else:
                geps_train_s = geps_mean_s

            try:
                amd_series = amd_cache.get_series(site, time_index[0], time_index[-1])
            except Exception as e:
                logger.debug("Bias training AMD load failed for %s: %s", site.name, e)
                continue

            amd_aligned = amd_series.reindex(time_index)
            for idx, event_date in enumerate(time_index):
                lead_days = int((event_date - pd.Timestamp(init_date)).days)
                if lead_days < 0:
                    continue
                if ignore_lead0 and lead_days == 0:
                    continue
                if lead_max_local is not None and lead_days > lead_max_local:
                    continue

                geps_val = float(geps_train_s.iloc[idx])
                amd_val = float(amd_aligned.iloc[idx]) if event_date in amd_aligned.index else float("nan")
                if not np.isfinite(geps_val) or not np.isfinite(amd_val):
                    continue

                err = amd_val - geps_val
                bias_errs[site.name][lead_days].append(err)
                bias_geps[site.name][lead_days].append(geps_val)
                bias_amd[site.name][lead_days].append(amd_val)

        ds.close()

    return bias_errs, bias_geps, bias_amd


def build_bias_tables(
    sites: list[Site],
    bias_errs: dict[str, dict[int, list[float]]],
    trim_rate: float,
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]], dict[str, dict[int, int]]]:
    bias_mean: dict[str, dict[int, float]] = {site.name: {} for site in sites}
    bias_trim: dict[str, dict[int, float]] = {site.name: {} for site in sites}
    bias_samples: dict[str, dict[int, int]] = {site.name: {} for site in sites}

    for site in sites:
        for lead, errs in bias_errs[site.name].items():
            arr = np.asarray(errs, dtype="float64")
            bias_samples[site.name][lead] = int(arr.size)
            bias_mean[site.name][lead] = float(np.nanmean(arr)) if arr.size else float("nan")
            bias_trim[site.name][lead] = trimmed_mean(arr, trim_rate)

    return bias_mean, bias_trim, bias_samples


def resolve_bias_value(
    mode: str,
    site_name: str,
    lead: int,
    geps_val: float,
    bias_mean: dict[str, dict[int, float]],
    bias_trim: dict[str, dict[int, float]],
    bias_samples: dict[str, dict[int, int]],
    bias_geps: dict[str, dict[int, list[float]]],
    bias_amd: dict[str, dict[int, list[float]]],
    min_samples: int,
    anchor_delta: float | None,
    fallback_logged: set[tuple[str, int, str]],
    logger: logging.Logger,
) -> tuple[float | None, int]:
    samples = int(bias_samples.get(site_name, {}).get(lead, 0))

    def fallback_value(reason: str) -> tuple[float | None, int]:
        key = (site_name, lead, reason)
        if key not in fallback_logged:
            logger.info("Bias fallback for %s lead=%d: %s", site_name, lead, reason)
            fallback_logged.add(key)
        if anchor_delta is not None and np.isfinite(anchor_delta):
            return float(anchor_delta), samples
        return 0.0, samples

    if mode == "lead_mean":
        if samples < min_samples:
            return fallback_value("insufficient samples for lead_mean")
        bias = bias_mean.get(site_name, {}).get(lead, float("nan"))
        if not np.isfinite(bias):
            return fallback_value("nan mean bias")
        return float(bias), samples

    if mode == "lead_trim":
        if samples < min_samples:
            return fallback_value("insufficient samples for lead_trim")
        bias = bias_trim.get(site_name, {}).get(lead, float("nan"))
        if not np.isfinite(bias):
            return fallback_value("nan trim bias")
        return float(bias), samples

    if mode == "lead_qm":
        if samples < min_samples:
            return resolve_bias_value(
                "lead_trim",
                site_name,
                lead,
                geps_val,
                bias_mean,
                bias_trim,
                bias_samples,
                bias_geps,
                bias_amd,
                min_samples,
                anchor_delta,
                fallback_logged,
                logger,
            )
        geps_vals = np.asarray(bias_geps.get(site_name, {}).get(lead, []), dtype="float64")
        amd_vals = np.asarray(bias_amd.get(site_name, {}).get(lead, []), dtype="float64")
        mapped = quantile_map_value(geps_val, geps_vals, amd_vals)
        if mapped is None or not np.isfinite(mapped):
            return resolve_bias_value(
                "lead_trim",
                site_name,
                lead,
                geps_val,
                bias_mean,
                bias_trim,
                bias_samples,
                bias_geps,
                bias_amd,
                min_samples,
                anchor_delta,
                fallback_logged,
                logger,
            )
        return float(mapped - geps_val), samples

    if mode == "anchor":
        if anchor_delta is None or not np.isfinite(anchor_delta):
            return None, 0
        return float(anchor_delta), 0

    raise ValueError(f"Unknown bias mode: {mode}")


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
    ap.add_argument(
        "--bias-mode",
        default="anchor",
        choices=["anchor", "lead_mean", "lead_trim", "lead_qm"],
        help="Bias correction mode",
    )
    ap.add_argument(
        "--bias-target",
        default="mean",
        choices=["mean", "p10"],
        help="学習時に誤差を取るGEPS代表値。meanはens_mean_daymin、p10はp10_dayminを使う",
    )
    ap.add_argument("--bias-window-days", type=int, default=90, help="Training window length (days)")
    ap.add_argument("--bias-min-samples", type=int, default=30, help="Min samples per site/lead")
    ap.add_argument("--bias-trim-rate", type=float, default=0.1, help="Trim rate for lead_trim")
    ap.add_argument("--lead-max", type=int, default=None, help="Max lead days to train/apply")
    ap.add_argument(
        "--ignore-lead0",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore lead=0 for training/output",
    )
    ap.add_argument("--no-extra-cols", action="store_true", help="Do not append bias columns")
    args = ap.parse_args(argv)

    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")
    logger = logging.getLogger("geps_frost_bias")

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
    indexer_cache: dict[str, dict[str, dict]] = {}
    amd_cache = AmdCache(args.amd_url, logger)
    fallback_logged: set[tuple[str, int, str]] = set()

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
        p10_name = f"{args.var}_p10_daymin"
        member_name = f"{args.var}_daymin"

        bias_window_days = max(0, int(args.bias_window_days))
        past_init_dates = [current - timedelta(days=i) for i in range(1, bias_window_days + 1)]
        bias_errs, bias_geps, bias_amd = collect_bias_samples(
            sites,
            past_init_dates,
            geps_root,
            args.var,
            amd_cache,
            args.bias_target,
            args.lead_max,
            args.ignore_lead0,
            indexer_cache,
            logger,
        )
        bias_mean, bias_trim, bias_samples = build_bias_tables(sites, bias_errs, args.bias_trim_rate)

        with open(out_csv, "a", newline="") as f:
            if header_needed:
                base_cols = (
                    "name,latitude,longitude,threshold,event_date,geps_init_date,lead_time_days,"
                    "event_flag,amd_tmp_min,percentile"
                )
                if args.no_extra_cols:
                    f.write(base_cols + "\n")
                else:
                    f.write(base_cols + ",bias_mode,bias_value,bias_samples,bias_target\n")
                header_needed = False

            p10_missing_warned = False
            for site in sites:
                try:
                    file_cache = indexer_cache.setdefault(str(geps_file), {})
                    indexer = file_cache.get(site.name)
                    if indexer is None:
                        indexer, chosen_lat, chosen_lon = build_indexer(ds, lat_name, lon_name, site.lat, site.lon)
                        file_cache[site.name] = indexer
                    else:
                        chosen_lat = site.lat
                        chosen_lon = site.lon
                except Exception as e:
                    logger.warning("Failed to pick nearest grid for %s: %s", site.name, e)
                    skipped_missing += 1
                    continue

                logger.debug("Nearest grid for %s: lat=%.4f lon=%.4f", site.name, chosen_lat, chosen_lon)

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
                    amd_series = amd_cache.get_series(site, time_index[0], time_index[-1])
                except Exception as e:
                    logger.warning("AMD load failed for %s: %s", site.name, e)
                    skipped_missing += 1
                    continue

                amd_aligned = amd_series.reindex(time_index)

                target_local = args.bias_target
                if target_local == "p10" and p10_name not in ds:
                    if not p10_missing_warned:
                        logger.warning("Bias target p10 unavailable in %s; fallback to mean", geps_file)
                        p10_missing_warned = True
                    target_local = "mean"
                if target_local == "p10":
                    try:
                        p10_da = ds[p10_name].isel(indexer)
                        geps_train = maybe_k_to_c(np.asarray(p10_da.values).astype("float64"))
                        geps_train_s = pd.Series(geps_train, index=time_index)
                    except Exception as e:
                        logger.warning("Bias target p10 failed in %s: %s; fallback to mean", geps_file, e)
                        geps_train_s = geps_mean_s
                else:
                    geps_train_s = geps_mean_s

                anchor_delta = None
                if time_index[0] in amd_aligned.index:
                    amd0 = float(amd_aligned.iloc[0])
                    geps0 = float(geps_mean_s.iloc[0])
                    if np.isfinite(amd0) and np.isfinite(geps0):
                        anchor_delta = float(amd0 - geps0)

                bias_vals = []
                bias_samples_vals = []
                for idx, event_date in enumerate(time_index):
                    lead_days = int((event_date - pd.Timestamp(current)).days)
                    if args.ignore_lead0 and lead_days == 0:
                        bias_vals.append(float("nan"))
                        bias_samples_vals.append(0)
                        continue
                    if args.lead_max is not None and lead_days > args.lead_max:
                        bias_vals.append(float("nan"))
                        bias_samples_vals.append(0)
                        continue

                    geps_val = float(geps_train_s.iloc[idx])
                    bias, samples = resolve_bias_value(
                        args.bias_mode,
                        site.name,
                        lead_days,
                        geps_val,
                        bias_mean,
                        bias_trim,
                        bias_samples,
                        bias_geps,
                        bias_amd,
                        args.bias_min_samples,
                        anchor_delta,
                        fallback_logged,
                        logger,
                    )
                    bias_vals.append(float("nan") if bias is None else float(bias))
                    bias_samples_vals.append(samples)

                bias_s = pd.Series(bias_vals, index=time_index, name="bias")
                geps_mean_bc = geps_mean_s + bias_s
                geps_members_bc = None
                if geps_members is not None:
                    geps_members_bc = geps_members.add(bias_s, axis=0)

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
                        lead_days = int((event_date - pd.Timestamp(current)).days)
                        if args.ignore_lead0 and lead_days == 0:
                            continue
                        if args.lead_max is not None and lead_days > args.lead_max:
                            continue

                        amd_val = float(amd_aligned.loc[event_date]) if event_date in amd_aligned.index else float("nan")
                        mean_val = float(geps_mean_bc.loc[event_date]) if event_date in geps_mean_bc.index else float("nan")
                        sigma_val = float(geps_spread_s.loc[event_date]) if event_date in geps_spread_s.index else float("nan")
                        bias_val = float(bias_s.loc[event_date]) if event_date in bias_s.index else float("nan")

                        if not np.isfinite(amd_val) or not np.isfinite(mean_val) or not np.isfinite(sigma_val):
                            logger.warning("NaN in inputs for %s on %s", site.name, event_date.date())
                            skipped_missing += 1
                            continue
                        if not np.isfinite(bias_val):
                            logger.warning("Bias NaN for %s on %s", site.name, event_date.date())
                            skipped_missing += 1
                            continue
                        if sigma_val <= 0:
                            logger.warning("Sigma <= 0 for %s on %s", site.name, event_date.date())
                            skipped_missing += 1
                            continue

                        event_flag = 1 if amd_val <= threshold else 0
                        pct_input = amd_val if event_flag == 1 else float(threshold)
                        pct = normal_cdf_percentile(pct_input, mean_val, sigma_val)

                        row = (
                            f"{site.name},{site.lat:.6f},{site.lon:.6f},"
                            f"{threshold:.1f},{event_date.strftime('%Y-%m-%d')},"
                            f"{current.strftime('%Y-%m-%d')},{lead_days},"
                            f"{event_flag},{amd_val:.2f},{pct:.1f}"
                        )
                        if args.no_extra_cols:
                            f.write(row + "\n")
                        else:
                            samples = int(bias_samples_vals[time_index.get_loc(event_date)])
                            row = (
                                row
                                + f",{args.bias_mode},{bias_val:.3f},{samples:d},{args.bias_target}"
                                + "\n"
                            )
                            f.write(row)
                        rows_written += 1

        ds.close()
        current += timedelta(days=1)

    logger.info("Done. rows_written=%d, skipped_missing=%d", rows_written, skipped_missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
