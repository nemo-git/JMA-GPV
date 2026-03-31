#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


POINTS = [
    ("Sapporo", 43.06, 141.35),
    ("Sendai", 38.27, 140.87),
    ("Tokyo", 35.68, 139.77),
    ("Osaka", 34.69, 135.50),
    ("Fukuoka", 33.59, 130.40),
    ("Naha", 26.21, 127.68),
]

VAR_SPECS = {
    "TMP": {
        "file": "TMP/ENS1M_hourly_{date}_TMP.nc",
        "main": "TMP_mean",
        "lower": "TMP_p10",
        "upper": "TMP_p90",
        "ylabel": "Temperature [K or degC]",
        "title": "TMP hourly mean",
    },
    "RH": {
        "file": "RH/ENS1M_hourly_{date}_RH.nc",
        "main": "RH_mean",
        "lower": "RH_p10",
        "upper": "RH_p90",
        "ylabel": "Relative Humidity [%]",
        "title": "RH hourly mean",
    },
    "APCP": {
        "file": "APCP/ENS1M_hourly_{date}_APCP.nc",
        "main": "APCP_mean",
        "lower": "APCP_p10",
        "upper": "APCP_p90",
        "ylabel": "Precipitation [mm/h]",
        "title": "APCP hourly mean",
    },
    "WS": {
        "file": "WS/ENS1M_hourly_{date}_WS.nc",
        "main": "WS_mean",
        "lower": "WS_p10",
        "upper": "WS_p90",
        "ylabel": "Wind Speed [m/s]",
        "title": "WS hourly mean",
    },
    "WD": {
        "file": "WD/ENS1M_hourly_{date}_WD.nc",
        "main": "WD_mean",
        "lower": "WD_p10",
        "upper": "WD_p90",
        "ylabel": "Wind Direction [deg]",
        "title": "WD hourly mean",
    },
}


def _nearest_point(ds: xr.Dataset, lat: float, lon: float) -> tuple[int, int, float, float]:
    lats = ds["latitude"].values.astype("float64")
    lons = ds["longitude"].values.astype("float64")
    lat_idx = int(np.abs(lats - lat).argmin())
    lon_idx = int(np.abs(lons - lon).argmin())
    return lat_idx, lon_idx, float(lats[lat_idx]), float(lons[lon_idx])


def _extract_series(ds: xr.Dataset, varname: str, lat_idx: int, lon_idx: int) -> np.ndarray:
    da = ds[varname].isel(latitude=lat_idx, longitude=lon_idx)
    if "ensemble" in da.dims:
        da = da.mean(dim="ensemble")
    return np.asarray(da.values, dtype="float64")


def _plot_one_variable(base_dir: Path, date_str: str, var_key: str, out_dir: Path) -> Path:
    spec = VAR_SPECS[var_key]
    nc_path = base_dir / date_str[:4] / spec["file"].format(date=date_str)
    ds = xr.open_dataset(nc_path)
    try:
        times = ds["time"].values
        fig, axes = plt.subplots(3, 2, figsize=(15, 10), dpi=150, sharex=True)
        axes = axes.ravel()

        for ax, (name, lat, lon) in zip(axes, POINTS):
            lat_idx, lon_idx, grid_lat, grid_lon = _nearest_point(ds, lat, lon)
            main = _extract_series(ds, spec["main"], lat_idx, lon_idx)
            lower = _extract_series(ds, spec["lower"], lat_idx, lon_idx)
            upper = _extract_series(ds, spec["upper"], lat_idx, lon_idx)

            ax.fill_between(times, lower, upper, color="tab:orange", alpha=0.2, label="p10-p90")
            ax.plot(times, main, color="tab:orange", linewidth=1.2, label="mean")
            ax.set_title(f"{name} ({grid_lat:.2f}, {grid_lon:.2f})", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

            if var_key == "WD":
                ax.set_ylim(0, 360)
                ax.set_yticks([0, 90, 180, 270, 360])

        for ax in axes[4:]:
            ax.set_xlabel("Date")
        for ax in axes[::2]:
            ax.set_ylabel(spec["ylabel"])

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(f"ENS1M {spec['title']} check plot ({date_str})", y=0.98, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ENS1M_hourly_check_{date_str}_{var_key}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path
    finally:
        ds.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plot hourly ENS1M check charts for selected Japan points.")
    ap.add_argument("--date", required=True, help="Target date YYYYMMDD")
    ap.add_argument("--base", default="data/ENS1M", help="Base directory containing ENS1M hourly outputs")
    ap.add_argument("--out-dir", default="plots/ens1m_check", help="Output directory for PNG files")
    args = ap.parse_args(argv)

    base_dir = Path(args.base).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    for var_key in ["TMP", "RH", "APCP", "WS", "WD"]:
        out_path = _plot_one_variable(base_dir, args.date, var_key, out_dir)
        print(f"[DONE] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
