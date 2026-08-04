#!/usr/bin/env python3
"""Compare ENS1M daily GSR with NARO Agro-Meteorological Grid Square Data."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jma-gsr-matplotlib")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "bin"))


# Bounds of the six AMD distribution areas. Small overlaps are intentional and
# are harmless because the same 1-km grid values are drawn over one another.
AMD_AREAS = {
    "Area1": (118 / 3, 46.0, 139.0, 146.0),
    "Area2": (208 / 6, 42.0, 137.0, 143.0),
    "Area3": (32.0, 232 / 6, 135.0, 142.0),
    "Area4": (196 / 6, 220 / 6, 130.0, 138.0),
    "Area5": (172 / 6, 106 / 3, 128.0, 133.0),
    "Area6": (24.0, 88 / 3, 122.0, 132.0),
}

PLOT_REGIONS = {
    "japan": {
        "areas": tuple(AMD_AREAS),
        "extent": (122.0, 146.0, 24.0, 46.0),
        "figsize": (15.0, 7.8),
        "label": "Japan",
    },
    "hokkaido": {
        "areas": ("Area1",),
        "extent": (139.0, 146.0, 39.3, 46.0),
        "figsize": (15.0, 7.0),
        "label": "Hokkaido",
    },
}


def _yyyymmdd(value: str) -> str:
    datetime.strptime(value, "%Y%m%d")
    return value


def _download_amd(date_text: str, cache_path: Path, amd_url: str | None) -> None:
    if amd_url:
        local_root = Path(amd_url).expanduser()
        local_dir = local_root / date_text[:4] / "eGSR"
        if local_dir.is_dir():
            _load_local_amd_tiles(date_text, local_dir, cache_path)
            return

    import AMD_Tools4 as AMD

    iso_date = datetime.strptime(date_text, "%Y%m%d").strftime("%Y-%m-%d")
    output = xr.Dataset()
    for area, (south, north, west, east) in AMD_AREAS.items():
        kwargs = {"area": area, "namuni": True}
        if amd_url:
            kwargs["url"] = amd_url
        values, times, lat, lon, name, unit = AMD.GetMetData_Area(
            "GSR",
            [iso_date, iso_date],
            [south, north, west, east],
            **kwargs,
        )
        array = np.asarray(values)
        if array.ndim != 3 or array.shape[0] != 1:
            raise RuntimeError(f"Unexpected AMD {area} GSR shape: {array.shape}")
        lat_dim = f"lat_{area}"
        lon_dim = f"lon_{area}"
        output[f"GSR_{area}"] = xr.DataArray(
            array[0].astype(np.float32),
            dims=(lat_dim, lon_dim),
            coords={lat_dim: np.asarray(lat), lon_dim: np.asarray(lon)},
            attrs={"long_name": str(name), "units": str(unit), "area": area},
        )
        print(f"[AMD] {area}: shape={array.shape}, unit={unit}", flush=True)

    output.attrs.update(
        {
            "title": "NARO Agro-Meteorological Grid Square Data GSR",
            "valid_date": iso_date,
            "units": "MJ m-2 day-1",
            "source": amd_url or "https://amd.rd.naro.go.jp/opendap/AMD/",
        }
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in output.data_vars}
    output.to_netcdf(cache_path, encoding=encoding)
    output.close()
    print(f"[DONE] AMD cache: {cache_path}", flush=True)


def _load_local_amd_tiles(date_text: str, local_dir: Path, cache_path: Path) -> None:
    """Read current first-mesh AMD files and assemble the six plot areas."""
    target = np.datetime64(datetime.strptime(date_text, "%Y%m%d").date())
    tiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    creation_dates: set[str] = set()
    source_files = sorted(local_dir.glob(f"AMDy{date_text[:4]}p*eGSR.nc"))
    if not source_files:
        raise FileNotFoundError(f"No local AMD GSR tiles found: {local_dir}")

    for path in source_files:
        try:
            with xr.open_dataset(path, decode_times=True, mask_and_scale=True) as ds:
                if "GSR" not in ds or "time" not in ds.coords:
                    continue
                try:
                    da = ds["GSR"].sel(time=target).load()
                except KeyError:
                    continue
                lat = np.asarray(da.lat.values, dtype=np.float64)
                lon = np.asarray(da.lon.values, dtype=np.float64)
                values = np.asarray(da.values, dtype=np.float32)
                # Current first-mesh files use 311.72 at non-target/sea cells
                # without declaring it as _FillValue. Daily GSR cannot exceed
                # daily extraterrestrial radiation (~50 MJ m-2 day-1), so mask
                # these undeclared sentinels explicitly.
                values[(values < 0.0) | (values > 50.0)] = np.nan
                if lat[0] > lat[-1]:
                    lat = lat[::-1]
                    values = values[::-1]
                tiles.append((lat, lon, values))
                if ds.attrs.get("creation_date"):
                    creation_dates.add(str(ds.attrs["creation_date"]))
        except (OSError, ValueError):
            # Some distributed placeholder files contain no NetCDF payload.
            continue

    if not tiles:
        raise RuntimeError(f"No readable AMD GSR data for {date_text}: {local_dir}")

    all_lat = np.unique(np.concatenate([item[0] for item in tiles]))
    all_lon = np.unique(np.concatenate([item[1] for item in tiles]))
    mosaic = np.full((all_lat.size, all_lon.size), np.nan, dtype=np.float32)
    for lat, lon, values in tiles:
        lat_index = np.searchsorted(all_lat, lat)
        lon_index = np.searchsorted(all_lon, lon)
        mosaic[np.ix_(lat_index, lon_index)] = values

    output = xr.Dataset()
    for area, (south, north, west, east) in AMD_AREAS.items():
        lat_mask = (all_lat >= south) & (all_lat <= north)
        lon_mask = (all_lon >= west) & (all_lon <= east)
        lat_dim = f"lat_{area}"
        lon_dim = f"lon_{area}"
        output[f"GSR_{area}"] = xr.DataArray(
            mosaic[np.ix_(lat_mask, lon_mask)],
            dims=(lat_dim, lon_dim),
            coords={lat_dim: all_lat[lat_mask], lon_dim: all_lon[lon_mask]},
            attrs={
                "long_name": "Global Solar Radiation",
                "units": "MJ/m2/day",
                "area": area,
            },
        )
        finite = int(np.isfinite(output[f"GSR_{area}"].values).sum())
        print(
            f"[AMD LOCAL] {area}: shape={output[f'GSR_{area}'].shape}, finite={finite:,}",
            flush=True,
        )

    output.attrs.update(
        {
            "title": "NARO Agro-Meteorological Grid Square Data GSR",
            "valid_date": datetime.strptime(date_text, "%Y%m%d").strftime("%Y-%m-%d"),
            "units": "MJ m-2 day-1",
            "source": str(local_dir),
            "source_tile_count": len(tiles),
            "source_creation_dates": " ; ".join(sorted(creation_dates)),
        }
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in output.data_vars}
    output.to_netcdf(cache_path, encoding=encoding)
    output.close()
    print(f"[DONE] AMD local cache: {cache_path}", flush=True)


def _load_ens(path: Path, date_text: str) -> xr.DataArray:
    ds = xr.open_dataset(path, decode_times=True, mask_and_scale=True)
    name = "GSR_mean" if "GSR_mean" in ds else "GSR"
    if name not in ds:
        ds.close()
        raise KeyError(f"GSR_mean/GSR not found: {path}")
    target = np.datetime64(datetime.strptime(date_text, "%Y%m%d").date())
    try:
        da = ds[name].sel(time=target)
    except KeyError as exc:
        available = pd.DatetimeIndex(ds.time.values).strftime("%Y-%m-%d").tolist()
        ds.close()
        raise KeyError(f"Valid date {date_text} is absent; available={available}") from exc
    if "ensemble" in da.dims:
        da = da.mean("ensemble")
    da = da.load()
    ds.close()
    return da


def _amd_array(ds: xr.Dataset, area: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    da = ds[f"GSR_{area}"]
    lat = np.asarray(da.coords[f"lat_{area}"].values, dtype=float)
    lon = np.asarray(da.coords[f"lon_{area}"].values, dtype=float)
    values = np.asarray(da.values, dtype=float)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        values = values[::-1]
    return lat, lon, values


def _statistics(
    ens: xr.DataArray,
    amd: xr.Dataset,
    areas: tuple[str, ...],
    extent: tuple[float, float, float, float],
    stride: int,
) -> dict[str, float]:
    ens_lat = np.asarray(ens.latitude.values, dtype=float)
    ens_lon = np.asarray(ens.longitude.values, dtype=float)
    ens_values = np.asarray(ens.values, dtype=float)
    if ens_lat[0] > ens_lat[-1]:
        ens_lat = ens_lat[::-1]
        ens_values = ens_values[::-1]
    interpolator = RegularGridInterpolator(
        (ens_lat, ens_lon), ens_values, bounds_error=False, fill_value=np.nan
    )

    west, east, south, north = extent
    ens_samples: list[np.ndarray] = []
    amd_samples: list[np.ndarray] = []
    for area in areas:
        lat, lon, values = _amd_array(amd, area)
        lat = lat[::stride]
        lon = lon[::stride]
        values = values[::stride, ::stride]
        yy, xx = np.meshgrid(lat, lon, indexing="ij")
        inside = (yy >= south) & (yy <= north) & (xx >= west) & (xx <= east)
        estimated = interpolator(np.column_stack((yy.ravel(), xx.ravel()))).reshape(yy.shape)
        valid = inside & np.isfinite(values) & np.isfinite(estimated)
        ens_samples.append(estimated[valid])
        amd_samples.append(values[valid])

    model = np.concatenate(ens_samples)
    reference = np.concatenate(amd_samples)
    difference = model - reference
    correlation = np.corrcoef(model, reference)[0, 1] if model.size > 1 else np.nan
    return {
        "n": float(model.size),
        "ens_mean": float(np.mean(model)),
        "amd_mean": float(np.mean(reference)),
        "bias": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "correlation": float(correlation),
    }


def _decorate_axis(ax, extent: tuple[float, float, float, float]) -> None:
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="10m", linewidth=0.7, color="#303030", zorder=5)
    gridlines = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.35,
        color="#777777",
        alpha=0.55,
        linestyle="--",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False


def _plot_region(
    ens: xr.DataArray,
    amd: xr.Dataset,
    date_text: str,
    init_date: str,
    region_name: str,
    output_path: Path,
    stride: int,
) -> dict[str, float]:
    config = PLOT_REGIONS[region_name]
    areas = config["areas"]
    extent = config["extent"]
    stats = _statistics(ens, amd, areas, extent, stride)

    levels = np.arange(0.0, 36.0, 2.0)
    cmap = plt.get_cmap("turbo")
    norm = BoundaryNorm(levels, cmap.N, clip=True)
    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=config["figsize"],
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )

    left = axes[0].pcolormesh(
        ens.longitude,
        ens.latitude,
        ens,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=projection,
        zorder=1,
    )
    axes[0].set_title(f"ENS1M GSR ensemble mean\ninit {init_date}, valid {date_text}")
    _decorate_axis(axes[0], extent)

    for area in areas:
        lat, lon, values = _amd_array(amd, area)
        axes[1].pcolormesh(
            lon,
            lat,
            values,
            cmap=cmap,
            norm=norm,
            shading="auto",
            transform=projection,
            zorder=1,
        )
    axes[1].set_title(f"NARO Agro-Meteorological Grid GSR\nvalid {date_text}")
    _decorate_axis(axes[1], extent)

    summary = (
        f"{config['label']} | spatial comparison at AMD grid points (stride={stride})\n"
        f"mean: ENS1M {stats['ens_mean']:.2f}, AMD {stats['amd_mean']:.2f}  |  "
        f"bias {stats['bias']:+.2f}, RMSE {stats['rmse']:.2f} MJ m$^{{-2}}$ day$^{{-1}}$  |  "
        f"r={stats['correlation']:.3f}, n={int(stats['n']):,}"
    )
    fig.suptitle(summary, fontsize=11)
    colorbar = fig.colorbar(left, ax=axes, orientation="horizontal", pad=0.055, shrink=0.75)
    colorbar.set_label("Daily global solar radiation (MJ m$^{-2}$ day$^{-1}$)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] {output_path}", flush=True)
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_yyyymmdd, help="Valid date YYYYMMDD")
    parser.add_argument("--ens-init-date", required=True, type=_yyyymmdd, help="ENS1M initialization YYYYMMDD")
    parser.add_argument("--ens-gsr-file", required=True, help="ENS1M daily GSR NetCDF")
    parser.add_argument("--amd-file", default=None, help="Cached AMD GSR NetCDF")
    parser.add_argument("--amd-url", default=None, help="AMD server URL or local AMD root")
    parser.add_argument("--refresh-amd", action="store_true", help="Rebuild AMD cache")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "plots" / "gsr_validation"))
    parser.add_argument("--stats-stride", type=int, default=4, help="AMD-grid stride for statistics")
    args = parser.parse_args(argv)

    output_dir = Path(args.out_dir).expanduser().resolve()
    amd_path = (
        Path(args.amd_file).expanduser().resolve()
        if args.amd_file
        else output_dir / "data" / f"AMD_GSR_{args.date}.nc"
    )
    if args.refresh_amd or not amd_path.exists():
        _download_amd(args.date, amd_path, args.amd_url)

    ens = _load_ens(Path(args.ens_gsr_file).expanduser().resolve(), args.date)
    with xr.open_dataset(amd_path, decode_times=True, mask_and_scale=True) as amd:
        for region in PLOT_REGIONS:
            _plot_region(
                ens,
                amd,
                args.date,
                args.ens_init_date,
                region,
                output_dir / f"GSR_ENS1M_vs_AMD_{args.date}_{region}.png",
                max(1, args.stats_stride),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
