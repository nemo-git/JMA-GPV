#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot station locations from a CSV on a simple lat/lon map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import shape


LABEL_OFFSETS = {
    "Honbetsu": (0.06, 0.04),
    "Date": (0.06, 0.02),
    "Kitami": (0.06, 0.03),
    "Chitose": (0.06, -0.02),
    "Rikubetsu": (0.08, 0.08),
    "Asyoro": (0.08, -0.08),
    "Kamishihoro": (0.08, 0.02),
    "Ikeda": (0.08, -0.03),
    "Sarabetsu": (0.08, 0.06),
    "Memuro": (0.08, -0.06),
    "Oketo": (0.08, -0.02),
    "Bihoro": (0.08, 0.04),
    "Mori": (0.08, -0.03),
}


def sanitize_filename(s: str) -> str:
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    for ch in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]:
        s = s.replace(ch, "-")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def load_stations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name") or cols.get("station")
    lat_col = cols.get("latitude") or cols.get("lat")
    lon_col = cols.get("longitude") or cols.get("lon")
    if not (name_col and lat_col and lon_col):
        raise ValueError("Station CSV must contain Name, latitude, longitude")

    out = df[[name_col, lat_col, lon_col]].copy()
    out.columns = ["name", "latitude", "longitude"]
    out["name"] = out["name"].astype(str).str.strip()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["name", "latitude", "longitude"])
    return out


def add_station_labels(ax, stations: pd.DataFrame) -> None:
    for _, row in stations.reset_index(drop=True).iterrows():
        dx, dy = LABEL_OFFSETS.get(row["name"], (0.06, 0.02))
        ax.text(
            float(row["longitude"]) + dx,
            float(row["latitude"]) + dy,
            row["name"],
            fontsize=8,
            ha="left",
            va="center",
            color="#1f1f1f",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.7},
            transform=ccrs.PlateCarree(),
            zorder=4,
        )


def load_boundary_geometries(boundary_path: Path) -> list:
    suffix = boundary_path.suffix.lower()
    if suffix in {".shp", ".dbf", ".shx"}:
        shp_path = boundary_path if suffix == ".shp" else boundary_path.with_suffix(".shp")
        reader = shapereader.Reader(str(shp_path))
        return [rec.geometry for rec in reader.records()]
    if suffix in {".geojson", ".json"}:
        data = json.loads(boundary_path.read_text(encoding="utf-8"))
        features = data.get("features", [])
        return [shape(feature["geometry"]) for feature in features if feature.get("geometry")]
    raise ValueError(f"Unsupported boundary file format: {boundary_path}")


def plot_station_map(
    stations: pd.DataFrame,
    out_path: Path,
    title: str | None,
    boundary_file: Path | None,
) -> None:
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_facecolor("#e8f4fb")
    ax.set_extent([139.2, 146.3, 41.2, 45.9], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f4f1e6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#d8ecf7", edgecolor="none", zorder=0)
    ax.coastlines(resolution="10m", linewidth=0.9, color="#4a4a4a", zorder=2)
    if boundary_file is not None:
        geometries = load_boundary_geometries(boundary_file)
        ax.add_geometries(
            geometries,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="#6b6b6b",
            linewidth=0.8,
            linestyle="--",
            zorder=2.5,
        )
    ax.scatter(
        stations["longitude"],
        stations["latitude"],
        s=70,
        color="#c44e52",
        edgecolor="white",
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    add_station_labels(ax, stations)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.8,
        color="#9eb3c2",
        alpha=0.45,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title or "Station Locations")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", required=True, help="Station CSV with Name,latitude,longitude")
    ap.add_argument(
        "--out-dir",
        default=str((Path(__file__).resolve().parent / ".." / "plots").resolve()),
        help="Output directory",
    )
    ap.add_argument("--title", default="Station Locations in Hokkaido", help="Plot title")
    ap.add_argument("--output-name", default="stn_info_station_map.png", help="Output PNG filename")
    ap.add_argument(
        "--boundary-file",
        default=None,
        help="Optional subpref boundary file (.shp or .geojson) to overlay",
    )
    args = ap.parse_args(argv)

    station_path = Path(args.stations)
    if not station_path.exists():
        raise FileNotFoundError(station_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / sanitize_filename(args.output_name)
    boundary_file = Path(args.boundary_file) if args.boundary_file else None
    if boundary_file is not None and not boundary_file.exists():
        raise FileNotFoundError(boundary_file)

    stations = load_stations(station_path)
    plot_station_map(stations, out_path, args.title, boundary_file)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
