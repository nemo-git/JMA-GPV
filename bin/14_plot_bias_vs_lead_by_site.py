#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot lead time vs bias value for each station from FROST_th_find CSV."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


REQUIRED_COLS = {"name", "lead_time_days", "bias_value"}
DEDUP_COLS = ["name", "geps_init_date", "event_date", "lead_time_days"]


def sanitize_filename(s: str) -> str:
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    for ch in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]:
        s = s.replace(ch, "-")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def load_station_order(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name") or cols.get("station")
    if not name_col:
        raise ValueError("Station CSV must contain Name or station column")
    return [str(v).strip() for v in df[name_col].dropna().tolist()]


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not REQUIRED_COLS.issubset(df.columns):
        missing = REQUIRED_COLS - set(df.columns)
        raise ValueError(f"Missing columns: {sorted(missing)}")

    available_dedup_cols = [col for col in DEDUP_COLS if col in df.columns]
    if available_dedup_cols:
        df = df.drop_duplicates(subset=available_dedup_cols)

    df = df.copy()
    df["name"] = df["name"].astype(str).str.strip()
    df["lead_time_days"] = pd.to_numeric(df["lead_time_days"], errors="coerce")
    df["bias_value"] = pd.to_numeric(df["bias_value"], errors="coerce")
    if "bias_samples" in df.columns:
        df["bias_samples"] = pd.to_numeric(df["bias_samples"], errors="coerce")
    df = df.dropna(subset=["name", "lead_time_days", "bias_value"])
    return df


def plot_bias_panels(
    df: pd.DataFrame,
    station_order: list[str],
    out_path: Path,
    title: str | None,
) -> None:
    site_names = [name for name in station_order if name in set(df["name"])]
    remaining = [name for name in sorted(df["name"].unique()) if name not in site_names]
    site_names.extend(remaining)

    n_sites = len(site_names)
    ncols = 4
    nrows = max(1, math.ceil(n_sites / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 3.5 * nrows), dpi=150, sharex=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    y_min = float(df["bias_value"].min())
    y_max = float(df["bias_value"].max())
    y_pad = max(0.3, 0.08 * max(abs(y_min), abs(y_max), y_max - y_min))

    for ax, site_name in zip(axes_flat, site_names):
        grp = df[df["name"] == site_name].sort_values(["lead_time_days", "geps_init_date", "event_date"])
        ax.scatter(
            grp["lead_time_days"],
            grp["bias_value"],
            s=16,
            alpha=0.35,
            color="#4c78a8",
            edgecolors="none",
        )
        lead_summary = grp.groupby("lead_time_days", as_index=False)["bias_value"].median()
        ax.plot(
            lead_summary["lead_time_days"],
            lead_summary["bias_value"],
            color="#d62728",
            linewidth=1.5,
        )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(site_name, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for ax in axes_flat[n_sites:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=0.995)
    fig.supxlabel("Lead time (days)")
    fig.supylabel("Bias value (C)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bias_single_site(
    grp: pd.DataFrame,
    site_name: str,
    out_path: Path,
    title: str | None,
    y_limits: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.scatter(
        grp["lead_time_days"],
        grp["bias_value"],
        s=18,
        alpha=0.35,
        color="#4c78a8",
        edgecolors="none",
    )
    lead_summary = grp.groupby("lead_time_days", as_index=False)["bias_value"].median()
    ax.plot(
        lead_summary["lead_time_days"],
        lead_summary["bias_value"],
        color="#d62728",
        linewidth=1.8,
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Bias value (C)")
    ax.set_title(title or site_name)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bias_all_sites_mean(
    df: pd.DataFrame,
    out_path: Path,
    title: str | None,
) -> None:
    lead_stats = (
        df.groupby("lead_time_days")["bias_value"]
        .agg(
            median="median",
            mean="mean",
            p25=lambda s: np.nanpercentile(s, 25),
            p75=lambda s: np.nanpercentile(s, 75),
        )
        .reset_index()
        .sort_values("lead_time_days")
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.scatter(
        df["lead_time_days"],
        df["bias_value"],
        s=12,
        alpha=0.18,
        color="#4c78a8",
        edgecolors="none",
        label="All station cases",
    )
    ax.fill_between(
        lead_stats["lead_time_days"],
        lead_stats["p25"],
        lead_stats["p75"],
        color="#f28e2b",
        alpha=0.25,
        label="Station IQR by lead",
    )
    ax.plot(
        lead_stats["lead_time_days"],
        lead_stats["median"],
        color="#d62728",
        linewidth=2.0,
        label="Median across stations",
    )
    ax.plot(
        lead_stats["lead_time_days"],
        lead_stats["mean"],
        color="#2ca02c",
        linewidth=1.6,
        linestyle="--",
        label="Mean across stations",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Bias value (C)")
    ax.set_title(title or "Average bias tendency across stations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV from 13_GEPS_FROST_th_find_biaslead.py")
    ap.add_argument("--stations", required=True, help="Station CSV used for plotting order")
    ap.add_argument(
        "--out-dir",
        default=str((Path(__file__).resolve().parent / ".." / "plots").resolve()),
        help="Output directory",
    )
    ap.add_argument("--title", default=None, help="Optional title")
    ap.add_argument("--output-name", default=None, help="Optional output PNG filename")
    ap.add_argument(
        "--per-site-dir",
        default=None,
        help="Optional directory name under out-dir for one-figure-per-station output",
    )
    ap.add_argument(
        "--aggregate-name",
        default=None,
        help="Optional output PNG filename for all-station aggregate tendency plot",
    )
    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    station_path = Path(args.stations)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not station_path.exists():
        raise FileNotFoundError(station_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(csv_path)
    station_order = load_station_order(station_path)
    y_min = float(df["bias_value"].min())
    y_max = float(df["bias_value"].max())
    y_pad = max(0.3, 0.08 * max(abs(y_min), abs(y_max), y_max - y_min))
    y_limits = (y_min - y_pad, y_max + y_pad)

    if args.output_name:
        out_path = out_dir / sanitize_filename(args.output_name)
    else:
        stem = sanitize_filename(csv_path.stem + "_bias_vs_lead_by_site")
        out_path = out_dir / f"{stem}.png"
    plot_bias_panels(df, station_order, out_path, args.title)

    outputs = [out_path]

    if args.per_site_dir:
        per_site_dir = out_dir / sanitize_filename(args.per_site_dir)
        per_site_dir.mkdir(parents=True, exist_ok=True)
        site_names = [name for name in station_order if name in set(df["name"])]
        site_names.extend([name for name in sorted(df["name"].unique()) if name not in site_names])
        for site_name in site_names:
            grp = df[df["name"] == site_name].sort_values(["lead_time_days", "geps_init_date", "event_date"])
            if grp.empty:
                continue
            site_out = per_site_dir / f"{sanitize_filename(site_name)}_bias_vs_lead.png"
            site_title = f"{site_name}" if args.title is None else f"{site_name} - {args.title}"
            plot_bias_single_site(grp, site_name, site_out, site_title, y_limits)
            outputs.append(site_out)

    if args.aggregate_name:
        agg_out = out_dir / sanitize_filename(args.aggregate_name)
        plot_bias_all_sites_mean(df, agg_out, args.title)
        outputs.append(agg_out)

    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
