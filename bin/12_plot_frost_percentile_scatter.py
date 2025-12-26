#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot percentile vs lead time scatter by threshold from FROST_th_find CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


REQUIRED_COLS = {
    "threshold",
    "lead_time_days",
    "percentile",
}


def sanitize_filename(s: str) -> str:
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    for ch in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]:
        s = s.replace(ch, "-")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _apply_yaxis_style(ax, y_max: int, minor_step: int) -> None:
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
    ax.grid(True, alpha=0.3, which="major", axis="y")
    ax.grid(True, alpha=0.15, which="minor", axis="y")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV from 11_GEPS_FROST_th_find.py")
    ap.add_argument("--out-dir", default=str((Path(__file__).resolve().parent / ".." / "plots").resolve()))
    ap.add_argument("--title", default=None, help="Optional plot title suffix")
    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if not REQUIRED_COLS.issubset(df.columns):
        missing = REQUIRED_COLS - set(df.columns)
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for threshold, grp in df.groupby("threshold"):
        x = grp["lead_time_days"].astype("float64")
        y = grp["percentile"].astype("float64")

        for y_max, minor_step, suffix in [(100, 5, ""), (20, 1, "_0-20")]:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
            ax.scatter(x, y, s=14, alpha=0.6, edgecolors="none")
            ax.set_xlabel("Lead time (days)")
            ax.set_ylabel("Percentile")
            _apply_yaxis_style(ax, y_max, minor_step)
            title = f"Threshold {threshold}"
            if args.title:
                title = f"{title} - {args.title}"
            ax.set_title(title)
            fig.tight_layout()

            stem = sanitize_filename(f"frost_scatter_th{threshold}{suffix}")
            out_path = out_dir / f"{stem}.png"
            fig.savefig(out_path)
            plt.close(fig)

    thresholds = sorted(df["threshold"].dropna().unique())
    if len(thresholds) >= 2:
        for y_max, minor_step, suffix in [(100, 5, ""), (20, 1, "_0-20")]:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
            colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
            for i, threshold in enumerate(reversed(thresholds)):
                grp = df[df["threshold"] == threshold]
                x = grp["lead_time_days"].astype("float64")
                y = grp["percentile"].astype("float64")
                ax.scatter(
                    x,
                    y,
                    s=14,
                    alpha=0.6,
                    edgecolors="none",
                    label=f"Threshold {threshold}",
                    color=colors[i % len(colors)],
                )
            ax.set_xlabel("Lead time (days)")
            ax.set_ylabel("Percentile")
            _apply_yaxis_style(ax, y_max, minor_step)
            title = "Thresholds combined"
            if args.title:
                title = f"{title} - {args.title}"
            ax.set_title(title)
            ax.legend(fontsize=8)
            fig.tight_layout()

            out_path = out_dir / f"frost_scatter_combined{suffix}.png"
            fig.savefig(out_path)
            plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
