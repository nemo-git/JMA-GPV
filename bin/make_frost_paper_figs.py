#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def require_cols(df, name, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}\nAvailable: {list(df.columns)}")

def compute_pzero(df, lead_max):
    d = df[(df["lead_time_days"] >= 1) & (df["lead_time_days"] <= lead_max)]
    ev = d[d["event_flag"] == 1]
    out = (
        ev.groupby(["threshold", "lead_time_days"])["percentile"]
          .apply(lambda s: float((s == 0).mean()))
          .reset_index(name="p_zero")
    )
    return out

def compute_q_miss_far(df, lead_max, q_alpha=0.05):
    """
    percentileの向きは、あなたのデータ解析結果に合わせて
      - 低温事例ほど percentile が大きい
    を前提にしています（したがって risk = (percentile >= q) ）。
    qは event側percentileの5%点（miss<=5%）で設定。
    """
    d = df[(df["lead_time_days"] >= 1) & (df["lead_time_days"] <= lead_max)].copy()

    rows = []
    for (th, lead), g in d.groupby(["threshold", "lead_time_days"]):
        ev = g[g["event_flag"] == 1]["percentile"].dropna().values
        ne = g[g["event_flag"] == 0]["percentile"].dropna().values

        n_ev, n_ne = len(ev), len(ne)
        if n_ev == 0 or n_ne == 0:
            continue

        q = float(np.quantile(ev, q_alpha))
        miss = float((ev < q).mean())   # miss = eventなのにriskにならない割合
        far  = float((ne >= q).mean())  # FAR = non-eventなのにriskになる割合

        rows.append([float(th), int(lead), n_ev, n_ne, q, miss, far])

    out = pd.DataFrame(rows, columns=["threshold","lead","n_event","n_nonevent","q","miss","far"])
    return out.sort_values(["threshold","lead"])

def plot_pzero(pre, post, outdir, lead_max):
    p0_pre  = compute_pzero(pre, lead_max).rename(columns={"p_zero":"p_zero_pre"})
    p0_post = compute_pzero(post, lead_max).rename(columns={"p_zero":"p_zero_post"})
    merged = pd.merge(p0_pre, p0_post, on=["threshold","lead_time_days"], how="outer")

    for th in sorted(merged["threshold"].dropna().unique()):
        g = merged[merged["threshold"] == th].sort_values("lead_time_days")
        x = g["lead_time_days"].values
        y1 = g["p_zero_pre"].values
        y2 = g["p_zero_post"].values

        plt.figure()
        plt.plot(
            x, y1,
            color="black", linestyle="-", linewidth=1.5,
            marker="o", markerfacecolor="white",
            label="pre (no bias-corr)"
        )
        plt.plot(
            x, y2,
            color="black", linestyle="--", linewidth=1.5,
            marker="s", markerfacecolor="white",
            label="post (lead-mean bias-corr)"
        )
        plt.ylim(-0.02, 1.02)
        plt.xlabel("Lead time (days)")
        plt.ylabel("p_zero = P(percentile = 0 | event=1)")
        plt.title(f"Fig1  p_zero vs lead  (threshold={th}°C)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        fn = os.path.join(outdir, f"Fig1_pzero_threshold_{th:+.0f}.png".replace("+","p").replace("-","m"))
        plt.tight_layout()
        plt.savefig(fn, dpi=200)
        plt.close()

    return merged

def plot_miss_far(post, outdir, lead_max):
    mf = compute_q_miss_far(post, lead_max, q_alpha=0.05)

    for th in sorted(mf["threshold"].unique()):
        g = mf[mf["threshold"] == th].sort_values("lead")
        x = g["lead"].values

        plt.figure()
        plt.plot(
            x, g["miss"].values,
            color="black", linestyle="-", linewidth=1.5,
            marker="o", markerfacecolor="white",
            label="miss (target<=0.05)"
        )
        plt.plot(
            x, g["far"].values,
            color="black", linestyle="--", linewidth=1.5,
            marker="^", markerfacecolor="white",
            label="FAR"
        )
        plt.ylim(-0.02, 1.02)
        plt.xlabel("Lead time (days)")
        plt.ylabel("Rate")
        plt.title(f"Fig2  miss & FAR vs lead  (threshold={th}°C, risk=percentile>=q)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        fn = os.path.join(outdir, f"Fig2_miss_far_threshold_{th:+.0f}.png".replace("+","p").replace("-","m"))
        plt.tight_layout()
        plt.savefig(fn, dpi=200)
        plt.close()

    return mf

def plot_box_by_lead(post, outdir, leads=(2,10), lead_max=30):
    # threshold別に色を指定しない（matplotlibデフォルト）で、同一図内に4箱（-1/3 × event/non）を並べる
    for lead in leads:
        d = post[(post["lead_time_days"] == lead)].copy()

        # 抽出
        groups = []
        labels = []
        box_meta = []
        thresholds = sorted(d["threshold"].dropna().unique())
        if len(thresholds) == 0:
            continue
        shades = np.linspace(0.25, 0.8, len(thresholds))
        for th_idx, th in enumerate(thresholds):
            for evflag, lab in [(0,"non-event"),(1,"event")]:
                vals = d[(d["threshold"]==th) & (d["event_flag"]==evflag)]["percentile"].dropna().values
                if len(vals) == 0:
                    vals = np.array([np.nan])
                groups.append(vals)
                labels.append(f"{th}C\n{lab}")
                box_meta.append({"shade": float(shades[th_idx]), "event": evflag})

        plt.figure()
        bp = plt.boxplot(
            groups, labels=labels, showfliers=False, patch_artist=True,
            medianprops={"color":"black"},
            boxprops={"linewidth":1.0, "color":"black"},
            whiskerprops={"color":"black"},
            capprops={"color":"black"}
        )
        for box, meta in zip(bp["boxes"], box_meta):
            shade = meta["shade"]
            hatch = "//" if meta["event"] == 1 else "xx"
            box.set_facecolor(str(shade))
            box.set_hatch(hatch)
        plt.ylabel("percentile (rank)")
        plt.title(f"Fig3  Percentile distributions at lead={lead} (post-correction)")
        plt.grid(True, axis="y", alpha=0.3)

        fn = os.path.join(outdir, f"Fig3_box_lead_{lead:02d}.png")
        plt.tight_layout()
        plt.savefig(fn, dpi=200)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="補正前CSV")
    ap.add_argument("--post", required=True, help="補正後CSV（mean補正）")
    ap.add_argument("--outdir", default="paper_figs", help="出力ディレクトリ")
    ap.add_argument("--lead-max", type=int, default=30, help="図表に含めるlead最大（1..lead-max）")
    ap.add_argument("--box-leads", default="2,10", help="Fig3で使うlead（カンマ区切り）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pre  = pd.read_csv(args.pre)
    post = pd.read_csv(args.post)

    need = ["threshold","lead_time_days","event_flag","percentile"]
    require_cols(pre,  "pre",  need)
    require_cols(post, "post", need)

    # 図1
    p0 = plot_pzero(pre, post, args.outdir, args.lead_max)
    p0.to_csv(os.path.join(args.outdir, "Table_pzero_pre_post.csv"), index=False)

    # 図2 + 表1（miss/far含む）
    mf = plot_miss_far(post, args.outdir, args.lead_max)

    # 表1：lead×thresholdで全部まとめ（p_zeroも結合）
    p0_post = compute_pzero(post, args.lead_max).rename(columns={"lead_time_days":"lead","p_zero":"p_zero_post"})
    p0_pre  = compute_pzero(pre,  args.lead_max).rename(columns={"lead_time_days":"lead","p_zero":"p_zero_pre"})
    table1 = mf.merge(p0_pre, on=["threshold","lead"], how="left").merge(p0_post, on=["threshold","lead"], how="left")
    table1 = table1.sort_values(["threshold","lead"])
    table1.to_csv(os.path.join(args.outdir, "Table1_threshold_lead_metrics.csv"), index=False)

    # 図3
    box_leads = tuple(int(x) for x in args.box_leads.split(",") if x.strip())
    plot_box_by_lead(post, args.outdir, leads=box_leads, lead_max=args.lead_max)

    print("Done. Outputs in:", args.outdir)
    print(" - Fig1_pzero_threshold_*.png")
    print(" - Fig2_miss_far_threshold_*.png")
    print(" - Fig3_box_lead_*.png")
    print(" - Table1_threshold_lead_metrics.csv")

if __name__ == "__main__":
    main()
