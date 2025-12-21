#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
import numpy as np
import xarray as xr
import pygrib


DEFAULT_DIR = "/Users/nemo/Dropbox/linux_work/JMA_GPV"


# --------------------- 常量 ---------------------

ALIASES = {
    "TMP": {"TMP","T2M","2T"},
    "RH": {"RH","2R"},
    "UGRD": {"UGRD","U10","U"},
    "VGRD": {"VGRD","V10","V"},
    "PRMSL": {"PRMSL","MSLP","MSL"},
    "TCDC": {"TCDC","TCC"},
    "APCP": {"APCP","TP"},
    # upper-only
    "HGT": {"HGT","GH","Z"},
    "VVEL": {"VVEL","W","OMEGA"},
}

UPPER_LEVELS = {
    925: ["HGT","UGRD","VGRD","TMP","RH"],
    850: ["HGT","UGRD","VGRD","TMP","RH"],
    700: ["TMP","RH","VVEL"],
    500: ["HGT","UGRD","VGRD","TMP"],
}

# 変数ごとの既定パッキング
PACK_DEFAULTS = {
    "TMP":   dict(dtype="int16", scale=0.01,  add=0.0,     fill=-32767),
    "RH":    dict(dtype="int16", scale=0.1,   add=0.0,     fill=-32767),
    "UGRD":  dict(dtype="int16", scale=0.01,  add=0.0,     fill=-32767),
    "VGRD":  dict(dtype="int16", scale=0.01,  add=0.0,     fill=-32767),
    "TCDC":  dict(dtype="int16", scale=1.0,   add=0.0,     fill=-32767),
    "PRMSL": dict(dtype="int16", scale=1.0,   add=100000.0, fill=-32767),  # Pa→ int16(+offset)
    "APCP":  dict(dtype="int16", scale=0.1,   add=0.0,     fill=-32767),   # 0.1 mm 相当
    "HGT":   dict(dtype="int16", scale=0.1,   add=0.0,     fill=-32767),   # 0.1 m (gpm相当でもOK)
    "VVEL":  dict(dtype="int16", scale=0.001, add=0.0,     fill=-32767),   # 0.001 Pa s-1
}

# --------------------- ヘルパ ---------------------

def _dbg(on, *a, **k):
    if on: print(*a, **k, flush=True)

def _normalize_var(v: str):
    vu = v.upper()
    for key, vals in ALIASES.items():
        if vu in vals: return key
    raise ValueError(f"--var {v} は未対応です。サポート: " + ", ".join(ALIASES.keys()))



def _normalize_ensemble_coord(da: xr.DataArray, debug: bool=False, tag: str=""):
    """Ensure ensemble coord is comparable across parts: cast to int when possible and sort."""
    if "ensemble" not in da.dims:
        return da
    ens = da.coords["ensemble"].values
    ens_new = ens
    try:
        ens_new = ens.astype(int)
    except Exception:
        pass
    if ens_new is not ens:
        da = da.assign_coords(ensemble=ens_new)
    # sort by ensemble to avoid ordering mismatches
    try:
        da = da.sortby("ensemble")
    except Exception:
        pass
    return da

def _to_1d_coords(lats2d: np.ndarray, lons2d: np.ndarray, tol=1e-10):
    ny, nx = lats2d.shape
    if not np.allclose(lats2d, np.repeat(lats2d[:, :1], nx, axis=1), atol=tol, rtol=0):
        raise ValueError("格子が曲線（latitude が列ごとに変化）→ 1D 座標不可")
    if not np.allclose(lons2d, np.repeat(lons2d[:1, :], ny, axis=0), atol=tol, rtol=0):
        raise ValueError("格子が曲線（longitude が行ごとに変化）→ 1D 座標不可")
    lat1d = lats2d[:, 0].copy()
    lon1d = lons2d[0, :].copy()
    lat_desc = np.any(np.diff(lat1d) < 0)
    return lat1d, lon1d, lat_desc

def packing_for(element: str, units: str):
    e = element.upper()
    cfg = dict(PACK_DEFAULTS.get(e, dict(dtype="int16", scale=0.01, add=0.0, fill=-32767)))
    u = (units or "").strip().lower()
    if e == "PRMSL":
        if "hpa" in u or u == "mb":
            return dict(dtype="int16", scale=0.1, add=1000.0, fill=-32767)
        return dict(dtype="int16", scale=1.0, add=100000.0, fill=-32767)
    if e == "APCP":
        if u == "m" or u.endswith(" m"):
            return dict(dtype="int16", scale=1e-4, add=0.0, fill=-32767)  # 0.1 mm
        if "kg m-2" in u or "kg/m2" in u or "mm" in u:
            return dict(dtype="int16", scale=0.1, add=0.0, fill=-32767)
    return cfg

def add_ens_summary_vars(ds: xr.Dataset, da_ens: xr.DataArray, base_name: str, debug: bool=False):
    """Add ensemble summary variables (mean, spread, percentiles) into ds.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to add variables into.
    da_ens : xr.DataArray
        DataArray with dimension 'ensemble'.
    base_name : str
        Base variable name (e.g. 'TMP', 'PRMSL', 'HGT500').
    """
    if "ensemble" not in da_ens.dims:
        raise ValueError("da_ens must have 'ensemble' dimension")

    # Mean and spread (std dev)
    da_mean = da_ens.mean(dim="ensemble")
    da_std  = da_ens.std(dim="ensemble", ddof=0)

    mean_name = f"{base_name}_mean"
    spr_name  = f"{base_name}_spread"

    da_mean.name = mean_name
    da_std.name  = spr_name

    # basic attrs
    units = str(da_ens.attrs.get("units", ""))
    if units:
        da_mean.attrs["units"] = units
        da_std.attrs["units"] = units
    da_mean.attrs["description"] = "Ensemble mean"
    da_std.attrs["description"]  = "Ensemble spread (standard deviation, ddof=0)"

    # Percentiles
    perc_list = [1, 5, 10, 20, 50, 80, 90, 95, 99]
    qs = [p/100.0 for p in perc_list]
    try:
        q_da = da_ens.quantile(qs, dim="ensemble", method="linear")
    except TypeError:
        # older xarray
        q_da = da_ens.quantile(qs, dim="ensemble", interpolation="linear")

    # q_da has dim 'quantile' (values = qs)
    for p, q in zip(perc_list, qs):
        vname = f"{base_name}_p{p:02d}"
        one = q_da.sel(quantile=q, method="nearest").drop_vars("quantile")
        one.name = vname
        if units:
            one.attrs["units"] = units
        one.attrs["description"] = f"Ensemble percentile {p}%"
        ds[vname] = one

    ds[mean_name] = da_mean
    ds[spr_name]  = da_std
    return ds

def _apcp_end_hour(m) -> int:
    sr = getattr(m, "stepRange", None)
    if sr and isinstance(sr, str) and "-" in sr:
        try: return int(sr.split("-")[-1])
        except Exception: pass
    ft = getattr(m, "forecastTime", None)
    try: return int(ft)
    except Exception: return 0

# --------------------- メッセージ選別 ---------------------

def _match_lsurf_msg(g, element: str):
    e = element.upper()
    sn = (getattr(g, "shortName", "") or "").lower()
    nm = (getattr(g, "name", "") or "").lower()
    tol = getattr(g, "typeOfLevel", None)
    try:
        lvl = int(getattr(g, "level", 0))
    except Exception:
        lvl = None
    disc = getattr(g, "discipline", None)
    cat  = getattr(g, "parameterCategory", None)
    num  = getattr(g, "parameterNumber", None)
    step_type = (getattr(g, "stepType", "") or "").lower()

    # name 由来の高さキーワード
    m2  = ("2 m above ground" in nm) or ("2m above ground" in nm)
    m10 = ("10 m above ground" in nm) or ("10m above ground" in nm)

    def is_2m():
        if tol == "heightAboveGround" and lvl == 2: return True
        if m2: return True
        # 1mEGPV で tol が surface/None でも name に 2m が出る場合に対応
        if tol in (None, "surface") and (lvl in (None, 0)) and m2: return True
        return False

    def is_10m():
        if tol == "heightAboveGround" and lvl == 10: return True
        if m10: return True
        if tol in (None, "surface") and (lvl in (None, 0)) and m10: return True
        return False

    # 上空/土壌を除外するための NG セット
    iso_levels = {"isobaricInhPa","isobaricInPa","isobaric","isobaricLayer"}
    soil_levels = {"depthBelowLandLayer","soil","depthBelowLand"}

    if e == "TMP":
        # メイン条件：2m かつ 温度
        if is_2m() and (sn in {"2t","t2m","tmp"} or "temperature" in nm):
            return True
        # フォールバック：GRIB2 コード (0-0-0 Temperature) ＆ 上空/土壌レベルを除外
        if (disc == 0 and cat == 0 and num == 0) and (tol not in iso_levels|soil_levels):
            # level=2 や name の 2m 表記が取れないケースの救済
            if is_2m() or (tol in (None,"surface","heightAboveGround") and (lvl in (None,0,2))):
                return True
        return False

    if e == "RH":
        return is_2m() and (sn in {"2r","rh"} or "relative humidity" in nm)

    if e == "UGRD":
        return is_10m() and (sn in {"10u","ugrd","u"} or "u component" in nm)

    if e == "VGRD":
        return is_10m() and (sn in {"10v","vgrd","v"} or "v component" in nm)

    if e == "PRMSL":
        if sn in {"msl","prmsl"} or "mean sea level" in nm or "pressure reduced to msl" in nm:
            return True
        return (disc == 0 and cat == 3 and num == 1)

    if e == "TCDC":
        if sn in {"tcc","tcdc","tcld"} or "total cloud cover" in nm:
            return True
        return (disc == 0 and cat == 6 and num == 1)

    if e == "APCP":
        if (sn in {"tp","apcp"} or "total precipitation" in nm):
            return step_type in {"accum","acc"} or step_type == ""
        return (disc == 0 and cat == 1 and num == 8 and (step_type in {"accum","acc"} or step_type == ""))

    return False



def _match_lpall_msg(g, element: str, level_hpa: int):
    """上空（等圧面）向けのフィルタ"""
    e = element.upper()
    sn = (getattr(g, "shortName", "") or "").lower()
    nm = (getattr(g, "name", "") or "").lower()
    tol = getattr(g, "typeOfLevel", None)
    lvl = None
    try: lvl = int(getattr(g, "level", 0))
    except Exception: pass

    # 等圧面の表記ゆれに対応
    is_iso = tol in ("isobaricInhPa","isobaricInPa","isobaric")
    if not (is_iso and lvl == int(level_hpa)):
        return False

    if e == "TMP":
        return (sn in {"t","tmp"} or "temperature" in nm)
    if e == "RH":
        return (sn in {"r","rh"} or "relative humidity" in nm)
    if e == "UGRD":
        return (sn in {"u","ugrd"} or "u component of wind" in nm or "u-component of wind" in nm)
    if e == "VGRD":
        return (sn in {"v","vgrd"} or "v component of wind" in nm or "v-component of wind" in nm)
    if e == "HGT":
        return (sn in {"gh","z","hgt"} or "geopotential height" in nm)
    if e == "VVEL":
        return (sn in {"w","omega","vvel"} or "vertical velocity" in nm)
    return False

# --------------------- 読み出し（Lsurf / Lpall） ---------------------

def _load_lsurf_element_ens_signed_1dcoords(grib_path: Path, element: str, invert_sign=False, debug=False) -> xr.DataArray:
    _dbg(debug, f"[READ Lsurf] {grib_path} ({element})")
    grbs = pygrib.open(str(grib_path))
    msgs = [g for g in grbs if _match_lsurf_msg(g, element)]
    if not msgs:
        if debug:
            try:
                grbs.seek(0)  # pygrib では seek(0) が安全
                src = grbs
            except Exception:
                grbs.close()
                src = pygrib.open(str(grib_path))  # 開き直し
            cnt = Counter(
                (
                    (getattr(gg,"shortName","") or "").lower(),
                    (getattr(gg,"name","") or "").lower()[:50],
                    getattr(gg,"typeOfLevel",None),
                    getattr(gg,"level",None),
                    getattr(gg,"discipline",None),
                    getattr(gg,"parameterCategory",None),
                    getattr(gg,"parameterNumber",None),
                )
                for gg in src
            )
            print(f"[DEBUG] {element}: no hits in {grib_path.name}. Top keys:")
            for (snm, nm50, tolv, lv, d, c, n), cntv in cnt.most_common(20):
                print(f"  {cntv:4d} sn={snm:6s} name={nm50:50s} tol={tolv} lv={lv} code={d}-{c}-{n}")
            try:
                src.close()
            except Exception:
                pass
        else:
            grbs.close()
        raise RuntimeError(f"{element}: 対象メッセージが見つかりません: {grib_path}")


    # --- 座標 ---
    lats2d, lons2d = msgs[0].latlons()
    ny, nx = lats2d.shape
    lat1d, lon1d, lat_desc = _to_1d_coords(lats2d, lons2d)

    # --- 時間 ---
    if element.upper() == "APCP":
        ts = []
        for m in msgs:
            ad = getattr(m, "analDate", None) or getattr(m, "validDate", None)
            ts.append(ad + timedelta(hours=_apcp_end_hour(m)))
        times = sorted(set(ts))
        t_index = {t:i for i,t in enumerate(times)}
        def _ti(m): return t_index[(getattr(m,"analDate",None) or getattr(m,"validDate",None)) + timedelta(hours=_apcp_end_hour(m))]
    else:
        times = sorted({m.validDate for m in msgs})
        t_index = {t:i for i,t in enumerate(times)}
        def _ti(m): return t_index[m.validDate]

    # --- アンサンブル最大番号を自動検出（1m は 12、1w2w は 25 が想定） ---
    nums = {int(m.number) if m.number is not None else 0 for m in msgs}
    max_n = max(n for n in nums if 0 <= n <= 100)  # 安全側
    nt = len(times)

    # バッファ (time, number=0..max_n, branch=0/1, y, x)
    buf = np.full((nt, max_n+1, 2, ny, nx), np.nan, dtype=np.float32)
    seen = defaultdict(int)
    for m in msgs:
        ti = _ti(m)
        n  = int(m.number) if m.number is not None else 0
        if not (0 <= n <= max_n): continue
        bi = seen[(ti, n)];  bi = bi if bi <= 1 else 1
        arr = m.values.astype(np.float32, copy=False)
        if arr.shape != (ny, nx):
            grbs.close(); raise RuntimeError(f"{element}: 格子サイズ混在")
        buf[ti, n, bi] = arr
        seen[(ti, n)] += 1
    grbs.close()

    if lat_desc:
        buf = buf[:, :, :, ::-1, :]
        lat1d = lat1d[::-1]

    # branch→±（branch0=+k, branch1=-k）
    sign_for_branch = np.array([+1, -1], dtype=np.int8)
    num_signed = np.array(list(range(-max_n,0)) + [0] + list(range(1,max_n+1)), dtype=np.int16)
    out = np.full((nt, len(num_signed), ny, nx), np.nan, dtype=np.float32)

    # 制御
    ctl = buf[:, 0]
    ctl_merged = np.where(np.isnan(ctl[:,0]), ctl[:,1], ctl[:,0])
    out[:, np.where(num_signed==0)[0][0]] = ctl_merged

    # 擾乱
    for k in range(1, max_n+1):
        for bi in (0,1):
            signed = sign_for_branch[bi]*k
            idx = int(np.where(num_signed==signed)[0][0])
            out[:, idx] = buf[:, k, bi]

    if element.upper() == "APCP":
        # Convert cumulative precip to step-wise increment (difference from previous time).
        prev = np.zeros_like(out[:1])
        out = np.diff(out, axis=0, prepend=prev)
        # The first step can be large due to cumulative base; mark as missing.
        out[0] = np.nan


    # 変数名・単位
    units = getattr(msgs[0], "units", "")
    name = element.upper()

    da = xr.DataArray(
        out,
        dims=("time", "ensemble", "latitude", "longitude"),
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "ensemble": num_signed.astype(np.int16),
            "latitude": lat1d,
            "longitude": lon1d,
        },
        name=name,
        attrs={"units": units, "long_name": msgs[0].name, "standard_name": name.lower()},
    )
    return da

def _load_lpall_element_ens_signed_1dcoords(grib_path: Path, element: str, level_hpa: int, debug=False) -> xr.DataArray:
    _dbg(debug, f"[READ Lpall] {grib_path} ({element}@{level_hpa}hPa)")
    grbs = pygrib.open(str(grib_path))
    msgs = [g for g in grbs if _match_lpall_msg(g, element, level_hpa)]
    if not msgs:
        if debug:
            try:
                grbs.seek(0)  # pygrib では seek(0) が安全
                src = grbs
            except Exception:
                grbs.close()
                src = pygrib.open(str(grib_path))  # 開き直し
            cnt = Counter(
                (
                    (getattr(gg,"shortName","") or "").lower(),
                    (getattr(gg,"name","") or "").lower()[:50],
                    getattr(gg,"typeOfLevel",None),
                    getattr(gg,"level",None),
                    getattr(gg,"discipline",None),
                    getattr(gg,"parameterCategory",None),
                    getattr(gg,"parameterNumber",None),
                )
                for gg in src
            )
            print(f"[DEBUG] {element}: no hits in {grib_path.name}. Top keys:")
            for (snm, nm50, tolv, lv, d, c, n), cntv in cnt.most_common(20):
                print(f"  {cntv:4d} sn={snm:6s} name={nm50:50s} tol={tolv} lv={lv} code={d}-{c}-{n}")
            try:
                src.close()
            except Exception:
                pass
        else:
            grbs.close()
        raise RuntimeError(f"{element}: 対象メッセージが見つかりません: {grib_path}")


    # --- 座標 ---
    lats2d, lons2d = msgs[0].latlons()
    ny, nx = lats2d.shape
    lat1d, lon1d, lat_desc = _to_1d_coords(lats2d, lons2d)

    times = sorted({m.validDate for m in msgs})
    t_index = {t:i for i,t in enumerate(times)}
    def _ti(m): return t_index[m.validDate]

    # --- アンサンブル最大番号を自動検出（1m は 12、1w2w は 25 が想定） ---
    nums = {int(m.number) if m.number is not None else 0 for m in msgs}
    max_n = max(n for n in nums if 0 <= n <= 100)  # 安全側
    nt = len(times)

    # バッファ (time, number=0..max_n, branch=0/1, y, x)
    buf = np.full((nt, max_n+1, 2, ny, nx), np.nan, dtype=np.float32)
    seen = defaultdict(int)
    for m in msgs:
        ti = _ti(m)
        n  = int(m.number) if m.number is not None else 0
        if not (0 <= n <= max_n): continue
        bi = seen[(ti, n)];  bi = bi if bi <= 1 else 1
        arr = m.values.astype(np.float32, copy=False)
        if arr.shape != (ny, nx):
            grbs.close(); raise RuntimeError(f"{element}: 格子サイズ混在")
        buf[ti, n, bi] = arr
        seen[(ti, n)] += 1
    grbs.close()

    if lat_desc:
        buf = buf[:, :, :, ::-1, :]
        lat1d = lat1d[::-1]

    # branch→±（branch0=+k, branch1=-k）
    sign_for_branch = np.array([+1, -1], dtype=np.int8)
    num_signed = np.array(list(range(-max_n,0)) + [0] + list(range(1,max_n+1)), dtype=np.int16)
    out = np.full((nt, len(num_signed), ny, nx), np.nan, dtype=np.float32)

    # 制御
    ctl = buf[:, 0]
    ctl_merged = np.where(np.isnan(ctl[:,0]), ctl[:,1], ctl[:,0])
    out[:, np.where(num_signed==0)[0][0]] = ctl_merged

    # 擾乱
    for k in range(1, max_n+1):
        for bi in (0,1):
            signed = sign_for_branch[bi]*k
            idx = int(np.where(num_signed==signed)[0][0])
            out[:, idx] = buf[:, k, bi]

    units = getattr(msgs[0], "units", "")
    name = element.upper()

    da = xr.DataArray(
        out,
        dims=("time", "ensemble", "latitude", "longitude"),
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "ensemble": num_signed.astype(np.int16),
            "latitude": lat1d,
            "longitude": lon1d,
        },
        name=name,
        attrs={"units": units, "long_name": f"{msgs[0].name} @ {level_hpa} hPa", "standard_name": name.lower()},
    )
    return da

# --------------------- ファイル探索 ---------------------

def _find_target_files_for_12utc_prevday(base_dir: Path, run_date_yyyymmdd: str, is_upper: bool, debug=False):
    from datetime import datetime, timezone, timedelta
    import re

    ymd = datetime.strptime(run_date_yyyymmdd, "%Y%m%d").replace(tzinfo=timezone.utc)
    target = (ymd - timedelta(days=1)).strftime("%Y%m%d")
    yyyy_prev = target[:4]

    # ★ Lsurf と L-pall でパターンを分ける（L-pall はハイフン有り/無しの両対応）
    if is_upper:
        patterns = [
            f"Z__C_*_{target}12*_L-pall_*_grib2.bin",
            f"*{target}12*L-pall*grib2.bin",
            f"Z__C_*_{target}12*_Lpall_*_grib2.bin",
            f"*{target}12*Lpall*grib2.bin",
        ]
    else:
        patterns = [
            f"Z__C_*_{target}12*_Lsurf_*_grib2.bin",
            f"*{target}12*Lsurf*grib2.bin",
        ]

    roots_1w = [base_dir/"jmadata"/"1wEGPV"/yyyy_prev, base_dir/"1wEGPV"/yyyy_prev]
    roots_2w = [base_dir/"jmadata"/"2wEGPV"/yyyy_prev, base_dir/"2wEGPV"/yyyy_prev]

    def collect(roots):
        hits = []
        for root in roots:
            if not root.exists():
                continue
            for pat in patterns:
                hits.extend(sorted(root.glob(pat)))
        return hits

    hits_1w = collect(roots_1w)
    hits_2w = collect(roots_2w)

    # フォールバック：--dir 以下を総当たり（パターンは上と同じ）
    if not hits_1w or not hits_2w:
        if debug:
            print("[FALLBACK] rglob:", base_dir, "(upper=" + str(is_upper) + ")")
        for pat in patterns:
            for p in base_dir.rglob(pat):
                s = str(p)
                if "1wEGPV" in s:
                    hits_1w.append(p)
                elif "2wEGPV" in s:
                    hits_2w.append(p)

    # FD の開始時刻でソート（L-pall: 0000-0512, 0518-1100 / 2w: 1106-1800）
    def fd_key(p: Path):
        m = re.search(r"FD(\d{4})-(\d{4})", p.name)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (999999, 999999)

    hits_1w = sorted(set(hits_1w), key=fd_key)
    hits_2w = sorted(set(hits_2w), key=fd_key)

    sel_1w = hits_1w[:2]   # 1wは2本
    sel_2w = hits_2w[:1]   # 2wは1本

    if debug:
        print(f"[SCAN] prev-day= {target}  layer={'L-pall' if is_upper else 'Lsurf'}")
        print("  1w candidates:"); [print("   ", p) for p in hits_1w]
        print("  2w candidates:"); [print("   ", p) for p in hits_2w]
        print("[PICK] 1w:", sel_1w)
        print("[PICK] 2w:", sel_2w)

    if len(sel_1w) < 1 or len(sel_2w) < 1:
        raise FileNotFoundError("必要なファイルが見つかりません（1日前12UTCの 1w×≥1, 2w×≥1 を期待）")
    return [*sel_1w, *sel_2w]

def _find_monthly_paths_for_date(base_dir: Path, target_yyyymmdd: str, is_upper: bool, debug=False):
    """
    1mEGPV/{yyyy} から target_yyyymmdd の 12UTC 初期の Lsurf/L-pall を収集。
    FD の分割本数は配信によって異なる可能性があるため、該当日付の全パートを拾って返す。
    """
    yyyy = target_yyyymmdd[:4]
    if is_upper:
        patterns = [
            f"Z__C_*_{target_yyyymmdd}12*_L-pall_*_grib2.bin",
            f"*{target_yyyymmdd}12*L-pall*grib2.bin",
            f"Z__C_*_{target_yyyymmdd}12*_Lpall_*_grib2.bin",
            f"*{target_yyyymmdd}12*Lpall*grib2.bin",
        ]
    else:
        patterns = [
            f"Z__C_*_{target_yyyymmdd}12*_Lsurf_*_grib2.bin",
            f"*{target_yyyymmdd}12*Lsurf*grib2.bin",
        ]

    roots = [base_dir/"jmadata"/"1mEGPV"/yyyy, base_dir/"1mEGPV"/yyyy]
    hits = []
    for root in roots:
        if not root.exists(): continue
        for pat in patterns:
            hits.extend(sorted(root.glob(pat)))

    # フォールバック（再帰）
    if not hits and base_dir.exists():
        for pat in patterns:
            hits.extend(sorted(base_dir.rglob(pat)))

    # FD 開始でソート
    def fd_key(p: Path):
        m = re.search(r"FD(\d{4})-(\d{4})", p.name)
        if m: return (int(m.group(1)), int(m.group(2)))
        return (999999, 999999)
    hits = sorted(set(hits), key=fd_key)

    if debug:
        layer = "L-pall" if is_upper else "Lsurf"
        print(f"[SCAN 1m] date={target_yyyymmdd} layer={layer}")
        for h in hits: print("   ", h)

    if not hits:
        raise FileNotFoundError(f"[1m] {target_yyyymmdd} のファイルが見つかりません")
    return hits




# --------------------- 連結 & 保存 ---------------------

def concat_time_and_save(paths, element: str, out_path: Path, debug=False, level_hpa: int|None=None):
    # 読み込み
    if level_hpa is None:
        loaders = [_load_lsurf_element_ens_signed_1dcoords(p, element, debug=debug) for p in paths]
    else:
        loaders = [_load_lpall_element_ens_signed_1dcoords(p, element, level_hpa, debug=debug) for p in paths]

    # 座標互換チェック
    lat0 = loaders[0].coords["latitude"].values
    lon0 = loaders[0].coords["longitude"].values
    ens0 = loaders[0].coords["ensemble"].values
    for i, da in enumerate(loaders[1:], start=2):
        if not np.array_equal(da.coords["latitude"].values, lat0) or not np.array_equal(da.coords["longitude"].values, lon0):
            raise ValueError(f"格子が一致しません: {paths[i-1]}")
        if not np.array_equal(da.coords["ensemble"].values, ens0):
            raise ValueError(f"アンサンブル座標が一致しません: {paths[i-1]}")

    da_all = xr.concat(loaders, dim="time").sortby("time")
    # 重複時刻を先勝ちで間引き
    tvals = da_all["time"].values
    _, uniq_idx = np.unique(tvals, return_index=True)
    if len(uniq_idx) != da_all.sizes["time"]:
        da_all = da_all.isel(time=np.sort(uniq_idx))

    # time → int hours since 1900-01-01
    origin = np.datetime64("1900-01-01T00:00:00Z")
    hours = (da_all["time"].values.astype("datetime64[ns]") - origin).astype("timedelta64[h]").astype("int64")

    da_all = da_all.assign_coords(time=("time", hours))
    da_all["time"].attrs.update({"units": "hours since 1900-01-01 00:00:00", "calendar": "proleptic_gregorian"})

    varname = element.upper() if level_hpa is None else f"{element.upper()}{int(level_hpa)}"
    ds = xr.Dataset({varname: da_all})
    # ---- Add ensemble summary variables ----
    ds = add_ens_summary_vars(ds, da_all, varname, debug=debug)

    units = str(da_all.attrs.get("units", ""))

    pk = packing_for(element.upper(), units)
    np_dtype = np.int16 if pk["dtype"] == "int16" else np.int32
    fill = np_dtype(pk["fill"])

    # encoding: apply the same packing to all data variables (raw + summary)
    encoding = {
        "time": {"dtype": "int32", "_FillValue": None},
        "ensemble": {"dtype": "int16", "_FillValue": None},
        "latitude": {"dtype": "float64", "_FillValue": None},
        "longitude": {"dtype": "float64", "_FillValue": None},
    }
    for vn in ds.data_vars:
        encoding[vn] = {
            "dtype": np_dtype,
            "scale_factor": float(pk["scale"]),
            "add_offset": float(pk["add"]),
            "_FillValue": fill,
            "zlib": True,
            "complevel": 4,
        }


    out_path.parent.mkdir(parents=True, exist_ok=True)
    _dbg(debug, f"[SAVE] {out_path} (units='{units}', pack={pk})")
    ds.to_netcdf(out_path, encoding=encoding)



# --------------------- CLI ---------------------


def concat_time_and_save_from_das(das, element: str, out_path: Path, debug=False, level_hpa: int|None=None):
    """Concat already-loaded DataArrays (each containing a subset of time) and save to NetCDF.

    Used for 1m processing where GRIB is split into multiple parts and we already loaded them.
    Each DataArray must have dims (time, ensemble, latitude, longitude).
    This function is robust to:
      - different ensemble ordering across parts (it sorts),
      - partially missing members across parts (it uses the intersection by default).
    """
    if not das:
        raise ValueError("das is empty")

    # Normalize ensemble coord & sort ordering for each part
    loaders = [_normalize_ensemble_coord(da, debug=debug, tag=f"part{i}") for i, da in enumerate(das)]

    # Grid compatibility checks
    lat0 = loaders[0].coords["latitude"].values
    lon0 = loaders[0].coords["longitude"].values
    for i, da in enumerate(loaders[1:], start=2):
        if (not np.array_equal(da.coords["latitude"].values, lat0)) or (not np.array_equal(da.coords["longitude"].values, lon0)):
            raise ValueError(f"格子が一致しません: das[{i-1}]")

    # Align ensemble members across parts (intersection)
    ens_sets = [set(np.asarray(da.coords["ensemble"].values).tolist()) for da in loaders if "ensemble" in da.dims]
    if ens_sets:
        ens_common = sorted(set.intersection(*ens_sets))
        if len(ens_common) == 0:
            raise ValueError("アンサンブル座標の共通部分が空です（partsでmember集合が一致しません）")
        if any(len(s) != len(ens_common) for s in ens_sets):
            _dbg(True, f"[WARN] parts have different ensemble member sets; using intersection size={len(ens_common)}")
        loaders = [da.sel(ensemble=ens_common) if "ensemble" in da.dims else da for da in loaders]

    # Concat by time and sort
    da_all = xr.concat(loaders, dim="time").sortby("time")

    # Deduplicate time (first occurrence wins)
    tvals = da_all["time"].values
    _, uniq_idx = np.unique(tvals, return_index=True)
    if len(uniq_idx) != da_all.sizes["time"]:
        da_all = da_all.isel(time=np.sort(uniq_idx))

    # time → int hours since 1900-01-01
    origin = np.datetime64("1900-01-01T00:00:00Z")
    hours = (da_all["time"].values.astype("datetime64[ns]") - origin).astype("timedelta64[h]").astype("int64")
    da_all = da_all.assign_coords(time=("time", hours))
    da_all["time"].attrs.update({"units": "hours since 1900-01-01 00:00:00", "calendar": "proleptic_gregorian"})

    varname = element.upper() if level_hpa is None else f"{element.upper()}{int(level_hpa)}"
    ds = xr.Dataset({varname: da_all})

    # Add ensemble summary variables
    ds = add_ens_summary_vars(ds, da_all, varname, debug=debug)

    # Packing / encoding
    units = str(da_all.attrs.get("units", ""))
    pk = packing_for(element.upper(), units)
    np_dtype = np.int16 if pk["dtype"] == "int16" else np.int32
    fill = np_dtype(pk["fill"])

    encoding = {
        "time": {"dtype": "int32", "_FillValue": None},
        "ensemble": {"dtype": "int16", "_FillValue": None},
        "latitude": {"dtype": "float64", "_FillValue": None},
        "longitude": {"dtype": "float64", "_FillValue": None},
    }
    for vn in ds.data_vars:
        encoding[vn] = {
            "dtype": np_dtype,
            "scale_factor": float(pk["scale"]),
            "add_offset": float(pk["add"]),
            "_FillValue": fill,
            "zlib": True,
            "complevel": 4,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _dbg(debug, f"[SAVE] {out_path} (units='{units}', pack={pk})")
    ds.to_netcdf(out_path, encoding=encoding)

def main(argv=None):
    import argparse, sys, re
    from pathlib import Path
    from datetime import datetime, timedelta

    ap = argparse.ArgumentParser(description="GEPS Lsurf/L-pall/1m → NetCDF（pygrib, 51/25ens, 1日前/週次/月次）")
    ap.add_argument("--date", required=True, help="基準日 yyyymmdd（1w2w: この1日前12UTC, 1m: 木曜に火/水12UTCを処理）")
    ap.add_argument("--dir", default=DEFAULT_DIR, help=f"ホームDIR（既定: {DEFAULT_DIR}）")
    ap.add_argument("--var", required=True, help="要素（Lsurf: TMP/RH/APCP/TCDC/PRMSL/UGRD/VGRD, L-pall: +HGT/VVEL）または短縮（例 TMP850）")
    ap.add_argument("--hgt", type=int, default=None, help="等圧面 (hPa)。指定時は L-pall を処理（例: 850）")
    ap.add_argument("--debug", action="store_true", help="ログ多め")
    ap.add_argument("--dry-run", action="store_true", help="保存せず読み込み・検証のみ")
    ap.add_argument("--skip-monthly", action="store_true", help="1ヶ月（1m）処理をスキップ")
    ap.add_argument("--only-monthly", action="store_true", help="1ヶ月（1m）のみ処理（1w2wはスキップ）")
    args = ap.parse_args(argv)

    # 例: --var TMP850 / HGT500 → var=TMP/HGT, hgt=850/500 として扱う（--hgt 明示が優先）
    m = re.fullmatch(r"([A-Za-z]+?)(\d{3})$", args.var)
    if m and args.hgt is None:
        args.var = m.group(1)
        args.hgt = int(m.group(2))

    # バリデーション
    if len(args.date) != 8 or not args.date.isdigit():
        print("[ERROR] --date は yyyymmdd", file=sys.stderr)
        return 2

    # 要素正規化
    try:
        element = _normalize_var(args.var)  # → 'TMP','RH',...,'HGT','VVEL'
    except ValueError as e:
        print("[ERROR]", e, file=sys.stderr)
        return 2

    # L-pall／Lsurf の選択
    is_upper = args.hgt is not None
    if is_upper:
        lvl = int(args.hgt)
        if lvl not in UPPER_LEVELS:
            print(f"[ERROR] --hgt {lvl} は未サポート。選択: {sorted(UPPER_LEVELS.keys())}", file=sys.stderr)
            return 2
        allowed = set(UPPER_LEVELS[lvl])
        if element not in allowed:
            print(f"[ERROR] {element}@{lvl} は未サポート。level {lvl} で選べる要素: {sorted(allowed)}", file=sys.stderr)
            return 2

    base_dir = Path(args.dir)
    proc_date = datetime.strptime(args.date, "%Y%m%d")
    is_thursday = (proc_date.weekday() == 3)  # 月=0 ... 木=3

    # ------------------ 1w + 2w 処理（必要なら） ------------------
    if not args.only_monthly:
        try:
            paths = _find_target_files_for_12utc_prevday(base_dir, args.date, is_upper=is_upper, debug=args.debug)
        except FileNotFoundError as e:
            print(f"[WARN] 1w2w: {e}")
            paths = []

        if paths:
            print("[FILES]"); [print("  ", p) for p in paths]

            # 読み込み検証
            try:
                if is_upper:
                    for p in paths:
                        _ = _load_lpall_element_ens_signed_1dcoords(p, element, lvl, debug=args.debug)
                else:
                    for p in paths:
                        _ = _load_lsurf_element_ens_signed_1dcoords(p, element, debug=args.debug)
                print(f"[INFO] 読み込み検証 OK ({element}{'' if not is_upper else f'@{lvl}'})")
            except Exception as e:
                print("[ERROR] 1w2w 読み込み中に失敗:", e, file=sys.stderr)
                return 1

            if not args.dry_run:
                yyyy = args.date[:4]
                elem_key = element if not is_upper else f"{element}{lvl}"
                out_dir = base_dir / "data" / "GEPS_NC" / yyyy / elem_key
                out_nc  = out_dir / f"GEPS_1w2w_{args.date}_{elem_key}.nc"
                try:
                    concat_time_and_save(paths, element, out_nc, debug=args.debug, level_hpa=lvl if is_upper else None)
                    print(f"[DONE] {out_nc}")
                except Exception as e:
                    print("[ERROR] 1w2w 保存時に失敗:", e, file=sys.stderr)
                    return 1
        else:
            print("[WARN] 1w2w: 対象ファイルが見つからないためスキップします。")

    if args.dry_run:
        print("[DRY-RUN] 保存せず終了")
        return 0

    # ------------------ 1ヶ月（1mEGPV）処理 ------------------
    do_monthly = (args.only_monthly or (is_thursday and not args.skip_monthly))
    if do_monthly:
        # 火曜/水曜 12UTC 初期のデータを処理
        target_tue = (proc_date - timedelta(days=2)).strftime("%Y%m%d")
        target_wed = (proc_date - timedelta(days=1)).strftime("%Y%m%d")

        plan = [
            # (元ファイル日付, 出力ファイル日付)
            (target_tue, (proc_date - timedelta(days=1)).strftime("%Y%m%d")),  # 火曜初期 → 水曜付
            (target_wed, args.date),                                           # 水曜初期 → 木曜付
        ]

        for tgt, out_date in plan:
            # 日付ごとの 1m ファイル一覧を収集
            try:
                paths_1m = _find_monthly_paths_for_date(base_dir, tgt, is_upper=is_upper, debug=args.debug)
            except FileNotFoundError as e:
                print(f"[WARN] 1m {tgt}: {e}")
                continue

            # 各パートを試し読み（当該要素がないパートはスキップ）
            das = []
            if is_upper:
                for p in paths_1m:
                    try:
                        da = _load_lpall_element_ens_signed_1dcoords(p, element, lvl, debug=args.debug)
                        das.append(da)
                    except RuntimeError as e:
                        if "対象メッセージが見つかりません" in str(e):
                            print(f"[WARN] 1m skip (no {element}@{lvl}) in {p.name}")
                        else:
                            raise
            else:
                for p in paths_1m:
                    try:
                        da = _load_lsurf_element_ens_signed_1dcoords(p, element, debug=args.debug)
                        das.append(da)
                    except RuntimeError as e:
                        if "対象メッセージが見つかりません" in str(e):
                            print(f"[WARN] 1m skip (no {element}) in {p.name}")
                        else:
                            raise

            if not das:
                print(f"[WARN] 1m {tgt}: {element}{'' if not is_upper else f'@{lvl}'} が1件も見つかりません。スキップします。")
                continue

            print(f"[INFO] 1m 読み込み検証 OK ({element}{'' if not is_upper else f'@'+str(lvl)}) for {tgt}  (used {len(das)}/{len(paths_1m)} parts)")

            # 保存（GEPS_1m_[yyyymmdd]_[要素].nc）
            yyyy_out = out_date[:4]
            elem_key = element if not is_upper else f"{element}{lvl}"
            out_dir = base_dir / "data" / "GEPS_NC" / yyyy_out / elem_key
            out_nc  = out_dir / f"GEPS_1m_{out_date}_{elem_key}.nc"

            try:
                concat_time_and_save_from_das(das, element, out_nc, debug=args.debug)
                print(f"[DONE] 1m {tgt} → {out_nc}")
            except Exception as e:
                print(f"[ERROR] 1m 保存時に失敗 ({tgt}):", e, file=sys.stderr)
                return 1
    else:
        _dbg(args.debug, "[INFO] 1m はスキップ（木曜でない、または --skip-monthly 指定）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
