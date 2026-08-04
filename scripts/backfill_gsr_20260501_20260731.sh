#!/usr/bin/env bash
# Create only daily GSR for ENS1M initialization dates 2026-05-01..2026-07-31.
# This does not run conversion, hourly/daily processing, wind, or plotting.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
geps_python="${GEPS_PYTHON:-/home/nemo/miniforge3/envs/geps/bin/python}"
ens1m_base="${ENS1M_BASE:-/mnt/e/data}"

if [[ ! -x "$geps_python" ]]; then
    echo "[ERROR] Python is not executable: $geps_python" >&2
    exit 2
fi

if [[ ! -d "$ens1m_base/ENS1M/2026/TCDC" ]]; then
    echo "[ERROR] Daily TCDC directory not found: $ens1m_base/ENS1M/2026/TCDC" >&2
    exit 2
fi

echo "[GSR BACKFILL] start=20260501 end=20260731"
echo "[GSR BACKFILL] input=$ens1m_base/ENS1M/2026/TCDC"
echo "[GSR BACKFILL] output=$ens1m_base/ENS1M/2026/GSR"

"$geps_python" \
    "$project_root/scripts/06_make_ens1m_daily_gsr.py" \
    --start 20260501 \
    --end 20260731 \
    --base "$ens1m_base" \
    --skip-missing

echo "[GSR BACKFILL DONE] start=20260501 end=20260731"
