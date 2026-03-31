#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared path settings for the ENS1M pipeline.

Edit this file when moving the pipeline to another environment, or override
the defaults with environment variables:

  ENS1M_JMADATA_ROOT
  ENS1M_OUT_ROOT

Legacy `GEPS_*` environment variables are also accepted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, legacy_name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    legacy_value = os.environ.get(legacy_name)
    if legacy_value:
        return Path(legacy_value).expanduser().resolve()
    return default.expanduser().resolve()


# Root containing ens_1w/ens_2w1m and/or jmadata/{ens_1w,ens_2w1m}.
DEFAULT_JMADATA_ROOT = _path_from_env("ENS1M_JMADATA_ROOT", "GEPS_JMADATA_ROOT", PROJECT_ROOT / "jmadata")

# Root containing ENS1M / legacy ENS1M_NC output.
DEFAULT_OUT_ROOT = _path_from_env("ENS1M_OUT_ROOT", "GEPS_OUT_ROOT", PROJECT_ROOT / "data")

# Current and legacy output subdirectories under DEFAULT_OUT_ROOT.
OUTPUT_SUBDIR = "ENS1M"
LEGACY_OUTPUT_SUBDIRS = ("ENS1M_NC", "GEPS_NC")


def output_subdirs_for_read() -> tuple[str, ...]:
    return (OUTPUT_SUBDIR, *LEGACY_OUTPUT_SUBDIRS)


def output_dir(out_root: Path, year: str, var: str, subdir: str | None = None) -> Path:
    chosen = subdir or OUTPUT_SUBDIR
    return out_root / chosen / year / var


def output_dirs_for_read(out_root: Path, year: str, var: str) -> tuple[Path, ...]:
    return tuple(output_dir(out_root, year, var, subdir=s) for s in output_subdirs_for_read())


def ens1m_file_candidates(out_root: Path, year: str, var: str, stem: str) -> list[Path]:
    candidates: list[Path] = []
    for subdir in output_subdirs_for_read():
        candidates.append(output_dir(out_root, year, var, subdir=subdir) / f"{stem}.nc")
    return candidates


def geps_file_candidates(out_root: Path, year: str, var: str, stem: str) -> list[Path]:
    return [output_dir(out_root, year, var, subdir="GEPS_NC") / f"{stem}.nc"]


def file_candidates(out_root: Path, year: str, var: str, ens1m_stem: str, geps_stem: str | None = None) -> list[Path]:
    candidates = ens1m_file_candidates(out_root, year, var, ens1m_stem)
    if geps_stem:
        candidates.extend(geps_file_candidates(out_root, year, var, geps_stem))
    return candidates


def first_existing(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None
