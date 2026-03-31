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

# Root containing ENS1M_NC output.
DEFAULT_OUT_ROOT = _path_from_env("ENS1M_OUT_ROOT", "GEPS_OUT_ROOT", PROJECT_ROOT / "data")
