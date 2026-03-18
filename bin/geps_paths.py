#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared path settings for the GEPS pipeline.

Edit this file when moving the pipeline to another environment, or override
the defaults with environment variables:

  GEPS_JMADATA_ROOT
  GEPS_OUT_ROOT
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default.expanduser().resolve()


# Root containing 1wEGPV/2wEGPV/1mEGPV and/or jmadata/{1wEGPV,2wEGPV,1mEGPV}.
DEFAULT_JMADATA_ROOT = _path_from_env("GEPS_JMADATA_ROOT", PROJECT_ROOT / "jmadata")

# Root containing GEPS_NC output.
DEFAULT_OUT_ROOT = _path_from_env("GEPS_OUT_ROOT", PROJECT_ROOT / "data")
