#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reset ENS1M status to 0 at midnight.

This script should be run via cron at midnight (00:00).
Example cron entry:
  0 0 * * * /path/to/scripts/reset_ens1m_status.py

Or as part of a longer command:
  0 0 * * * cd /home/nemo/JMA-GPV && python scripts/reset_ens1m_status.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from status_manager import update_status


def main() -> int:
    """Reset status to 0 for a new day."""
    try:
        update_status(0, "Daily reset at midnight")
        print("[SUCCESS] Status reset to 0 for new day", flush=True)
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to reset status: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
