#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Status file manager for ENS1M batch processing.

Status values:
  0: Before processing (set at midnight via cron)
  1: Batch started but required data not available yet
  2: Required data available, conversion in progress
  3: Completed through plot stage
  4: All processing completed
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

StatusCode = Literal[0, 1, 2, 3, 4]
STATUS_FILENAME = "ens1m_status.json"
LEGACY_STATUS_FILENAME = "ens1m_status"


def _get_status_file_path() -> Path:
    """Determine status file path based on environment.
    
    Development: JMA-GPV/status/ens1m_status.json
    Production: /var/www/html/status/ens1m_status.json
    """
    # Check if we're in production environment
    prod_status_path = Path("/var/www/html/status") / STATUS_FILENAME
    if prod_status_path.exists() or os.access(prod_status_path.parent, os.W_OK):
        return prod_status_path
    
    # Development environment: use symlink under JMA-GPV
    project_root = Path(__file__).resolve().parents[1]
    status_link = project_root / "status" / STATUS_FILENAME
    
    # Ensure directory exists
    status_link.parent.mkdir(parents=True, exist_ok=True)
    
    return status_link


def update_status(status: StatusCode, message: str = "") -> None:
    """Update the status file with current status and timestamp.
    
    Args:
        status: Status code (0-4)
        message: Optional message to include
    """
    status_path = _get_status_file_path()
    
    # Ensure directory exists
    status_path.parent.mkdir(parents=True, exist_ok=True)
    
    status_data = {
        "status": status,
        "datetime": datetime.now().isoformat(),
    }
    if message:
        status_data["message"] = message
    
    try:
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2, ensure_ascii=False)
        print(f"[STATUS] Updated: status={status} ({status_path})", flush=True)
    except Exception as e:
        print(f"[WARNING] Failed to update status file: {e}", flush=True)


def read_status() -> dict | None:
    """Read current status from file.
    
    Returns:
        Dict with status and datetime, or None if file doesn't exist
    """
    status_path = _get_status_file_path()
    
    legacy_path = status_path.with_name(LEGACY_STATUS_FILENAME)
    target_path = status_path if status_path.exists() else legacy_path
    if not target_path.exists():
        return None
    
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to read status file: {e}", flush=True)
        return None


def is_today_already_started() -> bool:
    """Check if processing for today has already been started.
    
    Returns:
        True if status file indicates processing has started today
    """
    status_data = read_status()
    if status_data is None:
        return False
    
    current_status = status_data.get("status", 0)
    status_datetime_str = status_data.get("datetime", "")
    
    if current_status == 0:
        return False
    
    try:
        status_dt = datetime.fromisoformat(status_datetime_str)
        today = datetime.now().date()
        return status_dt.date() == today and current_status >= 1
    except Exception:
        return False


if __name__ == "__main__":
    # Test
    update_status(0, "Manual reset for testing")
    print(read_status())
    update_status(1, "Testing batch start")
    print(read_status())
