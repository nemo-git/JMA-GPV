#!/bin/bash
# Setup symbolic link for status directory

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
STATUS_LINK="$PROJECT_ROOT/status"
TARGET_STATUS_FILE="/var/www/html/status/ens1m_status"

echo "Setting up status directory symlink for ENS1M batch processing..."
echo "  Project root: $PROJECT_ROOT"
echo "  Status link: $STATUS_LINK"
echo "  Target: $TARGET_STATUS_FILE"

# Check if we're in production environment (has write access to /var/www/html)
if [ -d "/var/www/html/status" ] && [ -w "/var/www/html/status" ]; then
    echo ""
    echo "[INFO] Production environment detected (/var/www/html exists and is writable)"
    echo "       Status file will be created at: $TARGET_STATUS_FILE"
    
    # Ensure production status directory exists
    sudo mkdir -p /var/www/html/status 2>/dev/null
    
    # Create status directory symlink if it doesn't exist
    if [ ! -e "$STATUS_LINK" ]; then
        echo "       Creating symlink: $STATUS_LINK -> $TARGET_STATUS_FILE"
        mkdir -p "$(dirname "$STATUS_LINK")"
        ln -s "$TARGET_STATUS_FILE" "$STATUS_LINK" || echo "[WARNING] Failed to create symlink (may require sudo)"
    else
        echo "       Symlink already exists: $STATUS_LINK"
    fi
else
    echo ""
    echo "[INFO] Development environment detected"
    echo "       Creating local status directory: $STATUS_LINK"
    
    # Create local status directory
    mkdir -p "$STATUS_LINK"
    
    # Initialize status file
    python3 -c "
import json
from datetime import datetime
from pathlib import Path

status_file = Path('$STATUS_LINK') / 'ens1m_status'
status_data = {
    'status': 0,
    'datetime': datetime.now().isoformat(),
    'message': 'Development environment - initial status'
}
status_file.parent.mkdir(parents=True, exist_ok=True)
with open(status_file, 'w') as f:
    json.dump(status_data, f, indent=2)
print(f'[OK] Created status file: {status_file}')
"
fi

echo ""
echo "[OK] Setup complete!"
echo ""
echo "Next steps:"
echo "1. For production, add this cron entry to reset status daily:"
echo "   0 0 * * * cd $PROJECT_ROOT && python3 scripts/reset_ens1m_status.py >> /var/log/ens1m_status_reset.log 2>&1"
echo ""
echo "2. Test the status system with:"
echo "   python3 -c \"import sys; sys.path.insert(0, '$PROJECT_ROOT/src'); from status_manager import read_status, update_status; update_status(1, 'test'); print(read_status())\""
