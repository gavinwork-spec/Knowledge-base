#!/bin/bash
# XAgent System Launcher for macOS

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start the XAgent system
echo "🚀 Launching XAgent Manufacturing Intelligence System..."
exec "$SCRIPT_DIR/start_system.sh"