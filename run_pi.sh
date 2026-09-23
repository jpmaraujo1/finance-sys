#!/bin/bash
# ========================================================
# Finance AI — Raspberry Pi 4 Autonomous Launcher
# Runs 24/7 Market Scanning, Sentiment & Web Dashboard
# (Zero GPU / Zero LLM Required)
# ========================================================

cd "$(dirname "$0")"

echo "========================================================"
echo "  Starting Finance AI on Raspberry Pi..."
echo "========================================================"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install lightweight dependencies
echo "Verifying lightweight Python dependencies..."
pip install --upgrade pip
pip install -r requirements_pi.txt

# Run the system (Signal Engine + Web Dashboard on port 8080)
echo "Launching 24/7 Market Engine..."
python3 run_local.py
