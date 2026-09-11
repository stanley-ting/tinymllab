#!/usr/bin/env bash
set -euo pipefail

# Raspberry Pi OS packages required before creating the virtual environment.
sudo apt update
sudo apt install -y python3-pip python3-venv python3-full

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_setup.py
