#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export GESTUREVOICE_YOLO11_POSE=1
export GESTUREVOICE_YOLO_MODEL=yolo11m-pose.pt
python app.py
