#!/usr/bin/env bash
set -e
source .venv/bin/activate 2>/dev/null || true
python -m pip install 'torch>=2.2' 'ultralytics>=8.3,<9'
echo 'Neural extras installed. To enable YOLO11 Pose, run:'
echo 'GESTUREVOICE_YOLO11_POSE=1 python app.py'
