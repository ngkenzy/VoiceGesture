#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_neural.txt
if [ ! -f models/yolox_s.onnx ]; then
  python download_yolox.py || echo "YOLOX download failed; app will still run with full-frame detection."
fi
export GESTUREVOICE_YOLO11_POSE=1
python app.py
