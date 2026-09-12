# GestureVoice v9.3 — YOLO11 Pose Upgrade

This build keeps the v9.2 8-example reliability workflow and false-trigger gate, but upgrades the vision stack.

## Primary vision stack

- **YOLO11s-pose**: primary person detection + body keypoints
- **MediaPipe Hands**: detailed left/right hand landmarks
- **MediaPipe Pose**: automatic fallback only if YOLO11 cannot load/detect

YOLOX is no longer used in the normal inference path.

## Why YOLO11s-pose

`yolo11s-pose.pt` is the default because GestureVoice needs good pose stability *and* real-time laptop responsiveness. A heavier `yolo11m-pose.pt` launcher is included if your machine has enough headroom.

## Run (recommended)

```bash
cd ~/Downloads/GestureVoice_Adaptive_v9_3_YOLO11
chmod +x run_mac.sh
./run_mac.sh
```

Then open `http://127.0.0.1:7860`.

The first run may download the YOLO11s-pose weights automatically.

## Heavier YOLO11m Pose

```bash
./run_yolo11m_mac.sh
```

Use this only if it remains smooth on your Mac. For a hackathon demo, stable frame rate matters more than using a larger model.

## Training

Keep the v9.2 reliability protocol:

- REST/NONE: 8 recordings
- each gesture: 8 recordings

If you trained an older model on a different pose backend, reset and retrain so the feature distribution matches YOLO11 Pose.
