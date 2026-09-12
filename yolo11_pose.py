from __future__ import annotations
import os


class YOLO11PoseBackend:
    """YOLO11 Pose backend used as GestureVoice's primary body model.

    Defaults to YOLO11s-pose because it is a strong accuracy/speed balance for
    real-time laptop inference. Set GESTUREVOICE_YOLO_MODEL=yolo11m-pose.pt if
    you want a heavier model, or GESTUREVOICE_YOLO11_POSE=0 to force fallback.
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.name = 'MediaPipe fallback'
        self.model_name = os.environ.get('GESTUREVOICE_YOLO_MODEL', 'yolo11s-pose.pt')
        if os.environ.get('GESTUREVOICE_YOLO11_POSE', '1') == '0':
            return
        try:
            from ultralytics import YOLO
            source = model_path if model_path and os.path.exists(model_path) else self.model_name
            self.model = YOLO(source)
            pretty = os.path.basename(source).replace('.pt', '')
            self.name = f'{pretty} Pose'
        except Exception as e:
            print(f'[GestureVoice] YOLO11 Pose unavailable; using MediaPipe fallback: {e}')

    @property
    def enabled(self):
        return self.model is not None

    def infer(self, frame, conf: float = 0.25):
        """Return the largest detected person's selected COCO keypoints.

        Output keypoints: nose, L/R shoulder, L/R elbow, L/R wrist, L/R hip.
        Coordinates are normalized to the full frame.
        """
        if not self.enabled:
            return None, None, 0.0
        try:
            result = self.model.predict(frame, verbose=False, conf=conf, imgsz=640)[0]
            if result.keypoints is None or len(result.keypoints.xy) == 0:
                return None, result, 0.0

            xy = result.keypoints.xy.cpu().numpy()
            kconf = (
                result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else None
            )
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
            box_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else None

            if boxes is not None and len(boxes):
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                j = int(areas.argmax())
            else:
                areas = []
                for pts in xy:
                    areas.append(float((pts[:, 0].max()-pts[:, 0].min()) * (pts[:, 1].max()-pts[:, 1].min())))
                j = int(max(range(len(areas)), key=lambda i: areas[i]))

            pts = xy[j]
            cf = kconf[j] if kconf is not None else [1.0] * len(pts)
            h, w = frame.shape[:2]
            ids = [0, 5, 6, 7, 8, 9, 10, 11, 12]
            selected = []
            for i in ids:
                selected.append((
                    float(pts[i, 0] / max(w, 1)),
                    float(pts[i, 1] / max(h, 1)),
                    0.0,
                    float(cf[i]),
                ))
            score = float(box_conf[j]) if box_conf is not None and len(box_conf) > j else float(sum(p[3] for p in selected) / len(selected))
            return selected, result, score
        except Exception as e:
            print(f'[GestureVoice] YOLO11 Pose inference fallback: {e}')
            return None, None, 0.0
