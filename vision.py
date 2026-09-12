from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp

from yolo11_pose import YOLO11PoseBackend

POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]


class VisionPipeline:
    """Primary: YOLO11s-pose for person/body keypoints.

    MediaPipe Hands adds detailed hand landmarks. MediaPipe Pose is retained as
    a reliability fallback when YOLO11 cannot load or does not detect a pose.
    """

    def __init__(self, **_legacy_kwargs):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.yolo11_pose = YOLO11PoseBackend()
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_backend_status = self.yolo11_pose.name if self.yolo11_pose.enabled else 'MediaPipe Pose fallback'
        self.last_pose_score = 0.0

    @staticmethod
    def _body_normalize(points, left_shoulder, right_shoulder, left_hip, right_hip):
        ls = np.array(left_shoulder[:2], dtype=np.float32)
        rs = np.array(right_shoulder[:2], dtype=np.float32)
        lh = np.array(left_hip[:2], dtype=np.float32)
        rh = np.array(right_hip[:2], dtype=np.float32)
        center = (ls + rs) / 2.0
        shoulder = np.linalg.norm(ls-rs)
        torso = np.linalg.norm(center - (lh+rh)/2.0)
        scale = max(float(shoulder), float(torso)*0.65, 0.08)
        out = []
        for x, y, z, v in points:
            out.extend([(x-center[0])/scale, (y-center[1])/scale, z/scale, v])
        return np.array(out, dtype=np.float32), center, scale

    def extract(self, frame):
        # YOLO11 sees the full frame so it performs both person localization and pose.
        yolo_pts, yolo_res, yolo_score = self.yolo11_pose.infer(frame)
        self.last_pose_score = yolo_score

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hres = self.hands.process(rgb)
        # Run MediaPipe Pose in parallel as a robustness signal for switch access.
        # YOLO11 remains the primary body backend; MediaPipe gives us a second
        # opinion when wrist keypoints are low-confidence or partially occluded.
        pres = self.pose.process(rgb)
        overlay = frame.copy()

        if yolo_pts is not None:
            pose_pts = yolo_pts
            body, center, scale = self._body_normalize(
                pose_pts, pose_pts[1], pose_pts[2], pose_pts[7], pose_pts[8]
            )
            pose_backend = self.yolo11_pose.name
        else:
            # Safe fallback: GestureVoice still works if ultralytics/model download fails.
            if not pres.pose_landmarks:
                return None, frame, {
                    'person': False,
                    'pose': False,
                    'hands': 0,
                    'pose_score': self.last_pose_score,
                    'pose_backend': 'none',
                }
            lm = pres.pose_landmarks.landmark
            pose_pts = [(lm[i].x, lm[i].y, lm[i].z, lm[i].visibility) for i in POSE_IDX]
            body, center, scale = self._body_normalize(
                pose_pts,
                (lm[11].x, lm[11].y), (lm[12].x, lm[12].y),
                (lm[23].x, lm[23].y), (lm[24].x, lm[24].y),
            )
            pose_backend = 'MediaPipe Pose fallback'

        # Detailed hand features remain body-relative so body and hands share a
        # stable coordinate system even if the user moves closer to the camera.
        hands_data = []
        if hres.multi_hand_landmarks:
            for hand in hres.multi_hand_landmarks:
                feats = []
                wrist_x = hand.landmark[0].x
                for p in hand.landmark:
                    feats.extend([
                        (p.x-center[0])/scale,
                        (p.y-center[1])/scale,
                        p.z/scale,
                    ])
                hands_data.append((wrist_x, feats, hand))
            hands_data.sort(key=lambda t: t[0])

        hand_vecs = []
        for idx in range(2):
            if idx < len(hands_data):
                hand_vecs.extend(hands_data[idx][1])
                hand_vecs.append(1.0)
            else:
                hand_vecs.extend([0.0] * (21*3))
                hand_vecs.append(0.0)

        feature = np.concatenate([body, np.asarray(hand_vecs, dtype=np.float32)])

        if yolo_res is not None and yolo_pts is not None:
            try:
                plotted = yolo_res.plot(labels=False, boxes=True)
                if plotted.shape == overlay.shape:
                    overlay = plotted
            except Exception:
                pass
        elif pres is not None and pres.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                overlay, pres.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

        if hres.multi_hand_landmarks:
            for hand in hres.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    overlay, hand, self.mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(
            overlay,
            f'{pose_backend} {self.last_pose_score:.2f}' if yolo_pts is not None else pose_backend,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (34, 197, 94),
            2,
            cv2.LINE_AA,
        )
        # Robust accessibility switch signal. We intentionally fuse several
        # independent cues instead of trusting a single YOLO wrist keypoint:
        #   1) YOLO11 wrist above shoulder
        #   2) MediaPipe Pose wrist above shoulder
        #   3) MediaPipe Hands landmarks visibly above the shoulder line
        # This makes the demo resilient to partial wrist occlusion and low pose confidence.
        margin = 0.012  # permissive enough for a comfortable hand raise
        yolo_left = yolo_right = False
        if yolo_pts is not None:
            yl_sy, yr_sy = float(yolo_pts[1][1]), float(yolo_pts[2][1])
            yl_wy, yr_wy = float(yolo_pts[5][1]), float(yolo_pts[6][1])
            yolo_left = bool(float(yolo_pts[1][3]) >= 0.20 and float(yolo_pts[5][3]) >= 0.20 and yl_wy < yl_sy - margin)
            yolo_right = bool(float(yolo_pts[2][3]) >= 0.20 and float(yolo_pts[6][3]) >= 0.20 and yr_wy < yr_sy - margin)

        mp_left = mp_right = False
        shoulder_line = min(float(pose_pts[1][1]), float(pose_pts[2][1]))
        if pres is not None and pres.pose_landmarks:
            mlm = pres.pose_landmarks.landmark
            mp_left = bool(mlm[11].visibility >= 0.20 and mlm[15].visibility >= 0.20 and mlm[15].y < mlm[11].y - margin)
            mp_right = bool(mlm[12].visibility >= 0.20 and mlm[16].visibility >= 0.20 and mlm[16].y < mlm[12].y - margin)
            shoulder_line = min(shoulder_line, float(mlm[11].y), float(mlm[12].y))

        hands_landmark_up = False
        hand_min_y = 1.0
        if hres.multi_hand_landmarks:
            for hand in hres.multi_hand_landmarks:
                ys = [float(pt.y) for pt in hand.landmark]
                if ys:
                    hand_min_y = min(hand_min_y, min(ys))
                    # Use the palm/wrist region plus fingertips: a hand can be raised
                    # even when the wrist itself is still near shoulder level.
                    above = sum(1 for y in ys if y < shoulder_line - 0.004)
                    if above >= 5:
                        hands_landmark_up = True

        left_hand_up = bool(yolo_left or mp_left)
        right_hand_up = bool(yolo_right or mp_right)
        hands_up = bool(left_hand_up or right_hand_up or hands_landmark_up)
        sources = []
        if yolo_left or yolo_right: sources.append('YOLO11 pose')
        if mp_left or mp_right: sources.append('MediaPipe pose')
        if hands_landmark_up: sources.append('MediaPipe hands')
        switch_source = ' + '.join(sources) if sources else 'none'

        if hands_up:
            cv2.putText(overlay, 'HAND DETECTED - SELECT READY', (16, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (34, 197, 94), 2, cv2.LINE_AA)
        else:
            cv2.putText(overlay, 'Raise either hand to SELECT', (16, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 158, 11), 2, cv2.LINE_AA)

        return feature, overlay, {
            'person': True,
            'pose': True,
            'hands': len(hands_data),
            'pose_score': self.last_pose_score,
            'pose_backend': pose_backend,
            'left_hand_up': left_hand_up,
            'right_hand_up': right_hand_up,
            'hands_landmark_up': hands_landmark_up,
            'hands_up': hands_up,
            'switch_source': switch_source,
            'switch_shoulder_line': round(float(shoulder_line), 4),
            'switch_hand_min_y': None if hand_min_y >= 1.0 else round(float(hand_min_y), 4),
        }
