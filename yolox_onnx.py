from __future__ import annotations
import os
import cv2
import numpy as np

COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]


def _nms(boxes, scores, iou_thr=0.45):
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2-x1+1) * (y2-y1+1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return keep


class YOLOXPersonDetector:
    """Small ONNX YOLOX person detector. If no model is present, caller can bypass it."""
    def __init__(self, model_path: str, input_size=(640, 640), score_thr=0.35, nms_thr=0.45):
        import onnxruntime as ort
        self.model_path = model_path
        self.input_size = input_size
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self._grids, self._expanded_strides = self._make_grids(input_size)

    @staticmethod
    def _make_grids(input_size):
        hsizes = [input_size[0] // s for s in (8,16,32)]
        wsizes = [input_size[1] // s for s in (8,16,32)]
        grids, strides = [], []
        for h, w, stride in zip(hsizes, wsizes, (8,16,32)):
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            grid = np.stack((xv,yv),2).reshape(1,-1,2)
            grids.append(grid)
            strides.append(np.full((*grid.shape[:2],1), stride))
        return np.concatenate(grids,1), np.concatenate(strides,1)

    def _preprocess(self, img):
        ih, iw = self.input_size
        h, w = img.shape[:2]
        r = min(ih/h, iw/w)
        resized = cv2.resize(img, (int(w*r), int(h*r)), interpolation=cv2.INTER_LINEAR)
        padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        rgb = padded[:, :, ::-1].astype(np.float32)
        blob = np.transpose(rgb, (2,0,1))[None, ...]
        return blob, r

    def detect_person(self, img):
        blob, ratio = self._preprocess(img)
        out = self.session.run(None, {self.input_name: blob})[0]
        if out.ndim == 2:
            out = out[None, ...]
        pred = out.copy()
        pred[..., :2] = (pred[..., :2] + self._grids) * self._expanded_strides
        pred[..., 2:4] = np.exp(pred[..., 2:4]) * self._expanded_strides
        boxes_xywh = pred[0, :, :4]
        obj = pred[0, :, 4:5]
        cls = pred[0, :, 5:]
        scores = obj * cls
        person_scores = scores[:, 0]
        mask = person_scores >= self.score_thr
        if not np.any(mask):
            return None, 0.0
        b = boxes_xywh[mask]
        s = person_scores[mask]
        boxes = np.empty_like(b)
        boxes[:,0] = b[:,0]-b[:,2]/2
        boxes[:,1] = b[:,1]-b[:,3]/2
        boxes[:,2] = b[:,0]+b[:,2]/2
        boxes[:,3] = b[:,1]+b[:,3]/2
        boxes /= ratio
        keep = _nms(boxes, s, self.nms_thr)
        if not keep:
            return None, 0.0
        best = keep[int(np.argmax(s[keep]))]
        box = boxes[best]
        h, w = img.shape[:2]
        x1,y1,x2,y2 = box
        # add margin so hands near body edges stay in frame
        bw, bh = x2-x1, y2-y1
        x1 -= 0.12*bw; x2 += 0.12*bw
        y1 -= 0.08*bh; y2 += 0.08*bh
        x1,y1 = max(0,int(x1)), max(0,int(y1))
        x2,y2 = min(w,int(x2)), min(h,int(y2))
        return (x1,y1,x2,y2), float(s[best])
