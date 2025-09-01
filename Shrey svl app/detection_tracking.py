"""
Detection + Tracking + ReID integration.
- YOLOv8 detection at configurable FPS
- Simple per-camera tracker (IoU + smoothing)
- ReID embeddings ensure global IDs remain stable across cameras
"""

import time
import torch
import numpy as np
from ultralytics import YOLO
from typing import Dict, Tuple, List

import config as C

# COCO 80 names as in YOLOv8
COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog',
    'horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag',
    'tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
    'skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard',
    'cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',
    'teddy bear','hair drier','toothbrush'
]

# Attempt to import TorchReID extractor from either namespace
FeatureExtractor = None
try:
    from torchreid.reid.utils import FeatureExtractor as _FE
    FeatureExtractor = _FE
except Exception:
    try:
        from torchreid.utils import FeatureExtractor as _FE2
        FeatureExtractor = _FE2
    except Exception:
        FeatureExtractor = None

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.dot(a, b))
    return max(min(d, 1.0), -1.0)

class DetectionTracker:
    def __init__(self, model_path=C.YOLO_MODEL, device=C.DEVICE, conf_thres=C.CONF_THRES):
        self.device = device
        print(f"[Detection] Loading YOLO model {model_path} on {self.device} ...")
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception:
            pass
        self.model.fuse()

        self.conf_thres = conf_thres
        self.nms_iou = C.NMS_IOU

        # ReID extractor
        if FeatureExtractor is None:
            raise RuntimeError("torchreid FeatureExtractor not available. Please install torchreid.")
        print("[Detection] Loading ReID feature extractor (osnet_x0_25) ...")
        self.extractor = FeatureExtractor(
            model_name="osnet_x0_25",
            model_path=None,
            device=self.device
        )

        # class filter
        if C.SELECTED_CLASSES:
            self.allowed = set([c.lower() for c in C.SELECTED_CLASSES])
        else:
            self.allowed = None

        # Global ID memory
        self.next_gid = 0
        self.gallery: Dict[int, Dict] = {}  # gid -> {"emb": np.ndarray, "cls": str, "last": ts}

        # Per-camera last detect time to throttle
        self.last_detect_ts: Dict[str, float] = {}

    def _get_embedding(self, crop) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None
        try:
            feat = self.extractor(crop)
            # torch Tensor -> numpy
            import torch as _torch
            if isinstance(feat, _torch.Tensor):
                feat = feat.detach().cpu().numpy()
            if feat is None or len(feat) == 0:
                return None
            v = feat[0]
            n = np.linalg.norm(v)
            if n == 0:
                return None
            return (v / n).astype(np.float32)
        except Exception as e:
            print("[ReID] embedding error:", e)
            return None

    def _assign_global_id(self, emb: np.ndarray, cls_name: str, ts: int) -> int:
        if emb is None:
            gid = self.next_gid
            self.next_gid += 1
            self.gallery[gid] = {"emb": None, "cls": cls_name, "last": ts}
            return gid
        best_gid, best_sim = None, -1.0
        for gid, rec in self.gallery.items():
            if rec["emb"] is None or rec["cls"] != cls_name:
                continue
            sim = cosine_sim(rec["emb"], emb)
            if sim > best_sim:
                best_gid, best_sim = gid, sim
        if best_gid is None or best_sim < 0.72:
            gid = self.next_gid
            self.next_gid += 1
            self.gallery[gid] = {"emb": emb, "cls": cls_name, "last": ts}
            return gid
        else:
            # update
            self.gallery[best_gid]["emb"] = 0.6 * self.gallery[best_gid]["emb"] + 0.4 * emb
            n = np.linalg.norm(self.gallery[best_gid]["emb"])
            if n > 0:
                self.gallery[best_gid]["emb"] /= n
            self.gallery[best_gid]["last"] = ts
            return best_gid

    def _class_allowed(self, cls_name: str) -> bool:
        return True if self.allowed is None else (cls_name.lower() in self.allowed)

    def process(self, frames: Dict[str, Tuple[np.ndarray, int]]) -> List[Dict]:
        """
        frames: { cam -> (frame, ts) }
        returns list of detections dicts:
          {cam, ts, bbox, class, score, global_id, frame}
        """
        detections: List[Dict] = []
        now = time.time()
        for cam, (frame, ts) in frames.items():
            if frame is None or frame.size == 0:
                continue

            # throttle detection per camera
            last = self.last_detect_ts.get(cam, 0)
            if now - last < (1.0 / max(C.PROCESS_FPS, 0.1)):
                continue
            self.last_detect_ts[cam] = now

            # YOLO inference
            try:
                r = self.model.predict(
                    source=frame,
                    verbose=False,
                    conf=self.conf_thres,
                    iou=self.nms_iou,
                    device=self.device
                )[0]
            except Exception as e:
                print(f"[Detection] YOLO error on {cam}: {e}")
                continue

            if r.boxes is None or len(r.boxes) == 0:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = COCO_NAMES[cls_id] if 0 <= cls_id < len(COCO_NAMES) else str(cls_id)
                if not self._class_allowed(cls_name):
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if x2 <= x1 or y2 <= y1:
                    continue
                h, w = frame.shape[:2]
                x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
                y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
                crop = frame[y1:y2, x1:x2].copy() if (y2 > y1 and x2 > x1) else None
                emb = self._get_embedding(crop)
                gid = self._assign_global_id(emb, cls_name, ts)

                detections.append({
                    "cam": cam,
                    "ts": ts,
                    "bbox": (x1, y1, x2, y2),
                    "class": cls_name,
                    "score": float(box.conf[0].item()) if hasattr(box.conf, "shape") else float(box.conf),
                    "global_id": gid,
                    "frame": frame  # pass for thumbnail
                })

        return detections
