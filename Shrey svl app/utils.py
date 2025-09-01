import os
import time
import cv2
import json
import numpy as np
from datetime import datetime

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def atomic_json_write(path: str, obj, indent=2):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)

def timestamp_str(ts_sec: float | int):
    return datetime.fromtimestamp(int(ts_sec)).strftime("%Y-%m-%d %H:%M:%S")

def draw_bbox(frame, box, color=(0,255,0), thick=2):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thick)

def put_label(frame, text, org, scale=0.6, color=(0,255,0), thick=2):
    cv2.putText(frame, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def resize_keep_aspect(img, target_w):
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    scale = target_w / float(w)
    return cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

class PerfLogger:
    def __init__(self, label="[Perf] UI FPS", interval_frames=60):
        self.label = label
        self.interval_frames = interval_frames
        self.t0 = time.time()
        self.n = 0

    def tick(self, extra=""):
        self.n += 1
        if self.n >= self.interval_frames:
            t1 = time.time()
            dt = t1 - self.t0
            fps = self.n / dt if dt > 0 else 0.0
            print(f"{self.label}: {fps:.1f} {extra}".rstrip())
            self.t0 = t1
            self.n = 0

def tile_frames(frames_list, max_width=1920):
    """Make a grid mosaic from a list of BGR frames (varied sizes ok)."""
    if not frames_list:
        return None
    # normalize heights
    target_h = min([f.shape[0] for f in frames_list])
    resized = []
    for f in frames_list:
        h, w = f.shape[:2]
        if h != target_h:
            scale = target_h / float(h)
            f = cv2.resize(f, (int(w*scale), target_h), interpolation=cv2.INTER_AREA)
        resized.append(f)

    # grid: 1xN if <=3, else 2 rows
    n = len(resized)
    if n == 1:
        out = resized[0]
    elif n == 2 or n == 3:
        out = np.hstack(resized)
    else:
        mid = (n + 1) // 2
        top = np.hstack(resized[:mid])
        bottom = np.hstack(resized[mid:])
        # pad if needed
        if top.shape[1] != bottom.shape[1]:
            pad = abs(top.shape[1] - bottom.shape[1])
            if top.shape[1] < bottom.shape[1]:
                top = cv2.copyMakeBorder(top, 0, 0, 0, pad, cv2.BORDER_REPLICATE)
            else:
                bottom = cv2.copyMakeBorder(bottom, 0, 0, 0, pad, cv2.BORDER_REPLICATE)
        out = np.vstack([top, bottom])

    h, w = out.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        out = cv2.resize(out, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return out
