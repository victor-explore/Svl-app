import os
import json
import cv2
import time
from typing import Dict, List, Tuple, Any, Optional
import glob
import datetime

from utils import ensure_dir, atomic_json_write

class Reporter:
    """
    Saves thumbnails with bounding box, links each sighting to a recording segment,
    and persists events.json keyed by global_id.
    """
    def __init__(self, outdir="events", recordings_dir="recordings", segment_duration=60, thumb_width=300, max_thumbs_per_id=1000):
        self.outdir = outdir
        self.thumbs_dir = os.path.join(outdir, "thumbs")
        self.events_file = os.path.join(outdir, "events.json")
        self.recordings_dir = recordings_dir
        self.segment_duration = int(segment_duration)
        self.thumb_width = int(thumb_width)
        self.max_thumbs_per_id = int(max_thumbs_per_id)
        ensure_dir(self.outdir)
        ensure_dir(self.thumbs_dir)
        ensure_dir(self.recordings_dir)
        if not os.path.exists(self.events_file):
            atomic_json_write(self.events_file, {})

        self._events: Dict[str, Dict[str, Any]] = self._load_events()
        self._seg_index: Dict[str, List[Tuple[int, str]]] = {}
        self._seg_last_scan: Dict[str, float] = {}

    def _load_events(self):
        try:
            with open(self.events_file, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def update(self, detections: List[Dict[str, Any]]):
        if not detections:
            return
        for d in detections:
            gid = str(d["global_id"])
            cam = d["cam"]
            ts = int(d["ts"])
            bbox = d.get("bbox")
            frame = d.get("frame")
            cls = d.get("class", "object")

            entry = self._events.get(gid)
            if entry is None:
                entry = {"class": cls, "first_seen": ts, "last_seen": ts, "thumb": None, "sightings": []}
                self._events[gid] = entry
            else:
                entry["class"] = entry.get("class", cls) or cls
                entry["first_seen"] = min(entry["first_seen"], ts)
                entry["last_seen"] = max(entry["last_seen"], ts)

            img_rel = None
            if isinstance(frame, type(None)) or frame is None or frame.size == 0 or bbox is None:
                pass
            else:
                img_rel = self._save_thumb(gid, cam, ts, frame, bbox)
                if img_rel and not entry.get("thumb"):
                    entry["thumb"] = img_rel

            video_rel, seg_start = self._find_segment(cam, ts)
            # de-dup
            if not any(s.get("cam") == cam and int(s.get("ts", -1)) == ts for s in entry["sightings"]):
                entry["sightings"].append({
                    "cam": cam,
                    "ts": ts,
                    "image": img_rel,
                    "video": video_rel,
                    "segment_start": seg_start
                })
                # cap length
                if self.max_thumbs_per_id and len(entry["sightings"]) > self.max_thumbs_per_id:
                    entry["sightings"] = entry["sightings"][-self.max_thumbs_per_id:]

        # sort sightings by time
        for gid, entry in self._events.items():
            entry["sightings"].sort(key=lambda s: s.get("ts", 0))

        atomic_json_write(self.events_file, self._events)

    def _save_thumb(self, gid, cam, ts, frame, bbox) -> Optional[str]:
        try:
            H, W = frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
            y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
            if x2 <= x1 or y2 <= y1:
                crop = frame.copy()
            else:
                crop = frame.copy()
                cv2.rectangle(crop, (x1,y1), (x2,y2), (0,255,255), 2)
                cv2.putText(crop, f"{cam}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            # resize
            if self.thumb_width > 0 and crop.shape[1] > self.thumb_width:
                scale = self.thumb_width / float(crop.shape[1])
                crop = cv2.resize(crop, (self.thumb_width, int(crop.shape[0]*scale)), interpolation=cv2.INTER_AREA)

            id_dir = os.path.join(self.thumbs_dir, str(gid))
            ensure_dir(id_dir)
            fname = f"{ts}_{cam}.jpg"
            abs_path = os.path.join(id_dir, fname)
            cv2.imwrite(abs_path, crop)
            return os.path.join("events", "thumbs", str(gid), fname).replace("\\", "/")
        except Exception as e:
            print("[Reporter] thumb error:", e)
            return None

    def _find_segment(self, cam: str, ts: int) -> Tuple[Optional[str], Optional[int]]:
        idx = self._scan_segments(cam)
        if not idx:
            return (None, None)
        # binary search
        lo, hi = 0, len(idx)-1
        while lo <= hi:
            mid = (lo + hi)//2
            st, rel = idx[mid]
            if st <= ts < st + self.segment_duration:
                return (rel, st)
            if ts < st:
                hi = mid - 1
            else:
                lo = mid + 1
        # lenient neighbors
        pos = max(0, min(len(idx)-1, lo))
        st, rel = idx[pos]
        if st <= ts < st + self.segment_duration + 5:
            return (rel, st)
        if pos > 0:
            st, rel = idx[pos-1]
            if st <= ts < st + self.segment_duration + 5:
                return (rel, st)
        return (None, None)

    def _scan_segments(self, cam: str) -> List[Tuple[int, str]]:
        now = time.time()
        last = self._seg_last_scan.get(cam, 0.0)
        if now - last < 5.0 and cam in self._seg_index:
            return self._seg_index[cam]
        cam_dir = os.path.join(self.recordings_dir, cam)
        if not os.path.isdir(cam_dir):
            self._seg_index[cam] = []
            self._seg_last_scan[cam] = now
            return []
        pairs = []
        for f in glob.glob(os.path.join(cam_dir, "*.mp4")):
            base = os.path.basename(f)
            st = self._parse_start(base)
            if st is None:
                continue
            pairs.append((st, os.path.join(cam, base).replace("\\","/")))
        pairs.sort(key=lambda x: x[0])
        self._seg_index[cam] = pairs
        self._seg_last_scan[cam] = now
        return pairs

    @staticmethod
    def _parse_start(fname: str) -> Optional[int]:
        try:
            stem = fname.split(".")[0]  # 20250819_121530
            y = int(stem[0:4]); m = int(stem[4:6]); d = int(stem[6:8])
            hh = int(stem[9:11]); mm = int(stem[11:13]); ss = int(stem[13:15])
            dt = datetime.datetime(y,m,d,hh,mm,ss)
            return int(dt.timestamp())
        except Exception:
            return None

    def finalize(self):
        # nothing special, but exists for symmetry
        pass
