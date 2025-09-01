import cv2
import threading
import queue
import subprocess
import os
import time
from datetime import datetime


class CameraWorker(threading.Thread):
    def __init__(self, name, url, outdir=None, segment_time=60):
        super().__init__()
        self.name = name
        self.url = url
        self.outdir = outdir
        self.segment_time = segment_time
        self.q = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self.proc = None

    def run(self):
        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(self.url)

            if not cap.isOpened():
                print(f"[{self.name}] Could not open stream {self.url}, retrying in 5s...")
                time.sleep(5)
                continue

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"[{self.name}] Frame grab failed, reconnecting...")
                    cap.release()
                    time.sleep(2)
                    break  # go back to outer loop to retry

                ts = datetime.now()
                if not self.q.full():
                    self.q.put((frame, ts))

            cap.release()

        print(f"[{self.name}] Stopped.")

    def get_frame(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None, None

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)   # ensure thread exits


class FfmpegRecorder(threading.Thread):
    def __init__(self, name, url, outdir="recordings", segment_time=60):
        super().__init__()
        self.name = name
        self.url = url
        self.outdir = os.path.join(outdir, name)
        self.segment_time = segment_time
        os.makedirs(self.outdir, exist_ok=True)
        self._stop_event = threading.Event()
        self.proc = None

    def run(self):
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-stimeout", "3000000",
            "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
            "-i", self.url,
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(self.segment_time),
            "-reset_timestamps", "1",
            "-strftime", "1",
            os.path.join(self.outdir, "%Y%m%d_%H%M%S.mp4")
        ]

        while not self._stop_event.is_set():
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                self.proc.wait()
            except Exception as e:
                print(f"[{self.name}][FFmpeg] error: {e}")

            if not self._stop_event.is_set():
                print(f"[{self.name}] Recorder restarting in 2s...")
                time.sleep(2)

        print(f"[{self.name}] Recorder stopped.")

    def stop(self):
        self._stop_event.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()
        self.join(timeout=2)


class CameraManager:
    def __init__(self, cameras, segment_time=60):
        self.workers = {}
        self.recorders = {}
        self.segment_time = segment_time
        for name, url in cameras.items():
            self.workers[name] = CameraWorker(name, url)
            self.recorders[name] = FfmpegRecorder(name, url, outdir="recordings", segment_time=segment_time)

    def start(self):
        for w in self.workers.values():
            w.start()
        for r in self.recorders.values():
            r.start()

    def get_frames(self):
        frames = {}
        for name, w in self.workers.items():
            frame, ts = w.get_frame()
            if frame is not None:
                frames[name] = (frame, ts)
        return frames

    def stop(self):
        for w in self.workers.values():
            w.stop()
        for r in self.recorders.values():
            r.stop()
