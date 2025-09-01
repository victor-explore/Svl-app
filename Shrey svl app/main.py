import signal
import sys
import time
from camera_io import CameraManager
from detection_tracking import Detector
from reporting import Reporter
from config import CAMERAS


def main():
    cam_mgr = CameraManager(CAMERAS, segment_time=60)
    detector = Detector()
    reporter = Reporter(outdir="events")

    cam_mgr.start()
    print("[System] Cameras and recorders started.")

    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        running = False
        print("\n[System] Ctrl+C received, shutting down...")

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while running:
            frames = cam_mgr.get_frames()
            if frames:
                try:
                    detections = detector.process(frames)
                    reporter.update(detections)
                except Exception as e:
                    print(f"[System] Detection/Reporting error: {e}")
            else:
                # no frames yet, wait and retry
                time.sleep(0.1)
    finally:
        cam_mgr.stop()
        print("[System] Shutdown complete.")


if __name__ == "__main__":
    main()
