"""
Central config for the CCTV system.
Tweak here—no code edits elsewhere.
"""

# Cameras (name -> RTSP URL)
CAMERAS = {
    "Cam1": "rtsp://admin:admin123@192.168.1.21/cam/realmonitor?channel=1&subtype=0",
    "Cam2": "rtsp://admin:admin123@192.168.1.15/cam/realmonitor?channel=1&subtype=0",
}

# Processing / detection
PROCESS_FPS = 5                        # YOLO per-camera processing rate (tracking runs every frame, detection throttled)
YOLO_MODEL = "yolov8s.pt"
CONF_THRES = 0.35
NMS_IOU = 0.5
DEVICE = "cuda:0"                      # "cuda:0" or "cpu"

# Classes to detect (names as per COCO 80). Empty/None => all classes.
SELECTED_CLASSES = []
# SELECTED_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

# Recording
RECORDINGS_DIR = "recordings"
SEGMENT_SECONDS = 60

# Events reporting
EVENTS_DIR = "events"
THUMB_WIDTH = 300
MAX_THUMBS_PER_ID = 1000

# HTTP server
HTTP_PORT = 8080

# Performance logging
UI_FPS_LOG_EVERY = 120
