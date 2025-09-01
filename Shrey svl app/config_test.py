"""
Test configuration for CCTV system using webcam
Use this for testing when RTSP cameras are not available
"""

# Test with webcam (use index 0 for default webcam)
CAMERAS = {
    "Webcam": "0",  # Use webcam index 0
    # Uncomment below to test with video file:
    # "TestVideo": "path/to/video.mp4",
}

# For actual RTSP cameras, use format like:
# CAMERAS = {
#     "Cam1": "rtsp://username:password@192.168.1.100:554/stream1",
#     "Cam2": "rtsp://username:password@192.168.1.101:554/stream1",
# }

# Processing / detection
PROCESS_FPS = 2                        # Lower FPS for testing
YOLO_MODEL = "yolov8s.pt"              # Small model for testing
CONF_THRES = 0.35
NMS_IOU = 0.5
DEVICE = "cpu"                         # Use GPU if available: "cuda:0"

# Classes to detect (for testing, detect common objects)
SELECTED_CLASSES = ['person', 'cell phone', 'laptop', 'mouse', 'keyboard', 'book', 'cup', 'bottle']

# Recording
RECORDINGS_DIR = "recordings"
SEGMENT_SECONDS = 30                   # Shorter segments for testing

# Events reporting
EVENTS_DIR = "events"
THUMB_WIDTH = 300
MAX_THUMBS_PER_ID = 100

# HTTP server
HTTP_PORT = 8080

# Performance logging
UI_FPS_LOG_EVERY = 60