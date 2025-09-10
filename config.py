"""
Enhanced configuration for the hybrid CCTV system.
Combines web interface capabilities with robust RTSP handling.
"""

# RTSP Connection Settings - Optimized for Performance
RTSP_TRANSPORT = "tcp"                     # Use TCP transport for reliability
RTSP_TIMEOUT_MS = 5000                     # Connection timeout in milliseconds (reduced from 10000 for faster failure detection)
RTSP_READ_TIMEOUT_MS = 3000                # Frame read timeout in milliseconds (reduced from 5000 to prevent hangs)
RTSP_RECONNECT_DELAY = 1                   # Seconds to wait before reconnecting (reduced from 2 for faster recovery)
RTSP_MAX_RECONNECT_ATTEMPTS = 10           # Maximum reconnection attempts before giving up
RTSP_RECONNECT_DELAY_MAX = 30              # Maximum delay between reconnection attempts

# Frame Processing Settings - Optimized for Performance
FRAME_QUEUE_SIZE = 10                      # Maximum frames to buffer per camera (increased from 2 for better throughput)
PROCESSING_FPS = 25                        # Target FPS for frame processing (increased from 10 for smoother streams)
JPEG_QUALITY = 70                          # JPEG compression quality for web streaming (reduced from 80 for faster encoding)


# Performance Settings
MAX_CAMERAS = 20                           # Maximum number of cameras supported
THREAD_CLEANUP_TIMEOUT = 5                 # Timeout for thread cleanup in seconds
STATUS_UPDATE_INTERVAL = 2                 # Interval for status updates in seconds

# Camera Deletion Settings - Senior Developer Optimistic Approach
DELETE_STRATEGY = "optimistic"             # "optimistic" for immediate UI response, "synchronous" for blocking
CLEANUP_TIMEOUT_PROGRESSIVE = [1, 2, 0]    # Progressive timeouts: [graceful, terminate, force_kill]
SHOW_DELETION_FEEDBACK_MS = 300            # Brief "deleting" state duration in milliseconds

# Web Interface Settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Camera Status Constants
CAMERA_STATUS_ONLINE = "online"
CAMERA_STATUS_OFFLINE = "offline"
CAMERA_STATUS_CONNECTING = "connecting"
CAMERA_STATUS_ERROR = "error"

# Default camera template for new cameras
DEFAULT_CAMERA_CONFIG = {
    'username': '',
    'password': '',
    'status': CAMERA_STATUS_CONNECTING,
    'auto_start': True,
}

# Error messages
ERROR_MESSAGES = {
    'connection_failed': 'Failed to connect to RTSP stream',
    'no_frames': 'RTSP stream opened but no frames received',
    'invalid_url': 'Invalid RTSP URL format',
    'timeout': 'Connection timeout exceeded'
}

# Detection Settings
PERSON_DETECTION_ENABLED = True                # Enable person detection globally
PERSON_DETECTION_MODEL = './yolov8n.pt'       # YOLOv8 model to use (yolov8n.pt is fastest)
PERSON_DETECTION_CONFIDENCE = 0.5             # Minimum confidence threshold for detections
PERSON_DETECTION_INTERVAL = 30                # Process every 30 frames (~1.2 seconds at 25 FPS)
PERSON_DETECTION_MAX_HISTORY = 100            # Maximum detection results to keep per camera
PERSON_DETECTION_DRAW_BOXES = True            # Draw bounding boxes on streamed video
PERSON_DETECTION_BOX_COLOR = (0, 255, 0)      # Bounding box color (BGR format)
PERSON_DETECTION_BOX_THICKNESS = 2            # Bounding box line thickness

# Person Detection Performance Settings
PERSON_DETECTION_RESIZE_ENABLED = True        # Enable frame resizing before inference
PERSON_DETECTION_RESIZE_WIDTH = 640           # Target width for inference (YOLO optimal)
PERSON_DETECTION_RESIZE_HEIGHT = 640          # Target height for inference
PERSON_DETECTION_MAINTAIN_ASPECT = True       # Maintain aspect ratio when resizing

# Database Storage Settings
DATABASE_ENABLED = True                        # Enable database storage of detections
DATABASE_URL = "sqlite:///detection_records.db"  # Database connection string
DATABASE_AUTO_INIT = True                     # Auto-initialize database on startup

# Image Storage Settings
DETECTION_IMAGE_STORAGE_ENABLED = True        # Enable saving detection images to disk
DETECTION_IMAGE_BASE_PATH = "detection_images"  # Base directory for detection images
DETECTION_IMAGE_QUALITY = 95                  # JPEG quality for detection images (0-100)

# Database Maintenance Settings
DATABASE_CLEANUP_ENABLED = False              # Enable automatic cleanup of old records
DATABASE_CLEANUP_DAYS = 30                   # Days to keep detection records
DATABASE_CLEANUP_INTERVAL_HOURS = 24         # Hours between cleanup runs

# Storage Throttling Settings
DETECTION_STORAGE_INTERVAL_SECONDS = 30      # Save images/DB every N seconds when person detected
DETECTION_STORAGE_THROTTLING_ENABLED = True  # Enable time-based storage throttling

# Map Configuration Settings
MAP_ENABLED = True                           # Enable offline map functionality
MAP_TILES_PATH = "static/map_tiles"         # Path to offline map tiles
MAP_DEFAULT_CENTER_LAT = 34.1642            # Default map center latitude (Leh, Ladakh)
MAP_DEFAULT_CENTER_LNG = 77.5847           # Default map center longitude (Leh, Ladakh)
MAP_DEFAULT_ZOOM = 14                       # Default zoom level for the map
MAP_MIN_ZOOM = 10                           # Minimum zoom level allowed
MAP_MAX_ZOOM = 18                           # Maximum zoom level allowed
MAP_TILE_URL_TEMPLATE = "/static/map_tiles/{z}/{x}/{y}.png"  # Template for offline tile URLs

# Camera Map Settings
CAMERA_DEFAULT_LATITUDE = None              # Default latitude for new cameras (None = user must specify)
CAMERA_DEFAULT_LONGITUDE = None             # Default longitude for new cameras (None = user must specify)
CAMERA_MARKER_COLORS = {                    # Map marker colors by camera status
    CAMERA_STATUS_ONLINE: "#22c55e",        # Green for online cameras
    CAMERA_STATUS_OFFLINE: "#ef4444",       # Red for offline cameras
    CAMERA_STATUS_CONNECTING: "#f59e0b",    # Orange for connecting cameras
    CAMERA_STATUS_ERROR: "#dc2626"          # Dark red for error cameras
}