"""
Enhanced configuration for the hybrid CCTV system.
Combines web interface capabilities with robust RTSP handling.
"""

# RTSP Connection Settings
RTSP_TRANSPORT = "tcp"                     # Use TCP transport for reliability
RTSP_TIMEOUT_MS = 10000                    # Connection timeout in milliseconds
RTSP_READ_TIMEOUT_MS = 5000                # Frame read timeout in milliseconds
RTSP_RECONNECT_DELAY = 2                   # Seconds to wait before reconnecting
RTSP_MAX_RECONNECT_ATTEMPTS = 10           # Maximum reconnection attempts before giving up
RTSP_RECONNECT_DELAY_MAX = 30              # Maximum delay between reconnection attempts

# Frame Processing Settings
FRAME_QUEUE_SIZE = 2                       # Maximum frames to buffer per camera
PROCESSING_FPS = 10                        # Target FPS for frame processing
JPEG_QUALITY = 80                          # JPEG compression quality for web streaming

# Recording Settings
RECORDINGS_DIR = "recordings"              # Directory to store recordings
SEGMENT_DURATION = 60                      # Duration of each recording segment in seconds
RECORDING_AUTO_START = False               # Whether to auto-start recording for new cameras

# FFmpeg Recording Settings
FFMPEG_LOGLEVEL = "warning"                # FFmpeg log level (quiet, error, warning, info, debug)
FFMPEG_STIMEOUT = 3000000                  # Stream timeout for FFmpeg (microseconds)
FFMPEG_RECONNECT = True                    # Enable FFmpeg automatic reconnection
FFMPEG_RECONNECT_DELAY_MAX = 5             # Maximum reconnection delay for FFmpeg

# Performance Settings
MAX_CAMERAS = 20                           # Maximum number of cameras supported
THREAD_CLEANUP_TIMEOUT = 5                 # Timeout for thread cleanup in seconds
STATUS_UPDATE_INTERVAL = 2                 # Interval for status updates in seconds

# Web Interface Settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Camera Status Constants
CAMERA_STATUS_ONLINE = "online"
CAMERA_STATUS_OFFLINE = "offline"
CAMERA_STATUS_CONNECTING = "connecting"
CAMERA_STATUS_RECORDING = "recording"
CAMERA_STATUS_ERROR = "error"

# Default camera template for new cameras
DEFAULT_CAMERA_CONFIG = {
    'username': '',
    'password': '',
    'status': CAMERA_STATUS_CONNECTING,
    'auto_start': True,
    'record_footage': False,
    'recording_status': 'stopped'
}

# Error messages
ERROR_MESSAGES = {
    'connection_failed': 'Failed to connect to RTSP stream',
    'no_frames': 'RTSP stream opened but no frames received',
    'ffmpeg_error': 'FFmpeg recording error',
    'invalid_url': 'Invalid RTSP URL format',
    'timeout': 'Connection timeout exceeded'
}