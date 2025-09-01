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