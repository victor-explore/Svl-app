# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Enhanced CCTV surveillance system with Flask web interface for monitoring RTSP and USB camera feeds. Features multi-threaded stream processing, YOLOv8 person detection, OSNet person re-identification, and offline geographic mapping.

## Critical Constraints
- **Desktop Only**: Built for desktop browsers - do not focus on mobile responsiveness
- **Server Management**: DO NOT run the Flask server - user handles server management
- **Architecture**: Multi-threaded camera workers with optimistic deletion for responsive UX

## Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python app.py  # Runs on http://0.0.0.0:5000
```

### Testing Commands
```bash
# Database initialization test
python -c "from database import db_manager; db_manager.initialize(); print('Database OK')"

# Individual test scripts
python test_database_integration.py    # Database models and operations
python test_detection_performance.py   # YOLOv8 detection performance
python test_detection_service.py       # Detection service integration
python test_single_yolo.py             # Single YOLOv8 inference
python test_reid_optimization.py       # Person re-identification performance
python test_reid_flask_context.py      # Re-ID model in Flask context

# Setup offline map tiles
python scripts/setup_map.py
```

## Project Architecture

### Core Components

**app.py** - Flask application
- REST API endpoints for camera management, streaming, and analytics
- In-memory camera list + SQLite database for detections and metadata
- Auto-initializes `EnhancedCameraManager` on startup
- Runtime settings management via `user_settings.json`

**camera_manager.py** - Multi-threaded camera handling
- `CameraWorker`: Dedicated thread per camera (RTSP or USB) with frame queue buffering
- `EnhancedCameraManager`: Coordinates all camera workers, status tracking, and frame distribution
- Supports RTSP (TCP transport) and USB cameras (OpenCV VideoCapture)
- Integrates with `DetectionService` for person detection frame submission

**person_detector.py** - YOLOv8 detection service
- `DetectionService`: Single shared YOLOv8n model processes frames from all cameras
- Frame queue submission from camera workers at configurable intervals
- Database storage throttling (saves every N seconds per camera)
- Detection pause/resume API for UX optimization during user interactions

**person_reid.py** - OSNet person re-identification
- `PersonReID`: Singleton OSNet model for finding similar person detections
- Feature extraction and cosine similarity matching with early stopping optimization
- Chronological path construction for person movement tracking
- Search-by-detection-ID and search-by-uploaded-image endpoints

**database.py** - SQLAlchemy persistence layer
- Models: `Camera`, `Detection`, `SavedCamera`
- `DatabaseManager`: Session management, detection history queries, hourly statistics
- SQLite database with automatic initialization

**detection_storage.py** - Image file management
- Date-based directory structure (YYYY/MM/DD) for detection images
- Annotated images with bounding boxes and timestamps
- Storage statistics and cleanup utilities

**config.py** - Centralized configuration
- RTSP/USB connection parameters, frame processing settings
- Detection parameters (confidence, interval, resize dimensions)
- Camera deletion strategy (optimistic vs synchronous)
- Database, storage, and map settings

### Critical Architecture Patterns

**Single Shared YOLOv8 Model** (`person_detector.py:DetectionService`)
- ONE YOLOv8n instance processes frames from ALL cameras via queue
- Camera workers submit frames at configurable intervals (default: every 30 frames)
- Prevents GPU memory exhaustion from per-camera model instances
- Detection results matched to frames via timestamp correlation

**Optimistic Camera Deletion** (`app.py:delete_camera`, `config.py:DELETE_STRATEGY`)
- Immediate UI response: Remove from in-memory list instantly
- Background cleanup: Simple synchronous `remove_camera_simple()` call
- Progressive timeout strategy configurable: [graceful → terminate → force_kill]
- Camera IDs use timestamps - never reused to prevent stale references

**Person Re-Identification Flow** (`person_reid.py`)
- Singleton `PersonReID` instance with OSNet model (`osnet_x0_25_imagenet.pth`)
- Same-day detection search with early stopping optimization (configurable `max_search`)
- Two search modes: by detection ID or by uploaded image
- Returns similarity scores + chronological person path across cameras

**Frame Queue Architecture** (`camera_manager.py:CameraWorker`)
- Display queue: Size-1 queue (latest frame only) for web streaming
- Detection queue: Submitted to `DetectionService` at intervals
- Prevents memory buildup while maintaining smooth streaming
- Thread-safe status tracking with locks

**Runtime Settings Management** (`app.py`, `user_settings.json`)
- Settings persisted to JSON, override `config.py` defaults
- Apply at runtime without restart for: RTSP timeouts, detection intervals, DB cleanup
- API endpoints: `GET/PUT /api/settings`, `POST /api/settings/reset`

### Key API Endpoints

**Camera Management**
- `GET/POST /api/cameras` - List or add cameras (RTSP/USB)
- `POST /api/cameras/test-connection` - Test RTSP or USB camera
- `GET /api/usb-devices` - Scan for available USB cameras (parallel scan)
- `DELETE /api/cameras/<id>` - Remove camera (optimistic deletion)
- `GET /api/cameras/<id>/stream` - MJPEG video stream

**Person Detection & Re-ID**
- `GET /api/detections/history` - Detection history with date/camera filters
- `POST /api/detections/search-similar/<detection_id>` - Find similar detections using Re-ID
- `POST /api/detections/search-by-image` - Upload image to search for person
- `DELETE /api/detections/<id>` - Delete detection record and image
- `GET /api/analytics/hourly-detections` - Hourly stats for charts
- `POST /api/detection/pause`, `/resume` - Pause/resume detection for UX

**System & Settings**
- `GET /api/system/status` - System-wide camera and detection status
- `GET/PUT /api/settings` - Runtime settings (RTSP timeouts, detection params)
- `POST /api/settings/reset` - Reset to defaults

**Pages**
- `GET /` or `/feed` - Main dashboard with camera grid
- `GET /tracking` - Person detection timeline and search
- `GET /sensor-analytics` - Detection database with pagination
- `GET /map` - Geographic surveillance map (offline tiles)

## Data Storage & Persistence

**Dual Storage Model**
- In-memory list (`cameras` in `app.py`): Active camera configs for real-time streaming
- SQLite database (`detection_records.db`): Camera metadata, detection records, saved configs
- Detection images: `detection_images/YYYY/MM/DD/` with annotated bounding boxes

**Database Models** (`database.py`)
- `Camera`: Metadata with GPS coordinates for map display
- `Detection`: Bbox, confidence, image path, person_id, timestamps
- `SavedCamera`: Reusable camera configurations

**Key Configuration** (`config.py`)
- Camera IDs: Timestamp-based (never reused)
- Camera statuses: `online`, `offline`, `connecting`, `error`
- Detection storage throttling: Save every N seconds per camera (default: 10s)
- RTSP: 5s connection timeout, 3s read timeout, TCP transport
- USB: Parallel device scanning with reduced timeout (500ms)

## Important Technical Details

**Camera Worker Lifecycle** (`camera_manager.py`)
1. `CameraWorker` thread starts, opens VideoCapture (RTSP or USB)
2. Captures frames → puts in display queue (size 1, latest only)
3. Every Nth frame → submits to `DetectionService` queue
4. On stop: Progressive cleanup strategy (graceful → terminate → force)

**Detection Processing Flow** (`person_detector.py`)
1. Camera workers submit frames to shared queue
2. `DetectionService` runs YOLO inference (single model instance)
3. Throttled storage: Save to DB/disk every 10s per camera (prevents spam)
4. Detection results returned via timestamp matching

**Person Re-ID Search Strategy** (`person_reid.py`)
- Extract OSNet features from detection images or uploaded image
- Search same-day detections only with early stopping (configurable `max_search`)
- Cosine similarity ranking with threshold filtering (default: 0.7)
- Returns chronological path showing person movement across cameras

**Auto-Initialization** (on startup)
- YOLOv8n model: Auto-downloaded by Ultralytics if missing
- OSNet model: Must exist at `./osnet_x0_25_imagenet.pth`
- Database tables: Auto-created via SQLAlchemy
- Re-ID instance: Tested at startup with detailed logging

## Important Implementation Notes

**FLASK_DEBUG Setting** (`config.py:FLASK_DEBUG = False`)
- Must be False in production to prevent auto-restart when cameras change
- Auto-reloader can cause unexpected shutdowns when camera list becomes empty

**USB Camera Support** (`app.py`, `camera_manager.py`)
- Parallel device scanning (configurable workers, default: 4)
- Quick scan mode: Skip resolution detection for faster enumeration
- Device index validation prevents duplicate USB camera additions

**Offline Map Functionality** (`/map` page)
- Pre-cached tiles in `static/map_tiles/{z}/{x}/{y}.png`
- Default center: Leh, Ladakh (configurable in `config.py`)
- Camera markers color-coded by status (green/red/orange)

## Frontend Stack
- **Tailwind CSS + DaisyUI**: Component styling
- **Server-side rendering**: Flask/Jinja2 templates
- **JavaScript polling**: Real-time camera status updates
- **No mobile optimization**: Desktop-only interface