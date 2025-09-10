# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an enhanced CCTV surveillance system built with Flask and Python. The application provides a web interface for monitoring multiple RTSP camera feeds with advanced features including real-time streaming and camera management.

## Directions from User
- Do not focus on making the code responsive to look good on mobile. The app is being built for a desktop browser only.
- DO NOT run the Flask server. The user is already running it and will handle server management.

## Development Commands

### Running the Application
```bash
cd Svl-app
python app.py
```
The application runs on `http://0.0.0.0:5000` by default.


### Install Dependencies
```bash
cd Svl-app
pip install -r requirements.txt
```

### Testing RTSP Connections
Use the built-in connection test API endpoint:
```bash
curl -X POST http://localhost:5000/api/cameras/test-connection \
  -H "Content-Type: application/json" \
  -d '{"rtsp_url": "your_rtsp_url_here"}'
```

## Project Architecture

### Core Components

**app.py** (Flask Application)
- Main Flask application with REST API endpoints
- Routes for camera management, streaming, and status monitoring
- In-memory camera storage (should be replaced with database in production)
- Auto-initializes cameras on startup using the `EnhancedCameraManager`

**camera_manager.py** (Enhanced Camera Management)
- `CameraWorker`: Threading-based RTSP frame capture with robust error handling
- `EnhancedCameraManager`: Central manager coordinating multiple cameras
- Implements frame queuing, status tracking, and performance metrics
- Note: No FFmpeg recording functionality currently implemented

**config.py** (Configuration Settings)
- RTSP connection parameters optimized for performance
- Frame processing settings (FPS, JPEG quality, buffer sizes)
- Thread management and cleanup timeouts
- Camera deletion strategy configuration (optimistic vs synchronous)
- Person detection parameters (confidence threshold, model settings)

**database.py** (SQLAlchemy Database Layer)
- `Camera` and `Detection` models for persistent storage
- `DatabaseManager`: Handles all database operations and connections
- SQLite database with automatic table creation
- Detection history querying with filtering and pagination

**person_detector.py** (YOLOv8 Person Detection)
- `PersonDetector`: YOLOv8-based person detection with confidence filtering
- `DetectionResult`: Data structure for detection results
- `PersonDetectionManager`: Multi-camera detection coordinator
- GPU/CPU optimization with performance tracking

**detection_storage.py** (Image Storage Management)
- `DetectionImageStorage`: Manages saving detection images to disk
- Date-based directory structure (YYYY/MM/DD)
- Image annotation with bounding boxes and timestamps

### Key Features
- **Multi-threaded RTSP handling**: Each camera runs in its own worker thread
- **Adaptive reconnection**: Configurable timeout and retry logic for unreliable streams  
- **Real-time streaming**: Optimized JPEG encoding for web streaming
- **Status monitoring**: Comprehensive camera health and performance metrics
- **Frame queue management**: Prevents memory buildup with configurable buffering
- **Person detection**: YOLOv8-powered real-time person detection with configurable confidence
- **Detection storage**: Persistent SQLite database for detection records and analytics
- **Image archival**: Automatic saving of detection images with annotations and metadata

### Directory Structure
```
Svl-app/
├── app.py                 # Main Flask application
├── camera_manager.py      # Core camera handling logic
├── config.py             # Configuration settings
├── database.py           # SQLAlchemy models and database operations
├── person_detector.py    # YOLOv8 person detection system
├── detection_storage.py  # Image storage management
├── templates/            # HTML templates
│   ├── base.html
│   ├── feed.html
│   ├── tracking.html     # Person detection interface
│   ├── sensor_analytics.html # Analytics dashboard
│   └── settings.html
├── static/               # CSS, JS, and static assets
├── detection_images/     # Stored detection images (auto-created)
├── yolov8n.pt           # YOLOv8 model file
├── detection_records.db  # SQLite database
└── requirements.txt      # Python dependencies
```

### API Endpoints

**Camera Management**
- `GET /api/cameras` - List all cameras with real-time status
- `POST /api/cameras` - Add new camera
- `POST /api/cameras/test-connection` - Test RTSP connection
- `DELETE /api/cameras/<id>` - Remove camera
- `PUT /api/cameras/<id>/status` - Update camera status
- `GET /api/cameras/<id>/stream` - Live video stream

**Person Detection**
- `POST /api/cameras/<id>/detection/enable` - Enable/disable person detection
- `GET /api/detections` - Get detection history with pagination
- `GET /api/detections/stats` - Detection statistics and analytics
- `GET /api/detections/hourly-stats` - Hourly detection data for charts
- `GET /api/detection-images/<path:filename>` - Serve detection images

**System**
- `GET /api/system/status` - Overall system status
- `GET /` - Main dashboard
- `GET /tracking` - Person detection interface  
- `GET /analytics` - Analytics and charts dashboard

### Configuration Notes
- **Camera data**: Stored in-memory for active management; persistent detection data in SQLite database
- **RTSP settings**: Timeout settings optimized for network reliability vs. speed
- **Image quality**: JPEG quality set to 70% for balance between quality and bandwidth
- **Detection model**: YOLOv8n model (lightweight, good for real-time processing)
- **Database**: SQLite for detection records with automatic table creation
- **Storage**: Detection images stored in `detection_images/` with date-based structure

### Dependencies
- Flask 3.1.2 for web framework
- OpenCV (opencv-python) for video processing
- Threading for concurrent camera handling
- YOLOv8 (ultralytics) for person detection
- SQLAlchemy for database operations
- Pillow for image processing

### Performance Considerations
- Frame queue size is configurable per camera (default: 10 frames)
- Processing FPS can be adjusted in config.py (default: 25 FPS)
- JPEG compression quality affects bandwidth and CPU usage
- Thread cleanup timeout prevents hanging on shutdown
- **Person detection**: YOLOv8 inference optimized for CPU/GPU with resizable input frames
- **Database queries**: Indexed detection queries with pagination for large datasets
- **Image storage**: Date-based directory structure prevents filesystem bottlenecks

## Critical Implementation Details

### Camera Deletion Strategy
The application implements an optimistic camera deletion pattern for enhanced UX:
- **DELETE_STRATEGY = "optimistic"**: Provides immediate UI feedback (<100ms)
- **Progressive timeout cleanup**: [1s graceful] → [2s terminate] → [0s force_kill]
- Background async cleanup prevents UI blocking during camera removal
- Error recovery restores UI state if backend deletion fails

### Thread Management Architecture
- Each camera runs in dedicated `CameraWorker` thread with frame queue buffering
- Progressive cleanup strategies prevent hanging threads during shutdown (graceful → terminate → force_kill)
- Status tracking with thread-safe locks for real-time monitoring

### RTSP Connection Handling  
- TCP transport for reliability with optimized timeout settings
- Adaptive reconnection with exponential backoff (max 30s delay)
- Frame queue management prevents memory buildup
- Connection health monitoring with performance metrics

## Frontend Framework
- Uses Tailwind CSS for styling and DaisyUI components
- Templates are rendered server-side with Flask/Jinja2
- JavaScript handles real-time UI updates and camera management

## Camera Status Management
- Camera statuses: `online`, `offline`, `connecting`, `error`
- Real-time status updates via API polling
- Status synchronization between in-memory storage and camera manager

## Development Notes
- Virtual environment setup recommended (venv folder present)
- Application designed for desktop browsers only (not mobile responsive)
- Uses in-memory camera storage with SQLite persistence for detection data
- **YOLOv8 model**: Downloaded automatically on first detection attempt
- **Database migrations**: Tables created automatically via SQLAlchemy
- **Detection images**: Stored locally with configurable retention policies

## Testing Detection Features
Use these endpoints to test person detection functionality:

### Test Database Connection
```bash
python -c "from database import db_manager; db_manager.initialize(); print('Database connection successful')"
```

### Test Person Detection
```bash
# Enable detection for camera ID 1
curl -X POST http://localhost:5000/api/cameras/1/detection/enable \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'

# Get detection history
curl http://localhost:5000/api/detections?limit=10
```

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.