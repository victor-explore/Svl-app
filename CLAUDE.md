# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an enhanced CCTV surveillance system built with Flask and Python. The application provides a web interface for monitoring multiple RTSP camera feeds with advanced features including real-time streaming, person detection, and geographic mapping.

## Critical Constraints
- **Desktop Only**: Do not focus on mobile responsiveness - built for desktop browsers only
- **Server Management**: DO NOT run the Flask server - user handles server management
- **Architecture**: Multi-threaded RTSP processing with optimistic deletion patterns for UX

## Development Commands

### Environment Setup
```bash
cd Svl-app
pip install -r requirements.txt
```

### Running the Application  
```bash
python app.py  # Runs on http://0.0.0.0:5000
```

### Testing Commands
```bash
# Test RTSP connection
curl -X POST http://localhost:5000/api/cameras/test-connection \
  -H "Content-Type: application/json" \
  -d '{"rtsp_url": "your_rtsp_url_here"}'

# Test database connection
python -c "from database import db_manager; db_manager.initialize(); print('Database OK')"

# Enable person detection for camera ID 1
curl -X POST http://localhost:5000/api/cameras/1/detection/enable \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
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
- GPU/CPU optimization with performance tracking

**detection_storage.py** (Image Storage Management)
- `DetectionImageStorage`: Manages saving detection images to disk
- Date-based directory structure (YYYY/MM/DD)
- Image annotation with bounding boxes and timestamps

### Key Features
- **Multi-threaded RTSP handling**: Each camera runs in dedicated worker thread with frame queue buffering
- **Optimistic camera deletion**: Immediate UI feedback (<100ms) with background async cleanup  
- **Real-time streaming**: Optimized JPEG encoding for web streaming
- **Person detection**: YOLOv8-powered detection with SQLite persistence and image archival
- **Geographic mapping**: Offline map tiles for surveillance device location tracking
- **Status monitoring**: Comprehensive camera health and performance metrics with real-time updates

### Critical Architecture Patterns

**Optimistic Deletion Strategy** (`config.py:DELETE_STRATEGY = "optimistic"`)
- Progressive timeout cleanup: [1s graceful] → [2s terminate] → [0s force_kill]
- Background async cleanup prevents UI blocking during camera removal
- Error recovery restores UI state if backend deletion fails

**Thread Management Architecture**
- Each camera runs in dedicated `CameraWorker` thread with frame queue buffering
- Progressive cleanup strategies prevent hanging threads during shutdown
- Status tracking with thread-safe locks for real-time monitoring

**RTSP Connection Handling**
- TCP transport for reliability with optimized timeout settings (5s connection, 3s read)
- Adaptive reconnection with exponential backoff (max 30s delay)
- Frame queue management prevents memory buildup (default: 10 frames per camera)

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
- `GET /map` - Geographic surveillance map with offline tiles

## Data Storage Architecture

**In-Memory Camera Management**
- Active camera data stored in-memory for real-time performance
- Camera statuses: `online`, `offline`, `connecting`, `error`
- Real-time status updates via API polling

**SQLite Persistence** 
- Detection records stored in `detection_records.db` with automatic table creation
- Detection images stored in `detection_images/` with date-based directory structure (YYYY/MM/DD)
- Image annotation with bounding boxes and timestamps

**Configuration Settings** (`config.py`)
- RTSP timeouts: 5s connection, 3s read (optimized for reliability vs speed)
- JPEG quality: 70% (balance between quality and bandwidth)
- Frame processing: 25 FPS, 10-frame buffer per camera
- YOLOv8n model: Lightweight, good for real-time processing

## Frontend Architecture
- **Tailwind CSS + DaisyUI**: Component styling framework
- **Server-side rendering**: Flask/Jinja2 templates
- **Real-time updates**: JavaScript polling for camera status and detection updates
- **Offline mapping**: Pre-cached map tiles in `static/map_tiles/` for geographic surveillance view

## Key Technical Patterns

**Camera Status Synchronization**
- Status tracking with thread-safe locks for real-time monitoring
- Status synchronization between in-memory storage and camera manager
- Real-time status updates via API polling

**Performance Optimizations**
- Frame queue management prevents memory buildup (configurable buffer sizes)
- YOLOv8 inference optimized for CPU/GPU with resizable input frames
- Database queries use pagination for large detection datasets
- Date-based directory structure prevents filesystem bottlenecks

**Auto-Initialization**
- YOLOv8 model downloaded automatically on first detection attempt
- Database tables created automatically via SQLAlchemy
- Cameras auto-initialize on startup using `EnhancedCameraManager`

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.