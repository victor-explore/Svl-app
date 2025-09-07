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

### Key Features
- **Multi-threaded RTSP handling**: Each camera runs in its own worker thread
- **Adaptive reconnection**: Configurable timeout and retry logic for unreliable streams  
- **Real-time streaming**: Optimized JPEG encoding for web streaming
- **Status monitoring**: Comprehensive camera health and performance metrics
- **Frame queue management**: Prevents memory buildup with configurable buffering

### Directory Structure
```
Svl-app/
├── app.py                 # Main Flask application
├── camera_manager.py      # Core camera handling logic
├── config.py             # Configuration settings
├── templates/            # HTML templates
│   ├── feed.html
│   └── base.html
├── static/css/           # CSS files (Tailwind, DaisyUI, custom)
└── requirements.txt      # Python dependencies
```

### API Endpoints
- `GET /api/cameras` - List all cameras with real-time status
- `POST /api/cameras` - Add new camera
- `POST /api/cameras/test-connection` - Test RTSP connection
- `DELETE /api/cameras/<id>` - Remove camera
- `PUT /api/cameras/<id>/status` - Update camera status
- `GET /api/cameras/<id>/stream` - Live video stream
- `GET /api/system/status` - Overall system status

### Configuration Notes
- Camera data is currently stored in-memory. For production use, implement database persistence.
- RTSP timeout settings are optimized for network reliability vs. speed
- JPEG quality is set to 70% for balance between quality and bandwidth

### Dependencies
- Flask 3.1.2 for web framework
- OpenCV (opencv-python) for video processing
- Threading for concurrent camera handling

### Performance Considerations
- Frame queue size is configurable per camera (default: 10 frames)
- Processing FPS can be adjusted in config.py (default: 25 FPS)
- JPEG compression quality affects bandwidth and CPU usage
- Thread cleanup timeout prevents hanging on shutdown

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
- Uses in-memory camera storage - consider database for production

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.