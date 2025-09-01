# CCTV Surveillance System with AI Detection

A real-time CCTV monitoring system with YOLOv8 object detection, person re-identification, and video recording capabilities.

## Features

- **Real-time Object Detection**: Uses YOLOv8 for detecting objects in video streams
- **Person Re-identification**: Tracks individuals across multiple cameras using TorchReID
- **Multi-camera Support**: Handles multiple RTSP camera streams simultaneously
- **Video Recording**: Records video segments with FFmpeg
- **Event Logging**: Saves detection events with thumbnails
- **Configurable Detection**: Filter specific object classes
- **CPU/GPU Support**: Works on both CPU and CUDA-enabled GPUs

## System Requirements

- Windows 10/11 (tested) or Linux
- Python 3.8+ (tested with 3.12.10)
- 4GB+ RAM minimum (8GB recommended)
- Webcam or IP cameras with RTSP support
- FFmpeg (for video recording features)

## Installation

### 1. Clone or Download the Project

```bash
cd "c:\1\Svl app\Shrey svl app"
```

### 2. Set Up Python Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
```

### 3. Install Python Dependencies

All required libraries are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python numpy torch torchvision ultralytics torchreid
pip install gdown tensorboard  # Additional dependencies for torchreid
```

### 4. Install FFmpeg (Optional but Recommended)

FFmpeg is required for video recording features. See `FFMPEG_INSTALL.md` for detailed instructions.

Quick installation:
1. Download from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to system PATH
4. Verify: `ffmpeg -version`

## Quick Start

### 1. Test Installation

Run the component test to verify everything is installed correctly:

```bash
python test_components.py
```

Expected output:
- [OK] All Python libraries imported
- [OK] YOLO model ready
- [OK] Directories created
- [OK/FAIL] FFmpeg (only needed for recording)

### 2. Run Webcam Demo

Test the system with your webcam:

```bash
python demo_webcam.py
```

Choose option 1 for live webcam detection. Press 'q' to quit.

### 3. Configure Cameras

Edit `config.py` to set up your cameras:

```python
# For RTSP cameras:
CAMERAS = {
    "Cam1": "rtsp://username:password@192.168.1.100:554/stream",
    "Cam2": "rtsp://username:password@192.168.1.101:554/stream",
}

# For testing with webcam:
CAMERAS = {
    "Webcam": "0",  # Use webcam index 0
}
```

### 4. Run Main Application

```bash
python main.py
```

Press Ctrl+C to stop the application gracefully.

## Configuration Options

Edit `config.py` to customize:

- **CAMERAS**: Dictionary of camera names and URLs
- **PROCESS_FPS**: Detection frame rate (default: 5)
- **YOLO_MODEL**: Model to use (yolov8s.pt, yolov8m.pt, etc.)
- **CONF_THRES**: Confidence threshold (0.0-1.0)
- **DEVICE**: "cpu" or "cuda:0" for GPU
- **SELECTED_CLASSES**: List of classes to detect (empty = all)

## Project Structure

```
Shrey svl app/
├── main.py                 # Main application entry point
├── config.py              # Configuration file
├── config_test.py         # Test configuration (webcam)
├── camera_io.py           # Camera management and recording
├── detection_tracking.py  # YOLOv8 detection and ReID
├── reporting.py           # Event logging and thumbnails
├── utils.py              # Utility functions
├── test_components.py    # Component testing script
├── demo_webcam.py        # Webcam demo script
├── requirements.txt      # Python dependencies
├── FFMPEG_INSTALL.md    # FFmpeg installation guide
├── README.md            # This file
├── yolov8s.pt          # YOLO model (auto-downloaded)
├── recordings/         # Video recordings directory
├── events/            # Detection events and thumbnails
└── venv/             # Python virtual environment
```

## Output Files

- **recordings/**: Segmented video files (MP4 format)
  - Format: `CameraName/YYYYMMDD_HHMMSS.mp4`
  
- **events/**: Detection events and thumbnails
  - `events.json`: All detection events with timestamps
  - `thumbs/`: Thumbnail images for each detection

## Troubleshooting

### Camera Connection Issues
- Verify RTSP URL is correct
- Check network connectivity
- Ensure camera credentials are correct
- Try VLC player to test RTSP stream

### High CPU Usage
- Reduce PROCESS_FPS in config.py
- Use smaller YOLO model (yolov8n.pt)
- Limit number of cameras
- Consider using GPU if available

### Import Errors
- Activate virtual environment: `venv\Scripts\activate`
- Reinstall dependencies: `pip install -r requirements.txt`
- Run test script: `python test_components.py`

### FFmpeg Not Found
- See `FFMPEG_INSTALL.md` for installation
- Recording will not work without FFmpeg
- Detection and monitoring still function

## Performance Tips

1. **GPU Acceleration**: 
   - Install CUDA-enabled PyTorch for faster processing
   - Set `DEVICE = "cuda:0"` in config.py

2. **Optimize Detection**:
   - Use smaller models for speed (yolov8n.pt)
   - Reduce PROCESS_FPS for lower CPU usage
   - Filter specific classes with SELECTED_CLASSES

3. **Network Optimization**:
   - Use wired connections for IP cameras
   - Reduce camera resolution if needed
   - Use local network for RTSP streams

## Common RTSP URLs

Different camera manufacturers use different RTSP formats:

- **Hikvision**: `rtsp://user:pass@IP:554/Streaming/Channels/101`
- **Dahua**: `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0`
- **Axis**: `rtsp://user:pass@IP/axis-media/media.amp`
- **Generic**: `rtsp://user:pass@IP:554/stream1`

## Support

For issues or questions:
1. Check the troubleshooting section
2. Run `python test_components.py` for diagnostics
3. Review error messages in console output
4. Ensure all dependencies are installed correctly

## License

This project uses open-source libraries:
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- TorchReID

Please respect their respective licenses when using this system.