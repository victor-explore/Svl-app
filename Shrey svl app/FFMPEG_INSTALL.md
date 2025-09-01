# FFmpeg Installation Guide for Windows

FFmpeg is required for video recording functionality in this CCTV application.

## Download FFmpeg

1. Visit the official FFmpeg download page: https://ffmpeg.org/download.html
2. Click on "Windows" under "Get packages & executable files"
3. Choose "Windows builds by BtbN" (recommended)
4. Download the latest release (ffmpeg-master-latest-win64-gpl.zip)

## Installation Steps

### Method 1: Quick Installation (Recommended)

1. Extract the downloaded ZIP file to `C:\ffmpeg`
2. The folder structure should look like:
   ```
   C:\ffmpeg\
   ├── bin\
   │   ├── ffmpeg.exe
   │   ├── ffplay.exe
   │   └── ffprobe.exe
   ├── doc\
   └── presets\
   ```

3. Add FFmpeg to Windows PATH:
   - Press `Win + X` and select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", find and select "Path", then click "Edit"
   - Click "New" and add: `C:\ffmpeg\bin`
   - Click "OK" on all windows

4. Verify installation:
   - Open a NEW Command Prompt (important: must be new to load PATH changes)
   - Run: `ffmpeg -version`
   - You should see version information

### Method 2: Local Installation (No Admin Rights)

If you can't modify system PATH:

1. Extract ffmpeg.exe directly to your project folder:
   ```
   c:\1\Svl app\Shrey svl app\ffmpeg.exe
   ```

2. Modify the FfmpegRecorder class in camera_io.py to use local path:
   ```python
   cmd = [
       "./ffmpeg.exe",  # Use local ffmpeg
       # ... rest of the command
   ]
   ```

## Verify Installation

Run the test script to verify FFmpeg is working:
```bash
python test_components.py
```

The FFmpeg test should now show "[OK] PASSED"

## Troubleshooting

### "ffmpeg is not recognized as an internal or external command"
- Make sure you opened a NEW command prompt after adding to PATH
- Check that the path `C:\ffmpeg\bin` is correctly added to system PATH
- Try running with full path: `C:\ffmpeg\bin\ffmpeg.exe -version`

### Permission errors
- Run Command Prompt as Administrator
- Or use Method 2 (local installation)

### Antivirus blocking
- Some antivirus software may flag ffmpeg.exe
- Add an exception for ffmpeg.exe in your antivirus settings

## Alternative: Using Chocolatey

If you have Chocolatey package manager installed:
```bash
choco install ffmpeg
```

## Testing Video Recording

Once FFmpeg is installed, the application will be able to:
- Record video segments from RTSP cameras
- Save recordings in MP4 format
- Create segmented recordings (default: 60-second segments)

## Note for Development

Without FFmpeg:
- Object detection and tracking will still work
- Live camera feeds will be processed
- Event logging and thumbnails will be saved
- Only video recording/saving features will be disabled