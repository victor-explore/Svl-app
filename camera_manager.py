"""
Advanced Camera Management System
Combines threading, robust RTSP handling, and FFmpeg recording capabilities.
Adapted from Shrey svl app with Flask integration enhancements.
"""

import cv2
import threading
import queue
import subprocess
import os
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraStatus:
    """Enum-like class for camera status constants"""
    ONLINE = CAMERA_STATUS_ONLINE
    OFFLINE = CAMERA_STATUS_OFFLINE
    CONNECTING = CAMERA_STATUS_CONNECTING
    RECORDING = CAMERA_STATUS_RECORDING
    ERROR = CAMERA_STATUS_ERROR


class CameraWorker(threading.Thread):
    """
    Enhanced Camera Worker Thread
    Handles RTSP connection, frame capture, and status reporting
    """
    
    def __init__(self, camera_id: int, name: str, rtsp_url: str, 
                 username: str = '', password: str = ''):
        super().__init__()
        self.camera_id = camera_id
        self.name = name
        self.rtsp_url = rtsp_url
        self.username = username
        self.password = password
        
        # Threading controls
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        
        # Status tracking
        self._status = CameraStatus.CONNECTING
        self._last_frame_time = None
        self._connection_attempts = 0
        self._last_error = None
        self.is_recording = False
        
        # Performance tracking
        self.frames_captured = 0
        self.frames_dropped = 0
        self.start_time = time.time()
        
        self.daemon = True
        logger.info(f"[{self.name}] Camera worker initialized")

    @property
    def status(self) -> str:
        """Thread-safe status getter"""
        with self._status_lock:
            return self._status

    def set_status(self, new_status: str, error_msg: str = None):
        """Thread-safe status setter"""
        with self._status_lock:
            old_status = self._status
            self._status = new_status
            if error_msg:
                self._last_error = error_msg
            if old_status != new_status:
                logger.info(f"[{self.name}] Status changed: {old_status} -> {new_status}")

    def get_stats(self) -> Dict[str, Any]:
        """Get camera performance statistics"""
        uptime = time.time() - self.start_time
        fps = self.frames_captured / uptime if uptime > 0 else 0
        
        return {
            'status': self.status,
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'fps': round(fps, 2),
            'uptime_seconds': round(uptime, 2),
            'last_frame_time': self._last_frame_time,
            'connection_attempts': self._connection_attempts,
            'last_error': self._last_error,
            'is_recording': self.is_recording
        }

    def _create_rtsp_url(self) -> str:
        """Create authenticated RTSP URL if credentials provided"""
        if self.username and self.password:
            # Insert credentials into RTSP URL
            if '://' in self.rtsp_url:
                protocol, rest = self.rtsp_url.split('://', 1)
                return f"{protocol}://{self.username}:{self.password}@{rest}"
        return self.rtsp_url

    def _setup_opencv_capture(self) -> cv2.VideoCapture:
        """Setup OpenCV VideoCapture with optimal settings - Performance Optimized"""
        url = self._create_rtsp_url()
        cap = cv2.VideoCapture(url)
        
        # OpenCV Performance Optimizations (based on documentation)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_TIMEOUT_MS)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS)
        
        # Critical: Minimize latency with smallest buffer
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # NEW: Hardware acceleration and performance settings
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            cap.set(cv2.CAP_PROP_FPS, PROCESSING_FPS)  # Set target FPS
            # Enable hardware decoding if available
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # Ensure RGB conversion
        except:
            pass  # Ignore if not supported
            
        return cap

    def run(self):
        """Main worker thread loop"""
        logger.info(f"[{self.name}] Camera worker thread started")
        
        while not self._stop_event.is_set():
            try:
                self._connection_loop()
            except Exception as e:
                logger.error(f"[{self.name}] Unexpected error in main loop: {e}")
                self.set_status(CameraStatus.ERROR, str(e))
                time.sleep(RTSP_RECONNECT_DELAY)
        
        logger.info(f"[{self.name}] Camera worker thread stopped")

    def _connection_loop(self):
        """Handle connection and frame capture loop"""
        self.set_status(CameraStatus.CONNECTING)
        self._connection_attempts += 1
        
        cap = self._setup_opencv_capture()
        
        if not cap.isOpened():
            error_msg = f"Could not open RTSP stream (attempt {self._connection_attempts})"
            logger.warning(f"[{self.name}] {error_msg}")
            self.set_status(CameraStatus.OFFLINE, error_msg)
            cap.release()
            time.sleep(min(RTSP_RECONNECT_DELAY * self._connection_attempts, RTSP_RECONNECT_DELAY_MAX))
            return

        logger.info(f"[{self.name}] Successfully connected to RTSP stream")
        self.set_status(CameraStatus.ONLINE)
        self._connection_attempts = 0  # Reset on successful connection

        consecutive_failures = 0
        max_consecutive_failures = 10
        
        # Frame capture loop
        while not self._stop_event.is_set():
            try:
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"[{self.name}] Too many frame read failures, reconnecting...")
                        break
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                self.frames_captured += 1
                self._last_frame_time = datetime.now()
                
                # Add frame to queue (non-blocking) - FFmpeg-style buffer management
                try:
                    # Efficient queue management: drop oldest frame if full
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                            self.frames_dropped += 1
                        except queue.Empty:
                            pass
                    
                    self.frame_queue.put((frame.copy(), self._last_frame_time), block=False)
                    
                except queue.Full:
                    self.frames_dropped += 1
                
                # REMOVED: Artificial delay - let natural RTSP timing control flow
                # OLD: time.sleep(1.0 / PROCESSING_FPS)  
                # This allows frames to flow at their natural rate for better performance
                
            except Exception as e:
                logger.error(f"[{self.name}] Frame capture error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break

        cap.release()
        
        if not self._stop_event.is_set():
            # Connection lost, will retry
            self.set_status(CameraStatus.OFFLINE, "Connection lost")
            time.sleep(RTSP_RECONNECT_DELAY)

    def get_latest_frame(self) -> Tuple[Optional[Any], Optional[datetime]]:
        """Get the most recent frame from the queue"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None, None

    def stop(self):
        """Stop the camera worker thread"""
        logger.info(f"[{self.name}] Stopping camera worker...")
        self._stop_event.set()
        
        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for thread to finish
        if self.is_alive():
            self.join(timeout=THREAD_CLEANUP_TIMEOUT)
            if self.is_alive():
                logger.warning(f"[{self.name}] Thread did not stop gracefully")


class FfmpegRecorder(threading.Thread):
    """
    Enhanced FFmpeg Recorder
    Handles professional recording with automatic restart and error recovery
    """
    
    def __init__(self, camera_id: int, name: str, rtsp_url: str, 
                 username: str = '', password: str = ''):
        super().__init__()
        self.camera_id = camera_id
        self.name = name
        self.rtsp_url = rtsp_url
        self.username = username
        self.password = password
        
        # Setup output directory
        self.output_dir = os.path.join(RECORDINGS_DIR, f"camera_{camera_id}_{name}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Threading controls
        self._stop_event = threading.Event()
        self._recording_active = False
        self.process = None
        
        # Status tracking
        self.recording_start_time = None
        self.segments_created = 0
        self.last_error = None
        
        self.daemon = True
        logger.info(f"[{self.name}] Recorder initialized, output: {self.output_dir}")

    def _create_authenticated_url(self) -> str:
        """Create authenticated RTSP URL for FFmpeg"""
        if self.username and self.password:
            if '://' in self.rtsp_url:
                protocol, rest = self.rtsp_url.split('://', 1)
                return f"{protocol}://{self.username}:{self.password}@{rest}"
        return self.rtsp_url

    def _build_ffmpeg_command(self) -> list:
        """Build FFmpeg command with all necessary parameters"""
        url = self._create_authenticated_url()
        output_pattern = os.path.join(self.output_dir, "%Y%m%d_%H%M%S.mp4")
        
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", FFMPEG_LOGLEVEL,
            "-rtsp_transport", RTSP_TRANSPORT,
            "-stimeout", str(FFMPEG_STIMEOUT),
        ]
        
        if FFMPEG_RECONNECT:
            cmd.extend([
                "-reconnect", "1", 
                "-reconnect_streamed", "1", 
                "-reconnect_delay_max", str(FFMPEG_RECONNECT_DELAY_MAX)
            ])
        
        cmd.extend([
            "-i", url,
            "-c", "copy",  # Copy streams without re-encoding
            "-f", "segment",
            "-segment_time", str(SEGMENT_DURATION),
            "-reset_timestamps", "1",
            "-strftime", "1",
            output_pattern
        ])
        
        return cmd

    def start_recording(self):
        """Start the recording process"""
        if not self._recording_active and not self.is_alive():
            self._recording_active = True
            self.recording_start_time = datetime.now()
            self.start()
            logger.info(f"[{self.name}] Recording started")
            return True
        return False

    def stop_recording(self):
        """Stop the recording process"""
        if self._recording_active:
            self._recording_active = False
            self._stop_event.set()
            
            # Terminate FFmpeg process
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"[{self.name}] Force killing FFmpeg process")
                    self.process.kill()
                except Exception as e:
                    logger.error(f"[{self.name}] Error stopping FFmpeg: {e}")
            
            # Wait for thread to finish
            if self.is_alive():
                self.join(timeout=THREAD_CLEANUP_TIMEOUT)
            
            logger.info(f"[{self.name}] Recording stopped")
            return True
        return False

    def run(self):
        """Main recording thread loop"""
        logger.info(f"[{self.name}] Recording thread started")
        
        while self._recording_active and not self._stop_event.is_set():
            try:
                cmd = self._build_ffmpeg_command()
                logger.info(f"[{self.name}] Starting FFmpeg with command: {' '.join(cmd)}")
                
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Monitor the process
                stdout, stderr = self.process.communicate()
                
                if self.process.returncode != 0 and self._recording_active:
                    error_msg = f"FFmpeg exited with code {self.process.returncode}: {stderr}"
                    logger.error(f"[{self.name}] {error_msg}")
                    self.last_error = error_msg
                    
                    if not self._stop_event.is_set():
                        logger.info(f"[{self.name}] Restarting recording in {RTSP_RECONNECT_DELAY}s...")
                        time.sleep(RTSP_RECONNECT_DELAY)
                
            except Exception as e:
                error_msg = f"Recording error: {e}"
                logger.error(f"[{self.name}] {error_msg}")
                self.last_error = error_msg
                
                if not self._stop_event.is_set():
                    time.sleep(RTSP_RECONNECT_DELAY)
        
        logger.info(f"[{self.name}] Recording thread stopped")

    def get_recording_stats(self) -> Dict[str, Any]:
        """Get recording statistics"""
        recording_duration = 0
        if self.recording_start_time:
            recording_duration = (datetime.now() - self.recording_start_time).total_seconds()
        
        return {
            'is_recording': self._recording_active,
            'recording_duration': round(recording_duration, 2),
            'segments_created': self.segments_created,
            'output_directory': self.output_dir,
            'last_error': self.last_error,
            'recording_start_time': self.recording_start_time.isoformat() if self.recording_start_time else None
        }

    def list_recordings(self) -> list:
        """List all recording files for this camera"""
        try:
            if not os.path.exists(self.output_dir):
                return []
            
            recordings = []
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.mp4'):
                    filepath = os.path.join(self.output_dir, filename)
                    stat = os.stat(filepath)
                    recordings.append({
                        'filename': filename,
                        'filepath': filepath,
                        'size_mb': round(stat.st_size / (1024 * 1024), 2),
                        'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            
            # Sort by creation time (newest first)
            recordings.sort(key=lambda x: x['created_time'], reverse=True)
            return recordings
            
        except Exception as e:
            logger.error(f"[{self.name}] Error listing recordings: {e}")
            return []


class EnhancedCameraManager:
    """
    Enhanced Camera Manager
    Manages multiple camera workers and recorders with Flask integration
    """
    
    def __init__(self):
        self.workers: Dict[int, CameraWorker] = {}
        self.recorders: Dict[int, FfmpegRecorder] = {}
        self._manager_lock = threading.Lock()
        logger.info("Enhanced Camera Manager initialized")

    def add_camera(self, camera_data: dict) -> bool:
        """Add a new camera to the manager"""
        camera_id = camera_data['id']
        
        with self._manager_lock:
            if camera_id in self.workers:
                logger.warning(f"Camera {camera_id} already exists")
                return False
            
            try:
                # Create worker thread
                worker = CameraWorker(
                    camera_id=camera_id,
                    name=camera_data['name'],
                    rtsp_url=camera_data['rtsp_url'],
                    username=camera_data.get('username', ''),
                    password=camera_data.get('password', '')
                )
                
                # Create recorder thread (not started yet)
                recorder = FfmpegRecorder(
                    camera_id=camera_id,
                    name=camera_data['name'],
                    rtsp_url=camera_data['rtsp_url'],
                    username=camera_data.get('username', ''),
                    password=camera_data.get('password', '')
                )
                
                self.workers[camera_id] = worker
                self.recorders[camera_id] = recorder
                
                # Start worker if auto_start is enabled
                if camera_data.get('auto_start', True):
                    worker.start()
                
                logger.info(f"Camera {camera_id} ({camera_data['name']}) added to manager")
                return True
                
            except Exception as e:
                logger.error(f"Error adding camera {camera_id}: {e}")
                return False

    def remove_camera(self, camera_id: int) -> bool:
        """Remove a camera from the manager"""
        with self._manager_lock:
            if camera_id not in self.workers:
                return False
            
            try:
                # Stop worker
                if camera_id in self.workers:
                    self.workers[camera_id].stop()
                    del self.workers[camera_id]
                
                # Stop recorder
                if camera_id in self.recorders:
                    if self.recorders[camera_id].is_alive():
                        self.recorders[camera_id].stop_recording()
                    del self.recorders[camera_id]
                
                logger.info(f"Camera {camera_id} removed from manager")
                return True
                
            except Exception as e:
                logger.error(f"Error removing camera {camera_id}: {e}")
                return False

    def get_camera_frame(self, camera_id: int) -> Optional[bytes]:
        """Get latest frame from camera as JPEG bytes - Performance Optimized"""
        if camera_id not in self.workers:
            return None
        
        frame, timestamp = self.workers[camera_id].get_latest_frame()
        if frame is None:
            return None
        
        try:
            # Optimized JPEG encoding settings (based on FFmpeg patterns)
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,        # Enable optimization
                cv2.IMWRITE_JPEG_PROGRESSIVE, 0      # Disable progressive (faster)
            ]
            
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            return buffer.tobytes() if success else None
        except Exception as e:
            logger.error(f"Error encoding frame for camera {camera_id}: {e}")
        
        return None

    def get_camera_status(self, camera_id: int) -> Optional[dict]:
        """Get detailed status for a camera"""
        if camera_id not in self.workers:
            return None
        
        worker_stats = self.workers[camera_id].get_stats()
        recorder_stats = self.recorders[camera_id].get_recording_stats() if camera_id in self.recorders else {}
        
        return {
            **worker_stats,
            'recording': recorder_stats
        }

    def start_recording(self, camera_id: int) -> bool:
        """Start recording for a specific camera"""
        if camera_id not in self.recorders:
            return False
        
        return self.recorders[camera_id].start_recording()

    def stop_recording(self, camera_id: int) -> bool:
        """Stop recording for a specific camera"""
        if camera_id not in self.recorders:
            return False
        
        return self.recorders[camera_id].stop_recording()

    def get_all_camera_statuses(self) -> Dict[int, dict]:
        """Get status for all cameras"""
        statuses = {}
        for camera_id in self.workers:
            statuses[camera_id] = self.get_camera_status(camera_id)
        return statuses

    def list_recordings(self, camera_id: int) -> list:
        """List recordings for a specific camera"""
        if camera_id not in self.recorders:
            return []
        
        return self.recorders[camera_id].list_recordings()

    def generate_video_stream(self, camera_id: int):
        """Generator for video streaming - Performance Optimized"""
        if camera_id not in self.workers:
            logger.warning(f"Camera {camera_id} not found for streaming")
            return
        
        worker = self.workers[camera_id]
        logger.info(f"Starting video stream for camera {camera_id}")
        
        last_frame_time = 0
        min_interval = 1.0 / PROCESSING_FPS
        
        # Optimized JPEG encoding settings (reused for consistency)
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0
        ]
        
        while True:
            try:
                current_time = time.time()
                
                # Throttle based on actual FPS target (prevents overwhelming client)
                if current_time - last_frame_time < min_interval:
                    time.sleep(min_interval - (current_time - last_frame_time))
                    continue
                    
                frame, timestamp = worker.get_latest_frame()
                if frame is not None:
                    # Use optimized encoding settings
                    success, buffer = cv2.imencode('.jpg', frame, encode_params)
                    if success:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
                        last_frame_time = current_time
                
            except Exception as e:
                logger.error(f"Error in video stream for camera {camera_id}: {e}")
                break

    def shutdown(self):
        """Shutdown all cameras and cleanup"""
        logger.info("Shutting down Enhanced Camera Manager...")
        
        with self._manager_lock:
            # Stop all recorders first
            for camera_id, recorder in self.recorders.items():
                if recorder.is_alive():
                    recorder.stop_recording()
            
            # Stop all workers
            for camera_id, worker in self.workers.items():
                worker.stop()
            
            # Clear collections
            self.workers.clear()
            self.recorders.clear()
        
        logger.info("Enhanced Camera Manager shutdown complete")