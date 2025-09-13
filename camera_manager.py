"""
Advanced Camera Management System
Combines threading and robust RTSP handling.
Adapted from Shrey svl app with Flask integration enhancements.
"""

import cv2
import threading
import queue
import os
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List
from config import *
from person_detector import DetectionResult, get_detection_service, get_detection_queue

# Import database and image storage only if enabled
if DATABASE_ENABLED:
    from database import db_manager
    
if DETECTION_IMAGE_STORAGE_ENABLED:
    from detection_storage import image_storage

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CameraStatus:
    """Enum-like class for camera status constants"""
    ONLINE = CAMERA_STATUS_ONLINE
    OFFLINE = CAMERA_STATUS_OFFLINE
    CONNECTING = CAMERA_STATUS_CONNECTING
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
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)  # Keep for backward compatibility
        self.display_queue = queue.Queue(maxsize=1)  # Size-1 queue for latest frame only
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()

        # Status tracking
        self._status = CameraStatus.CONNECTING
        self._last_frame_time = None
        self._connection_attempts = 0
        self._last_error = None

        # Performance tracking
        self.frames_captured = 0
        self.frames_dropped = 0
        self.start_time = time.time()

        # Person detection tracking
        self.accepting_detection_frames = PERSON_DETECTION_ENABLED
        self.frame_count_for_detection = 0
        self.total_persons_detected = 0
        self.last_detection_count = 0
        self.last_detection_time = None
        # Storage throttling tracking
        self.last_storage_time = 0  # Timestamp of last storage save

        # OpenCV VideoCapture object for proper cleanup
        self._video_capture = None
        self._capture_lock = threading.Lock()

        self.daemon = True
        logger.info(f"[{self.name}] Camera worker initialized with detection {'enabled' if self.accepting_detection_frames else 'disabled'}")

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
        
        stats = {
            'status': self.status,
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'fps': round(fps, 2),
            'uptime_seconds': round(uptime, 2),
            'last_frame_time': self._last_frame_time,
            'connection_attempts': self._connection_attempts,
            'last_error': self._last_error
        }
        
        # Add detection statistics if detection is enabled
        if self.accepting_detection_frames:
            stats.update({
                'accepting_detection_frames': True,
                'total_persons_detected': self.total_persons_detected,
                'last_detection_count': self.last_detection_count,
                'last_detection_time': self.last_detection_time
            })
        else:
            stats['accepting_detection_frames'] = False
            
        return stats

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
        
        # Store capture object for proper cleanup
        with self._capture_lock:
            self._video_capture = cap
        
        if not cap.isOpened():
            error_msg = f"Could not open RTSP stream (attempt {self._connection_attempts})"
            logger.warning(f"[{self.name}] {error_msg}")
            self.set_status(CameraStatus.OFFLINE, error_msg)
            cap.release()
            time.sleep(min(RTSP_RECONNECT_DELAY * self._connection_attempts, RTSP_RECONNECT_DELAY_MAX))
            return

        logger.info(f"[{self.name}] Successfully connected to RTSP stream")
        # Don't set ONLINE yet - wait for first frame
        self._connection_attempts = 0  # Reset on successful connection

        consecutive_failures = 0
        max_consecutive_failures = 10
        first_frame_captured = False
        
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
                
                # Only set ONLINE when first frame is successfully captured
                if not first_frame_captured:
                    logger.info(f"[{self.name}] First frame captured, camera is now ONLINE")
                    self.set_status(CameraStatus.ONLINE)
                    first_frame_captured = True
                
                consecutive_failures = 0
                self.frames_captured += 1
                self._last_frame_time = datetime.now()
                
                # Simple queue-based detection submission
                detections = []
                detection_count = 0

                # Submit frame to global detection queue if enabled
                if self.accepting_detection_frames:
                    self.frame_count_for_detection += 1

                    # Submit frame every Nth frame to control detection frequency
                    if self.frame_count_for_detection >= PERSON_DETECTION_INTERVAL:
                        detection_queue = get_detection_queue()

                        # Try to submit frame to global queue
                        submitted = detection_queue.submit_frame(self.camera_id, frame, self._last_frame_time)

                        if submitted:
                            self.frame_count_for_detection = 0
                            logger.debug(f"[{self.name}] Frame submitted to global detection queue")
                        else:
                            # Queue is full - keep counting frames until next interval
                            logger.debug(f"[{self.name}] Detection queue full, will retry next interval")
                else:
                    logger.debug(f"[{self.name}] Not accepting detection frames")
                
                # Add frame to display queue (immediate replacement for latest frame)
                try:
                    # Clear display queue to ensure only latest frame
                    while not self.display_queue.empty():
                        try:
                            self.display_queue.get_nowait()
                        except queue.Empty:
                            break

                    # Store frame data for camera feed display
                    display_frame_data = {
                        'frame': frame.copy(),
                        'timestamp': self._last_frame_time,
                        'detections': [],  # Detection results come from background worker
                        'detection_count': 0  # Will be updated by background worker
                    }

                    # Add latest frame to display queue (should never block with size=1)
                    self.display_queue.put(display_frame_data, block=False)

                except queue.Full:
                    # This should never happen with size=1 queue, but handle gracefully
                    self.frames_dropped += 1
                    logger.debug(f"[{self.name}] Display queue unexpectedly full")

                # Keep backward compatibility: also add to frame_queue for any legacy usage
                try:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                            self.frames_dropped += 1
                        except queue.Empty:
                            pass
                    self.frame_queue.put(display_frame_data.copy(), block=False)
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
        
        # Clear stored capture object
        with self._capture_lock:
            self._video_capture = None
        
        if not self._stop_event.is_set():
            # Connection lost, will retry
            self.set_status(CameraStatus.OFFLINE, "Connection lost")
            time.sleep(RTSP_RECONNECT_DELAY)

    def get_latest_frame(self) -> Tuple[Optional[Any], Optional[datetime], Optional[List], int]:
        """Get the most recent frame from the display queue (always latest frame)"""
        try:
            # Get from display queue for guaranteed latest frame
            frame_data = self.display_queue.get_nowait()
            if isinstance(frame_data, dict):
                # New format with detection data
                return (frame_data['frame'], frame_data['timestamp'],
                       frame_data.get('detections', []), frame_data.get('detection_count', 0))
            else:
                # Backward compatibility for old format (frame, timestamp tuple)
                frame, timestamp = frame_data
                return frame, timestamp, [], 0
        except queue.Empty:
            # No frame available in display queue
            return None, None, [], 0
    
    def _save_detections_to_storage(self, frame, detections):
        """Save detection data to database and images to disk with time-based throttling"""
        if not detections:
            return
        
        # Time-based throttling check
        if DETECTION_STORAGE_THROTTLING_ENABLED:
            current_time = time.time()
            time_since_last_storage = current_time - self.last_storage_time
            
            if time_since_last_storage < DETECTION_STORAGE_INTERVAL_SECONDS:
                logger.debug(f"[{self.name}] Storage throttled - {time_since_last_storage:.1f}s since last save (need {DETECTION_STORAGE_INTERVAL_SECONDS}s)")
                return  # Skip storage, not enough time has passed
            
            # Update last storage time
            self.last_storage_time = current_time
            logger.info(f"[{self.name}] Storage interval reached ({DETECTION_STORAGE_INTERVAL_SECONDS}s) - saving detection data")
        
        try:
            # Initialize database if enabled
            if DATABASE_ENABLED and hasattr(db_manager, 'initialize') and not db_manager._initialized:
                db_manager.initialize()
            
            # Save detection image with annotations (if enabled)
            image_path = None
            if DETECTION_IMAGE_STORAGE_ENABLED:
                try:
                    image_path = image_storage.save_full_frame_image(
                        frame=frame,
                        camera_id=self.camera_id,
                        camera_name=self.name,
                        timestamp=self._last_frame_time,
                        detections=detections
                    )
                    if image_path:
                        logger.debug(f"[{self.name}] Saved detection image: {image_path}")
                except Exception as e:
                    logger.error(f"[{self.name}] Error saving detection image: {e}")
            
            # Process each detection
            for detection in detections:
                try:
                    # Update detection object with image path
                    detection.image_path = image_path
                    
                    # Save to database (if enabled)
                    if DATABASE_ENABLED:
                        try:
                            detection_data = detection.to_database_dict(
                                camera_id=self.camera_id
                            )
                            detection_data['rtsp_url'] = self.rtsp_url
                            detection_data['camera_name'] = self.name
                            
                            db_record = db_manager.save_detection(
                                camera_id=self.camera_id,
                                detection_data=detection_data
                            )
                            
                            if db_record:
                                detection.person_id = db_record.person_id
                                logger.debug(f"[{self.name}] Saved detection to database: {db_record.person_id}")
                        except Exception as e:
                            logger.error(f"[{self.name}] Error saving detection to database: {e}")
                
                except Exception as e:
                    logger.error(f"[{self.name}] Error processing individual detection: {e}")
            
            logger.info(f"[{self.name}] Successfully processed {len(detections)} detection(s) for storage")
            
        except Exception as e:
            logger.error(f"[{self.name}] Error in detection storage process: {e}")
            import traceback
            logger.error(f"[{self.name}] Storage error traceback: {traceback.format_exc()}")

    def stop(self):
        """Stop the camera worker thread with proper resource cleanup"""
        logger.info(f"[{self.name}] Stopping camera worker...")
        self._stop_event.set()
        self.detection_pending = False  # Clear pending detection flag
        
        # Force release VideoCapture if still active
        with self._capture_lock:
            if self._video_capture is not None:
                try:
                    logger.info(f"[{self.name}] Force releasing VideoCapture...")
                    self._video_capture.release()
                    self._video_capture = None
                except Exception as e:
                    logger.error(f"[{self.name}] Error releasing VideoCapture: {e}")
        
        # Clear both queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        while not self.display_queue.empty():
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                break
        
        # Note: No cleanup needed for global detection queue
        # The queue-based architecture handles cleanup automatically
        
        # Wait for thread to finish
        if self.is_alive():
            self.join(timeout=THREAD_CLEANUP_TIMEOUT)
            if self.is_alive():
                logger.warning(f"[{self.name}] Thread did not stop gracefully")



class EnhancedCameraManager:
    """
    Enhanced Camera Manager
    Manages multiple camera workers with Flask integration
    """

    def __init__(self):
        self.workers: Dict[int, CameraWorker] = {}
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
                
                self.workers[camera_id] = worker
                
                # Start worker if auto_start is enabled
                if camera_data.get('auto_start', True):
                    worker.start()
                
                logger.info(f"Camera {camera_id} ({camera_data['name']}) added to manager")
                return True
                
            except Exception as e:
                logger.error(f"Error adding camera {camera_id}: {e}")
                return False

    def remove_camera_simple(self, camera_id: int) -> bool:
        """Simple camera removal - just the essentials (no YOLO cleanup needed)"""
        with self._manager_lock:
            if camera_id not in self.workers:
                logger.warning(f"Camera {camera_id} not found for removal")
                return False

            try:
                worker = self.workers[camera_id]

                # 1. Stop worker thread and release VideoCapture
                logger.info(f"Stopping camera worker for {camera_id}")
                worker.stop()  # This handles VideoCapture.release() - the critical part

                # 2. Wait reasonable time for cleanup (single timeout, no progressive complexity)
                if worker.is_alive():
                    logger.info(f"Waiting up to 3 seconds for camera {camera_id} thread to stop")
                    worker.join(timeout=3)  # Single 3-second timeout

                    if worker.is_alive():
                        logger.warning(f"Camera {camera_id} thread did not stop within 3 seconds, continuing anyway")

                # 3. Remove from tracking (even if thread didn't die)
                del self.workers[camera_id]

                logger.info(f"Camera {camera_id} removed using simple cleanup")
                return True

            except Exception as e:
                logger.error(f"Error in simple removal of camera {camera_id}: {e}")
                # Even on error, remove from tracking to prevent stuck state
                if camera_id in self.workers:
                    del self.workers[camera_id]
                return False

    def remove_camera(self, camera_id: int) -> bool:
        """Remove a camera from the manager - Legacy method for backward compatibility"""
        return self.remove_camera_simple(camera_id)

    def get_camera_frame(self, camera_id: int) -> Optional[bytes]:
        """Get latest frame from camera as JPEG bytes - Performance Optimized"""
        if camera_id not in self.workers:
            return None

        frame, timestamp, detections, detection_count = self.workers[camera_id].get_latest_frame()
        if frame is None:
            return None

        try:
            # Enhanced JPEG encoding settings for better performance
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,        # Enable optimization
                cv2.IMWRITE_JPEG_PROGRESSIVE, 0,     # Disable progressive (faster)
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422
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
        
        return worker_stats

    def get_all_camera_statuses(self) -> Dict[int, dict]:
        """Get status for all cameras"""
        statuses = {}
        for camera_id in self.workers:
            statuses[camera_id] = self.get_camera_status(camera_id)
        return statuses

    def generate_video_stream(self, camera_id: int):
        """Generator for video streaming - Performance Optimized (No FPS Throttling)"""
        if camera_id not in self.workers:
            logger.warning(f"Camera {camera_id} not found for streaming")
            return

        worker = self.workers[camera_id]
        logger.info(f"Starting video stream for camera {camera_id}")

        # Enhanced JPEG encoding settings for better performance
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422
        ]

        while True:
            try:
                frame, timestamp, detections, detection_count = worker.get_latest_frame()
                if frame is not None:
                    # Use optimized encoding settings
                    success, buffer = cv2.imencode('.jpg', frame, encode_params)
                    if success:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' +
                               buffer.tobytes() + b'\r\n')

            except Exception as e:
                logger.error(f"Error in video stream for camera {camera_id}: {e}")
                break

    def shutdown(self):
        """Shutdown all cameras and cleanup"""
        logger.info("Shutting down Enhanced Camera Manager...")
        
        with self._manager_lock:
            # Stop all workers
            for camera_id, worker in self.workers.items():
                worker.stop()
            
            # Clear collections
            self.workers.clear()
        
        logger.info("Enhanced Camera Manager shutdown complete")