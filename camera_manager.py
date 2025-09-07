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
from person_detector import detection_manager, DetectionResult

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
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
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
        self.detection_enabled = PERSON_DETECTION_ENABLED
        self.frame_count_for_detection = 0
        self.total_persons_detected = 0
        self.last_detection_count = 0
        self.last_detection_time = None
        
        # Storage throttling tracking
        self.last_storage_time = 0  # Timestamp of last storage save
        
        # Get detector instance from global manager
        if self.detection_enabled:
            detection_manager.get_detector(
                camera_id=self.camera_id,
                model_path=PERSON_DETECTION_MODEL,
                confidence_threshold=PERSON_DETECTION_CONFIDENCE
            )
            detection_manager.enable_detection(self.camera_id, True)
        
        self.daemon = True
        logger.info(f"[{self.name}] Camera worker initialized with detection {'enabled' if self.detection_enabled else 'disabled'}")

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
        if self.detection_enabled:
            detection_stats = detection_manager.get_detection_stats(self.camera_id)
            if detection_stats:
                stats.update({
                    'detection_enabled': True,
                    'total_persons_detected': self.total_persons_detected,
                    'last_detection_count': self.last_detection_count,
                    'last_detection_time': self.last_detection_time,
                    'detection_stats': detection_stats
                })
            else:
                stats['detection_enabled'] = False
        else:
            stats['detection_enabled'] = False
            
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
                
                # Person detection processing (if enabled)
                detections = []
                detection_count = 0
                
                logger.debug(f"[{self.name}] Frame processing - detection_enabled: {self.detection_enabled}")
                
                if (self.detection_enabled and 
                    detection_manager.is_detection_enabled(self.camera_id)):
                    
                    self.frame_count_for_detection += 1
                    logger.debug(f"[{self.name}] Detection frame count: {self.frame_count_for_detection}/{PERSON_DETECTION_INTERVAL}")
                    
                    # Process detection every Nth frame
                    if self.frame_count_for_detection >= PERSON_DETECTION_INTERVAL:
                        logger.info(f"[{self.name}] Starting person detection processing...")
                        
                        try:
                            detector = detection_manager.get_detector(self.camera_id)
                            logger.debug(f"[{self.name}] Got detector, model initialized: {detector.is_initialized}")
                            
                            detections, detection_count = detector.detect_persons(frame)
                            logger.info(f"[{self.name}] Detection completed - found {detection_count} person(s)")
                            
                            if detection_count > 0:
                                self.total_persons_detected += detection_count
                                self.last_detection_count = detection_count
                                self.last_detection_time = self._last_frame_time
                                
                                # Add to detection history
                                detection_manager.add_detection_result(self.camera_id, detections)
                                
                                # Store detections to database and save images
                                self._save_detections_to_storage(frame, detections)
                                
                                logger.info(f"[{self.name}] Detected {detection_count} person(s) - Total: {self.total_persons_detected}")
                            else:
                                logger.debug(f"[{self.name}] No persons detected in this frame")
                            
                            self.frame_count_for_detection = 0
                            
                        except Exception as e:
                            logger.error(f"[{self.name}] Detection error: {e}")
                            import traceback
                            logger.error(f"[{self.name}] Detection error traceback: {traceback.format_exc()}")
                else:
                    if not self.detection_enabled:
                        logger.debug(f"[{self.name}] Detection disabled at camera level")
                    elif not detection_manager.is_detection_enabled(self.camera_id):
                        logger.debug(f"[{self.name}] Detection disabled at manager level")
                
                # Add frame to queue (non-blocking) - optimized buffer management
                try:
                    # Efficient queue management: drop oldest frame if full
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                            self.frames_dropped += 1
                        except queue.Empty:
                            pass
                    
                    # Store frame with detection data
                    frame_data = {
                        'frame': frame.copy(),
                        'timestamp': self._last_frame_time,
                        'detections': detections,
                        'detection_count': detection_count
                    }
                    
                    self.frame_queue.put(frame_data, block=False)
                    
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

    def get_latest_frame(self) -> Tuple[Optional[Any], Optional[datetime], Optional[List], int]:
        """Get the most recent frame from the queue with detection data"""
        try:
            frame_data = self.frame_queue.get_nowait()
            if isinstance(frame_data, dict):
                # New format with detection data
                return (frame_data['frame'], frame_data['timestamp'], 
                       frame_data.get('detections', []), frame_data.get('detection_count', 0))
            else:
                # Backward compatibility for old format (frame, timestamp tuple)
                frame, timestamp = frame_data
                return frame, timestamp, [], 0
        except queue.Empty:
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
            
            # Save full frame image with detections (if enabled)
            full_image_path = None
            if DETECTION_IMAGE_STORAGE_ENABLED and DETECTION_SAVE_FULL_FRAMES:
                try:
                    full_image_path = image_storage.save_full_frame_image(
                        frame=frame,
                        camera_id=self.camera_id,
                        camera_name=self.name,
                        timestamp=self._last_frame_time,
                        detections=detections
                    )
                    if full_image_path:
                        logger.debug(f"[{self.name}] Saved full frame image: {full_image_path}")
                except Exception as e:
                    logger.error(f"[{self.name}] Error saving full frame image: {e}")
            
            # Process each detection
            for detection in detections:
                try:
                    # Save cropped person image (if enabled)
                    person_image_path = None
                    if DETECTION_IMAGE_STORAGE_ENABLED and DETECTION_SAVE_PERSON_CROPS:
                        try:
                            person_image_path = image_storage.save_person_crop(
                                frame=frame,
                                bbox=detection.bbox,
                                camera_id=self.camera_id,
                                camera_name=self.name,
                                timestamp=detection.timestamp,
                                confidence=detection.confidence
                            )
                            if person_image_path:
                                logger.debug(f"[{self.name}] Saved person crop: {person_image_path}")
                        except Exception as e:
                            logger.error(f"[{self.name}] Error saving person crop: {e}")
                    
                    # Update detection object with image paths
                    detection.full_image_path = full_image_path
                    detection.person_image_path = person_image_path
                    
                    # Save to database (if enabled)
                    if DATABASE_ENABLED:
                        try:
                            detection_data = detection.to_database_dict(
                                camera_id=self.camera_id,
                                camera_name=self.name,
                                camera_unique_id=f"camera_{self.camera_id}"
                            )
                            detection_data['rtsp_url'] = self.rtsp_url
                            
                            db_record = db_manager.save_detection(
                                camera_id=self.camera_id,
                                camera_name=self.name,
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
        """Stop the camera worker thread"""
        logger.info(f"[{self.name}] Stopping camera worker...")
        self._stop_event.set()
        
        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clean up detection resources
        if self.detection_enabled:
            detection_manager.cleanup_camera(self.camera_id)
        
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

    def remove_camera(self, camera_id: int) -> bool:
        """Remove a camera from the manager - Legacy method for backward compatibility"""
        return self.remove_camera_graceful(camera_id, THREAD_CLEANUP_TIMEOUT)

    def remove_camera_graceful(self, camera_id: int, timeout: int) -> bool:
        """Remove a camera with graceful shutdown and custom timeout"""
        with self._manager_lock:
            if camera_id not in self.workers:
                return False
            
            try:
                success = True
                
                # Stop worker gracefully
                if camera_id in self.workers:
                    worker = self.workers[camera_id]
                    worker.stop()
                    if worker.is_alive():
                        worker.join(timeout=timeout)
                        if worker.is_alive():
                            logger.warning(f"Worker thread for camera {camera_id} did not stop within {timeout}s")
                            success = False
                        else:
                            logger.info(f"Worker thread for camera {camera_id} stopped gracefully")
                    del self.workers[camera_id]
                
                
                if success:
                    logger.info(f"Camera {camera_id} removed gracefully")
                return success
                
            except Exception as e:
                logger.error(f"Error in graceful removal of camera {camera_id}: {e}")
                return False

    def remove_camera_terminate(self, camera_id: int, timeout: int) -> bool:
        """Remove a camera with process termination and shorter timeout"""
        with self._manager_lock:
            if camera_id not in self.workers:
                return False
            
            try:
                success = True
                
                
                # Stop worker with shorter timeout
                if camera_id in self.workers:
                    worker = self.workers[camera_id]
                    worker.stop()
                    worker.join(timeout=timeout)
                    if worker.is_alive():
                        logger.warning(f"Worker thread for camera {camera_id} still alive after {timeout}s")
                        success = False
                    del self.workers[camera_id]
                
                if success:
                    logger.info(f"Camera {camera_id} removed with termination strategy")
                return success
                
            except Exception as e:
                logger.error(f"Error in termination removal of camera {camera_id}: {e}")
                return False

    def remove_camera_force(self, camera_id: int) -> bool:
        """Force remove a camera immediately - no graceful shutdown"""
        with self._manager_lock:
            if camera_id not in self.workers:
                return False
            
            try:
                
                # Force stop worker thread
                if camera_id in self.workers:
                    worker = self.workers[camera_id]
                    worker._stop_event.set()  # Signal stop immediately
                    # Clear frame queue to unblock any operations
                    while not worker.frame_queue.empty():
                        try:
                            worker.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    # Don't wait for thread - let it die naturally
                    del self.workers[camera_id]
                
                logger.info(f"Camera {camera_id} force removed immediately")
                return True
                
            except Exception as e:
                logger.error(f"Error in force removal of camera {camera_id}: {e}")
                # Even if there's an error, remove from tracking
                if camera_id in self.workers:
                    del self.workers[camera_id]
                return True  # Return success since we removed from tracking

    def get_camera_frame(self, camera_id: int, draw_detections: bool = None) -> Optional[bytes]:
        """Get latest frame from camera as JPEG bytes - Performance Optimized"""
        if camera_id not in self.workers:
            return None
        
        frame, timestamp, detections, detection_count = self.workers[camera_id].get_latest_frame()
        if frame is None:
            return None
        
        try:
            # Draw detection boxes if requested and available
            if draw_detections is None:
                draw_detections = PERSON_DETECTION_DRAW_BOXES
                
            if draw_detections and detections and len(detections) > 0:
                detector = detection_manager.get_detector(camera_id)
                frame = detector.draw_detections(frame, detections)
            
            # Optimized JPEG encoding settings
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
        
        return worker_stats

    def get_all_camera_statuses(self) -> Dict[int, dict]:
        """Get status for all cameras"""
        statuses = {}
        for camera_id in self.workers:
            statuses[camera_id] = self.get_camera_status(camera_id)
        return statuses

    def generate_video_stream(self, camera_id: int, draw_detections: bool = None):
        """Generator for video streaming - Performance Optimized"""
        if camera_id not in self.workers:
            logger.warning(f"Camera {camera_id} not found for streaming")
            return
        
        worker = self.workers[camera_id]
        logger.info(f"Starting video stream for camera {camera_id}")
        
        # Use default detection drawing setting if not specified
        if draw_detections is None:
            draw_detections = PERSON_DETECTION_DRAW_BOXES
        
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
                    
                frame, timestamp, detections, detection_count = worker.get_latest_frame()
                if frame is not None:
                    # Draw detection boxes if requested and available
                    if draw_detections and detections and len(detections) > 0:
                        try:
                            detector = detection_manager.get_detector(camera_id)
                            frame = detector.draw_detections(frame, detections)
                        except Exception as e:
                            logger.error(f"Error drawing detections for camera {camera_id}: {e}")
                    
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
            # Stop all workers
            for camera_id, worker in self.workers.items():
                worker.stop()
            
            # Clear collections
            self.workers.clear()
        
        logger.info("Enhanced Camera Manager shutdown complete")