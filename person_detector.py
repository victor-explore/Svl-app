"""
Person Detection Module for CCTV Surveillance System
Uses YOLOv8 for real-time person detection in video streams
"""

import cv2
import numpy as np
import logging
import time
import traceback
import threading
import queue
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from config import (PERSON_DETECTION_RESIZE_ENABLED, PERSON_DETECTION_RESIZE_WIDTH,
                   PERSON_DETECTION_RESIZE_HEIGHT, PERSON_DETECTION_MAINTAIN_ASPECT,
                   DATABASE_ENABLED, DETECTION_IMAGE_STORAGE_ENABLED,
                   DETECTION_STORAGE_THROTTLING_ENABLED, DETECTION_STORAGE_INTERVAL_SECONDS,
                   DETECTION_PAUSE_ON_USER_ACTION, DETECTION_AUTO_RESUME_SECONDS,
                   DETECTION_PAUSE_MIN_DURATION, DETECTION_PAUSE_DRAIN_QUEUE)

# Import database and image storage only if enabled
if DATABASE_ENABLED:
    from database import db_manager

if DETECTION_IMAGE_STORAGE_ENABLED:
    from detection_storage import image_storage

# Import additional modules for diagnostics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class DetectionResult:
    """Represents a single person detection result"""
    
    def __init__(self, bbox: List[float], confidence: float, timestamp: datetime,
                 person_id: str = None, frame_width: int = None, frame_height: int = None):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.timestamp = timestamp
        self.person_id = person_id  # Unique identifier for this detection
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Storage paths (set after saving to disk)
        self.image_path = None
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            'bbox': self.bbox,
            'confidence': round(self.confidence, 3),
            'timestamp': self.timestamp.isoformat(),
            'person_id': self.person_id,
            'frame_dimensions': [self.frame_width, self.frame_height] if self.frame_width else None,
            'image_path': self.image_path
        }
    
    def to_database_dict(self, camera_id: int) -> Dict:
        """Convert to dictionary format for database storage"""
        return {
            'confidence': self.confidence,
            'bbox': self.bbox,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'image_path': self.image_path,
        }

class PersonDetector:
    """
    Person detection using YOLOv8 model
    Optimized for real-time video stream processing
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize the person detector

        Args:
            model_path: Path to YOLOv8 model file
            confidence_threshold: Minimum confidence for valid detection
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_initialized = False

        logger.info(f"PersonDetector created with model: {model_path}, confidence: {confidence_threshold}")
        logger.info("Model will be loaded lazily on first detection call")

    def initialize_model(self) -> bool:
        """
        Initialize the YOLOv8 model
        Returns True if successful, False otherwise
        """
        if self.is_initialized:
            return True

        try:
            logger.info(f"Loading YOLOv8 model: {self.model_path}")

            # GPU/Device detection
            if TORCH_AVAILABLE:
                cuda_available = torch.cuda.is_available()
                logger.info(f"CUDA Available: {cuda_available}")
                if not cuda_available:
                    logger.info("Running on CPU - this will be significantly slower")

            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.is_initialized = True

            logger.info("YOLOv8 model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8 model: {e}")
            self.is_initialized = False
            return False

    def detect_persons(self, frame: np.ndarray) -> Tuple[List[DetectionResult], int]:
        """
        Detect persons in a video frame
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            Tuple of (list of DetectionResult objects, total person count)
        """
        if not self.is_initialized and not self.initialize_model():
            return [], 0
            
        try:
            # Store original frame dimensions
            original_height, original_width = frame.shape[:2]

            # Resize frame for inference if enabled
            if PERSON_DETECTION_RESIZE_ENABLED:
                inference_frame = self._resize_frame_for_inference(frame)
            else:
                inference_frame = frame

            # Run inference - class 0 is 'person' in COCO dataset
            results = self.model(inference_frame, classes=[0], verbose=False)
            
            detections = []
            person_count = 0
            
            if results and len(results) > 0:
                result = results[0]  # First (and only) result
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
                    
                    current_time = datetime.now()
                    
                    for i in range(len(boxes)):
                        confidence = float(confidences[i])
                        
                        # Filter by confidence threshold
                        if confidence >= self.confidence_threshold:
                            bbox = boxes[i].tolist()  # [x1, y1, x2, y2]
                            
                            # Scale bounding boxes back to original frame dimensions
                            if PERSON_DETECTION_RESIZE_ENABLED:
                                bbox = self._scale_bbox_to_original(bbox, original_width, original_height)
                            
                            detection = DetectionResult(
                                bbox=bbox, 
                                confidence=confidence, 
                                timestamp=current_time,
                                frame_width=original_width,
                                frame_height=original_height
                            )
                            detections.append(detection)
                            person_count += 1
            
            # Only log when persons are actually detected to reduce noise
            if person_count > 0:
                logger.debug(f"Found {person_count} person(s)")

            return detections, person_count
            
        except Exception as e:
            logger.error(f"Error during person detection: {e}")
            return [], 0


    def _resize_frame_for_inference(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame for optimal YOLO inference performance"""
        target_size = (PERSON_DETECTION_RESIZE_WIDTH, PERSON_DETECTION_RESIZE_HEIGHT)
        
        if PERSON_DETECTION_MAINTAIN_ASPECT:
            return self._resize_with_padding(frame, target_size)
        else:
            return cv2.resize(frame, target_size)

    def _resize_with_padding(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frame maintaining aspect ratio with padding"""
        target_width, target_height = target_size
        h, w = frame.shape[:2]
        
        # Calculate scale to fit the frame within target size
        scale = min(target_width / w, target_height / h)
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Create target canvas and center the resized frame
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        # Store padding info for bbox scaling
        self._padding_info = {
            'scale': scale,
            'x_offset': x_offset,
            'y_offset': y_offset
        }
        
        return canvas

    def _scale_bbox_to_original(self, bbox: List[float], orig_width: int, orig_height: int) -> List[float]:
        """Scale bounding box coordinates back to original frame dimensions"""
        x1, y1, x2, y2 = bbox
        
        if PERSON_DETECTION_MAINTAIN_ASPECT and hasattr(self, '_padding_info'):
            # Remove padding offset and scale back
            padding = self._padding_info
            x1 = (x1 - padding['x_offset']) / padding['scale']
            y1 = (y1 - padding['y_offset']) / padding['scale']
            x2 = (x2 - padding['x_offset']) / padding['scale']
            y2 = (y2 - padding['y_offset']) / padding['scale']
        else:
            # Simple scaling without aspect ratio preservation
            width_scale = orig_width / PERSON_DETECTION_RESIZE_WIDTH
            height_scale = orig_height / PERSON_DETECTION_RESIZE_HEIGHT
            x1 *= width_scale
            y1 *= height_scale
            x2 *= width_scale
            y2 *= height_scale
        
        return [x1, y1, x2, y2]


    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold updated to {self.confidence_threshold}")


class DetectionService(threading.Thread):
    """
    Single detection thread that processes frames from all cameras.
    Maintains one YOLO model instance that never gets destroyed.
    """

    def __init__(self, model_path: str = './yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize the detection service thread

        Args:
            model_path: Path to YOLOv8 model file
            confidence_threshold: Minimum confidence for valid detection
        """
        super().__init__(daemon=True)
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        # Input queue for frames to process
        self.input_queue = queue.Queue(maxsize=50)

        # Single PersonDetector instance that will be reused
        self.detector = None
        self.is_running = False
        self.shutdown_event = threading.Event()

        # Storage throttling per camera
        self.camera_last_storage_time = {}  # Dict[camera_id, timestamp]

        # Pause mechanism for UX improvements
        self.is_paused = False
        self.pause_until = None  # Timestamp when auto-resume should happen
        self.pause_reason = None  # String describing why detection was paused
        self.pause_lock = threading.Lock()

        logger.info(f"DetectionService initialized with model: {model_path}")
        logger.info("Simplified architecture: Direct storage, no output queues")
        if DETECTION_PAUSE_ON_USER_ACTION:
            logger.info(f"Detection pause enabled: auto-resume after {DETECTION_AUTO_RESUME_SECONDS}s")

    def run(self):
        """Main detection loop - runs in separate thread"""
        logger.info("DetectionService thread started")
        self.is_running = True

        # Initialize the detector once when thread starts - SINGLE MODEL FOR ALL CAMERAS
        logger.info("========================================")
        logger.info("Initializing SINGLE YOLO model in DetectionService...")
        logger.info("This ONE model will be shared by ALL camera streams")
        logger.info("========================================")

        self.detector = PersonDetector(self.model_path, self.confidence_threshold)

        # Initialize model immediately
        if not self.detector.initialize_model():
            logger.error("Failed to initialize YOLO model in DetectionService")
            self.is_running = False
            return

        logger.info("========================================")
        logger.info("SUCCESS: Single YOLO model loaded and ready!")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info("All cameras will share this single model instance")
        logger.info("========================================")

        while self.is_running:
            try:
                # Check for shutdown
                if self.shutdown_event.is_set():
                    break

                # Check if detection is paused
                with self.pause_lock:
                    if self.is_paused:
                        # Check if auto-resume time has passed
                        if self.pause_until and time.time() >= self.pause_until:
                            logger.info(f"Auto-resuming detection after pause (reason: {self.pause_reason})")
                            self.is_paused = False
                            self.pause_until = None
                            self.pause_reason = None
                        else:
                            # Still paused - drain queue if configured
                            if DETECTION_PAUSE_DRAIN_QUEUE:
                                try:
                                    # Discard frames while paused to prevent stale frame buildup
                                    self.input_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            time.sleep(0.1)
                            continue

                # Get frame from queue with timeout
                try:
                    item = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None:  # Shutdown signal
                    break

                camera_id, frame, timestamp, camera_name, rtsp_url = item

                # Check if there are newer frames from the same camera in the queue
                # This helps prevent processing stale frames when the queue backs up
                queue_items = []
                newest_item_for_camera = (camera_id, frame, timestamp, camera_name, rtsp_url)

                try:
                    # Peek at remaining items in queue without blocking
                    while not self.input_queue.empty():
                        next_item = self.input_queue.get_nowait()
                        if next_item and next_item[0] == camera_id:
                            # Found a newer frame from same camera, use it instead
                            newest_item_for_camera = next_item
                            # Don't log frame skipping to reduce noise - this is normal operation
                        else:
                            # Different camera or shutdown signal, keep it for later
                            queue_items.append(next_item)
                except queue.Empty:
                    pass

                # Put back items from other cameras
                for item_to_restore in queue_items:
                    try:
                        self.input_queue.put_nowait(item_to_restore)
                    except queue.Full:
                        logger.warning("Could not restore item to queue")

                # Use the newest frame for this camera
                camera_id, frame, timestamp, camera_name, rtsp_url = newest_item_for_camera

                # Run detection
                try:
                    detections, person_count = self.detector.detect_persons(frame)

                    # Save detections directly if found
                    if detections:
                        self._save_detections_to_storage(
                            frame, detections, camera_id, camera_name, rtsp_url, timestamp
                        )
                        logger.info(f"[{camera_name}] Detected {len(detections)} person(s) - saved to storage")

                except Exception as e:
                    logger.error(f"Detection error for camera {camera_id}: {e}")

            except Exception as e:
                logger.error(f"DetectionService loop error: {e}")

        logger.info("DetectionService thread stopped")
        self.is_running = False

    def submit_frame(self, camera_id: int, frame: np.ndarray, camera_name: str, rtsp_url: str, timestamp: datetime = None) -> bool:
        """
        Submit a frame for detection processing

        Args:
            camera_id: ID of the camera
            frame: OpenCV frame to process
            camera_name: Name of the camera
            rtsp_url: RTSP URL of the camera
            timestamp: Timestamp of the frame (optional)

        Returns:
            True if frame was submitted, False if queue is full
        """
        if not self.is_running:
            logger.warning("DetectionService is not running")
            return False

        if timestamp is None:
            timestamp = datetime.now()

        try:
            self.input_queue.put_nowait((camera_id, frame, timestamp, camera_name, rtsp_url))
            return True
        except queue.Full:
            logger.debug(f"Input queue full, dropping frame from camera {camera_id}")
            return False

    def _save_detections_to_storage(self, frame, detections, camera_id: int, camera_name: str, rtsp_url: str, timestamp: datetime):
        """Save detection data to database and images to disk with time-based throttling"""
        if not detections:
            return

        # Time-based throttling check per camera
        if DETECTION_STORAGE_THROTTLING_ENABLED:
            current_time = time.time()
            last_storage_time = self.camera_last_storage_time.get(camera_id, 0)
            time_since_last_storage = current_time - last_storage_time

            if time_since_last_storage < DETECTION_STORAGE_INTERVAL_SECONDS:
                logger.debug(f"[{camera_name}] Storage throttled - {time_since_last_storage:.1f}s since last save (need {DETECTION_STORAGE_INTERVAL_SECONDS}s)")
                return  # Skip storage, not enough time has passed

            # Update last storage time
            self.camera_last_storage_time[camera_id] = current_time
            logger.info(f"[{camera_name}] Storage interval reached ({DETECTION_STORAGE_INTERVAL_SECONDS}s) - saving detection data")

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
                        camera_id=camera_id,
                        camera_name=camera_name,
                        timestamp=timestamp,
                        detections=detections
                    )
                    if image_path:
                        logger.debug(f"[{camera_name}] Saved detection image: {image_path}")
                except Exception as e:
                    logger.error(f"[{camera_name}] Error saving detection image: {e}")

            # Process each detection
            for detection in detections:
                try:
                    # Update detection object with image path
                    detection.image_path = image_path

                    # Save to database (if enabled)
                    if DATABASE_ENABLED:
                        try:
                            detection_data = detection.to_database_dict(
                                camera_id=camera_id
                            )
                            detection_data['rtsp_url'] = rtsp_url
                            detection_data['camera_name'] = camera_name

                            db_record = db_manager.save_detection(
                                camera_id=camera_id,
                                detection_data=detection_data
                            )

                            if db_record:
                                detection.person_id = db_record.person_id
                                logger.debug(f"[{camera_name}] Saved detection to database: {db_record.person_id}")
                        except Exception as e:
                            logger.error(f"[{camera_name}] Error saving detection to database: {e}")

                except Exception as e:
                    logger.error(f"[{camera_name}] Error processing individual detection: {e}")

            logger.info(f"[{camera_name}] Successfully processed {len(detections)} detection(s) for storage")

        except Exception as e:
            logger.error(f"[{camera_name}] Error in detection storage process: {e}")
            import traceback
            logger.error(f"[{camera_name}] Storage error traceback: {traceback.format_exc()}")

    # Output queue methods removed - DetectionService now handles storage directly

    def cleanup_camera_state(self, camera_id: int):
        """Clean up storage throttling state for removed camera"""
        if camera_id in self.camera_last_storage_time:
            del self.camera_last_storage_time[camera_id]
            logger.info(f"Cleaned storage throttling state for removed camera {camera_id}")
        else:
            logger.debug(f"No storage state to clean for camera {camera_id}")

    def pause_detection(self, reason: str = "user_action", duration_seconds: Optional[int] = None) -> Dict:
        """
        Pause detection processing to improve UX during user interactions

        Args:
            reason: Reason for pausing (e.g., "re-id_search", "page_navigation", "filter_apply")
            duration_seconds: How long to pause (None = use default auto-resume time)

        Returns:
            Dict with pause status and info
        """
        if not DETECTION_PAUSE_ON_USER_ACTION:
            return {
                'paused': False,
                'reason': 'pause_disabled_in_config'
            }

        with self.pause_lock:
            # Apply minimum pause duration
            if duration_seconds is None:
                duration_seconds = DETECTION_AUTO_RESUME_SECONDS
            else:
                duration_seconds = max(duration_seconds, DETECTION_PAUSE_MIN_DURATION)

            self.is_paused = True
            self.pause_until = time.time() + duration_seconds
            self.pause_reason = reason

            logger.info(f"Detection paused for {duration_seconds}s (reason: {reason})")

            return {
                'paused': True,
                'reason': reason,
                'pause_until': self.pause_until,
                'duration_seconds': duration_seconds
            }

    def resume_detection(self) -> Dict:
        """
        Manually resume detection processing

        Returns:
            Dict with resume status
        """
        with self.pause_lock:
            was_paused = self.is_paused
            previous_reason = self.pause_reason

            self.is_paused = False
            self.pause_until = None
            self.pause_reason = None

            if was_paused:
                logger.info(f"Detection manually resumed (was paused for: {previous_reason})")

            return {
                'resumed': was_paused,
                'was_paused_for': previous_reason
            }

    def get_pause_status(self) -> Dict:
        """
        Get current pause status

        Returns:
            Dict with pause state information
        """
        with self.pause_lock:
            if self.is_paused and self.pause_until:
                time_remaining = max(0, self.pause_until - time.time())
            else:
                time_remaining = 0

            return {
                'is_paused': self.is_paused,
                'reason': self.pause_reason,
                'time_remaining_seconds': time_remaining,
                'pause_until': self.pause_until
            }

    def get_stats(self) -> Dict:
        """Get detection service statistics"""
        pause_status = self.get_pause_status()

        stats = {
            'is_running': self.is_running,
            'model_initialized': self.detector is not None and self.detector.is_initialized,
            'input_queue_size': self.input_queue.qsize(),
            'is_paused': pause_status['is_paused'],
            'pause_reason': pause_status['reason'],
            'pause_time_remaining': pause_status['time_remaining_seconds']
        }

        return stats

    def shutdown(self):
        """Shutdown the detection service"""
        logger.info("Shutting down DetectionService...")
        self.shutdown_event.set()
        self.is_running = False

        # Put None to signal shutdown
        try:
            self.input_queue.put(None, timeout=1)
        except:
            pass

        # Wait for thread to stop
        if self.is_alive():
            self.join(timeout=5)

        logger.info("DetectionService shutdown complete")



# DetectionWorkerThread and GlobalDetectionQueue have been removed
# Using only DetectionService for single YOLO model shared across all cameras

# Global detection service instance (created but not started yet)
# Will be started by app.py at application startup
detection_service = None


def get_detection_service():
    """Get the global detection service instance"""
    return detection_service

def set_detection_service(service):
    """Set the global detection service instance"""
    global detection_service
    detection_service = service
    return detection_service

# Functions for DetectionWorkerThread removed - using DetectionService instead