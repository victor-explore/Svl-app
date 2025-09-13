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
                   DETECTION_QUEUE_MAX_SIZE, DETECTION_QUEUE_TIMEOUT)

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

        # Queues for communication
        self.input_queue = queue.Queue(maxsize=50)  # Frames to process
        self.output_queues = {}  # Dict[camera_id, Queue] for results

        # Single PersonDetector instance that will be reused
        self.detector = None
        self.is_running = False
        self.shutdown_event = threading.Event()

        # Detection settings per camera
        self.camera_settings = {}  # Dict[camera_id, dict] for per-camera settings

        logger.info(f"DetectionService initialized with model: {model_path}")

    def run(self):
        """Main detection loop - runs in separate thread"""
        logger.info("DetectionService thread started")
        self.is_running = True

        # Initialize the detector once when thread starts
        logger.info("Initializing YOLO model in DetectionService...")
        self.detector = PersonDetector(self.model_path, self.confidence_threshold)

        # Initialize model immediately
        if not self.detector.initialize_model():
            logger.error("Failed to initialize YOLO model in DetectionService")
            self.is_running = False
            return

        logger.info("YOLO model successfully loaded in DetectionService")

        while self.is_running:
            try:
                # Check for shutdown
                if self.shutdown_event.is_set():
                    break

                # Get frame from queue with timeout
                try:
                    item = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None:  # Shutdown signal
                    break

                camera_id, frame, timestamp = item

                # Check if camera has an output queue
                if camera_id not in self.output_queues:
                    logger.warning(f"No output queue for camera {camera_id}, skipping frame")
                    continue

                # Check if there are newer frames from the same camera in the queue
                # This helps prevent processing stale frames when the queue backs up
                queue_items = []
                newest_item_for_camera = (camera_id, frame, timestamp)

                try:
                    # Peek at remaining items in queue without blocking
                    while not self.input_queue.empty():
                        next_item = self.input_queue.get_nowait()
                        if next_item and next_item[0] == camera_id:
                            # Found a newer frame from same camera, use it instead
                            newest_item_for_camera = next_item
                            logger.debug(f"Skipping stale frame for camera {camera_id}, using newer one")
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
                camera_id, frame, timestamp = newest_item_for_camera

                # Run detection
                try:
                    detections, person_count = self.detector.detect_persons(frame)

                    # Put result in camera's output queue
                    result = {
                        'detections': detections,
                        'person_count': person_count,
                        'timestamp': timestamp,
                        'success': True
                    }

                    # Try to put result, don't block if queue is full
                    try:
                        self.output_queues[camera_id].put_nowait(result)
                    except queue.Full:
                        logger.warning(f"Output queue full for camera {camera_id}, dropping result")

                except Exception as e:
                    logger.error(f"Detection error for camera {camera_id}: {e}")
                    # Send error result
                    try:
                        error_result = {
                            'detections': [],
                            'person_count': 0,
                            'timestamp': timestamp,
                            'success': False,
                            'error': str(e)
                        }
                        self.output_queues[camera_id].put_nowait(error_result)
                    except queue.Full:
                        pass

            except Exception as e:
                logger.error(f"DetectionService loop error: {e}")

        logger.info("DetectionService thread stopped")
        self.is_running = False

    def submit_frame(self, camera_id: int, frame: np.ndarray, timestamp: datetime = None) -> bool:
        """
        Submit a frame for detection processing

        Args:
            camera_id: ID of the camera
            frame: OpenCV frame to process
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
            self.input_queue.put_nowait((camera_id, frame, timestamp))
            return True
        except queue.Full:
            logger.debug(f"Input queue full, dropping frame from camera {camera_id}")
            return False

    def get_result(self, camera_id: int, timeout: float = 0.05) -> Optional[Dict]:
        """
        Get detection result for a camera

        Args:
            camera_id: ID of the camera
            timeout: How long to wait for result

        Returns:
            Detection result dict or None if no result available
        """
        if camera_id not in self.output_queues:
            return None

        try:
            return self.output_queues[camera_id].get(timeout=timeout)
        except queue.Empty:
            return None

    def register_camera(self, camera_id: int, settings: Dict = None):
        """
        Register a camera with the detection service

        Args:
            camera_id: ID of the camera
            settings: Optional detection settings for this camera
        """
        if camera_id not in self.output_queues:
            self.output_queues[camera_id] = queue.Queue(maxsize=10)
            self.camera_settings[camera_id] = settings or {}
            logger.info(f"Registered camera {camera_id} with DetectionService")
        else:
            # Camera already registered, clear any stale results
            try:
                while not self.output_queues[camera_id].empty():
                    self.output_queues[camera_id].get_nowait()
                logger.debug(f"Cleared stale results for re-registered camera {camera_id}")
            except:
                pass

    def unregister_camera(self, camera_id: int):
        """
        Unregister a camera from the detection service

        Args:
            camera_id: ID of the camera
        """
        if camera_id in self.output_queues:
            # Clear any pending results
            try:
                while not self.output_queues[camera_id].empty():
                    self.output_queues[camera_id].get_nowait()
            except:
                pass

            # Clear any pending frames in input queue for this camera
            cleared_count = 0
            temp_items = []
            try:
                while not self.input_queue.empty():
                    item = self.input_queue.get_nowait()
                    if item and len(item) >= 1 and item[0] == camera_id:
                        cleared_count += 1  # Skip frames for this camera
                    else:
                        temp_items.append(item)  # Keep frames for other cameras
            except queue.Empty:
                pass

            # Restore frames for other cameras
            for item in temp_items:
                try:
                    self.input_queue.put_nowait(item)
                except queue.Full:
                    pass

            if cleared_count > 0:
                logger.info(f"Cleared {cleared_count} pending detection frames for camera {camera_id}")

            del self.output_queues[camera_id]

        if camera_id in self.camera_settings:
            del self.camera_settings[camera_id]

        logger.info(f"Unregistered camera {camera_id} from DetectionService")

    def update_camera_settings(self, camera_id: int, settings: Dict):
        """Update detection settings for a camera"""
        if camera_id in self.camera_settings:
            self.camera_settings[camera_id].update(settings)
            logger.info(f"Updated settings for camera {camera_id}: {settings}")

    def get_stats(self) -> Dict:
        """Get detection service statistics"""
        stats = {
            'is_running': self.is_running,
            'model_initialized': self.detector is not None and self.detector.is_initialized,
            'input_queue_size': self.input_queue.qsize(),
            'registered_cameras': list(self.output_queues.keys()),
            'camera_count': len(self.output_queues)
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



class GlobalDetectionQueue:
    """
    Centralized queue for all camera detection frames
    Fixed size queue to prevent memory issues and provide backpressure control
    """

    def __init__(self, max_size: int = DETECTION_QUEUE_MAX_SIZE):
        self.max_size = max_size
        self.queue = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()
        self.accepting_frames = True
        logger.info(f"GlobalDetectionQueue initialized with max_size={max_size}")

    def submit_frame(self, camera_id: int, frame: np.ndarray, timestamp: datetime) -> bool:
        """Submit a frame for detection processing"""
        try:
            if not self.accepting_frames:
                return False

            frame_data = {
                'camera_id': camera_id,
                'frame': frame.copy(),
                'timestamp': timestamp
            }

            self.queue.put(frame_data, block=False)
            logger.debug(f"Frame submitted to detection queue from camera {camera_id}")
            return True

        except queue.Full:
            logger.debug(f"Detection queue full, rejecting frame from camera {camera_id}")
            return False
        except Exception as e:
            logger.error(f"Error submitting frame to detection queue: {e}")
            return False

    def get_frame(self, timeout: float = DETECTION_QUEUE_TIMEOUT) -> Optional[dict]:
        """Get the next frame for processing"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error getting frame from detection queue: {e}")
            return None

    def task_done(self):
        """Mark a queue task as done"""
        self.queue.task_done()

    def is_full(self) -> bool:
        """Check if the queue is full"""
        return self.queue.full()

    def is_empty(self) -> bool:
        """Check if the queue is empty"""
        return self.queue.empty()

    def qsize(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()

    def set_accepting_frames(self, accepting: bool):
        """Set whether to accept new frames"""
        with self.lock:
            self.accepting_frames = accepting
            logger.info(f"Detection queue accepting_frames set to {accepting}")


class DetectionWorkerThread(threading.Thread):
    """
    Background thread that processes frames from the global detection queue
    Handles YOLO inference and result storage
    """

    def __init__(self, detection_queue: GlobalDetectionQueue):
        super().__init__()
        self.detection_queue = detection_queue
        self.shared_detector = None
        self.running = False
        self.daemon = True
        self.name = "DetectionWorker"
        logger.info("DetectionWorkerThread initialized")

    def start_processing(self):
        """Start the detection worker thread"""
        self.running = True
        self.start()
        logger.info("DetectionWorkerThread started")

    def stop_processing(self):
        """Stop the detection worker thread"""
        self.running = False
        logger.info("DetectionWorkerThread stopping...")


    def process_frame(self, frame_data: dict):
        """Process a single frame through YOLO detection"""
        camera_id = frame_data['camera_id']
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']

        try:
            # Initialize shared detector if not already done
            if self.shared_detector is None:
                self.shared_detector = PersonDetector()
                logger.info("Created shared PersonDetector for all cameras")

            # Run detection
            detections, person_count = self.shared_detector.detect_persons(frame)

            if detections:
                logger.info(f"Detected {person_count} person(s) in camera {camera_id}")


                # Save detections to database and storage
                from detection_storage import DetectionImageStorage
                storage = DetectionImageStorage()

                # Save detection images with annotations
                for detection in detections:
                    try:
                        # Draw bounding box on frame copy
                        annotated_frame = frame.copy()
                        x1, y1, x2, y2 = detection.bbox
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Save annotated image
                        storage.save_detection_image(annotated_frame, camera_id, timestamp, detection.confidence)

                    except Exception as e:
                        logger.error(f"Error saving detection image: {e}")

            else:
                logger.debug(f"No persons detected in camera {camera_id}")

        except Exception as e:
            logger.error(f"Error processing frame from camera {camera_id}: {e}")
            logger.error(f"Detection processing error traceback: {traceback.format_exc()}")

    def run(self):
        """Main worker thread loop"""
        logger.info("DetectionWorkerThread started processing")

        while self.running:
            # Get frame from queue
            frame_data = self.detection_queue.get_frame(timeout=1.0)

            if frame_data is None:
                # Timeout - continue loop
                continue

            try:
                # Process the frame
                self.process_frame(frame_data)

            finally:
                # Mark task as done
                self.detection_queue.task_done()

        logger.info("DetectionWorkerThread stopped processing")


# Global detection queue and worker
global_detection_queue = GlobalDetectionQueue()
detection_worker = None

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

def get_detection_queue():
    """Get the global detection queue instance"""
    return global_detection_queue

def start_detection_worker():
    """Start the global detection worker thread"""
    global detection_worker
    if detection_worker is None or not detection_worker.is_alive():
        detection_worker = DetectionWorkerThread(global_detection_queue)
        detection_worker.start_processing()
        logger.info("Global detection worker started")
    return detection_worker

def stop_detection_worker():
    """Stop the global detection worker thread"""
    global detection_worker
    if detection_worker and detection_worker.is_alive():
        detection_worker.stop_processing()
        detection_worker.join(timeout=5)
        logger.info("Global detection worker stopped")