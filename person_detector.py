"""
Person Detection Module for CCTV Surveillance System
Uses YOLOv8 for real-time person detection in video streams
"""

import cv2
import numpy as np
import logging
import time
import traceback
import inspect
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
        # DIAGNOSTIC LOGGING - Track when and from where PersonDetector is created
        current_time = datetime.now()
        logger.warning(f"🚨 DIAGNOSTIC: PersonDetector.__init__ called at {current_time.strftime('%H:%M:%S.%f')[:-3]}")
        logger.warning(f"🚨 DIAGNOSTIC: Model path: {model_path}, confidence: {confidence_threshold}")

        # Log the call stack to see what triggered this initialization
        try:
            frame = inspect.currentframe()
            call_stack = []
            while frame:
                filename = frame.f_code.co_filename
                function_name = frame.f_code.co_name
                line_number = frame.f_lineno
                call_stack.append(f"{filename}:{line_number} in {function_name}()")
                frame = frame.f_back

            logger.warning("🚨 DIAGNOSTIC: Call stack for PersonDetector creation:")
            for i, call in enumerate(call_stack[:8]):  # Show top 8 stack frames
                logger.warning(f"🚨   [{i}] {call}")

        except Exception as e:
            logger.error(f"Failed to get call stack: {e}")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_initialized = False

        # Performance tracking
        self.total_detections = 0
        self.total_frames_processed = 0
        self.average_inference_time = 0.0
        self.last_detection_time = None
        self.first_inference = True

        logger.info(f"PersonDetector created with model: {model_path}, confidence: {confidence_threshold}")
        logger.info("Model will be loaded lazily on first detection call")
        logger.warning(f"🚨 DIAGNOSTIC: PersonDetector.__init__ completed at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} - NO MODEL LOADING")

    def initialize_model(self) -> bool:
        """
        Initialize the YOLOv8 model
        Returns True if successful, False otherwise
        """
        if self.is_initialized:
            logger.debug("Model already initialized, skipping")
            return True

        # DIAGNOSTIC LOGGING - Track model initialization timing
        start_time = datetime.now()
        logger.warning(f"🚨 DIAGNOSTIC: initialize_model() started at {start_time.strftime('%H:%M:%S.%f')[:-3]}")

        try:
            logger.info(f"Downloading and loading YOLOv8 model: {self.model_path}")
            logger.info("This may take a few moments on first run as the model is downloaded...")
            logger.warning(f"🚨 DIAGNOSTIC: About to load YOLO model from {self.model_path}")
            
            # GPU/Device detection
            if TORCH_AVAILABLE:
                cuda_available = torch.cuda.is_available()
                cuda_device_count = torch.cuda.device_count()
                
                logger.info(f"CUDA Available: {cuda_available}")
                if cuda_available:
                    logger.info(f"CUDA Devices: {cuda_device_count}")
                    for i in range(cuda_device_count):
                        logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
                else:
                    logger.info("Running on CPU - this will be significantly slower")
            else:
                logger.warning("PyTorch not available - cannot detect GPU")
            
            from ultralytics import YOLO
            logger.warning(f"🚨 DIAGNOSTIC: Importing YOLO and creating model object...")
            model_load_start = datetime.now()
            self.model = YOLO(self.model_path)
            model_load_end = datetime.now()
            model_load_duration = (model_load_end - model_load_start).total_seconds()
            logger.warning(f"🚨 DIAGNOSTIC: YOLO model object created in {model_load_duration:.3f} seconds")

            self.is_initialized = True

            logger.info("YOLOv8 model loaded successfully!")
            logger.info(f"Model classes: {len(self.model.names)} (person is class 0)")

            # DIAGNOSTIC LOGGING - Track total initialization time
            total_duration = (datetime.now() - start_time).total_seconds()
            logger.warning(f"🚨 DIAGNOSTIC: initialize_model() completed in {total_duration:.3f} seconds")
            
            # Log model device after loading
            try:
                model_device = getattr(self.model, 'device', 'unknown')
                logger.info(f"Model loaded on device: {model_device}")
            except:
                logger.info("Could not determine model device")
            
            # Model configuration analysis
            try:
                logger.info(f"Model task: {getattr(self.model, 'task', 'unknown')}")
                logger.info(f"Model mode: {getattr(self.model, 'mode', 'unknown')}")
                
                # Try to get model parameters
                if hasattr(self.model, 'model'):
                    model_parameters = sum(p.numel() for p in self.model.model.parameters())
                    logger.info(f"Model parameters: {model_parameters:,}")
            except Exception as e:
                logger.debug(f"Could not determine model details: {e}")
            
            return True
        except Exception as e:
            error_time = datetime.now()
            logger.error(f"Failed to initialize YOLOv8 model: {e}")
            logger.warning(f"🚨 DIAGNOSTIC: Model initialization FAILED at {error_time.strftime('%H:%M:%S.%f')[:-3]}")
            error_duration = (error_time - start_time).total_seconds()
            logger.warning(f"🚨 DIAGNOSTIC: Failed initialization took {error_duration:.3f} seconds")
            import traceback
            logger.error(f"Model initialization traceback: {traceback.format_exc()}")
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
            start_time = time.time()
            
            # Store original frame dimensions
            original_height, original_width = frame.shape[:2]
            logger.debug(f"Original frame dimensions: {original_width}x{original_height}")
            
            # Resize frame for inference if enabled
            if PERSON_DETECTION_RESIZE_ENABLED:
                inference_frame = self._resize_frame_for_inference(frame)
                logger.debug(f"Resized frame dimensions: {PERSON_DETECTION_RESIZE_WIDTH}x{PERSON_DETECTION_RESIZE_HEIGHT}")
            else:
                inference_frame = frame
            
            # First-run detection tracking
            if self.first_inference:
                logger.info("=== FIRST INFERENCE - Expected to be slower ===")
                logger.info("Subsequent inferences should be faster")
            
            # Pre-inference diagnostics
            logger.info(f"=== YOLO Inference Debug Start ===")
            logger.info(f"Frame shape: {inference_frame.shape}")
            logger.info(f"Frame dtype: {inference_frame.dtype}")
            logger.info(f"Frame min/max values: {inference_frame.min()}/{inference_frame.max()}")
            
            # Model device check
            try:
                model_device = getattr(self.model, 'device', 'unknown')
                logger.info(f"Model device: {model_device}")
            except:
                logger.info("Could not determine model device")
            
            # System resource check
            if PSUTIL_AVAILABLE:
                try:
                    memory_info = psutil.virtual_memory()
                    logger.info(f"Available RAM: {memory_info.available / 1024**3:.1f}GB / {memory_info.total / 1024**3:.1f}GB ({memory_info.percent}% used)")
                    logger.info(f"CPU usage: {psutil.cpu_percent()}%")
                except Exception as e:
                    logger.debug(f"Could not get system info: {e}")
            
            # Detailed timing
            inference_start = time.time()
            logger.info(f"Starting YOLO inference at {inference_start}")
            
            # Run inference - class 0 is 'person' in COCO dataset
            results = self.model(inference_frame, classes=[0], verbose=False)
            
            inference_end = time.time()
            inference_time = inference_end - start_time
            inference_duration = inference_end - inference_start
            logger.info(f"YOLO inference completed in {inference_duration:.3f} seconds")
            logger.info(f"=== YOLO Inference Debug End ===")
            
            # Mark first inference as complete
            if self.first_inference:
                self.first_inference = False
                logger.info("=== FIRST INFERENCE COMPLETED ===")
                logger.info("Future inferences should be significantly faster")
            self._update_performance_stats(inference_time)
            
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
            
            self.total_detections += person_count
            self.last_detection_time = datetime.now()
            
            logger.debug(f"Detection completed in {inference_time:.3f}s - found {person_count} person(s)")
            
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

    def get_stats(self) -> Dict:
        """Get detection performance statistics"""
        return {
            'is_initialized': self.is_initialized,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'total_detections': self.total_detections,
            'total_frames_processed': self.total_frames_processed,
            'average_inference_time_ms': round(self.average_inference_time * 1000, 2),
            'last_detection_time': self.last_detection_time.isoformat() if self.last_detection_time else None
        }

    def _update_performance_stats(self, inference_time: float):
        """Update internal performance statistics"""
        self.total_frames_processed += 1
        
        # Calculate running average of inference time
        if self.total_frames_processed == 1:
            self.average_inference_time = inference_time
        else:
            # Exponential moving average with alpha = 0.1
            alpha = 0.1
            self.average_inference_time = (alpha * inference_time + 
                                         (1 - alpha) * self.average_inference_time)

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

        if self.detector:
            stats['detector_stats'] = self.detector.get_stats()

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
        self.detector_cache: Dict[int, PersonDetector] = {}
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

    def get_detector(self, camera_id: int) -> PersonDetector:
        """Get or create detector for camera"""
        if camera_id not in self.detector_cache:
            self.detector_cache[camera_id] = PersonDetector()
            logger.info(f"Created PersonDetector for camera {camera_id} in worker thread")
        return self.detector_cache[camera_id]

    def process_frame(self, frame_data: dict):
        """Process a single frame through YOLO detection"""
        camera_id = frame_data['camera_id']
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']

        try:
            # Get detector for this camera
            detector = self.get_detector(camera_id)

            # Run detection
            detections, person_count = detector.detect_persons(frame)

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