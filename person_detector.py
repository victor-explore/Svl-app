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
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from config import (PERSON_DETECTION_RESIZE_ENABLED, PERSON_DETECTION_RESIZE_WIDTH,
                   PERSON_DETECTION_RESIZE_HEIGHT, PERSON_DETECTION_MAINTAIN_ASPECT)

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

    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: OpenCV frame
            detections: List of detection results
            
        Returns:
            Frame with drawn bounding boxes
        """
        frame_with_boxes = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            confidence = detection.confidence
            
            # Draw bounding box
            color = (0, 255, 0)  # Green color
            thickness = 2
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence label
            label = f"Person {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame_with_boxes, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(frame_with_boxes, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame_with_boxes

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

class PersonDetectionManager:
    """
    Manages person detection for multiple cameras
    Provides centralized configuration and statistics
    """
    
    def __init__(self):
        self.detectors: Dict[int, PersonDetector] = {}
        self.detection_enabled: Dict[int, bool] = {}
        self.detection_histories: Dict[int, List[DetectionResult]] = {}
        self.max_history_size = 100  # Keep last 100 detections per camera
        
        logger.info("PersonDetectionManager initialized")

    def get_detector(self, camera_id: int, **kwargs) -> PersonDetector:
        """Get or create detector for camera"""

        # DIAGNOSTIC LOGGING - Track when detector is requested
        current_time = datetime.now()
        logger.warning(f"🚨 DIAGNOSTIC: get_detector() called for camera {camera_id} at {current_time.strftime('%H:%M:%S.%f')[:-3]}")
        logger.warning(f"🚨 DIAGNOSTIC: Current detectors: {list(self.detectors.keys())}")
        logger.warning(f"🚨 DIAGNOSTIC: Kwargs: {kwargs}")

        # Log call stack for detector requests
        try:
            frame = inspect.currentframe()
            call_stack = []
            for i in range(5):  # Show top 5 stack frames
                if frame:
                    filename = frame.f_code.co_filename.split('\\')[-1]  # Just filename
                    function_name = frame.f_code.co_name
                    line_number = frame.f_lineno
                    call_stack.append(f"{filename}:{line_number} in {function_name}()")
                    frame = frame.f_back

            logger.warning("🚨 DIAGNOSTIC: Call stack for get_detector():")
            for i, call in enumerate(call_stack):
                logger.warning(f"🚨   [{i}] {call}")
        except Exception as e:
            logger.error(f"Failed to get call stack: {e}")

        if camera_id not in self.detectors:
            logger.warning(f"🚨 DIAGNOSTIC: Creating NEW PersonDetector for camera {camera_id}")
            self.detectors[camera_id] = PersonDetector(**kwargs)
            self.detection_enabled[camera_id] = True
            self.detection_histories[camera_id] = []
            logger.warning(f"🚨 DIAGNOSTIC: PersonDetector created for camera {camera_id}")
        else:
            logger.warning(f"🚨 DIAGNOSTIC: Returning EXISTING PersonDetector for camera {camera_id}")

        return self.detectors[camera_id]

    def enable_detection(self, camera_id: int, enabled: bool = True):
        """Enable/disable detection for a camera"""
        self.detection_enabled[camera_id] = enabled
        logger.info(f"Camera {camera_id} detection {'enabled' if enabled else 'disabled'}")

    def is_detection_enabled(self, camera_id: int) -> bool:
        """Check if detection is enabled for camera"""
        return self.detection_enabled.get(camera_id, False)

    def add_detection_result(self, camera_id: int, detections: List[DetectionResult]):
        """Add detection results to history"""
        if camera_id not in self.detection_histories:
            self.detection_histories[camera_id] = []
        
        history = self.detection_histories[camera_id]
        history.extend(detections)
        
        # Maintain history size limit
        if len(history) > self.max_history_size:
            self.detection_histories[camera_id] = history[-self.max_history_size:]

    def get_recent_detections(self, camera_id: int, limit: int = 10) -> List[DetectionResult]:
        """Get recent detection results for camera"""
        if camera_id not in self.detection_histories:
            return []
        
        history = self.detection_histories[camera_id]
        return history[-limit:] if limit > 0 else history

    def get_detection_stats(self, camera_id: int) -> Optional[Dict]:
        """Get detection statistics for camera"""
        if camera_id not in self.detectors:
            return None
        
        detector_stats = self.detectors[camera_id].get_stats()
        recent_detections = self.get_recent_detections(camera_id, 1)
        
        stats = detector_stats.copy()
        stats.update({
            'detection_enabled': self.is_detection_enabled(camera_id),
            'recent_detection_count': len(recent_detections),
            'last_person_count': len(recent_detections) if recent_detections else 0
        })
        
        return stats

    def cleanup_camera(self, camera_id: int):
        """Clean up resources for a camera"""

        # DIAGNOSTIC LOGGING - Track cleanup timing and state
        cleanup_time = datetime.now()
        logger.warning(f"🚨 DIAGNOSTIC: cleanup_camera() called for camera {camera_id} at {cleanup_time.strftime('%H:%M:%S.%f')[:-3]}")
        logger.warning(f"🚨 DIAGNOSTIC: Detectors before cleanup: {list(self.detectors.keys())}")
        logger.warning(f"🚨 DIAGNOSTIC: Detection enabled before cleanup: {list(self.detection_enabled.keys())}")

        cleanup_actions = []
        if camera_id in self.detectors:
            del self.detectors[camera_id]
            cleanup_actions.append("detector")
        if camera_id in self.detection_enabled:
            del self.detection_enabled[camera_id]
            cleanup_actions.append("detection_enabled")
        if camera_id in self.detection_histories:
            del self.detection_histories[camera_id]
            cleanup_actions.append("detection_histories")

        logger.warning(f"🚨 DIAGNOSTIC: Cleaned up: {cleanup_actions} for camera {camera_id}")
        logger.warning(f"🚨 DIAGNOSTIC: Detectors after cleanup: {list(self.detectors.keys())}")
        logger.info(f"Cleaned up detection resources for camera {camera_id}")

# Global detection manager instance
detection_manager = PersonDetectionManager()

# DIAGNOSTIC LOGGING - Track when the global detection manager is accessed
original_get_detector = detection_manager.get_detector

def logged_get_detector(camera_id: int, **kwargs):
    logger.warning(f"🚨 DIAGNOSTIC: Global detection_manager.get_detector() called for camera {camera_id}")
    return original_get_detector(camera_id, **kwargs)

detection_manager.get_detector = logged_get_detector