"""
Person Detection Module for CCTV Surveillance System
Uses YOLOv8 for real-time person detection in video streams
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime

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
    
    def to_database_dict(self, camera_id: int, camera_unique_id: str = None) -> Dict:
        """Convert to dictionary format for database storage"""
        return {
            'confidence': self.confidence,
            'bbox': self.bbox,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'image_path': self.image_path,
            'camera_unique_id': camera_unique_id or f"camera_{camera_id}"
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
        
        # Performance tracking
        self.total_detections = 0
        self.total_frames_processed = 0
        self.average_inference_time = 0.0
        self.last_detection_time = None
        
        logger.info(f"PersonDetector initializing with model: {model_path}, confidence: {confidence_threshold}")
        
        # Force immediate model initialization instead of lazy loading
        success = self.initialize_model()
        if success:
            logger.info(f"PersonDetector successfully initialized with model: {model_path}")
        else:
            logger.error(f"PersonDetector failed to initialize model: {model_path}")

    def initialize_model(self) -> bool:
        """
        Initialize the YOLOv8 model
        Returns True if successful, False otherwise
        """
        if self.is_initialized:
            logger.debug("Model already initialized, skipping")
            return True
            
        try:
            logger.info(f"Downloading and loading YOLOv8 model: {self.model_path}")
            logger.info("This may take a few moments on first run as the model is downloaded...")
            
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.is_initialized = True
            
            logger.info("YOLOv8 model loaded successfully!")
            logger.info(f"Model classes: {len(self.model.names)} (person is class 0)")
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8 model: {e}")
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
            
            # Run inference - class 0 is 'person' in COCO dataset
            results = self.model(frame, classes=[0], verbose=False)
            
            inference_time = time.time() - start_time
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
                            
                            # Get frame dimensions
                            frame_height, frame_width = frame.shape[:2]
                            
                            detection = DetectionResult(
                                bbox=bbox, 
                                confidence=confidence, 
                                timestamp=current_time,
                                frame_width=frame_width,
                                frame_height=frame_height
                            )
                            detections.append(detection)
                            person_count += 1
            
            self.total_detections += person_count
            self.last_detection_time = datetime.now()
            
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
        if camera_id not in self.detectors:
            self.detectors[camera_id] = PersonDetector(**kwargs)
            self.detection_enabled[camera_id] = True
            self.detection_histories[camera_id] = []
        
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
        if camera_id in self.detectors:
            del self.detectors[camera_id]
        if camera_id in self.detection_enabled:
            del self.detection_enabled[camera_id]
        if camera_id in self.detection_histories:
            del self.detection_histories[camera_id]
        
        logger.info(f"Cleaned up detection resources for camera {camera_id}")

# Global detection manager instance
detection_manager = PersonDetectionManager()