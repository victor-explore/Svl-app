"""
Detection Image Storage Management System
Handles saving and managing detection images to disk storage
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Tuple, Optional, List
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)

class DetectionImageStorage:
    """Manages storage of detection images on disk"""
    
    def __init__(self, base_storage_path: str = "detection_images"):
        """
        Initialize image storage manager
        
        Args:
            base_storage_path: Base directory for storing detection images
        """
        self.base_storage_path = base_storage_path
        self.ensure_base_directory()
        
    def ensure_base_directory(self):
        """Ensure the base storage directory exists"""
        if not os.path.exists(self.base_storage_path):
            os.makedirs(self.base_storage_path)
            logger.info(f"Created base storage directory: {self.base_storage_path}")
    
    def get_date_directory(self, timestamp: datetime) -> str:
        """Get directory path based on date (YYYY/MM/DD structure)"""
        date_path = timestamp.strftime("%Y/%m/%d")
        full_path = os.path.join(self.base_storage_path, date_path)
        
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            logger.debug(f"Created date directory: {full_path}")
        
        return full_path
    
    def generate_filename(self, camera_id: int, camera_name: str, 
                         timestamp: datetime, file_type: str = "full") -> str:
        """
        Generate unique filename for detection images
        
        Args:
            camera_id: Camera identifier
            camera_name: Camera name (sanitized for filesystem)
            timestamp: Detection timestamp
            file_type: Type of file ("full" for full frame, "person" for cropped person)
        
        Returns:
            Unique filename
        """
        # Sanitize camera name for filesystem
        safe_camera_name = "".join(c for c in camera_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_camera_name = safe_camera_name.replace(' ', '_')
        
        # Generate timestamp string
        time_str = timestamp.strftime("%H%M%S_%f")[:-3]  # Include milliseconds
        
        # Create filename
        filename = f"{file_type}_cam{camera_id}_{safe_camera_name}_{time_str}.jpg"
        return filename
    
    def save_full_frame_image(self, frame: np.ndarray, camera_id: int, 
                            camera_name: str, timestamp: datetime, 
                            detections: List = None) -> str:
        """
        Save full frame image with optional detection annotations
        
        Args:
            frame: OpenCV frame (BGR format)
            camera_id: Camera identifier
            camera_name: Camera name
            timestamp: Detection timestamp
            detections: Optional list of detection results for annotation
        
        Returns:
            Relative path to saved image
        """
        try:
            # Get date-based directory
            date_dir = self.get_date_directory(timestamp)
            
            # Generate filename
            filename = self.generate_filename(camera_id, camera_name, timestamp, "full")
            full_path = os.path.join(date_dir, filename)
            
            # Add detection annotations if provided
            if detections:
                frame = self._add_detection_annotations(frame, detections, timestamp)
            
            # Save image with high quality
            success = cv2.imwrite(full_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                # Return relative path from base storage directory
                relative_path = os.path.relpath(full_path, self.base_storage_path)
                logger.debug(f"Saved full frame image: {relative_path}")
                return relative_path
            else:
                logger.error(f"Failed to save full frame image: {full_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving full frame image: {e}")
            return None
    
    def save_person_crop(self, frame: np.ndarray, bbox: List[float], 
                        camera_id: int, camera_name: str, 
                        timestamp: datetime, confidence: float) -> str:
        """
        Save cropped person image from detection bounding box
        
        Args:
            frame: OpenCV frame (BGR format)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            camera_id: Camera identifier
            camera_name: Camera name
            timestamp: Detection timestamp
            confidence: Detection confidence score
        
        Returns:
            Relative path to saved cropped image
        """
        try:
            # Get date-based directory
            date_dir = self.get_date_directory(timestamp)
            
            # Generate filename
            filename = self.generate_filename(camera_id, camera_name, timestamp, "person")
            full_path = os.path.join(date_dir, filename)
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            
            # Crop person region
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                logger.warning(f"Empty crop region for bbox {bbox}")
                return None
            
            # Add confidence label to cropped image
            person_crop = self._add_confidence_label(person_crop, confidence)
            
            # Save cropped image
            success = cv2.imwrite(full_path, person_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            if success:
                # Return relative path from base storage directory
                relative_path = os.path.relpath(full_path, self.base_storage_path)
                logger.debug(f"Saved person crop image: {relative_path}")
                return relative_path
            else:
                logger.error(f"Failed to save person crop: {full_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving person crop: {e}")
            return None
    
    def _add_detection_annotations(self, frame: np.ndarray, detections: List, 
                                 timestamp: datetime) -> np.ndarray:
        """Add detection bounding boxes and metadata to frame"""
        annotated_frame = frame.copy()
        
        try:
            for i, detection in enumerate(detections):
                if hasattr(detection, 'bbox') and hasattr(detection, 'confidence'):
                    bbox = detection.bbox
                    confidence = detection.confidence
                else:
                    # Handle different detection formats
                    bbox = detection.get('bbox', [])
                    confidence = detection.get('confidence', 0.0)
                
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw confidence label
                    label = f"Person {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(annotated_frame, 
                                 (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), 
                                 (0, 255, 0), -1)
                    
                    # Draw label text
                    cv2.putText(annotated_frame, label, 
                               (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add timestamp to image
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated_frame, timestamp_str, 
                       (10, annotated_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            logger.error(f"Error adding annotations: {e}")
            return frame  # Return original frame on error
        
        return annotated_frame
    
    def _add_confidence_label(self, image: np.ndarray, confidence: float) -> np.ndarray:
        """Add confidence score label to cropped person image"""
        labeled_image = image.copy()
        
        try:
            label = f"{confidence:.2f}"
            
            # Add label at top of image
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Draw label background
            cv2.rectangle(labeled_image, 
                         (5, 5), 
                         (15 + label_size[0], 25 + label_size[1]), 
                         (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(labeled_image, label, 
                       (10, 20 + label_size[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        except Exception as e:
            logger.error(f"Error adding confidence label: {e}")
            return image  # Return original image on error
        
        return labeled_image
    
    def get_image_path(self, relative_path: str) -> str:
        """Get full path from relative path"""
        return os.path.join(self.base_storage_path, relative_path)
    
    def image_exists(self, relative_path: str) -> bool:
        """Check if image file exists"""
        if not relative_path:
            return False
        full_path = self.get_image_path(relative_path)
        return os.path.exists(full_path)
    
    def delete_image(self, relative_path: str) -> bool:
        """Delete image file"""
        try:
            if not relative_path:
                return False
            
            full_path = self.get_image_path(relative_path)
            if os.path.exists(full_path):
                os.remove(full_path)
                logger.debug(f"Deleted image: {relative_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting image {relative_path}: {e}")
            return False
    
    def cleanup_empty_directories(self):
        """Remove empty date directories (maintenance function)"""
        try:
            for root, dirs, files in os.walk(self.base_storage_path, topdown=False):
                # Skip base directory
                if root == self.base_storage_path:
                    continue
                
                # Remove empty directories
                if not files and not dirs:
                    os.rmdir(root)
                    logger.debug(f"Removed empty directory: {root}")
        except Exception as e:
            logger.error(f"Error cleaning up directories: {e}")
    
    def get_storage_stats(self) -> dict:
        """Get storage statistics"""
        try:
            total_files = 0
            total_size = 0
            
            for root, dirs, files in os.walk(self.base_storage_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(root, file)
                        total_files += 1
                        total_size += os.path.getsize(file_path)
            
            return {
                'total_images': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'base_path': self.base_storage_path
            }
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {'total_images': 0, 'total_size_bytes': 0}

# Global image storage instance
image_storage = DetectionImageStorage()