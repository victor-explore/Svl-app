"""
Person Re-Identification System using OSNet_x0_25
Provides functionality to find similar person detections using deep learning embeddings
"""

import os
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
from datetime import datetime

logger = logging.getLogger(__name__)

# Get detection image base directory from config if available
try:
    from config import DETECTION_IMAGE_BASE_PATH as DETECTION_IMAGE_DIR
except ImportError:
    DETECTION_IMAGE_DIR = 'detection_images'

try:
    import torch
    import torchvision.transforms as T
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    logger.warning("torchreid not installed. Person Re-ID features will be disabled.")

class PersonReID:
    """Person Re-Identification using OSNet_x0_25 model"""

    def __init__(self):
        """Initialize the OSNet model for person re-identification"""
        self.model = None
        self.device = None
        self.transform = None
        self.initialized = False

        if not TORCHREID_AVAILABLE:
            logger.error("torchreid is not available. Install it with: pip install torchreid torch torchvision")
            return

        try:
            # Set device (GPU if available, else CPU)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")

            # Load OSNet_x0_25 model
            self.model = torchreid.models.build_model(
                name='osnet_x0_25',
                num_classes=1000,  # Dummy value, we use features only
                pretrained=True
            )

            # Set to evaluation mode
            self.model.eval()
            self.model.to(self.device)

            # Define image preprocessing transforms
            self.transform = T.Compose([
                T.Resize((256, 128)),  # Standard size for person Re-ID
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.initialized = True
            logger.info("PersonReID model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PersonReID model: {e}")
            self.initialized = False

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract feature embedding from a person detection image

        Args:
            image_path: Path to the detection image

        Returns:
            Normalized feature vector (512 dimensions for OSNet_x0_25)
        """
        if not self.initialized:
            logger.error("PersonReID model not initialized")
            return None

        # Handle relative paths - prepend detection_images directory if needed
        if not os.path.isabs(image_path):
            # Convert backslashes to forward slashes for consistency
            image_path = image_path.replace('\\', '/')
            # Prepend the detection_images directory
            full_path = os.path.join(DETECTION_IMAGE_DIR, image_path)
        else:
            full_path = image_path

        # Normalize the path for the current OS
        full_path = os.path.normpath(full_path)

        if not os.path.exists(full_path):
            logger.error(f"Image not found: {full_path} (original: {image_path})")
            return None

        try:
            # Load and preprocess image
            image = Image.open(full_path).convert('RGB')

            # Crop person region if bounding box is available
            # For now, we use the full detection image which should already be cropped

            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)

                # Normalize features for cosine similarity
                features = torch.nn.functional.normalize(features, p=2, dim=1)

                # Convert to numpy array
                embedding = features.cpu().numpy().flatten()

            return embedding

        except Exception as e:
            logger.error(f"Error extracting embedding from {image_path}: {e}")
            return None

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First feature vector
            embedding2: Second feature vector

        Returns:
            Similarity score between 0 and 1 (higher is more similar)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        try:
            # Compute cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Ensure similarity is in [0, 1] range (cosine can be negative)
            # Convert from [-1, 1] to [0, 1]
            similarity = (similarity + 1) / 2

            return float(similarity)

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def find_similar_detections(self,
                              detection_id: int,
                              all_detections: List[Dict[str, Any]],
                              threshold: float = 0.7,
                              top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Find similar person detections from a list of detections

        Args:
            detection_id: ID of the source detection to search for
            all_detections: List of detection dictionaries from same day
            threshold: Minimum similarity threshold (0-1)
            top_k: Maximum number of results to return

        Returns:
            List of similar detections sorted by similarity score
        """
        if not self.initialized:
            logger.error("PersonReID model not initialized")
            return []

        # Find source detection
        source_detection = None
        for det in all_detections:
            if det['id'] == detection_id:
                source_detection = det
                break

        if not source_detection:
            logger.error(f"Source detection {detection_id} not found")
            return []

        # Extract embedding for source detection
        source_image_path = source_detection['image_path']
        source_embedding = self.extract_embedding(source_image_path)

        if source_embedding is None:
            logger.error(f"Failed to extract embedding for source detection {detection_id}")
            return []

        # Compare with all detections (including the source for verification)
        similar_detections = []

        for detection in all_detections:
            try:
                # Extract embedding for current detection
                image_path = detection['image_path']
                embedding = self.extract_embedding(image_path)

                if embedding is None:
                    continue

                # Compute similarity
                similarity = self.compute_similarity(source_embedding, embedding)

                # Apply threshold
                if similarity >= threshold:
                    # Create result entry
                    result = {
                        'id': detection['id'],
                        'person_id': detection.get('person_id'),
                        'camera_id': detection.get('camera_id'),
                        'camera_name': detection.get('camera_name', f"Camera {detection.get('camera_id')}"),
                        'timestamp': detection['created_at'].strftime('%m/%d/%Y, %I:%M:%S %p'),
                        'image_path': detection['image_path'],
                        'similarity': round(similarity, 3),
                        'is_source': detection['id'] == detection_id,
                        'bbox': detection.get('bbox', []),
                        'confidence': detection.get('confidence', 0)
                    }

                    similar_detections.append(result)

            except Exception as e:
                logger.error(f"Error processing detection {detection.get('id')}: {e}")
                continue

        # Sort by similarity score (descending)
        similar_detections.sort(key=lambda x: x['similarity'], reverse=True)

        # Limit to top_k results
        if len(similar_detections) > top_k:
            similar_detections = similar_detections[:top_k]

        logger.info(f"Found {len(similar_detections)} similar detections for ID {detection_id}")
        return similar_detections

    def find_person_path(self, similar_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Organize similar detections into a chronological path across cameras

        Args:
            similar_detections: List of similar detections

        Returns:
            List of detections sorted chronologically showing person's movement
        """
        # Sort by timestamp to show movement path
        path = sorted(similar_detections, key=lambda x: x['timestamp'])

        # Add movement information
        for i, detection in enumerate(path):
            if i > 0:
                prev_camera = path[i-1]['camera_name']
                curr_camera = detection['camera_name']
                if prev_camera != curr_camera:
                    detection['movement'] = f"Moved from {prev_camera} to {curr_camera}"
                else:
                    detection['movement'] = f"Still at {curr_camera}"
            else:
                detection['movement'] = f"First seen at {detection['camera_name']}"

        return path

    def batch_extract_embeddings(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for multiple images efficiently

        Args:
            image_paths: List of paths to detection images

        Returns:
            Dictionary mapping image paths to embeddings
        """
        if not self.initialized:
            logger.error("PersonReID model not initialized")
            return {}

        embeddings = {}

        for image_path in image_paths:
            # The extract_embedding method will handle path resolution
            embedding = self.extract_embedding(image_path)
            if embedding is not None:
                embeddings[image_path] = embedding

        logger.info(f"Extracted embeddings for {len(embeddings)}/{len(image_paths)} images")
        return embeddings

# Global instance (lazy initialization)
_reid_instance = None

def get_reid_instance():
    """Get or create the global PersonReID instance"""
    global _reid_instance
    if _reid_instance is None:
        _reid_instance = PersonReID()
    return _reid_instance