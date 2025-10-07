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

# Get detection image base directory and model path from config if available
try:
    from config import DETECTION_IMAGE_BASE_PATH as DETECTION_IMAGE_DIR
    from config import PERSON_REID_MODEL_PATH
except ImportError:
    DETECTION_IMAGE_DIR = 'detection_images'
    PERSON_REID_MODEL_PATH = './osnet_x0_25_imagenet.pth'

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

            # Load OSNet_x0_25 model from local file (using config path)
            if os.path.isabs(PERSON_REID_MODEL_PATH):
                local_model_path = PERSON_REID_MODEL_PATH
            else:
                local_model_path = os.path.join(os.path.dirname(__file__), PERSON_REID_MODEL_PATH)

            if os.path.exists(local_model_path):
                # Build model without pretrained weights
                self.model = torchreid.models.build_model(
                    name='osnet_x0_25',
                    num_classes=1000,  # Dummy value, we use features only
                    pretrained=False
                )

                # Load weights from local file
                torchreid.utils.load_pretrained_weights(self.model, local_model_path)
                logger.info(f"Loaded OSNet weights from local file: {local_model_path}")
            else:
                # Fallback to auto-download if local file doesn't exist
                logger.warning(f"Local model file not found at {local_model_path}, falling back to auto-download")
                self.model = torchreid.models.build_model(
                    name='osnet_x0_25',
                    num_classes=1000,  # Dummy value, we use features only
                    pretrained=True
                )
                logger.info("Loaded OSNet weights from online (auto-download)")

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

    def extract_embedding_from_upload(self, image_file) -> np.ndarray:
        """
        Extract feature embedding from an uploaded image file

        Args:
            image_file: File object or file path from upload

        Returns:
            Normalized feature vector (512 dimensions for OSNet_x0_25)
        """
        if not self.initialized:
            logger.error("PersonReID model not initialized")
            return None

        try:
            # Load image from file object or path
            if hasattr(image_file, 'read'):
                # File object from upload
                image = Image.open(image_file).convert('RGB')
            else:
                # File path
                image = Image.open(image_file).convert('RGB')

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
            logger.error(f"Error extracting embedding from uploaded image: {e}")
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
                              top_k: int = 50,
                              max_search: int = 200) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Find similar person detections from a list of detections with early stopping optimization

        Args:
            detection_id: ID of the source detection to search for
            all_detections: List of detection dictionaries from same day
            threshold: Minimum similarity threshold (0-1)
            top_k: Maximum number of results to return
            max_search: Maximum number of detections to search through

        Returns:
            Tuple of (similar detections sorted by similarity, search statistics)
        """
        if not self.initialized:
            logger.error("PersonReID model not initialized")
            return [], {'searched': 0, 'found': 0}

        # Find source detection
        source_detection = None
        for det in all_detections:
            if det['id'] == detection_id:
                source_detection = det
                break

        if not source_detection:
            logger.error(f"Source detection {detection_id} not found")
            return [], {'searched': 0, 'found': 0}

        # Extract embedding for source detection
        source_image_path = source_detection['image_path']
        source_embedding = self.extract_embedding(source_image_path)

        if source_embedding is None:
            logger.error(f"Failed to extract embedding for source detection {detection_id}")
            return [], {'searched': 0, 'found': 0}

        # Sort detections by timestamp (newest first) for more relevant results
        sorted_detections = sorted(all_detections,
                                 key=lambda x: x['created_at'],
                                 reverse=True)

        # Limit search to max_search detections
        detections_to_search = sorted_detections[:max_search]

        similar_detections = []
        searched_count = 0
        found_count = 0

        for detection in detections_to_search:
            try:
                searched_count += 1

                # Extract embedding for current detection
                image_path = detection['image_path']
                embedding = self.extract_embedding(image_path)

                if embedding is None:
                    continue

                # Compute similarity
                similarity = self.compute_similarity(source_embedding, embedding)

                # Apply threshold
                if similarity >= threshold:
                    found_count += 1

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

                    # Early stopping: if we have enough high-quality matches, we can stop
                    # Only stop if we have at least top_k matches with good similarity
                    if found_count >= top_k * 1.5:  # Search a bit more to ensure best matches
                        logger.info(f"Early stopping: Found {found_count} matches after searching {searched_count} detections")
                        break

            except Exception as e:
                logger.error(f"Error processing detection {detection.get('id')}: {e}")
                continue

        # Sort by similarity score (descending)
        similar_detections.sort(key=lambda x: x['similarity'], reverse=True)

        # Limit to top_k results
        if len(similar_detections) > top_k:
            similar_detections = similar_detections[:top_k]

        stats = {
            'searched': searched_count,
            'found': found_count,
            'total_available': len(all_detections),
            'returned': len(similar_detections)
        }

        logger.info(f"Search stats for ID {detection_id}: searched={searched_count}, found={found_count}, returned={len(similar_detections)}")
        return similar_detections, stats

    def find_similar_by_embedding(self,
                                 source_embedding: np.ndarray,
                                 all_detections: List[Dict[str, Any]],
                                 threshold: float = 0.7,
                                 top_k: int = 50,
                                 max_search: int = 200) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Find similar person detections using a pre-computed embedding with early stopping

        Args:
            source_embedding: Pre-computed embedding to search with
            all_detections: List of detection dictionaries to search through
            threshold: Minimum similarity threshold (0-1)
            top_k: Maximum number of results to return
            max_search: Maximum number of detections to search through

        Returns:
            Tuple of (similar detections sorted by similarity, search statistics)
        """
        if not self.initialized:
            logger.error("PersonReID model not initialized")
            return [], {'searched': 0, 'found': 0}

        if source_embedding is None:
            logger.error("Source embedding is None")
            return [], {'searched': 0, 'found': 0}

        # Sort detections by timestamp (newest first) for more relevant results
        sorted_detections = sorted(all_detections,
                                 key=lambda x: x['created_at'],
                                 reverse=True)

        # Limit search to max_search detections
        detections_to_search = sorted_detections[:max_search]

        similar_detections = []
        searched_count = 0
        found_count = 0

        # Search through limited detections with early stopping
        for detection in detections_to_search:
            try:
                searched_count += 1

                # Extract embedding for current detection
                image_path = detection['image_path']
                embedding = self.extract_embedding(image_path)

                if embedding is None:
                    continue

                # Compute similarity
                similarity = self.compute_similarity(source_embedding, embedding)

                # Apply threshold
                if similarity >= threshold:
                    found_count += 1

                    # Create result entry
                    result = {
                        'id': detection['id'],
                        'person_id': detection.get('person_id'),
                        'camera_id': detection.get('camera_id'),
                        'camera_name': detection.get('camera_name', f"Camera {detection.get('camera_id')}"),
                        'timestamp': detection['created_at'].strftime('%m/%d/%Y, %I:%M:%S %p'),
                        'image_path': detection['image_path'],
                        'similarity': round(similarity, 3),
                        'is_source': False,  # No source detection when searching by image
                        'bbox': detection.get('bbox', []),
                        'confidence': detection.get('confidence', 0)
                    }

                    similar_detections.append(result)

                    # Early stopping: if we have enough high-quality matches
                    if found_count >= top_k * 1.5:  # Search a bit more to ensure best matches
                        logger.info(f"Early stopping: Found {found_count} matches after searching {searched_count} detections")
                        break

            except Exception as e:
                logger.error(f"Error processing detection {detection.get('id')}: {e}")
                continue

        # Sort by similarity score (descending)
        similar_detections.sort(key=lambda x: x['similarity'], reverse=True)

        # Limit to top_k results
        if len(similar_detections) > top_k:
            similar_detections = similar_detections[:top_k]

        stats = {
            'searched': searched_count,
            'found': found_count,
            'total_available': len(all_detections),
            'returned': len(similar_detections)
        }

        logger.info(f"Search stats for uploaded image: searched={searched_count}, found={found_count}, returned={len(similar_detections)}")
        return similar_detections, stats

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