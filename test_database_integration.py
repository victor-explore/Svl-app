#!/usr/bin/env python3
"""
Test script for database storage integration
Tests database models, image storage, and API endpoints
"""

import os
import sys
import numpy as np
import cv2
from datetime import datetime
import logging

# Set up logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_models():
    """Test database models and operations"""
    print("\n=== Testing Database Models ===")
    try:
        from database import db_manager, DatabaseManager
        
        # Test database initialization
        print("+ Database imports successful")
        
        # Initialize database
        test_db = DatabaseManager("sqlite:///test_detection_records.db")
        test_db.initialize()
        print("+ Database initialization successful")
        
        # Test camera creation
        camera = test_db.create_or_get_camera(
            camera_id=999,
            camera_name="Test Camera",
            camera_unique_id="test_camera_999",
            rtsp_url="rtsp://test.example.com/stream"
        )
        print(f"+ Camera created: {camera.name} (ID: {camera.id})")
        
        # Test detection storage
        test_detection_data = {
            'confidence': 0.85,
            'bbox': [100, 150, 250, 400],
            'timestamp': datetime.now(),
            'frame_width': 1920,
            'frame_height': 1080,
            'full_image_path': "test/path/to/full_image.jpg",
            'person_image_path': "test/path/to/person_crop.jpg",
            'camera_unique_id': "test_camera_999"
        }
        
        detection = test_db.save_detection(
            camera_id=999,
            camera_name="Test Camera",
            detection_data=test_detection_data
        )
        print(f"+ Detection saved with person ID: {detection.person_id}")
        
        # Test detection retrieval
        history = test_db.get_detection_history(camera_id=999, limit=10)
        print(f"+ Retrieved {len(history)} detection record(s)")
        
        # Test statistics
        stats = test_db.get_detection_stats(camera_id=999)
        print(f"+ Stats retrieved - Total detections: {stats.get('total_detections', 0)}")
        
        # Clean up test database
        if os.path.exists("test_detection_records.db"):
            os.remove("test_detection_records.db")
            print("+ Test database cleaned up")
        
        return True
        
    except Exception as e:
        print(f"- Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_storage():
    """Test image storage system"""
    print("\n=== Testing Image Storage ===")
    try:
        from detection_storage import DetectionImageStorage
        
        # Create test storage instance
        test_storage = DetectionImageStorage("test_detection_images")
        print("+ Image storage initialized")
        
        # Create test image (simple colored square)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:, :] = [100, 150, 200]  # BGR color
        
        # Draw a rectangle to simulate a person detection
        cv2.rectangle(test_frame, (200, 150), (400, 350), (0, 255, 0), 3)
        cv2.putText(test_frame, "Test Person", (210, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        print("+ Test frame created")
        
        # Test full frame saving
        timestamp = datetime.now()
        full_path = test_storage.save_full_frame_image(
            frame=test_frame,
            camera_id=999,
            camera_name="Test Camera",
            timestamp=timestamp
        )
        if full_path:
            print(f"+ Full frame saved: {full_path}")
        else:
            print("- Full frame save failed")
            
        # Test person crop saving
        bbox = [200, 150, 400, 350]  # x1, y1, x2, y2
        crop_path = test_storage.save_person_crop(
            frame=test_frame,
            bbox=bbox,
            camera_id=999,
            camera_name="Test Camera",
            timestamp=timestamp,
            confidence=0.85
        )
        if crop_path:
            print(f"+ Person crop saved: {crop_path}")
        else:
            print("- Person crop save failed")
        
        # Test storage statistics
        stats = test_storage.get_storage_stats()
        print(f"+ Storage stats - Total images: {stats.get('total_images', 0)}")
        
        # Clean up test images
        import shutil
        if os.path.exists("test_detection_images"):
            shutil.rmtree("test_detection_images")
            print("+ Test images cleaned up")
        
        return True
        
    except Exception as e:
        print(f"- Image storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detection_result():
    """Test DetectionResult class enhancements"""
    print("\n=== Testing DetectionResult Class ===")
    try:
        from person_detector import DetectionResult
        
        # Create test detection
        detection = DetectionResult(
            bbox=[100, 150, 250, 400],
            confidence=0.85,
            timestamp=datetime.now(),
            frame_width=1920,
            frame_height=1080
        )
        
        print("+ DetectionResult created")
        
        # Test to_dict method
        dict_data = detection.to_dict()
        required_fields = ['bbox', 'confidence', 'timestamp', 'person_id', 'frame_dimensions']
        for field in required_fields:
            if field not in dict_data:
                print(f"- Missing field in to_dict: {field}")
                return False
        print("+ to_dict() method works correctly")
        
        # Test to_database_dict method
        db_dict = detection.to_database_dict(
            camera_id=999,
            camera_name="Test Camera",
            camera_unique_id="test_camera_999"
        )
        required_db_fields = ['confidence', 'bbox', 'timestamp', 'camera_unique_id']
        for field in required_db_fields:
            if field not in db_dict:
                print(f"- Missing field in to_database_dict: {field}")
                return False
        print("+ to_database_dict() method works correctly")
        
        return True
        
    except Exception as e:
        print(f"- DetectionResult test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_settings():
    """Test configuration settings"""
    print("\n=== Testing Configuration ===")
    try:
        from config import (
            DATABASE_ENABLED, DATABASE_URL, DATABASE_AUTO_INIT,
            DETECTION_IMAGE_STORAGE_ENABLED, DETECTION_IMAGE_BASE_PATH,
            DETECTION_SAVE_FULL_FRAMES, DETECTION_SAVE_PERSON_CROPS
        )
        
        print(f"+ DATABASE_ENABLED: {DATABASE_ENABLED}")
        print(f"+ DATABASE_URL: {DATABASE_URL}")
        print(f"+ DETECTION_IMAGE_STORAGE_ENABLED: {DETECTION_IMAGE_STORAGE_ENABLED}")
        print(f"+ DETECTION_IMAGE_BASE_PATH: {DETECTION_IMAGE_BASE_PATH}")
        print(f"+ DETECTION_SAVE_FULL_FRAMES: {DETECTION_SAVE_FULL_FRAMES}")
        print(f"+ DETECTION_SAVE_PERSON_CROPS: {DETECTION_SAVE_PERSON_CROPS}")
        
        return True
        
    except Exception as e:
        print(f"- Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Starting Database Storage Integration Tests")
    print("=" * 50)
    
    test_results = []
    
    # Run individual tests
    test_results.append(("Config Settings", test_config_settings()))
    test_results.append(("DetectionResult Class", test_detection_result()))
    test_results.append(("Database Models", test_database_models()))
    test_results.append(("Image Storage", test_image_storage()))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:25} {status}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("*** ALL TESTS PASSED! Database storage system is ready. ***")
        print("\nNext steps:")
        print("1. Install new dependencies: pip install -r requirements.txt")
        print("2. Start your Flask application")
        print("3. Test person detection with a live camera feed")
        print("4. Check database records at: /api/detections/history")
        print("5. View stored images in: detection_images/ directory")
    else:
        print("XXX Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main()