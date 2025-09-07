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
            'frame_width': 1920,
            'frame_height': 1080,
            'image_path': "test/path/to/image.jpg",
        }
        
        test_detection_data['camera_name'] = "Test Camera"
        detection = test_db.save_detection(
            camera_id=999,
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
        required_fields = ['bbox', 'confidence', 'person_id', 'frame_dimensions']
        for field in required_fields:
            if field not in dict_data:
                print(f"- Missing field in to_dict: {field}")
                return False
        print("+ to_dict() method works correctly")
        
        # Test to_database_dict method
        db_dict = detection.to_database_dict(
            camera_id=999
        )
        required_db_fields = ['confidence', 'bbox']
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

def test_unique_id_aggregation():
    """Test unique_id based aggregation for camera re-registration"""
    print("\n=== Testing Unique ID Aggregation ===")
    try:
        from database import DatabaseManager
        
        # Initialize test database
        test_db = DatabaseManager("sqlite:///test_unique_id_aggregation.db")
        test_db.initialize()
        print("+ Test database initialized")
        
        # Create multiple camera records with same unique_id (simulating camera reconnections)
        unique_camera_id = "test_camera_reconnect"
        
        # First camera session
        camera1 = test_db.create_or_get_camera(
            camera_id=101,
            camera_name="Security Camera Front Door - Session 1",
            camera_unique_id=unique_camera_id,
            rtsp_url="rtsp://192.168.1.100/stream1"
        )
        print(f"+ Camera session 1 created: ID {camera1.id}")
        
        # Add detections for first session
        for i in range(3):
            detection_data = {
                'confidence': 0.8 + (i * 0.05),
                'bbox': [100 + i*10, 150 + i*10, 250 + i*10, 400 + i*10],
                'frame_width': 1920,
                'frame_height': 1080,
                'image_path': f"test/session1/detection_{i}.jpg",
                'camera_name': "Security Camera Front Door - Session 1"
            }
            test_db.save_detection(camera_id=101, detection_data=detection_data)
        print("+ Added 3 detections for session 1")
        
        # Second camera session (same unique_id, different database id)
        camera2 = test_db.create_or_get_camera(
            camera_id=102,
            camera_name="Security Camera Front Door - Session 2",
            camera_unique_id=unique_camera_id,  # Same unique_id!
            rtsp_url="rtsp://192.168.1.100/stream1"
        )
        print(f"+ Camera session 2 created: ID {camera2.id}")
        
        # Add detections for second session
        for i in range(2):
            detection_data = {
                'confidence': 0.9 + (i * 0.02),
                'bbox': [150 + i*15, 200 + i*15, 300 + i*15, 450 + i*15],
                'frame_width': 1920,
                'frame_height': 1080,
                'image_path': f"test/session2/detection_{i}.jpg",
                'camera_name': "Security Camera Front Door - Session 2"
            }
            test_db.save_detection(camera_id=102, detection_data=detection_data)
        print("+ Added 2 detections for session 2")
        
        # Test aggregated detection retrieval
        all_detections = test_db.get_detections_by_unique_id(unique_camera_id, limit=10)
        print(f"+ Retrieved {len(all_detections)} total detections across both sessions")
        
        if len(all_detections) != 5:
            print(f"- Expected 5 detections, got {len(all_detections)}")
            return False
        
        # Test aggregated statistics
        stats = test_db.get_detection_stats_by_unique_id(unique_camera_id)
        print(f"+ Aggregated stats - Total detections: {stats['total_detections']}")
        print(f"+ Camera records count: {stats['camera_records_count']}")
        
        if stats['total_detections'] != 5:
            print(f"- Expected 5 total detections in stats, got {stats['total_detections']}")
            return False
            
        if stats['camera_records_count'] != 2:
            print(f"- Expected 2 camera records, got {stats['camera_records_count']}")
            return False
        
        # Test that individual camera stats still work
        individual_stats_1 = test_db.get_detection_stats(camera_id=101)
        individual_stats_2 = test_db.get_detection_stats(camera_id=102)
        print(f"+ Individual stats - Session 1: {individual_stats_1['total_detections']} detections")
        print(f"+ Individual stats - Session 2: {individual_stats_2['total_detections']} detections")
        
        if individual_stats_1['total_detections'] != 3 or individual_stats_2['total_detections'] != 2:
            print("- Individual camera stats incorrect")
            return False
        
        # Clean up test database
        if os.path.exists("test_unique_id_aggregation.db"):
            os.remove("test_unique_id_aggregation.db")
            print("+ Test database cleaned up")
        
        print("+ All unique_id aggregation tests passed!")
        return True
        
    except Exception as e:
        print(f"- Unique ID aggregation test failed: {e}")
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
            DETECTION_IMAGE_QUALITY
        )
        
        print(f"+ DATABASE_ENABLED: {DATABASE_ENABLED}")
        print(f"+ DATABASE_URL: {DATABASE_URL}")
        print(f"+ DETECTION_IMAGE_STORAGE_ENABLED: {DETECTION_IMAGE_STORAGE_ENABLED}")
        print(f"+ DETECTION_IMAGE_BASE_PATH: {DETECTION_IMAGE_BASE_PATH}")
        print(f"+ DETECTION_IMAGE_QUALITY: {DETECTION_IMAGE_QUALITY}")
        
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
    test_results.append(("Unique ID Aggregation", test_unique_id_aggregation()))
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