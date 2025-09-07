#!/usr/bin/env python3
"""
Simple test to verify unique_id aggregation functionality
"""

import os
import sys

def test_unique_id_functionality():
    """Test that unique_id aggregation works correctly"""
    print("Testing unique_id functionality...")
    
    try:
        # Import our database module
        from database import DatabaseManager
        
        # Initialize fresh test database
        test_db = DatabaseManager("sqlite:///test_unique_functionality.db")
        test_db.initialize()
        print("+ Database initialized")
        
        # Create camera with unique_id
        camera = test_db.create_or_get_camera(
            camera_id=1,
            camera_name="Test Camera 1",
            camera_unique_id="camera_001",
            rtsp_url="rtsp://test.com/stream1"
        )
        print(f"+ Camera 1 created: {camera.name}")
        
        # Create another camera with same unique_id
        camera2 = test_db.create_or_get_camera(
            camera_id=2,
            camera_name="Test Camera 2",
            camera_unique_id="camera_001",  # Same unique_id!
            rtsp_url="rtsp://test.com/stream1"
        )
        print(f"+ Camera 2 created: {camera2.name} (same unique_id)")
        
        # Add detection to first camera
        detection_data1 = {
            'confidence': 0.85,
            'bbox': [100, 150, 250, 400],
            'frame_width': 1920,
            'frame_height': 1080,
            'image_path': "test1.jpg",
            'camera_name': "Test Camera 1"
        }
        detection1 = test_db.save_detection(camera_id=1, detection_data=detection_data1)
        print(f"+ Detection 1 saved: {detection1.person_id}")
        
        # Add detection to second camera
        detection_data2 = {
            'confidence': 0.90,
            'bbox': [150, 200, 300, 450],
            'frame_width': 1920,
            'frame_height': 1080,
            'image_path': "test2.jpg",
            'camera_name': "Test Camera 2"
        }
        detection2 = test_db.save_detection(camera_id=2, detection_data=detection_data2)
        print(f"+ Detection 2 saved: {detection2.person_id}")
        
        # Test aggregated detection retrieval
        all_detections = test_db.get_detections_by_unique_id("camera_001")
        print(f"+ Retrieved {len(all_detections)} aggregated detections")
        
        # Test aggregated stats
        stats = test_db.get_detection_stats_by_unique_id("camera_001")
        print(f"+ Aggregated stats: {stats['total_detections']} detections, {stats['camera_records_count']} camera records")
        
        # Verify results
        if len(all_detections) == 2 and stats['total_detections'] == 2 and stats['camera_records_count'] == 2:
            print("+ All tests passed!")
            return True
        else:
            print("- Test results incorrect")
            return False
            
    except Exception as e:
        print(f"- Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists("test_unique_functionality.db"):
            try:
                os.remove("test_unique_functionality.db")
                print("+ Test database cleaned up")
            except:
                print("! Could not clean up test database (file may be locked)")

if __name__ == "__main__":
    success = test_unique_id_functionality()
    print("\n" + "="*50)
    if success:
        print("*** UNIQUE_ID IMPLEMENTATION SUCCESSFUL! ***")
        print("\nThe implementation allows:")
        print("- Multiple camera records with same unique_id")
        print("- Aggregated detection retrieval by unique_id")
        print("- Aggregated statistics by unique_id")
        print("- Camera re-registration without errors")
    else:
        print("XXX TESTS FAILED - Check errors above")
    print("="*50)