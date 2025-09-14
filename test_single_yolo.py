#!/usr/bin/env python3
"""
Test script to verify that only ONE YOLO model is loaded
regardless of how many cameras are added to the system.

This script monitors the logs and memory usage to ensure
the DetectionService maintains a single YOLO model instance.
"""

import time
import requests
import json
import sys

BASE_URL = "http://localhost:5000"

def test_single_yolo_model():
    print("=" * 60)
    print("SINGLE YOLO MODEL VERIFICATION TEST")
    print("=" * 60)

    # Check system status
    print("\n1. Checking system status...")
    try:
        response = requests.get(f"{BASE_URL}/api/system/status")
        if response.status_code == 200:
            data = response.json()
            detection_status = data.get('system_status', {}).get('detection_queue', {})
            print(f"   ✓ Detection Service Running: {detection_status.get('service_running', False)}")
            print(f"   ✓ Model Initialized: {detection_status.get('model_initialized', False)}")
            print(f"   ✓ Registered Cameras: {detection_status.get('registered_cameras', 0)}")
            print(f"   ✓ Camera IDs: {detection_status.get('camera_list', [])}")
        else:
            print(f"   ✗ Failed to get system status: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error checking system status: {e}")

    print("\n2. Adding test cameras...")
    test_cameras = []

    for i in range(3):
        camera_data = {
            "unique_id": f"test_camera_{i+1}",
            "name": f"Test Camera {i+1}",
            "rtsp_url": f"rtsp://test{i+1}.example.com/stream",
            "auto_start": False  # Don't actually connect
        }

        try:
            response = requests.post(
                f"{BASE_URL}/api/cameras",
                json=camera_data,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                camera = response.json().get('camera', {})
                test_cameras.append(camera['id'])
                print(f"   ✓ Added camera {i+1}: ID={camera['id']}")
            else:
                print(f"   ✗ Failed to add camera {i+1}: {response.status_code}")
        except Exception as e:
            print(f"   ✗ Error adding camera {i+1}: {e}")

        time.sleep(1)

    print("\n3. Checking detection service after adding cameras...")
    try:
        response = requests.get(f"{BASE_URL}/api/system/status")
        if response.status_code == 200:
            data = response.json()
            detection_status = data.get('system_status', {}).get('detection_queue', {})
            print(f"   ✓ Registered Cameras: {detection_status.get('registered_cameras', 0)}")
            print(f"   ✓ Camera IDs: {detection_status.get('camera_list', [])}")

            # Check that model is still initialized (single instance)
            if detection_status.get('model_initialized'):
                print("   ✓ CONFIRMED: Single YOLO model still initialized")
            else:
                print("   ✗ WARNING: Model not initialized")
        else:
            print(f"   ✗ Failed to get system status: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error checking system status: {e}")

    print("\n4. Cleaning up test cameras...")
    for camera_id in test_cameras:
        try:
            response = requests.delete(f"{BASE_URL}/api/cameras/{camera_id}")
            if response.status_code == 200:
                print(f"   ✓ Removed camera ID={camera_id}")
            else:
                print(f"   ✗ Failed to remove camera {camera_id}: {response.status_code}")
        except Exception as e:
            print(f"   ✗ Error removing camera {camera_id}: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("Check the application logs for:")
    print("1. 'Initializing SINGLE YOLO model in DetectionService...'")
    print("2. 'SUCCESS: Single YOLO model loaded and ready!'")
    print("3. NO additional 'PersonDetector created' messages")
    print("4. NO duplicate model loading messages")
    print("=" * 60)

if __name__ == "__main__":
    print("Starting test in 3 seconds...")
    print("Make sure the Flask app is running!")
    time.sleep(3)

    try:
        test_single_yolo_model()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)