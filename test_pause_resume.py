"""
Test script to verify detection pause/resume functionality
Simulates user interactions and checks if detection is properly paused
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import time
import json

BASE_URL = "http://localhost:5000"

def test_pause_resume():
    """Test the pause/resume detection API endpoints"""

    print("=" * 60)
    print("Testing Detection Pause/Resume Functionality")
    print("=" * 60)

    # Test 1: Pause detection
    print("\n1. Testing PAUSE detection...")
    pause_data = {
        "reason": "test_script",
        "duration_seconds": 5
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/detection/pause",
            json=pause_data,
            timeout=5
        )

        print(f"   Status Code: {response.status_code}")
        result = response.json()
        print(f"   Response: {json.dumps(result, indent=2)}")

        if result.get('success'):
            print("   ✓ Pause successful!")
            pause_status = result.get('pause_status', {})
            print(f"   - Paused: {pause_status.get('paused')}")
            print(f"   - Reason: {pause_status.get('reason')}")
            print(f"   - Duration: {pause_status.get('duration_seconds')}s")
        else:
            print(f"   ✗ Pause failed: {result.get('error')}")
            return False

    except Exception as e:
        print(f"   ✗ Error calling pause API: {e}")
        return False

    # Test 2: Check system status while paused
    print("\n2. Checking system status (should show paused)...")
    time.sleep(1)

    try:
        response = requests.get(f"{BASE_URL}/api/system/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            detection_status = status.get('detection', {})
            print(f"   Detection paused: {detection_status.get('is_paused')}")
            print(f"   Pause reason: {detection_status.get('pause_reason')}")
            print(f"   Time remaining: {detection_status.get('time_remaining_seconds', 0):.1f}s")
    except Exception as e:
        print(f"   ✗ Error getting status: {e}")

    # Test 3: Wait for auto-resume
    print("\n3. Waiting for auto-resume (5 seconds)...")
    for i in range(5, 0, -1):
        print(f"   {i}...", end='\r')
        time.sleep(1)
    print("   Done!    ")

    # Test 4: Check status after auto-resume
    print("\n4. Checking status after auto-resume...")
    time.sleep(0.5)

    try:
        response = requests.get(f"{BASE_URL}/api/system/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            detection_status = status.get('detection', {})
            is_paused = detection_status.get('is_paused')
            print(f"   Detection paused: {is_paused}")

            if not is_paused:
                print("   ✓ Auto-resume worked!")
            else:
                print("   ✗ Still paused - auto-resume may have failed")
    except Exception as e:
        print(f"   ✗ Error getting status: {e}")

    # Test 5: Manual pause and resume
    print("\n5. Testing MANUAL RESUME...")

    # Pause again
    print("   Pausing for 30 seconds...")
    pause_data = {
        "reason": "manual_resume_test",
        "duration_seconds": 30
    }
    requests.post(f"{BASE_URL}/api/detection/pause", json=pause_data, timeout=5)
    time.sleep(1)

    # Manually resume before auto-resume time
    print("   Manually resuming (before 30s timeout)...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/detection/resume",
            json={},
            timeout=5
        )

        result = response.json()
        print(f"   Response: {json.dumps(result, indent=2)}")

        if result.get('success'):
            print("   ✓ Manual resume successful!")
            resume_status = result.get('resume_status', {})
            print(f"   - Was paused: {resume_status.get('resumed')}")
            print(f"   - Previous reason: {resume_status.get('was_paused_for')}")
        else:
            print(f"   ✗ Resume failed: {result.get('error')}")

    except Exception as e:
        print(f"   ✗ Error calling resume API: {e}")

    # Test 6: Verify resumed
    print("\n6. Final status check...")
    time.sleep(0.5)

    try:
        response = requests.get(f"{BASE_URL}/api/system/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            detection_status = status.get('detection', {})
            is_paused = detection_status.get('is_paused')

            if not is_paused:
                print("   ✓ Detection is running normally")
                print("\n" + "=" * 60)
                print("ALL TESTS PASSED! ✓")
                print("=" * 60)
                return True
            else:
                print("   ✗ Detection still paused")
                return False
    except Exception as e:
        print(f"   ✗ Error getting status: {e}")
        return False

def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/api/system/status", timeout=3)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    print("\nChecking if Flask server is running...")
    if not check_server():
        print("✗ Server is not running at http://localhost:5000")
        print("  Please start the server with: python app.py")
        exit(1)

    print("✓ Server is running\n")

    success = test_pause_resume()

    if not success:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED ✗")
        print("=" * 60)
        exit(1)
