"""
Test script to verify all components are working correctly
"""

import sys
import os

def test_imports():
    """Test if all required libraries can be imported"""
    print("Testing library imports...")
    
    try:
        import cv2
        print("[OK] OpenCV imported successfully")
    except ImportError as e:
        print(f"[FAIL] OpenCV import failed: {e}")
        return False
    
    try:
        import torch
        print(f"[OK] PyTorch imported successfully (version: {torch.__version__})")
        print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    except ImportError as e:
        print(f"[FAIL] PyTorch import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("[OK] Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"[FAIL] Ultralytics import failed: {e}")
        return False
    
    try:
        from torchreid.reid.utils import FeatureExtractor
        print("[OK] TorchReID FeatureExtractor imported successfully")
    except ImportError as e:
        print(f"[FAIL] TorchReID import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"[OK] NumPy imported successfully (version: {np.__version__})")
    except ImportError as e:
        print(f"[FAIL] NumPy import failed: {e}")
        return False
    
    return True

def test_yolo_model():
    """Test YOLO model loading"""
    print("\nTesting YOLO model...")
    try:
        from ultralytics import YOLO
        import os
        
        model_path = "yolov8s.pt"
        if os.path.exists(model_path):
            print(f"[OK] YOLO model file found: {model_path}")
            model = YOLO(model_path)
            print("[OK] YOLO model loaded successfully")
            return True
        else:
            print(f"[FAIL] YOLO model file not found: {model_path}")
            print("  The model will be downloaded on first run")
            return True
    except Exception as e:
        print(f"[FAIL] YOLO model test failed: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    print("\nTesting directories...")
    
    dirs_to_check = ["recordings", "events"]
    all_exist = True
    
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            print(f"[OK] Directory exists: {dir_name}/")
        else:
            print(f"[FAIL] Directory missing: {dir_name}/")
            all_exist = False
    
    return all_exist

def test_ffmpeg():
    """Test FFmpeg availability"""
    print("\nTesting FFmpeg...")
    import subprocess
    
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            first_line = result.stdout.split('\n')[0]
            print(f"[OK] FFmpeg is installed: {first_line}")
            return True
        else:
            print("[FAIL] FFmpeg command failed")
            return False
    except FileNotFoundError:
        print("[FAIL] FFmpeg not found in PATH")
        print("  Recording features will not work")
        print("  Download from: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"[FAIL] FFmpeg test failed: {e}")
        return False

def test_webcam():
    """Test if a webcam is available"""
    print("\nTesting webcam access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("[OK] Webcam is accessible")
                print(f"  Frame shape: {frame.shape}")
                return True
            else:
                print("[FAIL] Could not read from webcam")
                return False
        else:
            print("[FAIL] No webcam detected or cannot access")
            return False
    except Exception as e:
        print(f"[FAIL] Webcam test failed: {e}")
        return False

def test_config():
    """Test configuration file"""
    print("\nTesting configuration...")
    try:
        import config
        print("[OK] Config file loaded successfully")
        print(f"  Device: {config.DEVICE}")
        print(f"  YOLO Model: {config.YOLO_MODEL}")
        print(f"  Process FPS: {config.PROCESS_FPS}")
        print(f"  Number of cameras configured: {len(config.CAMERAS)}")
        
        if config.CAMERAS:
            for name, url in config.CAMERAS.items():
                print(f"    - {name}: {url[:50]}...")
        
        return True
    except Exception as e:
        print(f"[FAIL] Config test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("CCTV Application Component Test")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Library Imports", test_imports()))
    results.append(("YOLO Model", test_yolo_model()))
    results.append(("Directories", test_directories()))
    results.append(("FFmpeg", test_ffmpeg()))
    results.append(("Webcam", test_webcam()))
    results.append(("Configuration", test_config()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[OK] PASSED" if result else "[FAIL] FAILED"
        print(f"{test_name:20} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All components are working correctly!")
        print("You can now run: python main.py")
    else:
        print("\n[WARNING]  Some components need attention.")
        print("Please fix the issues above before running the main application.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)