"""
Simple demo script to test the CCTV system with webcam
This allows testing without RTSP cameras
"""

import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import numpy as np

def run_webcam_demo():
    """Run a simple detection demo with webcam"""
    print("=" * 60)
    print("CCTV System - Webcam Demo")
    print("=" * 60)
    print("Press 'q' to quit, 's' to save screenshot")
    print()
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO("yolov8s.pt")
    print("Model loaded successfully!")
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam!")
        print("Please check:")
        print("1. Webcam is connected")
        print("2. No other application is using the webcam")
        print("3. Webcam drivers are installed")
        return
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Webcam opened successfully!")
    print("Starting detection loop...")
    print()
    
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    # Detection settings
    conf_threshold = 0.35
    selected_classes = ['person', 'cell phone', 'laptop', 'mouse', 
                       'keyboard', 'book', 'cup', 'bottle']
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break
        
        frame_count += 1
        
        # Run detection every 5 frames to reduce CPU load
        if frame_count % 5 == 0:
            # Run YOLO detection
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # Draw detections
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get class and confidence
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Get class name
                        class_name = model.names[cls]
                        
                        # Filter by selected classes (if specified)
                        if selected_classes and class_name not in selected_classes:
                            continue
                        
                        # Draw bounding box
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                        
                        # Draw label background
                        cv2.rectangle(frame, 
                                    (x1, label_y - label_size[1] - 4),
                                    (x1 + label_size[0], label_y + 4),
                                    color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (x1, label_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calculate FPS
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - fps_time)
            fps_time = current_time
        
        # Draw FPS and timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, timestamp, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("CCTV Webcam Demo - Press 'q' to quit", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            # Save screenshot
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo ended.")

def test_detection_on_image():
    """Test detection on a single image"""
    print("\nTesting detection on sample image...")
    
    # Create a sample image with shapes
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Sample Test Image", (200, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Run detection
    model = YOLO("yolov8s.pt")
    results = model(img, verbose=False)
    
    print("Detection test completed!")
    return True

if __name__ == "__main__":
    import sys
    
    print("Choose demo mode:")
    print("1. Webcam live detection")
    print("2. Test detection on sample image")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        try:
            run_webcam_demo()
        except Exception as e:
            print(f"\n[ERROR] Demo failed: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure webcam is connected")
            print("2. Check if another app is using the webcam")
            print("3. Try running: python test_components.py")
    elif choice == "2":
        test_detection_on_image()
    else:
        print("Exiting...")
        sys.exit(0)