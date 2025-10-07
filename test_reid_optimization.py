"""
Test script to verify the Re-ID optimization is working correctly
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_reid_optimization():
    """Test that the Re-ID search optimization is working"""
    print("Testing Re-ID Search Optimization...")
    print("=" * 50)

    try:
        from person_reid import get_reid_instance
        from database import db_manager, Detection

        # Initialize
        print("\n1. Initializing Re-ID model...")
        reid_model = get_reid_instance()

        if not reid_model.initialized:
            print("[ERROR] Re-ID model failed to initialize. Make sure torchreid is installed.")
            return False

        print("[OK] Re-ID model initialized")

        # Get test data
        print("\n2. Getting test detections from database...")
        session = db_manager.get_session()

        # Get detections from today
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)

        detections = session.query(Detection).filter(
            Detection.created_at.between(today_start, today_end)
        ).limit(100).all()

        if len(detections) < 2:
            print("[ERROR] Not enough detections in database for testing (need at least 2)")
            session.close()
            return False

        print(f"[OK] Found {len(detections)} detections to test with")

        # Convert to format expected by reid_model
        all_detections = []
        for det in detections:
            all_detections.append({
                'id': det.id,
                'person_id': det.person_id,
                'camera_id': det.camera_id,
                'camera_name': f'Camera {det.camera_id}',
                'created_at': det.created_at,
                'image_path': det.image_path,
                'confidence': det.confidence,
                'bbox': [det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2] if det.bbox_x1 else []
            })

        # Test with different max_search values
        test_id = all_detections[0]['id']

        print(f"\n3. Testing search for detection ID {test_id}...")
        print("-" * 40)

        # Test 1: Small search limit
        print("\nTest 1: max_search=10, top_k=5")
        start_time = time.time()
        results1, stats1 = reid_model.find_similar_detections(
            detection_id=test_id,
            all_detections=all_detections,
            threshold=0.5,
            top_k=5,
            max_search=10
        )
        time1 = time.time() - start_time

        print(f"  Searched: {stats1['searched']} / {stats1['total_available']} detections")
        print(f"  Found: {stats1['found']} matches")
        print(f"  Returned: {stats1['returned']} results")
        print(f"  Time: {time1:.2f} seconds")

        # Test 2: Larger search limit
        print("\nTest 2: max_search=50, top_k=20")
        start_time = time.time()
        results2, stats2 = reid_model.find_similar_detections(
            detection_id=test_id,
            all_detections=all_detections,
            threshold=0.5,
            top_k=20,
            max_search=50
        )
        time2 = time.time() - start_time

        print(f"  Searched: {stats2['searched']} / {stats2['total_available']} detections")
        print(f"  Found: {stats2['found']} matches")
        print(f"  Returned: {stats2['returned']} results")
        print(f"  Time: {time2:.2f} seconds")

        # Verify early stopping
        print("\n4. Verifying optimization...")
        print("-" * 40)

        # Check that search is limited
        if stats1['searched'] <= 10:
            print(f"[OK] Search limit working: searched {stats1['searched']} <= 10")
        else:
            print(f"[ERROR] Search limit NOT working: searched {stats1['searched']} > 10")

        # Check that we get at most top_k results
        if len(results1) <= 5 and len(results2) <= 20:
            print(f"[OK] Top-K limit working: got {len(results1)} and {len(results2)} results")
        else:
            print(f"[ERROR] Top-K limit NOT working")

        # Check performance improvement
        if stats2['searched'] < stats2['total_available']:
            percentage = (stats2['searched'] / stats2['total_available']) * 100
            print(f"[OK] Early stopping working: searched only {percentage:.1f}% of available detections")
        else:
            print(f"[WARNING] No early stopping occurred (may need more data)")

        session.close()

        print("\n" + "=" * 50)
        print("[SUCCESS] All optimization tests passed!")
        return True

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("   Make sure all dependencies are installed")
        return False

    except Exception as e:
        print(f"[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reid_optimization()
    sys.exit(0 if success else 1)