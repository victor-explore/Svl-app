"""
Test PersonReID initialization in Flask application context
This simulates how the Re-ID model is initialized in app.py
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("Testing PersonReID in Flask Application Context")
print("=" * 80)

# Test lazy initialization (as done in app.py)
print("\n[TEST] Importing get_reid_instance (lazy import)...")
try:
    from person_reid import get_reid_instance
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test getting the instance (lazy initialization)
print("\n[TEST] Calling get_reid_instance() for first time...")
try:
    reid_model = get_reid_instance()
    print("✓ get_reid_instance() returned")
    print(f"  Instance type: {type(reid_model)}")
    print(f"  Initialized: {reid_model.initialized}")

    if reid_model.initialized:
        print("✓ PersonReID initialized successfully!")
        print(f"  Device: {reid_model.device}")
        print(f"  Model loaded: {reid_model.model is not None}")
    else:
        print("✗ PersonReID.initialized = False")
        print("\nThis is the error you're seeing in the Flask app!")

except Exception as e:
    print(f"✗ Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test calling it again (should return same instance)
print("\n[TEST] Calling get_reid_instance() second time (cached)...")
try:
    reid_model2 = get_reid_instance()
    print(f"✓ Second call successful")
    print(f"  Same instance: {reid_model is reid_model2}")
    print(f"  Initialized: {reid_model2.initialized}")
except Exception as e:
    print(f"✗ Exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

if reid_model.initialized:
    print("✓ PersonReID works correctly in Flask context!")
    print("\nIf you're still seeing errors in the Flask app:")
    print("  1. Restart the Flask server")
    print("  2. Check for import errors in the server logs")
    print("  3. Verify the detection_images directory exists")
else:
    print("✗ PersonReID initialization failed!")
    print("\nThe Re-ID features will not work until this is fixed.")
    print("Check the error messages above for details.")

print("=" * 80)
