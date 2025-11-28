"""
Check Person Re-ID initialization in Flask startup context
This simulates the exact startup sequence of app.py
"""

import sys
import os
import logging

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')

# Set up logging exactly like app.py
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 80)
print("Person Re-ID Startup Diagnostic")
print("=" * 80)

# Step 1: Test imports exactly as they happen in app.py
print("\n[STEP 1] Testing imports in Flask startup order...")

print("  Importing camera_manager...")
try:
    from camera_manager import EnhancedCameraManager, CameraStatus
    print("  ✓ camera_manager imported")
except Exception as e:
    print(f"  ✗ Failed to import camera_manager: {e}")

print("  Importing config...")
try:
    from config import *
    print("  ✓ config imported")
except Exception as e:
    print(f"  ✗ Failed to import config: {e}")

print("  Importing person_detector...")
try:
    import person_detector
    print("  ✓ person_detector imported")
except Exception as e:
    print(f"  ✗ Failed to import person_detector: {e}")

# Step 2: Test Re-ID initialization (lazy import like in app.py)
print("\n[STEP 2] Testing Person Re-ID lazy initialization...")
try:
    from person_reid import get_reid_instance
    print("  ✓ get_reid_instance imported")

    reid_instance = get_reid_instance()
    print(f"  ✓ get_reid_instance() called")
    print(f"  Initialized: {reid_instance.initialized}")

    if reid_instance.initialized:
        print(f"  ✓ Person Re-ID model initialized successfully!")
        print(f"    Device: {reid_instance.device}")
        print(f"    Model loaded: {reid_instance.model is not None}")
    else:
        print("  ✗ Person Re-ID model initialized = False")
        print("  This is the issue you're seeing in the Flask app!")

except Exception as e:
    print(f"  ✗ Exception during Re-ID initialization: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

# Step 3: Check TORCHREID_AVAILABLE flag
print("\n[STEP 3] Checking torchreid availability...")
try:
    from person_reid import TORCHREID_AVAILABLE
    print(f"  TORCHREID_AVAILABLE = {TORCHREID_AVAILABLE}")

    if not TORCHREID_AVAILABLE:
        print("  ✗ This is why Re-ID is failing!")
        print("  The import of torch/torchvision/torchreid failed")
except Exception as e:
    print(f"  ✗ Cannot check TORCHREID_AVAILABLE: {e}")

# Step 4: Direct import test
print("\n[STEP 4] Testing direct imports of torch/torchreid...")
try:
    import torch
    print(f"  ✓ torch version: {torch.__version__}")
except ImportError as e:
    print(f"  ✗ torch import failed: {e}")

try:
    import torchvision
    print(f"  ✓ torchvision version: {torchvision.__version__}")
except ImportError as e:
    print(f"  ✗ torchvision import failed: {e}")

try:
    import torchreid
    print(f"  ✓ torchreid imported")
except ImportError as e:
    print(f"  ✗ torchreid import failed: {e}")

# Step 5: Check for import order issues
print("\n[STEP 5] Checking for circular import issues...")
print("  Import order: camera_manager → person_detector → database")
print("  This could cause initialization to fail if database import fails")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

if 'reid_instance' in locals() and reid_instance.initialized:
    print("✓ Person Re-ID works correctly in startup context")
    print("\nIf you're still seeing errors in Flask:")
    print("  1. Stop the Flask server completely (Ctrl+C)")
    print("  2. Kill any remaining Python processes")
    print("  3. Restart Flask: python app.py")
    print("  4. Check the startup logs for detailed error messages")
else:
    print("✗ Person Re-ID initialization failed")
    print("\nThis explains the error you're seeing!")
    print("Check the error messages above to identify the root cause.")

print("=" * 80)
