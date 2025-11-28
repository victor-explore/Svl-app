"""
Diagnostic script to test PersonReID initialization
Tests torchreid import, model loading, and embedding extraction
"""

import sys
import os
import logging

# Fix Windows console encoding for UTF-8 characters
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("PersonReID Initialization Diagnostic")
print("=" * 80)

# Test 1: Check torchreid import
print("\n[TEST 1] Checking torchreid import...")
try:
    import torch
    print(f"✓ torch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ Failed to import torch: {e}")
    sys.exit(1)

try:
    import torchvision
    print(f"✓ torchvision version: {torchvision.__version__}")
except ImportError as e:
    print(f"✗ Failed to import torchvision: {e}")
    sys.exit(1)

try:
    import torchreid
    print(f"✓ torchreid imported successfully")
    print(f"  torchreid version: {torchreid.__version__ if hasattr(torchreid, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"✗ Failed to import torchreid: {e}")
    print("\nPossible solutions:")
    print("  1. Reinstall torchreid: pip install --upgrade torchreid")
    print("  2. Check for dependency conflicts: pip check")
    sys.exit(1)

# Test 2: Check model file
print("\n[TEST 2] Checking OSNet model file...")
import os
from config import PERSON_REID_MODEL_PATH

if os.path.isabs(PERSON_REID_MODEL_PATH):
    model_path = PERSON_REID_MODEL_PATH
else:
    model_path = os.path.join(os.path.dirname(__file__), PERSON_REID_MODEL_PATH)

print(f"Model path: {model_path}")
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"✓ Model file exists ({file_size:.2f} MB)")
else:
    print(f"✗ Model file not found at: {model_path}")
    print("\nWill attempt to download from online...")

# Test 3: Initialize PersonReID model
print("\n[TEST 3] Initializing PersonReID model...")
try:
    from person_reid import PersonReID

    print("Creating PersonReID instance...")
    reid = PersonReID()

    if reid.initialized:
        print("✓ PersonReID initialized successfully!")
        print(f"  Device: {reid.device}")
        print(f"  Model: {reid.model.__class__.__name__}")
        print(f"  Transform: {reid.transform is not None}")
    else:
        print("✗ PersonReID initialization failed (initialized=False)")
        print("\nCheck the logs above for detailed error messages")
        sys.exit(1)

except Exception as e:
    print(f"✗ Exception during PersonReID initialization: {e}")
    import traceback
    print("\nFull traceback:")
    print(traceback.format_exc())
    sys.exit(1)

# Test 4: Test embedding extraction (if model initialized)
print("\n[TEST 4] Testing embedding extraction...")
try:
    import numpy as np
    from PIL import Image

    # Create a simple test image (dummy person image)
    test_image = Image.new('RGB', (256, 128), color=(128, 128, 128))
    test_path = 'test_reid_image.jpg'
    test_image.save(test_path)
    print(f"Created test image: {test_path}")

    # Extract embedding
    embedding = reid.extract_embedding(test_path)

    if embedding is not None:
        print(f"✓ Embedding extracted successfully!")
        print(f"  Shape: {embedding.shape}")
        print(f"  Type: {type(embedding)}")
        print(f"  Sample values: {embedding[:5]}")
    else:
        print("✗ Failed to extract embedding (returned None)")

    # Clean up test image
    os.remove(test_path)
    print(f"Cleaned up test image")

except Exception as e:
    print(f"✗ Exception during embedding extraction: {e}")
    import traceback
    print("\nFull traceback:")
    print(traceback.format_exc())

# Test 5: Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

if reid.initialized:
    print("✓ PersonReID is working correctly!")
    print("\nThe Re-ID model should work in your application.")
    print("If you still see errors, check the Flask application logs for runtime issues.")
else:
    print("✗ PersonReID initialization failed")
    print("\nReview the error messages above to identify the issue.")
    print("Common solutions:")
    print("  1. Reinstall dependencies: pip install --force-reinstall torchreid torch torchvision")
    print("  2. Check Python version compatibility (Python 3.8-3.11 recommended)")
    print("  3. Ensure model file is not corrupted (try re-downloading)")

print("=" * 80)
