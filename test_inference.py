"""
Test the inference functionality
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

from frontend_web.inference import get_classifier
from PIL import Image
import numpy as np

# Test with a synthetic image
print("=" * 70)
print("TESTING INFERENCE MODULE")
print("=" * 70)

# Get classifier instance
classifier = get_classifier()

# Print model info
info = classifier.get_model_info()
print("\nModel Information:")
for key, value in info.items():
    if key != "classes":
        print(f"  {key}: {value}")

if not classifier.model_loaded:
    print("\n[ERROR] Model not loaded!")
    print("The model needs to be trained first using FL training.")
    exit(1)

# Create a test image (random 32x32 RGB)
print("\nCreating synthetic test image...")
test_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
img = Image.fromarray(test_img)
test_path = "uploads/test_image.png"
img.save(test_path)
print(f"  Saved to: {test_path}")

# Make prediction
print("\nRunning prediction...")
result = classifier.predict(test_path)

if result["success"]:
    print("\n[SUCCESS] Prediction completed!")
    print(f"  Predicted Class: {result['predicted_class']}")
    print(
        f"  Confidence: {result['confidence']:.4f} ({result['confidence'] * 100:.2f}%)"
    )
    if result.get("superclass"):
        print(f"  Superclass: {result['superclass']}")
    print(f"\n  Top 5 Predictions:")
    for i, pred in enumerate(result["top_5"], 1):
        print(f"    {i}. {pred['class_name']:20s} - {pred['probability'] * 100:5.2f}%")
else:
    print(f"\n[ERROR] Prediction failed: {result.get('error', 'Unknown error')}")

# Cleanup
try:
    os.remove(test_path)
    print(f"\n  Cleaned up test image")
except:
    pass

print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)
