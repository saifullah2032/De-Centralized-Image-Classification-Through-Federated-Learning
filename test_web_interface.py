"""
Web Interface Test Script
Tests the prediction functionality without needing a browser
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import sys
from pathlib import Path
import numpy as np
from PIL import Image

print("=" * 70)
print("WEB INTERFACE FUNCTIONALITY TEST")
print("=" * 70)
print()

# Import the inference module
try:
    from frontend_web.inference import ImageClassifier, get_classifier

    print("[OK] Inference module imported successfully")
except Exception as e:
    print(f"[X] Failed to import inference module: {e}")
    sys.exit(1)

# Check if model exists
model_path = Path("models/global_model.h5")
if not model_path.exists():
    print(f"[X] Model not found: {model_path}")
    print("   Run federated training first to generate the model.")
    sys.exit(1)

print(f"[OK] Model found: {model_path}")
print()

# Load the model
print("Loading trained model...")
try:
    classifier = ImageClassifier()
    if not classifier.model_loaded:
        print("[X] Failed to load model")
        sys.exit(1)
    model = classifier.model
    print(f"[OK] Model loaded successfully")
    print(f"  - Parameters: {model.count_params():,}")
    print()
except Exception as e:
    print(f"[X] Failed to load model: {e}")
    sys.exit(1)

# Test with CIFAR-10 test images
print("Testing predictions with CIFAR-10 test images...")
print()

from backend_fl.data_utils import get_test_set
from backend_fl.config import CIFAR10_LABELS

X_test, y_test = get_test_set()

# Test on 5 random images
test_indices = np.random.choice(len(X_test), 5, replace=False)

print("=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)

correct = 0
for i, idx in enumerate(test_indices, 1):
    test_image = X_test[idx]
    true_label_idx = np.argmax(y_test[idx])
    true_label = CIFAR10_LABELS[true_label_idx]

    # Make prediction
    predictions = model.predict(np.expand_dims(test_image, axis=0), verbose=0)
    pred_idx = np.argmax(predictions[0])
    pred_label = CIFAR10_LABELS[pred_idx]
    confidence = predictions[0][pred_idx] * 100

    # Check if correct
    is_correct = pred_idx == true_label_idx
    if is_correct:
        correct += 1

    status = "[CORRECT]" if is_correct else "[WRONG]"

    print(f"\nTest {i}:")
    print(f"  True label:      {true_label}")
    print(f"  Predicted label: {pred_label}")
    print(f"  Confidence:      {confidence:.2f}%")
    print(f"  Status:          {status}")

    # Show top 3 predictions
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    print(f"  Top 3 predictions:")
    for j, idx in enumerate(top3_indices, 1):
        print(f"    {j}. {CIFAR10_LABELS[idx]}: {predictions[0][idx] * 100:.2f}%")

print()
print("=" * 70)
print(f"Accuracy on sample: {correct}/5 ({correct / 5 * 100:.0f}%)")
print("=" * 70)
print()

# Test model evaluation on full test set
print("Evaluating model on full test set (10,000 images)...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"[OK] Evaluation complete:")
print(f"  - Loss:     {loss:.4f}")
print(f"  - Accuracy: {accuracy:.2%} ({accuracy * 100:.2f}%)")
print()

# Save a test image for web interface testing
print("Creating sample test images for web interface...")
sample_dir = Path("test_images")
sample_dir.mkdir(exist_ok=True)

# Save 5 test images with labels
for i in range(5):
    idx = np.random.randint(0, len(X_test))
    img = X_test[idx]
    true_label = CIFAR10_LABELS[np.argmax(y_test[idx])]

    # Convert from normalized float to uint8
    img_uint8 = (img * 255).astype(np.uint8)

    # Create PIL image
    pil_img = Image.fromarray(img_uint8)

    # Save with label in filename
    filename = f"test_{i + 1}_{true_label}.png"
    pil_img.save(sample_dir / filename)

print(f"[OK] Sample images saved to: {sample_dir}/")
print()

print("=" * 70)
print("WEB INTERFACE TEST SUMMARY")
print("=" * 70)
print()
print("[OK] Model loading works")
print("[OK] Prediction functionality works")
print("[OK] Model evaluation works")
print(f"[OK] Current model accuracy: {accuracy * 100:.2f}%")
print("[OK] Sample test images created")
print()
print("Next steps:")
print("  1. Start web interface: python run_web.py")
print("  2. Visit: http://localhost:5000")
print("  3. Login: admin / admin123")
print("  4. Upload images from test_images/ folder")
print("  5. See predictions in real-time")
print()
print("=" * 70)
