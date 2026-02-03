"""
Quick test script to verify all modules load correctly
"""

import os
import sys

# Fix encoding for Windows console
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.environ["KERAS_BACKEND"] = "jax"

print("Testing module imports...")

# Test config
print("1. Testing config...")
from backend_fl import config

print("   [OK] Config loaded")

# Test model
print("2. Testing model...")
from backend_fl.model import get_model

model = get_model()
print(f"   [OK] Model created ({model.count_params():,} params)")

# Test data utils
print("3. Testing data utils...")
from backend_fl.data_utils import load_cifar10

print("   Loading CIFAR-10 (this may take a moment)...")
X_train, y_train, X_test, y_test = load_cifar10()
print(f"   [OK] Data loaded ({X_train.shape[0]} train, {X_test.shape[0]} test)")

# Test auth
print("4. Testing auth...")
from frontend_web.auth import get_user

user = get_user("admin")
print(f"   [OK] Auth module loaded (admin user: {user})")

# Test inference
print("5. Testing inference...")
from frontend_web.inference import ImageClassifier

classifier = ImageClassifier()
print(f"   [OK] Inference module loaded (model loaded: {classifier.model_loaded})")

print("\n" + "=" * 60)
print("[SUCCESS] ALL MODULES LOADED SUCCESSFULLY!")
print("=" * 60)
print("\nYou can now start:")
print("  1. FL Server:  python run_server.py")
print("  2. FL Clients: python run_client.py --client-id 0")
print("  3. Web App:    python run_web.py")
