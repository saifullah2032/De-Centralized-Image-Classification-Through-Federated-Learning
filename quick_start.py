"""
Quick Start Script for Enhanced Federated Learning Training
Automates the process of continuing training with improvements
"""

import os
import sys
import subprocess
from pathlib import Path

print("=" * 70)
print("ENHANCED FEDERATED LEARNING - QUICK START")
print("=" * 70)
print()

# Check if model exists
model_path = Path("models/global_model.h5")
history_path = Path("models/model_history.json")

if model_path.exists():
    print("[OK] Found existing model: models/global_model.h5")

    if history_path.exists():
        import json

        with open(history_path) as f:
            history = json.load(f)
        rounds_completed = len(history["rounds"])
        final_accuracy = history["accuracies"][-1] * 100

        print(f"[OK] Training history found:")
        print(f"  - Rounds completed: {rounds_completed}")
        print(f"  - Current accuracy: {final_accuracy:.2f}%")
        print()
        print("[INFO] You can continue training from this checkpoint!")
        print()
else:
    print("[INFO] No existing model found - will start fresh training")
    print()

print("-" * 70)
print("IMPROVEMENTS IMPLEMENTED:")
print("-" * 70)
print("✓ Enhanced model architecture (2.3M params, was 871K)")
print("✓ Optimized hyperparameters (5 epochs, batch=64, lr=0.0005)")
print("✓ Better regularization (L2 + 3x BatchNorm)")
print("✓ Data augmentation support")
print()

print("-" * 70)
print("RECOMMENDED TRAINING OPTIONS:")
print("-" * 70)
print()
print("Option 1: Continue for 10 more rounds (RECOMMENDED)")
print("  - Starting point: 53.86% accuracy")
print("  - Expected result: 70-75% accuracy")
print("  - Time: ~2 hours")
print("  - Command: python run_server.py --num-rounds 10 --min-clients 2")
print()

print("Option 2: Fresh 20-round training")
print("  - Starting point: 0% (random)")
print("  - Expected result: 75-85% accuracy")
print("  - Time: ~4-5 hours")
print(
    "  - First backup models: mkdir models_backup && move models\\*.h5 models_backup\\"
)
print()

print("Option 3: Extended 20 more rounds")
print("  - Starting point: 53.86% accuracy")
print("  - Expected result: 80-85% accuracy")
print("  - Time: ~4 hours")
print("  - Command: python run_server.py --num-rounds 20 --min-clients 2")
print()

print("=" * 70)
print("TO START TRAINING:")
print("=" * 70)
print()
print("Open 3 terminals and run:")
print()
print("Terminal 1 (Server):")
print("  .\\venv\\Scripts\\Activate.ps1")
print("  python run_server.py --num-rounds 10 --min-clients 2")
print()
print("Terminal 2 (Client 0):")
print("  .\\venv\\Scripts\\Activate.ps1")
print("  python run_client.py --client-id 0 --num-clients 2")
print()
print("Terminal 3 (Client 1):")
print("  .\\venv\\Scripts\\Activate.ps1")
print("  python run_client.py --client-id 1 --num-clients 2")
print()

print("=" * 70)
print("AFTER TRAINING:")
print("=" * 70)
print()
print("1. Visualize results:")
print("   python visualize_training.py")
print()
print("2. Test predictions:")
print("   python run_web.py")
print("   Visit: http://localhost:5000")
print("   Login: admin / admin123")
print()

print("=" * 70)
print()

# Ask if user wants to proceed
try:
    choice = input("Would you like me to show the full commands? (y/n): ")

    if choice.lower() == "y":
        print()
        print("=" * 70)
        print("FULL COMMAND SEQUENCE:")
        print("=" * 70)
        print()
        print("# In PowerShell Terminal 1:")
        print(".\\venv\\Scripts\\Activate.ps1")
        print("python run_server.py --num-rounds 10 --min-clients 2")
        print()
        print("# In PowerShell Terminal 2:")
        print(".\\venv\\Scripts\\Activate.ps1")
        print("python run_client.py --client-id 0 --num-clients 2")
        print()
        print("# In PowerShell Terminal 3:")
        print(".\\venv\\Scripts\\Activate.ps1")
        print("python run_client.py --client-id 1 --num-clients 2")
        print()
        print("=" * 70)

except KeyboardInterrupt:
    print()
    print("Exiting...")
    sys.exit(0)

print()
print("Good luck with training! 🚀")
print()
