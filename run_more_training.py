"""
Run Additional Federated Learning Training
Continues training the model for more rounds to improve accuracy
"""

import os
import sys
os.environ["KERAS_BACKEND"] = "jax"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variables for extended training
os.environ["NUM_ROUNDS"] = "10"  # Additional 10 rounds
os.environ["NUM_CLIENTS"] = "5"

from backend_fl.fl_server import main

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADDITIONAL FEDERATED LEARNING TRAINING")
    print("="*70)
    print("Running 10 more rounds to improve model accuracy")
    print("="*70 + "\n")
    main()
