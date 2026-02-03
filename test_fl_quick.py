"""
Quick Federated Learning Test - Fast validation
Tests the FL workflow with minimal training for quick verification
"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # Set before any keras imports

import sys
import numpy as np
from pathlib import Path

# Import FL components
from backend_fl.config import *
from backend_fl.model import get_model
from backend_fl.data_utils import load_cifar10, partition_data_non_iid

print("=" * 70)
print("QUICK FL TEST - Validating System Components")
print("=" * 70)
print(f"Configuration:")
print(f"  - Clients: 2")
print(f"  - Rounds: 1 (quick test)")
print(f"  - Local Epochs: 1 (quick test)")
print(f"  - Batch Size: {BATCH_SIZE}")
print("=" * 70)
print()

# Step 1: Load and partition data
print("[1/4] Loading CIFAR-10 dataset...")
X_train, y_train, X_test, y_test = load_cifar10(normalize=True)

# Use only subset for quick test
print("  Using subset of data for quick test...")
X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:1000]
y_test = y_test[:1000]
print(f"  - Training samples: {X_train.shape[0]}")
print(f"  - Test samples: {X_test.shape[0]}")

# Partition data for 2 clients
num_clients = 2
client_data = partition_data_non_iid(
    X_train, y_train, num_clients=num_clients, alpha=ALPHA
)
print(f"  - Data partitioned into {num_clients} Non-IID clients")
for i, data in enumerate(client_data):
    print(f"    Client {i}: {len(data['y'])} samples")
print()

# Step 2: Initialize global model
print("[2/4] Initializing global model...")
global_model = get_model()
print(f"  - Model created with {global_model.count_params():,} parameters")
print()

# Step 3: Simulate one round of federated training
print("[3/4] Testing federated training (1 round)...")

client_weights = []
client_num_samples = []

for client_id in range(num_clients):
    print(f"\n  [Client {client_id}] Training...")

    # Create client model
    client_model = get_model()
    client_model.set_weights(global_model.get_weights())

    # Get client's data
    X_client = client_data[client_id]["X"]
    y_client = client_data[client_id]["y"]

    print(f"    - Training on {len(X_client)} samples for 1 epoch...")

    # Train for just 1 epoch
    history = client_model.fit(
        X_client,
        y_client,
        batch_size=BATCH_SIZE,
        epochs=1,
        verbose=0,
        validation_split=0.1,
    )

    final_loss = history.history["loss"][-1]
    final_acc = history.history["accuracy"][-1]

    print(f"    - Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")

    client_weights.append(client_model.get_weights())
    client_num_samples.append(len(X_client))

print(f"\n  [Server] Aggregating client models...")

# FedAvg aggregation
total_samples = sum(client_num_samples)
aggregated_weights = []

for layer_idx in range(len(client_weights[0])):
    layer_weighted_sum = np.zeros_like(client_weights[0][layer_idx], dtype=np.float32)

    for client_id in range(num_clients):
        weight = client_num_samples[client_id] / total_samples
        layer_weighted_sum += weight * client_weights[client_id][layer_idx]

    aggregated_weights.append(layer_weighted_sum)

global_model.set_weights(aggregated_weights)
print(f"    - Aggregated weights from {num_clients} clients")

# Evaluate
print(f"\n  [Server] Evaluating global model...")
test_loss, test_accuracy = global_model.evaluate(X_test, y_test, verbose=0)
print(f"    - Global model loss: {test_loss:.4f}")
print(f"    - Global model accuracy: {test_accuracy:.4f} ({test_accuracy:.1%})")
print()

# Step 4: Save model
print("[4/4] Saving model...")
model_path = Path("models") / "global_model.h5"
global_model.save(model_path)
print(f"  - Model saved: {model_path}")
print()

print("=" * 70)
print("[OK] Quick FL test completed successfully!")
print("=" * 70)
print()
print("Summary:")
print(f"  [OK] Data loading and partitioning works")
print(f"  [OK] Model creation and training works")
print(f"  [OK] FedAvg aggregation works")
print(f"  [OK] Model evaluation works")
print(f"  [OK] Model saving works")
print()
print(f"Model accuracy after 1 round: {test_accuracy:.1%}")
print(f"(Note: This is a quick test with limited data and 1 epoch)")
print()
print("Next step: Try running the full client-server setup with:")
print("  Terminal 1: python run_server.py --num-rounds 3 --min-clients 2")
print("  Terminal 2: python run_client.py --client-id 0 --num-clients 2")
print("  Terminal 3: python run_client.py --client-id 1 --num-clients 2")
