"""
Automated Federated Learning Simulation Script
Tests the complete FL workflow in a single process for easier testing.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # Set before any keras imports

import sys
import time
import threading
import numpy as np
from pathlib import Path

# Import FL components
from backend_fl.config import *
from backend_fl.model import get_model
from backend_fl.data_utils import load_cifar10, partition_data_non_iid, get_test_set

print("=" * 70)
print("FEDERATED LEARNING SIMULATION TEST")
print("=" * 70)
print(f"Configuration:")
print(f"  - Clients: 2")
print(f"  - Rounds: 3")
print(f"  - Local Epochs: {LOCAL_EPOCHS}")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Non-IID Alpha: {ALPHA}")
print("=" * 70)
print()

# Step 1: Load and partition data
print("[STEP 1/5] Loading and partitioning CIFAR-10 dataset...")
X_train, y_train, X_test, y_test = load_cifar10(normalize=True)
print(f"  - Training samples: {X_train.shape[0]}")
print(f"  - Test samples: {X_test.shape[0]}")

# Partition data for 2 clients with Non-IID distribution
num_clients = 2
client_data = partition_data_non_iid(
    X_train, y_train, num_clients=num_clients, alpha=ALPHA
)
print(f"  - Data partitioned into {num_clients} Non-IID clients")
for i, data in enumerate(client_data):
    print(f"    Client {i}: {len(data['y'])} samples")
print()

# Step 2: Initialize global model
print("[STEP 2/5] Initializing global model...")
global_model = get_model()
print(f"  - Model created with {global_model.count_params():,} parameters")
print()

# Step 3: Simulate federated training
print("[STEP 3/5] Starting federated training simulation...")
print()

num_rounds = 3
training_history = {
    "rounds": [],
    "global_accuracy": [],
    "global_loss": [],
    "client_metrics": [],
}

for round_num in range(1, num_rounds + 1):
    print(f"{'=' * 70}")
    print(f"ROUND {round_num}/{num_rounds}")
    print(f"{'=' * 70}")

    # Store client weights for aggregation
    client_weights = []
    client_num_samples = []
    round_client_metrics = []

    # Each client trains locally
    for client_id in range(num_clients):
        print(f"\n[Client {client_id}] Starting local training...")

        # Create a fresh model copy for this client
        client_model = get_model()

        # Set client model weights to current global weights
        client_model.set_weights(global_model.get_weights())

        # Get client's data
        X_client = client_data[client_id]["X"]
        y_client = client_data[client_id]["y"]

        print(f"  - Training on {len(X_client)} local samples")
        print(f"  - Training for {LOCAL_EPOCHS} epochs...")

        # Train client model
        history = client_model.fit(
            X_client,
            y_client,
            batch_size=BATCH_SIZE,
            epochs=LOCAL_EPOCHS,
            verbose=0,  # Silent training
            validation_split=0.1,
        )

        # Get final metrics
        final_loss = history.history["loss"][-1]
        final_acc = history.history["accuracy"][-1]

        print(f"  - Final training loss: {final_loss:.4f}")
        print(f"  - Final training accuracy: {final_acc:.4f}")

        # Store weights and sample count for aggregation
        client_weights.append(client_model.get_weights())
        client_num_samples.append(len(X_client))
        round_client_metrics.append(
            {
                "client_id": client_id,
                "loss": float(final_loss),
                "accuracy": float(final_acc),
                "num_samples": len(X_client),
            }
        )

    print(f"\n[Server] Aggregating client models using FedAvg...")

    # FedAvg: Weighted average of client weights
    total_samples = sum(client_num_samples)

    # Initialize aggregated weights
    aggregated_weights = []

    # For each layer's weights
    for layer_idx in range(len(client_weights[0])):
        # Weighted sum of this layer's weights from all clients
        layer_weighted_sum = np.zeros_like(
            client_weights[0][layer_idx], dtype=np.float32
        )

        for client_id in range(num_clients):
            weight = client_num_samples[client_id] / total_samples
            layer_weighted_sum += weight * client_weights[client_id][layer_idx]

        aggregated_weights.append(layer_weighted_sum)

    # Update global model with aggregated weights
    global_model.set_weights(aggregated_weights)
    print(f"  - Aggregated weights from {num_clients} clients")

    # Evaluate global model on test set
    print(f"\n[Server] Evaluating global model on test set...")
    test_loss, test_accuracy = global_model.evaluate(X_test, y_test, verbose=0)

    print(f"  - Global model loss: {test_loss:.4f}")
    print(f"  - Global model accuracy: {test_accuracy:.4f}")

    # Save metrics
    training_history["rounds"].append(round_num)
    training_history["global_accuracy"].append(float(test_accuracy))
    training_history["global_loss"].append(float(test_loss))
    training_history["client_metrics"].append(round_client_metrics)

    # Save model checkpoint
    round_model_path = MODELS_DIR / f"model_round_{round_num}.h5"
    global_model.save(round_model_path)
    print(f"  - Model saved: {round_model_path}")

    print()

print(f"{'=' * 70}")
print("TRAINING COMPLETED")
print(f"{'=' * 70}")
print()

# Step 4: Save final model
print("[STEP 4/5] Saving final global model...")
final_model_path = MODELS_DIR / "global_model.h5"
global_model.save(final_model_path)
print(f"  - Final model saved: {final_model_path}")
print()

# Step 5: Save training history
print("[STEP 5/5] Saving training history...")
import json

history_path = MODELS_DIR / "model_history.json"
with open(history_path, "w") as f:
    json.dump(training_history, f, indent=2)
print(f"  - History saved: {history_path}")
print()

# Display summary
print(f"{'=' * 70}")
print("TRAINING SUMMARY")
print(f"{'=' * 70}")
print(f"\nAccuracy progression:")
for i, (round_num, acc) in enumerate(
    zip(training_history["rounds"], training_history["global_accuracy"])
):
    improvement = ""
    if i > 0:
        diff = acc - training_history["global_accuracy"][i - 1]
        improvement = f" ({diff:+.2%} from previous round)"
    print(f"  Round {round_num}: {acc:.2%}{improvement}")

print(f"\nFinal Results:")
print(f"  - Initial Accuracy: {training_history['global_accuracy'][0]:.2%}")
print(f"  - Final Accuracy: {training_history['global_accuracy'][-1]:.2%}")
print(
    f"  - Total Improvement: {training_history['global_accuracy'][-1] - training_history['global_accuracy'][0]:+.2%}"
)

print(f"\nFiles created:")
print(f"  - {final_model_path}")
print(f"  - {history_path}")
for round_num in range(1, num_rounds + 1):
    print(f"  - {MODELS_DIR / f'model_round_{round_num}.h5'}")

print(f"\n{'=' * 70}")
print("[OK] Federated Learning simulation completed successfully!")
print(f"{'=' * 70}")
