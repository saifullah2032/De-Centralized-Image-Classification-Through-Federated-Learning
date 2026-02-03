"""
Federated Learning Client
Trains model locally on private data and sends updates to server
"""

import argparse
from typing import Dict, Tuple, List

import flwr as fl
import numpy as np

from backend_fl.config import (
    FL_CLIENT_SERVER_ADDRESS,
    LOCAL_EPOCHS,
    BATCH_SIZE,
    NUM_CLIENTS,
)
from backend_fl.model import get_model
from backend_fl.data_utils import load_dataset, partition_data_non_iid, get_client_data


class CIFARClient(fl.client.NumPyClient):
    """
    Federated Learning client for CIFAR-10 classification

    Each client:
    1. Receives global model weights from server
    2. Trains locally on private data for E epochs
    3. Sends updated weights back to server
    4. Never shares raw data
    """

    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        local_epochs: int = LOCAL_EPOCHS,
        batch_size: int = BATCH_SIZE,
    ):
        """
        Initialize client

        Args:
            client_id: Unique client identifier
            X_train: Local training images
            y_train: Local training labels (one-hot)
            X_test: Local test images
            y_test: Local test labels (one-hot)
            local_epochs: Number of local training epochs per round
            batch_size: Training batch size
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        # Create model
        self.model = get_model(pretrained=False)

        # Print client info
        print(f"\n{'=' * 70}")
        print(f"CLIENT {client_id} INITIALIZED")
        print(f"{'=' * 70}")
        print(f"  Training samples:   {len(X_train)}")
        print(f"  Test samples:       {len(X_test)}")
        print(f"  Local epochs:       {local_epochs}")
        print(f"  Batch size:         {batch_size}")
        print(f"{'=' * 70}\n")

        # Calculate class distribution
        y_labels = np.argmax(y_train, axis=1)
        unique, counts = np.unique(y_labels, return_counts=True)
        print(f"Class distribution:")
        for class_id, count in zip(unique, counts):
            print(
                f"  Class {class_id}: {count} samples ({count / len(y_train) * 100:.1f}%)"
            )
        print()

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters (weights) as NumPy arrays

        Args:
            config: Configuration dictionary from server

        Returns:
            List of NumPy arrays containing model weights
        """
        return self.model.get_weights()

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model locally on private data

        Args:
            parameters: Model weights from server
            config: Configuration dictionary from server

        Returns:
            Tuple of (updated_weights, num_samples, metrics)
        """
        # Update local model with global weights
        self.model.set_weights(parameters)

        # Get current round number
        current_round = config.get("server_round", 0)

        print(f"\n{'=' * 70}")
        print(f"CLIENT {self.client_id} - ROUND {current_round} TRAINING")
        print(f"{'=' * 70}")

        # Train locally
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=2,
        )

        # Get training metrics
        final_loss = float(history.history["loss"][-1])
        final_acc = float(history.history["accuracy"][-1])

        print(f"\nClient {self.client_id} Training Complete:")
        print(f"  Loss:     {final_loss:.4f}")
        print(f"  Accuracy: {final_acc:.4f} ({final_acc * 100:.2f}%)")
        print(f"{'=' * 70}\n")

        # Return updated weights and metrics
        return (
            self.model.get_weights(),
            len(self.X_train),
            {"loss": final_loss, "accuracy": final_acc, "client_id": self.client_id},
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local test data

        Args:
            parameters: Model weights from server
            config: Configuration dictionary from server

        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        # Update local model with global weights
        self.model.set_weights(parameters)

        # Evaluate on local test set
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)

        print(
            f"Client {self.client_id} Evaluation: Loss={loss:.4f}, Accuracy={accuracy:.4f}"
        )

        return (float(loss), len(self.X_test), {"accuracy": float(accuracy)})


def start_client(
    client_id: int,
    server_address: str = None,
    num_clients: int = NUM_CLIENTS,
):
    """
    Start a federated learning client

    Args:
        client_id: Unique client identifier (0-indexed)
        server_address: Server address to connect to
        num_clients: Total number of clients for data partitioning
    """
    # Use default if not specified
    if server_address is None:
        from backend_fl.config import FL_CLIENT_SERVER_ADDRESS

        server_address = FL_CLIENT_SERVER_ADDRESS

    print(f"\n{'=' * 70}")
    print(f"STARTING FEDERATED LEARNING CLIENT {client_id}")
    print(f"{'=' * 70}")
    print(f"  Server address: {server_address}")
    print(f"  Client ID:      {client_id}")
    print(f"  Total clients:  {num_clients}")
    print(f"{'=' * 70}\n")

    # Load and partition data
    print("Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_dataset()

    print(f"Partitioning data for {num_clients} clients...")
    partitions = partition_data_non_iid(X_train, y_train, num_clients=num_clients)

    # Get this client's data
    X_train_client, y_train_client = get_client_data(client_id, partitions)

    # Use a portion of test set for local validation
    # Each client gets an equal share of the test set
    test_partition_size = len(X_test) // num_clients
    start_idx = client_id * test_partition_size
    end_idx = start_idx + test_partition_size
    X_test_client = X_test[start_idx:end_idx]
    y_test_client = y_test[start_idx:end_idx]

    # Create client
    client = CIFARClient(
        client_id=client_id,
        X_train=X_train_client,
        y_train=y_train_client,
        X_test=X_test_client,
        y_test=y_test_client,
    )

    # Connect to server and start training
    print(f"Connecting to server at {server_address}...")

    try:
        fl.client.start_numpy_client(server_address=server_address, client=client)

        print(f"\n{'=' * 70}")
        print(f"CLIENT {client_id} - TRAINING COMPLETED")
        print(f"{'=' * 70}\n")

    except KeyboardInterrupt:
        print(f"\n\nClient {client_id} interrupted by user. Disconnecting...")
    except Exception as e:
        print(f"\n\n❌ Client {client_id} error: {e}")
        raise


def main():
    """Main entry point for client"""
    parser = argparse.ArgumentParser(description="Federated Learning Client")

    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        help="Unique client identifier (0-indexed)",
    )

    parser.add_argument(
        "--server-address",
        type=str,
        default=FL_CLIENT_SERVER_ADDRESS,
        help=f"Server address (default: {FL_CLIENT_SERVER_ADDRESS})",
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=NUM_CLIENTS,
        help=f"Total number of clients (default: {NUM_CLIENTS})",
    )

    args = parser.parse_args()

    # Validate client ID
    if args.client_id < 0 or args.client_id >= args.num_clients:
        print(f"❌ Error: Client ID must be between 0 and {args.num_clients - 1}")
        return

    # Start client
    start_client(
        client_id=args.client_id,
        server_address=args.server_address,
        num_clients=args.num_clients,
    )


if __name__ == "__main__":
    main()
