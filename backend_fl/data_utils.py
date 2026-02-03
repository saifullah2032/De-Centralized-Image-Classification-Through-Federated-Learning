"""
Data Loading and Partitioning Utilities
Handles CIFAR-10/CIFAR-100 dataset loading and Non-IID data partitioning using Dirichlet distribution
"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # Must be set before importing Keras

import numpy as np
import keras
from keras.datasets import cifar10, cifar100
from keras.utils import to_categorical
from typing import Tuple, List, Dict

from backend_fl.config import NUM_CLASSES, NUM_CLIENTS, ALPHA, DATASET


def load_cifar10(
    normalize=True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 dataset

    Args:
        normalize (bool): Whether to normalize pixel values to [0, 1]

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    print("Loading CIFAR-10 dataset...")

    # Download and load CIFAR-10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize pixel values to [0, 1]
    if normalize:
        X_train = X_train.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0

    # Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(f"[OK] Dataset loaded successfully!")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples:     {X_test.shape[0]}")
    print(f"  Image shape:      {X_train.shape[1:]}")

    return X_train, y_train, X_test, y_test


def load_cifar100(
    normalize=True, label_mode="fine"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIFAR-100 dataset

    Args:
        normalize (bool): Whether to normalize pixel values to [0, 1]
        label_mode (str): 'fine' for 100 classes or 'coarse' for 20 superclasses

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    print(f"Loading CIFAR-100 dataset (label_mode={label_mode})...")

    # Download and load CIFAR-100
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode=label_mode)

    # Normalize pixel values to [0, 1]
    if normalize:
        X_train = X_train.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0

    # Convert labels to categorical (one-hot encoding)
    num_classes = 100 if label_mode == "fine" else 20
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(f"[OK] Dataset loaded successfully!")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples:     {X_test.shape[0]}")
    print(f"  Image shape:      {X_train.shape[1:]}")
    print(f"  Classes:          {num_classes} ({label_mode})")

    return X_train, y_train, X_test, y_test


def load_dataset(
    dataset_name: str = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unified dataset loader - automatically selects CIFAR-10 or CIFAR-100

    Args:
        dataset_name (str): 'CIFAR10' or 'CIFAR100' (defaults to config.DATASET)
        normalize (bool): Whether to normalize pixel values to [0, 1]

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    if dataset_name is None:
        dataset_name = DATASET

    if dataset_name == "CIFAR100":
        return load_cifar100(normalize=normalize, label_mode="fine")
    elif dataset_name == "CIFAR10":
        return load_cifar10(normalize=normalize)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Use 'CIFAR10' or 'CIFAR100'"
        )


def partition_data_non_iid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_clients: int = NUM_CLIENTS,
    alpha: float = ALPHA,
) -> List[Dict[str, np.ndarray]]:
    """
    Partition training data into Non-IID distributions using Dirichlet distribution

    Args:
        X_train: Training images
        y_train: Training labels (one-hot encoded)
        num_clients: Number of clients to partition data for
        alpha: Dirichlet distribution concentration parameter
               Lower alpha = more heterogeneous distribution

    Returns:
        List of dictionaries, each containing 'X' and 'y' for a client
    """
    print(f"\nPartitioning data for {num_clients} clients (alpha={alpha})...")

    # Convert one-hot labels to class indices
    y_train_labels = np.argmax(y_train, axis=1)

    # Initialize partitions
    client_partitions = [{"indices": []} for _ in range(num_clients)]

    # For each class, partition samples using Dirichlet distribution
    for class_idx in range(NUM_CLASSES):
        # Get indices of samples belonging to this class
        class_indices = np.where(y_train_labels == class_idx)[0]
        num_samples = len(class_indices)

        # Draw proportions from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)

        # Assign samples to clients based on proportions
        proportions = (np.cumsum(proportions) * num_samples).astype(int)
        proportions = [0] + proportions.tolist()

        # Shuffle indices within this class
        np.random.shuffle(class_indices)

        # Distribute to clients
        for client_id in range(num_clients):
            start_idx = proportions[client_id]
            end_idx = proportions[client_id + 1]
            client_partitions[client_id]["indices"].extend(
                class_indices[start_idx:end_idx].tolist()
            )

    # Create final partitions with actual data
    partitions = []
    for client_id in range(num_clients):
        indices = np.array(client_partitions[client_id]["indices"])

        # Shuffle indices
        np.random.shuffle(indices)

        # Extract data
        X_client = X_train[indices]
        y_client = y_train[indices]

        partitions.append({"X": X_client, "y": y_client, "num_samples": len(indices)})

        # Print statistics for this client
        y_client_labels = np.argmax(y_client, axis=1)
        class_counts = [np.sum(y_client_labels == i) for i in range(NUM_CLASSES)]

        print(
            f"  Client {client_id}: {len(indices)} samples, "
            f"distribution: {class_counts}"
        )

    # Validate Non-IID distribution
    validate_non_iid_distribution(partitions)

    return partitions


def validate_non_iid_distribution(partitions: List[Dict[str, np.ndarray]]):
    """
    Validate that the data distribution is truly Non-IID
    Uses Coefficient of Variation (CV) to measure heterogeneity

    Args:
        partitions: List of client partitions
    """
    print("\nValidating Non-IID distribution...")

    # Calculate class distribution for each client
    class_distributions = []

    for partition in partitions:
        y_labels = np.argmax(partition["y"], axis=1)
        class_counts = [np.sum(y_labels == i) for i in range(NUM_CLASSES)]
        class_distributions.append(class_counts)

    class_distributions = np.array(class_distributions)

    # Calculate Coefficient of Variation for each class
    cvs = []
    for class_idx in range(NUM_CLASSES):
        class_across_clients = class_distributions[:, class_idx]
        mean = np.mean(class_across_clients)
        std = np.std(class_across_clients)
        cv = std / mean if mean > 0 else 0
        cvs.append(cv)

    avg_cv = np.mean(cvs)

    print(f"  Coefficient of Variation (CV): {avg_cv:.4f}")

    if avg_cv > 0.3:
        print(f"  [OK] Distribution is Non-IID (CV > 0.3)")
    else:
        print(f"  ⚠ Warning: Distribution may not be sufficiently Non-IID (CV ≤ 0.3)")

    return avg_cv


def get_client_data(
    client_id: int, partitions: List[Dict[str, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get training data for a specific client

    Args:
        client_id: Client identifier (0-indexed)
        partitions: List of client partitions

    Returns:
        Tuple of (X_train_client, y_train_client)
    """
    if client_id >= len(partitions):
        raise ValueError(
            f"Client ID {client_id} out of range. "
            f"Only {len(partitions)} clients available."
        )

    partition = partitions[client_id]
    return partition["X"], partition["y"]


def get_test_set() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and return the test set for server-side evaluation

    Returns:
        Tuple of (X_test, y_test)
    """
    _, _, X_test, y_test = load_dataset(normalize=True)
    return X_test, y_test


def analyze_data_distribution(partitions: List[Dict[str, np.ndarray]]):
    """
    Analyze and visualize data distribution across clients

    Args:
        partitions: List of client partitions
    """
    print("\n" + "=" * 70)
    print("DATA DISTRIBUTION ANALYSIS")
    print("=" * 70)

    from backend_fl.config import CIFAR10_LABELS

    # Create distribution matrix
    distribution_matrix = []

    for client_id, partition in enumerate(partitions):
        y_labels = np.argmax(partition["y"], axis=1)
        class_counts = [np.sum(y_labels == i) for i in range(NUM_CLASSES)]
        distribution_matrix.append(class_counts)

        print(f"\nClient {client_id} ({partition['num_samples']} samples):")
        for class_idx, count in enumerate(class_counts):
            percentage = (count / partition["num_samples"]) * 100
            bar = "█" * int(percentage / 2)
            print(
                f"  {CIFAR10_LABELS[class_idx]:12s}: {count:4d} ({percentage:5.1f}%) {bar}"
            )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    """Test data loading and partitioning"""
    print("Testing data utilities...")

    # Load CIFAR-10
    X_train, y_train, X_test, y_test = load_dataset()

    # Create Non-IID partitions
    partitions = partition_data_non_iid(X_train, y_train, num_clients=5, alpha=0.5)

    # Analyze distribution
    analyze_data_distribution(partitions)

    # Test data access
    X_client, y_client = get_client_data(0, partitions)
    print(f"\nClient 0 data shape: X={X_client.shape}, y={y_client.shape}")

    print("\n[OK] Data utilities test passed!")
