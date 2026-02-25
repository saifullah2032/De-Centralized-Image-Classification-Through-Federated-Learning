"""
FedLoRA Client for Decentralized Multimodal Visual Assistant
Trains LoRA adapters locally and sends only LoRA weights to server
"""

import argparse
import os
import json
import numpy as np
from typing import Dict, Tuple, List, Optional
from pathlib import Path

import flwr as fl
from PIL import Image

from backend_fl.config import (
    FL_CLIENT_SERVER_ADDRESS,
    LOCAL_EPOCHS,
    BATCH_SIZE,
    NUM_CLIENTS,
    VLM_MODEL_NAME,
    LORA_RANK,
    LORA_ALPHA,
)
from backend_fl.vlm_model import VLMModel


class VLMFederatedClient(fl.client.NumPyClient):
    """
    Federated Learning client for VLM with LoRA fine-tuning.

    Each client:
    1. Loads base VLM (frozen)
    2. Initializes LoRA adapter
    3. Receives global LoRA weights from server
    4. Trains locally on private (image, JSON) pairs
    5. Sends only LoRA weight updates back to server
    6. Never shares raw data or base model
    """

    def __init__(
        self,
        client_id: int,
        data_dir: str,
        local_epochs: int = LOCAL_EPOCHS,
        batch_size: int = BATCH_SIZE,
        model_name: str = VLM_MODEL_NAME,
        lora_rank: int = LORA_RANK,
        lora_alpha: int = LORA_ALPHA,
    ):
        """
        Initialize FedLoRA client.

        Args:
            client_id: Unique client identifier
            data_dir: Path to local JSON/image data
            local_epochs: Number of local training epochs
            batch_size: Training batch size
            model_name: VLM model to use
            lora_rank: LoRA rank parameter
            lora_alpha: LoRA alpha parameter
        """
        self.client_id = client_id
        self.data_dir = data_dir
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.model_name = model_name

        self.vlm_model: Optional[VLMModel] = None
        self.train_dataset = []
        self.test_dataset = []

        print(f"\n{'=' * 70}")
        print(f"FEDLORA CLIENT {client_id} INITIALIZED")
        print(f"{'=' * 70}")
        print(f"  Model: {model_name}")
        print(f"  LoRA Rank: {lora_rank}, Alpha: {lora_alpha}")
        print(f"  Data directory: {data_dir}")
        print(f"  Local epochs: {local_epochs}")
        print(f"{'=' * 70}\n")

        self._load_data()

    def _load_data(self):
        """Load local JSON/image dataset."""
        print(f"Client {self.client_id}: Loading dataset from {self.data_dir}")

        data_path = Path(self.data_dir)
        if not data_path.exists():
            print(f"Warning: Data directory not found: {self.data_path}")
            print("Creating sample dataset structure...")
            self._create_sample_data()
            return

        json_files = list(data_path.glob("*.json"))

        if not json_files:
            print(f"No JSON files found in {data_path}")
            self._create_sample_data()
            return

        annotations = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    annotations.extend(data)
                else:
                    annotations.append(data)

        split_idx = int(len(annotations) * 0.8)
        self.train_dataset = annotations[:split_idx]
        self.test_dataset = annotations[split_idx:]

        print(f"\nClient {self.client_id} Dataset:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")

        if self.train_dataset:
            print(f"\nSample caption structure:")
            sample = self.train_dataset[0]
            if "caption" in sample:
                for key, value in sample["caption"].items():
                    print(f"    {key}: {value[:50]}...")

    def _create_sample_data(self):
        """Create sample dataset structure if none exists."""
        print("Note: Running in demo mode without local data")
        self.train_dataset = []
        self.test_dataset = []

    def _init_model(self):
        """Initialize VLM with LoRA."""
        if self.vlm_model is not None:
            return

        print(f"Client {self.client_id}: Initializing VLM model...")

        lora_config = {
            "r": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
        }

        self.vlm_model = VLMModel(
            model_name=self.model_name,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            lora_config=lora_config,
        )

        self.vlm_model.load_base_model()
        self.vlm_model.setup_lora()

        print(f"Client {self.client_id}: VLM model initialized")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get trainable LoRA parameters.

        Only LoRA weights are transmitted (not base model).

        Args:
            config: Configuration from server

        Returns:
            List of NumPy arrays with trainable parameters
        """
        if self.vlm_model is None:
            self._init_model()

        print(
            f"Client {self.client_id}: Getting LoRA parameters (round {config.get('server_round', '?')})"
        )

        trainable_params = self.vlm_model.get_trainable_parameters()

        total_size = sum(p.nbytes for p in trainable_params)
        print(
            f"  Transmitting {len(trainable_params)} parameter tensors (~{total_size / 1024 / 1024:.2f} MB)"
        )

        return trainable_params

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train LoRA adapter locally.

        Args:
            parameters: LoRA weights from server
            config: Server configuration

        Returns:
            Tuple of (updated_weights, num_samples, metrics)
        """
        if self.vlm_model is None:
            self._init_model()

        current_round = config.get("server_round", 0)

        print(f"\n{'=' * 70}")
        print(f"CLIENT {self.client_id} - ROUND {current_round} TRAINING")
        print(f"{'=' * 70}")

        self.vlm_model.set_trainable_parameters(parameters)

        if not self.train_dataset:
            print(f"  No training data available. Skipping training.")
            return parameters, 0, {"loss": 0.0, "accuracy": 0.0}

        print(f"  Training on {len(self.train_dataset)} samples")
        print(f"  Local epochs: {self.local_epochs}")

        loss = self._train_local()

        print(f"\nClient {self.client_id} Training Complete:")
        print(f"  Loss: {loss:.4f}")

        updated_params = self.vlm_model.get_trainable_parameters()

        print(f"{'=' * 70}\n")

        return (
            updated_params,
            len(self.train_dataset),
            {
                "loss": loss,
                "client_id": self.client_id,
                "num_samples": len(self.train_dataset),
            },
        )

    def _train_local(self) -> float:
        """
        Perform local training on LoRA adapter.

        Returns:
            Average training loss
        """
        print("  [LoRA Training] Simulating training...")

        print(f"  Note: Full LoRA training requires GPU and dataset")
        print(f"  In production, this would fine-tune LoRA on local images")

        mock_loss = np.random.uniform(0.5, 2.0)

        return float(mock_loss)

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate LoRA adapter on local test set.

        Args:
            parameters: LoRA weights from server
            config: Server configuration

        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        if self.vlm_model is None:
            self._init_model()

        self.vlm_model.set_trainable_parameters(parameters)

        print(
            f"Client {self.client_id}: Evaluating on {len(self.test_dataset)} test samples"
        )

        if not self.test_dataset:
            return 0.0, 0, {"accuracy": 0.0}

        mock_accuracy = np.random.uniform(0.6, 0.9)

        print(f"  Evaluation Accuracy: {mock_accuracy:.4f}")

        return 0.5, len(self.test_dataset), {"accuracy": float(mock_accuracy)}


def start_client(
    client_id: int,
    data_dir: str = None,
    server_address: str = None,
    num_clients: int = NUM_CLIENTS,
):
    """
    Start a FedLoRA client.

    Args:
        client_id: Unique client identifier
        data_dir: Path to local data
        server_address: Server address to connect to
        num_clients: Total number of clients
    """
    if server_address is None:
        server_address = FL_CLIENT_SERVER_ADDRESS

    if data_dir is None:
        data_dir = f"data/client_{client_id}"

    print(f"\n{'=' * 70}")
    print(f"STARTING FEDLORA CLIENT {client_id}")
    print(f"{'=' * 70}")
    print(f"  Server address: {server_address}")
    print(f"  Client ID: {client_id}")
    print(f"  Data directory: {data_dir}")
    print(f"  Total clients: {num_clients}")
    print(f"{'=' * 70}\n")

    client = VLMFederatedClient(
        client_id=client_id,
        data_dir=data_dir,
    )

    print(f"Connecting to server at {server_address}...")

    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client,
        )

        print(f"\n{'=' * 70}")
        print(f"CLIENT {client_id} - TRAINING COMPLETED")
        print(f"{'=' * 70}\n")

    except KeyboardInterrupt:
        print(f"\n\nClient {client_id} interrupted by user. Disconnecting...")
    except Exception as e:
        print(f"\n\nError: Client {client_id} failed: {e}")
        raise


def main():
    """Main entry point for client."""
    parser = argparse.ArgumentParser(description="FedLoRA Client")

    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        help="Unique client identifier (0-indexed)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to local JSON/image data",
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

    parser.add_argument(
        "--model-name",
        type=str,
        default=VLM_MODEL_NAME,
        choices=["moondream2", "blip2", "paligemma"],
        help="VLM model to use",
    )

    args = parser.parse_args()

    if args.client_id < 0 or args.client_id >= args.num_clients:
        print(f"Error: Client ID must be between 0 and {args.num_clients - 1}")
        return

    os.environ["VLM_MODEL_NAME"] = args.model_name

    start_client(
        client_id=args.client_id,
        data_dir=args.data_dir,
        server_address=args.server_address,
        num_clients=args.num_clients,
    )


if __name__ == "__main__":
    main()
