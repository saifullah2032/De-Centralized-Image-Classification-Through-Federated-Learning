"""
FedLoRA Server for Decentralized Multimodal Visual Assistant
Orchestrates federated LoRA training across multiple clients
"""

import argparse
import os
import pickle
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any

import flwr as fl
import numpy as np

from backend_fl.config import (
    FL_SERVER_ADDRESS,
    NUM_ROUNDS,
    MIN_AVAILABLE_CLIENTS,
    LORA_RANK,
    LORA_ALPHA,
    LORA_WEIGHTS_DIR,
)


class FedLoRAStrategy:
    """
    Custom FedLoRA strategy for aggregating LoRA adapter weights.

    Unlike standard FedAvg which aggregates full model weights,
    FedLoRA only aggregates the trainable LoRA parameters.
    """

    def __init__(
        self,
        lora_weights_dir: str = LORA_WEIGHTS_DIR,
        min_available_clients: int = MIN_AVAILABLE_CLIENTS,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
    ):
        self.lora_weights_dir = lora_weights_dir
        self.min_available_clients = min_available_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.current_round = 0
        self.history = {
            "rounds": [],
            "losses": [],
            "accuracies": [],
            "clients": [],
        }

        os.makedirs(lora_weights_dir, exist_ok=True)

    def aggregate_fit(self, server_round: int, results: List, failures: List):
        """Aggregate LoRA weights from clients."""
        self.current_round = server_round

        if failures:
            print(f"Round {server_round}: {len(failures)} client failures")

        if not results:
            return None

        print(f"\n{'=' * 70}")
        print(f"FEDLORA AGGREGATION - ROUND {server_round}")
        print(f"{'=' * 70}")

        weighted_weights = []
        total_samples = 0

        for client, fit_results in results:
            num_samples = fit_results.num_examples
            client_id = fit_results.metrics.get("client_id", "unknown")

            print(f"  Client {client_id}: {num_samples} samples")

            weighted_weights.append((fit_results.parameters, num_samples))
            total_samples += num_samples

        aggregated_weights = self._compute_weighted_average(
            weighted_weights, total_samples
        )

        print(f"  Aggregated {len(aggregated_weights)} LoRA parameter tensors")

        weights_size = sum(w.nbytes for w in aggregated_weights)
        print(f"  Total LoRA weights size: {weights_size / 1024 / 1024:.2f} MB")

        self._save_global_lora(aggregated_weights, server_round)

        print(f"{'=' * 70}\n")

        return aggregated_weights

    def _compute_weighted_average(
        self,
        weighted_weights: List,
        total_samples: int,
    ) -> List[np.ndarray]:
        """Compute weighted average of LoRA weights."""
        aggregated = []

        if not weighted_weights:
            return aggregated

        for i in range(len(weighted_weights[0][0])):
            weighted_sum = None

            for weights, num_samples in weighted_weights:
                weight_factor = num_samples / total_samples

                if weighted_sum is None:
                    weighted_sum = weights[i] * weight_factor
                else:
                    weighted_sum += weights[i] * weight_factor

            aggregated.append(weighted_sum)

        return aggregated

    def _save_global_lora(
        self,
        weights: List[np.ndarray],
        round_num: int,
    ):
        """Save aggregated LoRA weights to disk."""
        round_dir = os.path.join(self.lora_weights_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        weights_path = os.path.join(round_dir, "lora_weights.pt")

        try:
            import torch

            state_dict = {
                f"lora_layer_{i}": torch.from_numpy(w) for i, w in enumerate(weights)
            }
            torch.save(state_dict, weights_path)
            print(f"  LoRA weights saved to: {weights_path}")
        except ImportError:
            weights_path = weights_path.replace(".pt", ".pkl")
            with open(weights_path, "wb") as f:
                pickle.dump(weights, f)
            print(f"  LoRA weights saved to: {weights_path}")

        latest_link = os.path.join(self.lora_weights_dir, "latest")
        if os.path.exists(latest_link):
            os.remove(latest_link)
        if os.path.exists(weights_path):
            shutil.copy(weights_path, latest_link)

    def aggregate_evaluate(self, server_round: int, results: List, failures: List):
        """Aggregate evaluation metrics from clients."""
        if failures:
            print(f"Round {server_round}: {len(failures)} evaluation failures")

        if not results:
            return None

        total_loss = 0.0
        total_samples = 0
        accuracies = []

        for client, eval_results in results:
            loss = eval_results.loss
            num_samples = eval_results.num_examples
            accuracy = eval_results.metrics.get("accuracy", 0.0)

            total_loss += loss * num_samples
            total_samples += num_samples
            accuracies.append(accuracy)

        avg_loss = total_loss / total_samples
        avg_accuracy = np.mean(accuracies)

        self.history["rounds"].append(server_round)
        self.history["losses"].append(avg_loss)
        self.history["accuracies"].append(avg_accuracy)
        self.history["clients"].append(len(results))

        print(f"  Round {server_round} Evaluation:")
        print(f"    Loss: {avg_loss:.4f}")
        print(f"    Accuracy: {avg_accuracy:.4f}")

        return avg_loss


class FedLoRAServer:
    """Flower server wrapper for FedLoRA."""

    def __init__(self, strategy: FedLoRAStrategy):
        self.strategy = strategy


def start_server(
    server_address: str = FL_SERVER_ADDRESS,
    num_rounds: int = NUM_ROUNDS,
    min_clients: int = MIN_AVAILABLE_CLIENTS,
    lora_weights_dir: str = LORA_WEIGHTS_DIR,
):
    """
    Start the FedLoRA server.

    Args:
        server_address: Server address (host:port)
        num_rounds: Number of training rounds
        min_clients: Minimum number of clients required
        lora_weights_dir: Directory to save LoRA weights
    """
    print("\n" + "=" * 70)
    print("FEDLORA SERVER - Multimodal Visual Assistant")
    print("=" * 70)
    print(f"  Server address:     {server_address}")
    print(f"  Training rounds:    {num_rounds}")
    print(f"  Min clients:        {min_clients}")
    print(f"  LoRA rank:          {LORA_RANK}")
    print(f"  LoRA alpha:         {LORA_ALPHA}")
    print(f"  LoRA weights dir:   {lora_weights_dir}")
    print(f"  Start time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    strategy = FedLoRAStrategy(
        lora_weights_dir=lora_weights_dir,
        min_available_clients=min_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    server = FedLoRAServer(strategy=strategy)

    print(f"Starting Flower server on {server_address}...")
    print(f"Waiting for {min_clients} clients to connect...\n")

    try:
        history = fl.server.start_server(
            server_address=server_address,
            server=server,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
        )

        print("\n" + "=" * 70)
        print("FEDLORA TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        if strategy.history["rounds"]:
            final_round = strategy.history["rounds"][-1]
            final_acc = strategy.history["accuracies"][-1]
            final_loss = strategy.history["losses"][-1]

            print(f"  Final Round:        {final_round}")
            print(f"  Final Accuracy:     {final_acc:.4f} ({final_acc * 100:.2f}%)")
            print(f"  Final Loss:         {final_loss:.4f}")
            print(f"  LoRA saved:        {lora_weights_dir}/round_{final_round}/")
            print("=" * 70 + "\n")

        return history

    except KeyboardInterrupt:
        print("\n\nServer interrupted by user. Shutting down gracefully...")
    except Exception as e:
        print(f"\n\nError during server execution: {e}")
        raise


def main():
    """Main entry point for server."""
    parser = argparse.ArgumentParser(description="FedLoRA Server")

    parser.add_argument(
        "--server-address",
        type=str,
        default=FL_SERVER_ADDRESS,
        help=f"Server address (default: {FL_SERVER_ADDRESS})",
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=NUM_ROUNDS,
        help=f"Number of training rounds (default: {NUM_ROUNDS})",
    )

    parser.add_argument(
        "--min-clients",
        type=int,
        default=MIN_AVAILABLE_CLIENTS,
        help=f"Minimum number of clients (default: {MIN_AVAILABLE_CLIENTS})",
    )

    parser.add_argument(
        "--lora-weights-dir",
        type=str,
        default=LORA_WEIGHTS_DIR,
        help=f"Directory to save LoRA weights (default: {LORA_WEIGHTS_DIR})",
    )

    args = parser.parse_args()

    start_server(
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        min_clients=args.min_clients,
        lora_weights_dir=args.lora_weights_dir,
    )


if __name__ == "__main__":
    main()
