"""
Custom Federated Averaging Strategy
Extends Flower's FedAvg with model saving, evaluation, and metrics tracking
"""

import json
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union
from logging import WARNING

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np

from backend_fl.config import (
    MODEL_PATH,
    MODEL_HISTORY_PATH,
    NUM_CLASSES,
    MIN_AVAILABLE_CLIENTS,
    MIN_FIT_CLIENTS,
    FRACTION_FIT,
    FRACTION_EVALUATE,
    TRAINING_LOG_PATH,
)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy that:
    1. Aggregates client model weights using weighted averaging
    2. Evaluates the global model on server-side test set
    3. Saves the global model after each round
    4. Tracks training history and metrics
    """

    def __init__(self, model_fn, X_test: np.ndarray, y_test: np.ndarray, **kwargs):
        """
        Initialize the strategy

        Args:
            model_fn: Function that returns a compiled Keras model
            X_test: Test images for server-side evaluation
            y_test: Test labels (one-hot encoded)
            **kwargs: Additional arguments for FedAvg
        """
        super().__init__(
            min_available_clients=kwargs.get(
                "min_available_clients", MIN_AVAILABLE_CLIENTS
            ),
            min_fit_clients=kwargs.get("min_fit_clients", MIN_FIT_CLIENTS),
            fraction_fit=kwargs.get("fraction_fit", FRACTION_FIT),
            fraction_evaluate=kwargs.get("fraction_evaluate", FRACTION_EVALUATE),
        )

        self.model_fn = model_fn
        self.X_test = X_test
        self.y_test = y_test
        self.history = {
            "rounds": [],
            "losses": [],
            "accuracies": [],
            "aggregation_times": [],
            "timestamps": [],
        }

        # Create model instance for evaluation
        self.model = model_fn()

        print(f"\n{'=' * 70}")
        print("FEDERATED AVERAGING STRATEGY INITIALIZED")
        print(f"{'=' * 70}")
        print(f"  Min available clients: {MIN_AVAILABLE_CLIENTS}")
        print(f"  Min fit clients:       {MIN_FIT_CLIENTS}")
        print(f"  Fraction fit:          {FRACTION_FIT}")
        print(f"  Test set size:         {len(X_test)}")
        print(f"{'=' * 70}\n")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model weights from clients and evaluate

        Args:
            server_round: Current round number
            results: List of (client, fit_result) tuples
            failures: List of failed clients

        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        start_time = datetime.now()

        # Log round start
        self._log_event(
            f"\n{'=' * 70}\nROUND {server_round} - AGGREGATION STARTED\n{'=' * 70}"
        )

        if failures:
            self._log_event(f"⚠ Warning: {len(failures)} clients failed")

        # Call parent's aggregate_fit to perform FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert parameters to numpy arrays
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

            # Update model with aggregated weights
            self.model.set_weights(aggregated_ndarrays)

            # Evaluate on server test set
            loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)

            # Save model
            self._save_model(server_round, loss, accuracy)

            # Update history
            aggregation_time = (datetime.now() - start_time).total_seconds()
            self._update_history(server_round, loss, accuracy, aggregation_time)

            # Log results
            self._log_event(
                f"\nRound {server_round} Results:\n"
                f"  Loss:              {loss:.4f}\n"
                f"  Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)\n"
                f"  Aggregation time:  {aggregation_time:.2f}s\n"
                f"  Participating clients: {len(results)}\n"
                f"{'=' * 70}"
            )

            # Add metrics to return dict
            aggregated_metrics["loss"] = loss
            aggregated_metrics["accuracy"] = accuracy
            aggregated_metrics["aggregation_time"] = aggregation_time

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients

        Args:
            server_round: Current round number
            results: List of (client, evaluate_result) tuples
            failures: List of failed clients

        Returns:
            Tuple of (aggregated_loss, metrics)
        """
        if not results:
            return None, {}

        # Calculate weighted average of client losses
        total_samples = sum([r.num_examples for _, r in results])
        aggregated_loss = (
            sum([r.loss * r.num_examples for _, r in results]) / total_samples
        )

        # Calculate weighted average of accuracies if available
        if results[0][1].metrics and "accuracy" in results[0][1].metrics:
            aggregated_accuracy = (
                sum([r.metrics["accuracy"] * r.num_examples for _, r in results])
                / total_samples
            )

            return aggregated_loss, {"accuracy": aggregated_accuracy}

        return aggregated_loss, {}

    def _save_model(self, round_num: int, loss: float, accuracy: float):
        """
        Save the global model to disk

        Args:
            round_num: Current round number
            loss: Model loss
            accuracy: Model accuracy
        """
        # Save current model
        self.model.save(MODEL_PATH)

        # Save round-specific model
        round_model_path = f"models/global_model_round_{round_num}.h5"
        self.model.save(round_model_path)

        # Save metadata
        metadata = {
            "round": round_num,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "timestamp": datetime.now().isoformat(),
            "model_path": round_model_path,
        }

        metadata_path = f"models/model_round_{round_num}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self._log_event(f"[OK] Model saved: {round_model_path}")

    def _update_history(
        self, round_num: int, loss: float, accuracy: float, agg_time: float
    ):
        """
        Update training history

        Args:
            round_num: Current round number
            loss: Model loss
            accuracy: Model accuracy
            agg_time: Aggregation time in seconds
        """
        self.history["rounds"].append(round_num)
        self.history["losses"].append(float(loss))
        self.history["accuracies"].append(float(accuracy))
        self.history["aggregation_times"].append(float(agg_time))
        self.history["timestamps"].append(datetime.now().isoformat())

        # Save history to file
        with open(MODEL_HISTORY_PATH, "w") as f:
            json.dump(self.history, f, indent=2)

    def _log_event(self, message: str):
        """
        Log event to console and file

        Args:
            message: Message to log
        """
        print(message)

        # Ensure log directory exists
        os.makedirs(os.path.dirname(TRAINING_LOG_PATH), exist_ok=True)

        # Append to log file
        with open(TRAINING_LOG_PATH, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")

    def get_history(self) -> Dict:
        """
        Get training history

        Returns:
            Dictionary containing training history
        """
        return self.history


if __name__ == "__main__":
    """Test strategy"""
    print("Testing SaveModelStrategy...")

    # This would be tested in integration tests with actual FL server
    print("[OK] Strategy module loaded successfully")
    print("  Note: Full strategy testing requires FL server and clients")
