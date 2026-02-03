"""
Training Status Module
Provides utilities for tracking and persisting training status
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Default status file path
DEFAULT_STATUS_FILE = "train_status.json"


class TrainingStatus:
    """
    Class to manage training status updates
    """

    def __init__(self, status_file=DEFAULT_STATUS_FILE):
        """
        Initialize training status tracker

        Args:
            status_file: Path to status JSON file
        """
        self.status_file = status_file
        self.current_status = {
            "status": "idle",
            "current_round": 0,
            "total_rounds": 0,
            "accuracy": 0.0,
            "loss": 0.0,
            "timestamp": datetime.utcnow().isoformat(),
            "clients_completed": 0,
            "clients_total": 0,
            "message": "Waiting for training to start",
        }

    def update(self, **kwargs):
        """
        Update training status with new values

        Args:
            **kwargs: Key-value pairs to update in status
        """
        self.current_status.update(kwargs)
        self.current_status["timestamp"] = datetime.utcnow().isoformat()
        self.save()

    def set_training_started(self, total_rounds, total_clients):
        """Set status when training starts"""
        self.update(
            status="training",
            current_round=0,
            total_rounds=total_rounds,
            clients_total=total_clients,
            message="Training in progress...",
        )

    def set_round_completed(self, round_num, accuracy, loss, clients_completed):
        """Set status when a round completes"""
        self.update(
            current_round=round_num,
            accuracy=float(accuracy),
            loss=float(loss),
            clients_completed=clients_completed,
            message=f"Round {round_num} completed",
        )

    def set_training_completed(self, final_accuracy, final_loss):
        """Set status when training completes"""
        self.update(
            status="completed",
            accuracy=final_accuracy,
            loss=final_loss,
            message="Training completed successfully",
        )

    def set_training_failed(self, error_message):
        """Set status when training fails"""
        self.update(status="failed", message=f"Training failed: {error_message}")

    def save(self):
        """Save status to file"""
        try:
            with open(self.status_file, "w") as f:
                json.dump(self.current_status, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save training status: {e}")

    @staticmethod
    def load(status_file=DEFAULT_STATUS_FILE):
        """
        Load status from file

        Args:
            status_file: Path to status JSON file

        Returns:
            Dictionary with status or None if file doesn't exist
        """
        try:
            if os.path.exists(status_file):
                with open(status_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load training status: {e}")

        return None

    @staticmethod
    def get_current(status_file=DEFAULT_STATUS_FILE):
        """Get current training status"""
        status = TrainingStatus.load(status_file)
        if status:
            return status

        # Return default idle status
        return {
            "status": "idle",
            "current_round": 0,
            "total_rounds": 0,
            "accuracy": 0.0,
            "loss": 0.0,
            "timestamp": datetime.utcnow().isoformat(),
            "clients_completed": 0,
            "clients_total": 0,
            "message": "Waiting for training to start",
        }


# Convenience functions
def update_training_status(**kwargs):
    """Update global training status"""
    status = TrainingStatus()
    status.update(**kwargs)


def get_training_status():
    """Get current training status"""
    return TrainingStatus.get_current()


def training_started(total_rounds, total_clients):
    """Mark training as started"""
    status = TrainingStatus()
    status.set_training_started(total_rounds, total_clients)


def round_completed(round_num, accuracy, loss, clients_completed):
    """Mark a round as completed"""
    status = TrainingStatus()
    status.set_round_completed(round_num, accuracy, loss, clients_completed)


def training_completed(final_accuracy, final_loss):
    """Mark training as completed"""
    status = TrainingStatus()
    status.set_training_completed(final_accuracy, final_loss)


def training_failed(error_message):
    """Mark training as failed"""
    status = TrainingStatus()
    status.set_training_failed(error_message)


if __name__ == "__main__":
    """Test training status module"""
    print("Testing training status module...")

    # Test creating and updating status
    status = TrainingStatus()
    status.set_training_started(10, 5)
    print("Status saved:", status.status_file)

    # Test loading status
    loaded = TrainingStatus.load()
    print("Loaded status:", json.dumps(loaded, indent=2))

    # Test round completion
    status.set_round_completed(1, 0.65, 0.45, 5)
    loaded = TrainingStatus.load()
    print("After round 1:", json.dumps(loaded, indent=2))

    print("\nTraining status module test completed!")
