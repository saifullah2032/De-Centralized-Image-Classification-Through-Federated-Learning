"""
Federated Learning Server
Orchestrates federated training across multiple clients using Flower framework
"""

import argparse
from datetime import datetime

import flwr as fl

from backend_fl.config import (
    FL_SERVER_ADDRESS,
    NUM_ROUNDS,
    MIN_AVAILABLE_CLIENTS,
)
from backend_fl.model import get_model
from backend_fl.data_utils import get_test_set
from backend_fl.strategies import SaveModelStrategy


def start_server(
    server_address: str = FL_SERVER_ADDRESS,
    num_rounds: int = NUM_ROUNDS,
    min_clients: int = MIN_AVAILABLE_CLIENTS,
):
    """
    Start the Federated Learning server

    Args:
        server_address: Server address (host:port)
        num_rounds: Number of training rounds
        min_clients: Minimum number of clients required
    """
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING SERVER")
    print("=" * 70)
    print(f"  Server address:     {server_address}")
    print(f"  Training rounds:    {num_rounds}")
    print(f"  Min clients:        {min_clients}")
    print(f"  Start time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    # Load test set for server-side evaluation
    print("Loading test dataset for server-side evaluation...")
    X_test, y_test = get_test_set()
    print(f"[OK] Test set loaded: {len(X_test)} samples\n")

    # Create model function
    def model_fn():
        return get_model(pretrained=False)

    # Initialize custom FedAvg strategy
    strategy = SaveModelStrategy(
        model_fn=model_fn,
        X_test=X_test,
        y_test=y_test,
        min_available_clients=min_clients,
        min_fit_clients=min_clients,
        fraction_fit=1.0,  # Use all available clients
        fraction_evaluate=1.0,
    )

    # Configure server
    config = fl.server.ServerConfig(num_rounds=num_rounds)

    print(f"Starting Flower server on {server_address}...")
    print(f"Waiting for {min_clients} clients to connect...\n")

    # Start server
    try:
        history = fl.server.start_server(
            server_address=server_address,
            config=config,
            strategy=strategy,
        )

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        # Print final results
        if strategy.history["rounds"]:
            final_round = strategy.history["rounds"][-1]
            final_acc = strategy.history["accuracies"][-1]
            final_loss = strategy.history["losses"][-1]

            print(f"  Final Round:        {final_round}")
            print(f"  Final Accuracy:     {final_acc:.4f} ({final_acc * 100:.2f}%)")
            print(f"  Final Loss:         {final_loss:.4f}")
            print(f"  Model saved:        models/global_model.h5")
            print("=" * 70 + "\n")

            # Check if target accuracy reached
            from backend_fl.config import TARGET_ACCURACY

            if final_acc >= TARGET_ACCURACY:
                print(
                    f"🎉 SUCCESS: Target accuracy of {TARGET_ACCURACY * 100:.1f}% achieved!"
                )
            else:
                print(
                    f"⚠ Note: Target accuracy of {TARGET_ACCURACY * 100:.1f}% not reached."
                )
                print(f"  Consider training for more rounds or tuning hyperparameters.")

        return history

    except KeyboardInterrupt:
        print("\n\nServer interrupted by user. Shutting down gracefully...")
    except Exception as e:
        print(f"\n\n❌ Error during server execution: {e}")
        raise


def main():
    """Main entry point for server"""
    parser = argparse.ArgumentParser(description="Federated Learning Server")

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

    args = parser.parse_args()

    # Start server
    start_server(
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        min_clients=args.min_clients,
    )


if __name__ == "__main__":
    main()
