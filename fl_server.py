import flwr as fl
import tensorflow as tf
import os
from models import create_model

# Custom Strategy to save model weights after training
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Call the base aggregation logic
        aggregated_weights, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_weights is not None:
            print(f"Saving round {server_round} weights...")
            # Convert Flower weights back to NumPy
            weights = fl.common.parameters_to_ndarrays(aggregated_weights)
            
            # Load weights into a temporary model to save as .h5
            model = create_model()
            model.set_weights(weights)
            model.save_weights("global_model_weights.h5")
            
            # Save a status file for the Flask Dashboard to read
            with open("train_status.txt", "w") as f:
                f.write(f"Round {server_round} complete. Accuracy updating...")

        return aggregated_weights, aggregated_metrics

if __name__ == "__main__":
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )
    
    print("Federated Server starting on port 8080...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )