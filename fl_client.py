import os
import flwr as fl
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

# Make TensorFlow quiet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_partition(partition_id):
    """Loads a specific 1/10th slice of CIFAR-10 for a client."""
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    # Partitioning the data (50,000 images / 10 = 5,000 per client)
    step = 5000
    start = partition_id * step
    end = (partition_id + 1) * step
    
    x_train, y_train = x_train[start:end], y_train[start:end]
    
    # CRITICAL FOR ACCURACY: Normalization
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    return x_train, y_train, x_test, y_test

def create_model():
    model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), weights=None, classes=10)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

class CifarClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.model = create_model()
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # Better Accuracy: Use 2-3 local epochs per round
        self.model.fit(self.x_train, self.y_train, epochs=2, batch_size=32, verbose=0)
        print("Local training finished. Sending updates to server...")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}

if __name__ == "__main__":
    # Change this ID (0-9) for each terminal you open to simulate different clients
    client_id = int(input("Enter Client ID (0-9): "))
    x_train, y_train, x_test, y_test = load_partition(client_id)
    
    print(f"Client {client_id} started with {len(x_train)} images.")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient(x_train, y_train, x_test, y_test))