"""
Shared Model Architecture for Federated Learning
Implements enhanced MobileNetV2-based image classifier for CIFAR-10
"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # Must be set before importing Keras

import keras
from keras import layers, models
from keras.applications import MobileNetV2
from keras.optimizers import Adam
from keras import regularizers

from backend_fl.config import (
    INPUT_SHAPE,
    NUM_CLASSES,
    LEARNING_RATE,
    MODEL_ALPHA,
    USE_PRETRAINED,
)


def get_model(pretrained=None, alpha=None):
    """
    Creates and compiles an enhanced MobileNetV2-based model for CIFAR-10 classification

    Args:
        pretrained (bool): Whether to use ImageNet pre-trained weights (default: from config)
        alpha (float): Width multiplier (default: from config, 1.0 for full capacity)

    Returns:
        keras.Model: Compiled model ready for training
    """
    # Use config defaults if not specified
    if pretrained is None:
        pretrained = USE_PRETRAINED
    if alpha is None:
        alpha = MODEL_ALPHA

    # Load MobileNetV2 base with larger capacity
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet" if pretrained else None,
        alpha=alpha,  # Width multiplier (1.0 = full capacity, 0.5 = 50% of channels)
    )

    # Fine-tune the last few layers
    base_model.trainable = True

    # Build enhanced classification head with more capacity
    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(name="global_avg_pool"),
            layers.BatchNormalization(name="batch_norm_1"),
            layers.Dropout(0.3, name="dropout_1"),
            layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=regularizers.l2(0.001),
                name="dense_1",
            ),
            layers.BatchNormalization(name="batch_norm_2"),
            layers.Dropout(0.4, name="dropout_2"),
            layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=regularizers.l2(0.001),
                name="dense_2",
            ),
            layers.BatchNormalization(name="batch_norm_3"),
            layers.Dropout(0.3, name="dropout_3"),
            layers.Dense(NUM_CLASSES, activation="softmax", name="predictions"),
        ],
        name="Enhanced_MobileNetV2_CIFAR10",
    )

    # Compile model with optimized settings
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_size(model):
    """
    Calculate model size in MB

    Args:
        model: Keras model

    Returns:
        float: Model size in MB
    """
    import numpy as np

    # Count total parameters
    total_params = model.count_params()

    # Estimate size (4 bytes per float32 parameter)
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 * 1024)

    return size_mb


def print_model_summary(model):
    """
    Print detailed model summary

    Args:
        model: Keras model
    """
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 70)

    model.summary()

    print("\n" + "-" * 70)
    print("MODEL STATISTICS")
    print("-" * 70)

    total_params = model.count_params()
    trainable_params = sum(
        [keras.backend.count_params(w) for w in model.trainable_weights]
    )
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters:        {total_params:,}")
    print(f"Trainable parameters:    {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Model size:              {get_model_size(model):.2f} MB")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    """Test model creation"""
    print("Testing model architecture...")

    # Create model
    model = get_model(pretrained=False)

    # Print summary
    print_model_summary(model)

    # Test forward pass
    import numpy as np

    dummy_input = np.random.rand(1, 32, 32, 3).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum:   {output.sum():.4f} (should be ~1.0)")
    print(f"Prediction:   Class {output.argmax()}")

    print("\n[OK] Model architecture test passed!")
