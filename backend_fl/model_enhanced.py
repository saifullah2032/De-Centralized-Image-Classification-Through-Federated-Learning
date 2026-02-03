"""
Enhanced Model Architecture for Federated Learning
Implements higher capacity networks with improved performance for CIFAR-10
"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # Must be set before importing Keras

import keras
from keras import layers, models
from keras.applications import MobileNetV2, EfficientNetB0, ResNet50V2
from keras.optimizers import Adam
from keras import regularizers

from backend_fl.config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE


def get_enhanced_mobilenet(pretrained=False, alpha=1.0):
    """
    Enhanced MobileNetV2 with larger capacity

    Args:
        pretrained (bool): Whether to use ImageNet pre-trained weights
        alpha (float): Width multiplier (0.5, 0.75, 1.0, 1.3, 1.4)

    Returns:
        keras.Model: Compiled model
    """
    # Load MobileNetV2 base with larger alpha
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet" if pretrained else None,
        alpha=alpha,  # Increased from 0.5 to 1.0 for more capacity
    )

    # Fine-tune the last layers
    base_model.trainable = True

    # Enhanced classification head with more capacity
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

    # Compile with optimized settings
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_resnet_model(pretrained=False):
    """
    ResNet50V2-based model for higher accuracy

    Args:
        pretrained (bool): Whether to use ImageNet pre-trained weights

    Returns:
        keras.Model: Compiled model
    """
    # Load ResNet50V2 base
    base_model = ResNet50V2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet" if pretrained else None,
    )

    # Fine-tune last layers
    base_model.trainable = True

    # Freeze early layers for faster convergence
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Build classification head
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))(
        x
    )
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(
        x
    )
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet50V2_CIFAR10")

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_efficientnet_model(pretrained=False):
    """
    EfficientNetB0-based model for balanced performance

    Args:
        pretrained (bool): Whether to use ImageNet pre-trained weights

    Returns:
        keras.Model: Compiled model
    """
    # Load EfficientNetB0 base
    base_model = EfficientNetB0(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet" if pretrained else None,
    )

    # Fine-tune
    base_model.trainable = True

    # Build classification head
    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(name="global_avg_pool"),
            layers.BatchNormalization(name="batch_norm_1"),
            layers.Dropout(0.3, name="dropout_1"),
            layers.Dense(
                384,
                activation="relu",
                kernel_regularizer=regularizers.l2(0.001),
                name="dense_1",
            ),
            layers.BatchNormalization(name="batch_norm_2"),
            layers.Dropout(0.4, name="dropout_2"),
            layers.Dense(
                192,
                activation="relu",
                kernel_regularizer=regularizers.l2(0.001),
                name="dense_2",
            ),
            layers.BatchNormalization(name="batch_norm_3"),
            layers.Dropout(0.3, name="dropout_3"),
            layers.Dense(NUM_CLASSES, activation="softmax", name="predictions"),
        ],
        name="EfficientNetB0_CIFAR10",
    )

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_custom_cnn_model():
    """
    Custom CNN optimized for CIFAR-10
    Designed specifically for 32x32 images

    Returns:
        keras.Model: Compiled model
    """
    model = models.Sequential(
        [
            # Block 1
            layers.Conv2D(
                64, (3, 3), padding="same", activation="relu", input_shape=INPUT_SHAPE
            ),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # Block 2
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            # Block 3
            layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            # Classification head
            layers.GlobalAveragePooling2D(),
            layers.Dense(
                512, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(
                256, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.Dropout(0.3),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="Custom_CNN_CIFAR10",
    )

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model(architecture="enhanced_mobilenet", pretrained=False, **kwargs):
    """
    Unified interface to get any model architecture

    Args:
        architecture (str): Model type to create
            - "enhanced_mobilenet": Enhanced MobileNetV2 (default)
            - "resnet50": ResNet50V2 model
            - "efficientnet": EfficientNetB0 model
            - "custom_cnn": Custom CNN for CIFAR-10
        pretrained (bool): Use ImageNet weights
        **kwargs: Additional architecture-specific parameters

    Returns:
        keras.Model: Compiled model
    """
    print(f"\n[INFO] Creating model: {architecture}")

    if architecture == "enhanced_mobilenet":
        alpha = kwargs.get("alpha", 1.0)
        model = get_enhanced_mobilenet(pretrained=pretrained, alpha=alpha)
        print(f"  - MobileNetV2 with alpha={alpha}")

    elif architecture == "resnet50":
        model = get_resnet_model(pretrained=pretrained)
        print("  - ResNet50V2 architecture")

    elif architecture == "efficientnet":
        model = get_efficientnet_model(pretrained=pretrained)
        print("  - EfficientNetB0 architecture")

    elif architecture == "custom_cnn":
        model = get_custom_cnn_model()
        print("  - Custom CNN for CIFAR-10")

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Print model info
    total_params = model.count_params()
    trainable_params = sum(
        [keras.backend.count_params(w) for w in model.trainable_weights]
    )
    size_mb = (total_params * 4) / (1024 * 1024)

    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: {size_mb:.2f} MB")

    return model


def get_model_size(model):
    """
    Calculate model size in MB

    Args:
        model: Keras model

    Returns:
        float: Model size in MB
    """
    total_params = model.count_params()
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
    """Test enhanced model architectures"""
    import numpy as np

    print("Testing Enhanced Model Architectures...")
    print("=" * 70)

    architectures = [
        ("enhanced_mobilenet", {"alpha": 1.0}),
        ("custom_cnn", {}),
    ]

    for arch_name, arch_kwargs in architectures:
        print(f"\n\nTesting: {arch_name}")
        print("-" * 70)

        try:
            # Create model
            model = get_model(architecture=arch_name, pretrained=False, **arch_kwargs)

            # Test forward pass
            dummy_input = np.random.rand(1, 32, 32, 3).astype(np.float32)
            output = model.predict(dummy_input, verbose=0)

            print(f"  Input shape:  {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output sum:   {output.sum():.4f} (should be ~1.0)")
            print(f"  [OK] {arch_name} works!")

        except Exception as e:
            print(f"  [X] {arch_name} failed: {e}")

    print("\n" + "=" * 70)
    print("[OK] Enhanced model architecture tests complete!")
