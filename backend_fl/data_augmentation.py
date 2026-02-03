"""
Data Augmentation Module for Federated Learning
Implements various augmentation techniques to improve model generalization
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers
import numpy as np


def get_augmentation_layer():
    """
    Creates a Keras Sequential model with data augmentation layers

    Returns:
        keras.Sequential: Augmentation pipeline
    """
    augmentation = keras.Sequential(
        [
            # Random horizontal flip
            layers.RandomFlip("horizontal"),
            # Random rotation (±15 degrees)
            layers.RandomRotation(0.15),
            # Random zoom (±10%)
            layers.RandomZoom(0.1),
            # Random translation
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            # Random brightness adjustment
            layers.RandomBrightness(factor=0.2),
            # Random contrast adjustment
            layers.RandomContrast(factor=0.2),
        ],
        name="data_augmentation",
    )

    return augmentation


def augment_batch(x_batch, y_batch, augmentation_layer):
    """
    Apply data augmentation to a batch of images

    Args:
        x_batch: Input images (batch_size, 32, 32, 3)
        y_batch: Labels (batch_size, num_classes)
        augmentation_layer: Keras augmentation layer

    Returns:
        tuple: Augmented (x_batch, y_batch)
    """
    # Apply augmentation only to images
    x_augmented = augmentation_layer(x_batch, training=True)
    return x_augmented, y_batch


def create_augmented_dataset(x_train, y_train, batch_size=32, shuffle=True):
    """
    Create a tf.data.Dataset with data augmentation

    Args:
        x_train: Training images
        y_train: Training labels
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Returns:
        tf.data.Dataset: Augmented dataset
    """
    # Create base dataset
    dataset = keras.utils.PyDataset(
        x_train, y_train, batch_size=batch_size, shuffle=shuffle
    )

    # Note: Augmentation is applied during model training via augmentation layer
    return dataset


def mixup(x_batch, y_batch, alpha=0.2):
    """
    Apply MixUp data augmentation
    MixUp creates synthetic training examples by mixing pairs of examples

    Args:
        x_batch: Batch of images
        y_batch: Batch of labels
        alpha: MixUp hyperparameter (Beta distribution parameter)

    Returns:
        tuple: Mixed (x_batch, y_batch)
    """
    batch_size = len(x_batch)

    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.maximum(lam, 1 - lam)  # Ensure lambda >= 0.5

    # Reshape lambda for broadcasting
    lam_x = lam.reshape(batch_size, 1, 1, 1)
    lam_y = lam.reshape(batch_size, 1)

    # Shuffle indices
    indices = np.random.permutation(batch_size)

    # Mix images and labels
    x_mixed = lam_x * x_batch + (1 - lam_x) * x_batch[indices]
    y_mixed = lam_y * y_batch + (1 - lam_y) * y_batch[indices]

    return x_mixed, y_mixed


def cutmix(x_batch, y_batch, alpha=1.0):
    """
    Apply CutMix data augmentation
    CutMix cuts and pastes patches from one image to another

    Args:
        x_batch: Batch of images
        y_batch: Batch of labels
        alpha: CutMix hyperparameter

    Returns:
        tuple: Mixed (x_batch, y_batch)
    """
    batch_size = len(x_batch)
    image_height, image_width = x_batch.shape[1:3]

    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Random crop box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(image_height * cut_ratio)
    cut_w = int(image_width * cut_ratio)

    # Random position
    cx = np.random.randint(image_width)
    cy = np.random.randint(image_height)

    bbx1 = np.clip(cx - cut_w // 2, 0, image_width)
    bby1 = np.clip(cy - cut_h // 2, 0, image_height)
    bbx2 = np.clip(cx + cut_w // 2, 0, image_width)
    bby2 = np.clip(cy + cut_h // 2, 0, image_height)

    # Shuffle indices
    indices = np.random.permutation(batch_size)

    # Create mixed batch
    x_mixed = x_batch.copy()
    x_mixed[:, bby1:bby2, bbx1:bbx2, :] = x_batch[indices, bby1:bby2, bbx1:bbx2, :]

    # Adjust lambda based on actual cut area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_height * image_width))

    # Mix labels
    y_mixed = lam * y_batch + (1 - lam) * y_batch[indices]

    return x_mixed, y_mixed


def apply_augmentation_strategy(x_batch, y_batch, strategy="standard", **kwargs):
    """
    Apply different augmentation strategies

    Args:
        x_batch: Batch of images
        y_batch: Batch of labels
        strategy: Augmentation strategy ("standard", "mixup", "cutmix", "none")
        **kwargs: Additional parameters for specific strategies

    Returns:
        tuple: Augmented (x_batch, y_batch)
    """
    if strategy == "mixup":
        alpha = kwargs.get("alpha", 0.2)
        return mixup(x_batch, y_batch, alpha=alpha)

    elif strategy == "cutmix":
        alpha = kwargs.get("alpha", 1.0)
        return cutmix(x_batch, y_batch, alpha=alpha)

    elif strategy == "standard":
        # Standard augmentation via Keras layers (applied in model)
        return x_batch, y_batch

    elif strategy == "none":
        return x_batch, y_batch

    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy}")


if __name__ == "__main__":
    """Test data augmentation"""
    import matplotlib.pyplot as plt
    from keras.datasets import cifar10

    print("Testing Data Augmentation...")
    print("=" * 70)

    # Load sample data
    (x_train, y_train), _ = cifar10.load_data()
    x_sample = x_train[:4] / 255.0  # Normalize
    y_sample = keras.utils.to_categorical(y_train[:4], 10)

    print(f"Sample batch shape: {x_sample.shape}")
    print(f"Label shape: {y_sample.shape}")
    print()

    # Test standard augmentation
    print("Testing standard augmentation layer...")
    aug_layer = get_augmentation_layer()
    x_aug = aug_layer(x_sample, training=True)
    print(f"  [OK] Augmented shape: {x_aug.shape}")
    print()

    # Test MixUp
    print("Testing MixUp augmentation...")
    x_mixed, y_mixed = mixup(x_sample, y_sample, alpha=0.2)
    print(f"  [OK] Mixed shape: {x_mixed.shape}")
    print(f"  Label mixing: {y_mixed[0][:5]}")
    print()

    # Test CutMix
    print("Testing CutMix augmentation...")
    x_cut, y_cut = cutmix(x_sample, y_sample, alpha=1.0)
    print(f"  [OK] CutMix shape: {x_cut.shape}")
    print()

    print("=" * 70)
    print("[OK] Data augmentation tests passed!")
