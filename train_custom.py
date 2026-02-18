"""
Custom Image Classification Training Script
Trains on custom dataset with data augmentation
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
from PIL import Image
import keras
from keras import layers, models
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import shutil
import random

DATASET_DIR = "dataset/train"
MODEL_DIR = "models"
NUM_CLASSES = 12
CLASS_NAMES = sorted(
    [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
)
BATCH_SIZE = 16
EPOCHS = 3
TRAINING_ROUNDS = 3

INPUT_SIZE = (224, 224)


def load_custom_dataset():
    """Load custom dataset"""
    print("\n" + "=" * 60)
    print("LOADING CUSTOM DATASET")
    print("=" * 60)

    images = []
    labels = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATASET_DIR, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(INPUT_SIZE)
                images.append(np.array(img))
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)

    print(f"Classes: {CLASS_NAMES}")
    print(f"Total images: {len(images)}")
    print(f"Images per class: {len(images) / NUM_CLASSES:.1f}")

    return images, labels


def create_model():
    """Create MobileNetV2 based model"""
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)

    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Model created with {NUM_CLASSES} classes")
    return model


def augment_image(img):
    """Apply data augmentation to a single image"""
    if random.random() > 0.5:
        img = np.fliplr(img)

    if random.random() > 0.5:
        shift = random.randint(-20, 20)
        img = np.roll(img, shift, axis=1)

    if random.random() > 0.5:
        shift = random.randint(-20, 20)
        img = np.roll(img, shift, axis=0)

    brightness = random.uniform(0.8, 1.2)
    img = np.clip(img * brightness, 0, 1)

    return img


def create_augmented_data(images, labels, multiplier=10):
    """Create augmented dataset"""
    aug_images = []
    aug_labels = []

    for img, label in zip(images, labels):
        aug_images.append(img)
        aug_labels.append(label)

        for _ in range(multiplier):
            aug_img = augment_image(img.copy())
            aug_images.append(aug_img)
            aug_labels.append(label)

    return np.array(aug_images), np.array(aug_labels)


def train_model(round_num):
    """Train model for one round"""
    print(f"\n{'=' * 60}")
    print(f"TRAINING ROUND {round_num}")
    print(f"{'=' * 60}")

    X, y = load_custom_dataset()

    print("\nCreating augmented data...")
    X_aug, y_aug = create_augmented_data(X, y, multiplier=15)
    print(f"Augmented dataset size: {len(X_aug)} images")

    model = create_model()

    model_path = os.path.join(MODEL_DIR, f"custom_model_round{round_num}.h5")

    model.fit(
        X_aug,
        y_aug,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )

    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"Final accuracy on original images: {acc * 100:.2f}%")

    return model, acc


def main():
    print("\n" + "#" * 60)
    print("#  CUSTOM IMAGE CLASSIFICATION TRAINING")
    print(f"#  Training Rounds: {TRAINING_ROUNDS}")
    print(f"#  Epochs per round: {EPOCHS}")
    print("#" * 60)

    best_acc = 0
    best_model = None

    for round_num in range(1, TRAINING_ROUNDS + 1):
        model, acc = train_model(round_num)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            shutil.copy(
                os.path.join(MODEL_DIR, f"custom_model_round{round_num}.h5"),
                os.path.join(MODEL_DIR, "custom_model_best.h5"),
            )

    print("\n" + "#" * 60)
    print(f"TRAINING COMPLETE")
    print(f"Best accuracy: {best_acc * 100:.2f}%")
    print(f"Model saved to: {MODEL_DIR}/custom_model_best.h5")
    print("#" * 60)


if __name__ == "__main__":
    main()
