"""
Model Inference Module
Handles image classification predictions using pre-trained ImageNet model (MobileNetV2)
This enables classification of ANY image into 1000 ImageNet categories
"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # Must be set before importing Keras

import numpy as np
from PIL import Image
import keras
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

from backend_fl.config import (
    MODEL_PATH,
    INPUT_SHAPE,
    NUM_CLASSES,
    DATASET,
    USE_IMAGENET_PRETRAINED,
)


class ImageClassifier:
    """
    Image classifier using pre-trained MobileNetV2 on ImageNet
    Can classify any image into 1000 different categories
    """

    def __init__(self, model_path=MODEL_PATH, use_pretrained_imagenet=True):
        """
        Initialize the classifier

        Args:
            model_path: Path to custom trained model file (.h5) - used for CIFAR mode
            use_pretrained_imagenet: If True, use pre-trained ImageNet model (default)
        """
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        self.use_imagenet = (
            use_pretrained_imagenet or USE_IMAGENET_PRETRAINED or DATASET == "IMAGENET"
        )

        # ImageNet input size
        self.input_size = (224, 224)

        # Try to load model
        self.load_model()

    def load_model(self):
        """Load the model - either pre-trained ImageNet or custom trained"""
        try:
            if self.use_imagenet:
                print("Loading pre-trained MobileNetV2 model with ImageNet weights...")
                # Load MobileNetV2 with ImageNet weights - includes the top classification layer
                self.model = MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=True,  # Include the classification head
                    weights="imagenet",  # Pre-trained on ImageNet
                    classes=1000,
                )
                self.model_loaded = True
                print("[OK] Pre-trained ImageNet model loaded successfully")
                print(f"    - Model: MobileNetV2")
                print(f"    - Classes: 1000 (ImageNet categories)")
                print(f"    - Input size: 224x224")
            else:
                # Fall back to custom trained model (CIFAR)
                if os.path.exists(self.model_path):
                    print(f"Loading custom model from {self.model_path}...")
                    self.model = keras.models.load_model(self.model_path)
                    self.model_loaded = True
                    print("[OK] Custom model loaded successfully")
                else:
                    print(f"[!] Model not found at {self.model_path}")
                    print("  Using pre-trained ImageNet model instead...")
                    self._load_imagenet_fallback()

        except Exception as e:
            print(f"[X] Error loading model: {e}")
            print("  Attempting to load pre-trained ImageNet model as fallback...")
            self._load_imagenet_fallback()

    def _load_imagenet_fallback(self):
        """Load ImageNet model as fallback"""
        try:
            self.model = MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=True,
                weights="imagenet",
                classes=1000,
            )
            self.use_imagenet = True
            self.model_loaded = True
            print("[OK] Fallback to pre-trained ImageNet model successful")
        except Exception as e:
            print(f"[X] Failed to load fallback model: {e}")
            self.model_loaded = False

    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image as NumPy array with shape (1, 224, 224, 3)
        """
        try:
            # Open and convert to RGB
            img = Image.open(image_path).convert("RGB")

            # Resize to ImageNet dimensions (224x224) or CIFAR (32x32)
            target_size = (
                self.input_size
                if self.use_imagenet
                else (INPUT_SHAPE[0], INPUT_SHAPE[1])
            )
            img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            # Apply appropriate preprocessing
            if self.use_imagenet:
                # MobileNetV2 specific preprocessing (scales to [-1, 1])
                img_array = preprocess_input(img_array)
            else:
                # Simple normalization for CIFAR models
                img_array = img_array / 255.0

            return img_array

        except Exception as e:
            print(f"[X] Error preprocessing image: {e}")
            raise

    def predict(self, image_path):
        """
        Predict the class of an image using ImageNet pre-trained model

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing:
                - success: Whether prediction succeeded
                - predicted_class: Name of the predicted class
                - predicted_class_id: ID of the predicted class
                - confidence: Confidence score (0-1)
                - all_predictions: List of all class probabilities (top 10)
                - top_5: Top 5 predictions with class names
                - error: Error message (if failed)
        """
        if not self.model_loaded:
            return {
                "success": False,
                "error": "Model not loaded. Please check the installation.",
            }

        try:
            # Preprocess image
            img_array = self.preprocess_image(image_path)

            # Make prediction
            if self.model is None:
                return {"success": False, "error": "Model not initialized"}
            predictions = self.model.predict(img_array, verbose=0)

            if self.use_imagenet:
                # Use Keras decode_predictions for ImageNet labels
                # Returns list of (class_id, class_name, probability) tuples
                decoded = decode_predictions(predictions, top=10)[0]

                # Get top prediction
                top_pred = decoded[0]
                predicted_class_id = top_pred[
                    0
                ]  # ImageNet class ID (e.g., 'n02124075')
                predicted_class = (
                    top_pred[1].replace("_", " ").title()
                )  # Human readable name
                confidence = float(top_pred[2])

                # Format all predictions
                all_predictions = [
                    {
                        "class_id": pred[0],
                        "class_name": pred[1].replace("_", " ").title(),
                        "probability": float(pred[2]),
                    }
                    for pred in decoded
                ]

                # Top 5 predictions
                top_5 = all_predictions[:5]

                return {
                    "success": True,
                    "predicted_class": predicted_class,
                    "predicted_class_id": predicted_class_id,
                    "confidence": confidence,
                    "confidence_percent": f"{confidence * 100:.2f}%",
                    "superclass": None,  # ImageNet doesn't use superclasses in the same way
                    "all_predictions": all_predictions,
                    "top_5": top_5,
                    "model_type": "ImageNet (MobileNetV2)",
                    "total_classes": 1000,
                }
            else:
                # CIFAR mode (kept for backward compatibility)
                from backend_fl.config import (
                    LABELS,
                    CIFAR100_COARSE_LABELS,
                    CIFAR100_FINE_TO_COARSE,
                )

                predicted_class_id = int(np.argmax(predictions[0]))
                predicted_class = (
                    LABELS[predicted_class_id]
                    if LABELS
                    else f"Class {predicted_class_id}"
                )
                confidence = float(predictions[0][predicted_class_id])

                # Get superclass for CIFAR-100
                superclass = None
                if (
                    DATASET == "CIFAR100"
                    and predicted_class_id in CIFAR100_FINE_TO_COARSE
                ):
                    superclass_id = CIFAR100_FINE_TO_COARSE[predicted_class_id]
                    superclass = CIFAR100_COARSE_LABELS[superclass_id]

                # Get all class probabilities
                all_predictions = [
                    {
                        "class_id": i,
                        "class_name": LABELS[i] if LABELS else f"Class {i}",
                        "probability": float(predictions[0][i]),
                    }
                    for i in range(len(predictions[0]))
                ]

                # Sort by probability (descending)
                all_predictions.sort(key=lambda x: x["probability"], reverse=True)

                return {
                    "success": True,
                    "predicted_class": predicted_class,
                    "predicted_class_id": predicted_class_id,
                    "confidence": confidence,
                    "confidence_percent": f"{confidence * 100:.2f}%",
                    "superclass": superclass,
                    "all_predictions": all_predictions[:10],
                    "top_5": all_predictions[:5],
                    "model_type": f"CIFAR ({DATASET})",
                    "total_classes": NUM_CLASSES,
                }

        except Exception as e:
            print(f"[X] Prediction error: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def get_model_info(self):
        """
        Get information about the loaded model

        Returns:
            Dictionary containing model information
        """
        info = {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path
            if not self.use_imagenet
            else "Pre-trained (ImageNet)",
            "model_type": "MobileNetV2 (ImageNet)"
            if self.use_imagenet
            else f"Custom ({DATASET})",
        }

        if self.model_loaded and self.model is not None:
            info["total_params"] = self.model.count_params()
            info["input_shape"] = (
                "(224, 224, 3)" if self.use_imagenet else str(INPUT_SHAPE)
            )
            info["num_classes"] = 1000 if self.use_imagenet else NUM_CLASSES
            info["dataset"] = "ImageNet" if self.use_imagenet else DATASET

            if self.use_imagenet:
                info["description"] = (
                    "Pre-trained on 1.2M images, can classify 1000 different objects"
                )
                info["categories_include"] = [
                    "Animals (dogs, cats, birds, fish, insects...)",
                    "Vehicles (cars, planes, boats, bikes...)",
                    "Food (fruits, vegetables, dishes...)",
                    "Objects (furniture, electronics, tools...)",
                    "Nature (plants, landscapes...)",
                    "And many more!",
                ]

        return info


# Global classifier instance
_classifier_instance = None


def get_classifier():
    """
    Get or create the global classifier instance (singleton pattern)

    Returns:
        ImageClassifier instance
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ImageClassifier()
    return _classifier_instance


def reset_classifier():
    """
    Reset the global classifier instance
    Useful when switching between ImageNet and CIFAR modes
    """
    global _classifier_instance
    _classifier_instance = None


if __name__ == "__main__":
    """Test inference module"""
    print("=" * 60)
    print("Testing ImageNet Inference Module")
    print("=" * 60)

    # Create classifier
    classifier = ImageClassifier()

    # Get model info
    info = classifier.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")

    if classifier.model_loaded:
        print("\n[OK] Inference module ready!")
        print("  The model can classify images into 1000 ImageNet categories")
        print("  Including: animals, vehicles, food, objects, plants, and more!")
    else:
        print("\n[X] Model failed to load")
