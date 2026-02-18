"""
Model Inference Module
Handles image classification predictions using MobileNetV2
Supports switching between:
  - ImageNet pre-trained model (1000 classes)
  - Custom CIFAR-100 trained model (100 classes)
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
    AVAILABLE_MODELS,
    CIFAR100_MODEL_PATH,
    CIFAR100_FINE_LABELS,
    CIFAR100_COARSE_LABELS,
    CIFAR100_FINE_TO_COARSE,
)


class ImageClassifier:
    """
    Image classifier using MobileNetV2
    Supports switching between:
      - ImageNet pre-trained model (1000 classes)
      - Custom CIFAR-100 trained model (100 classes)
    """

    def __init__(self, model_type="imagenet"):
        """
        Initialize the classifier

        Args:
            model_type: "imagenet" or "cifar100"
        """
        self.model = None
        self.model_loaded = False
        self.current_model_type = model_type

        # Model configuration
        self.model_config = AVAILABLE_MODELS.get(
            model_type, AVAILABLE_MODELS["imagenet"]
        )
        self.input_size = self.model_config["input_size"]

        # Try to load model
        self.load_model()

    def switch_model(self, model_type):
        """
        Switch to a different model type

        Args:
            model_type: "imagenet" or "cifar100"

        Returns:
            bool: True if switch was successful
        """
        if model_type not in AVAILABLE_MODELS:
            print(f"[X] Unknown model type: {model_type}")
            return False

        if model_type == self.current_model_type and self.model_loaded:
            print(f"[i] Already using {model_type} model")
            return True

        print(
            f"\n[*] Switching model from {self.current_model_type} to {model_type}..."
        )

        # Update configuration
        self.current_model_type = model_type
        self.model_config = AVAILABLE_MODELS[model_type]
        self.input_size = self.model_config["input_size"]

        # Clear existing model
        self.model = None
        self.model_loaded = False

        # Load new model
        self.load_model()

        return self.model_loaded

    def load_model(self):
        """Load the model based on current_model_type"""
        try:
            if self.current_model_type == "imagenet":
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

            elif self.current_model_type == "cifar100":
                model_path = CIFAR100_MODEL_PATH
                if os.path.exists(model_path):
                    print(f"Loading CIFAR-100 model from {model_path}...")
                    self.model = keras.models.load_model(model_path)
                    self.model_loaded = True
                    print("[OK] CIFAR-100 model loaded successfully")
                    print(f"    - Model: Custom MobileNetV2")
                    print(f"    - Classes: 100 (CIFAR-100 categories)")
                    print(f"    - Accuracy: ~64.23%")
                else:
                    print(f"[!] CIFAR-100 model not found at {model_path}")
                    print("  Falling back to ImageNet model...")
                    self._load_imagenet_fallback()

            elif self.current_model_type == "custom":
                model_path = "models/custom_model_best.h5"
                if os.path.exists(model_path):
                    print(f"Loading custom model from {model_path}...")
                    self.model = keras.models.load_model(model_path)
                    self.model_loaded = True
                    print("[OK] Custom model loaded successfully")
                    print(f"    - Model: Custom MobileNetV2")
                    print(f"    - Classes: 12 (Your images)")
                else:
                    print(f"[!] Custom model not found at {model_path}")
                    print("  Falling back to ImageNet model...")
                    self._load_imagenet_fallback()
            else:
                print(f"[!] Unknown model type: {self.current_model_type}")
                self._load_imagenet_fallback()

        except Exception as e:
            print(f"[X] Error loading model: {e}")
            print("  Attempting to load pre-trained ImageNet model as fallback...")
            self._load_imagenet_fallback()
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
            self.current_model_type = "imagenet"
            self.model_config = AVAILABLE_MODELS["imagenet"]
            self.input_size = (224, 224)
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
            Preprocessed image as NumPy array
        """
        try:
            # Open and convert to RGB
            img = Image.open(image_path).convert("RGB")

            # For CIFAR-100 model, we need to resize to the model's expected input
            # The CIFAR-100 model was trained on images upscaled to 224x224
            if self.current_model_type in ["cifar100", "custom"]:
                # CIFAR-100 and Custom models expect 224x224 (trained with upscaled images)
                target_size = (224, 224)
            else:
                # ImageNet model expects 224x224
                target_size = (224, 224)

            img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            # Apply appropriate preprocessing
            if self.current_model_type == "imagenet":
                # MobileNetV2 specific preprocessing (scales to [-1, 1])
                img_array = preprocess_input(img_array)
            else:
                # Simple normalization for CIFAR and Custom models (same as training)
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

            if self.current_model_type == "imagenet":
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
                # CIFAR-100 or Custom mode - use our trained model
                predicted_class_id = int(np.argmax(predictions[0]))

                # Get custom labels based on model type
                if self.current_model_type == "custom":
                    custom_labels = [
                        "airplane",
                        "bus",
                        "cat",
                        "chair",
                        "cinema",
                        "deer",
                        "dog",
                        "flower",
                        "fruit",
                        "horse",
                        "library",
                        "truck",
                    ]
                    predicted_class = (
                        custom_labels[predicted_class_id]
                        if predicted_class_id < len(custom_labels)
                        else f"Class {predicted_class_id}"
                    )
                    model_type_name = "Custom (Your Images)"
                    total_classes = 12
                    superclass = None
                else:
                    predicted_class = (
                        CIFAR100_FINE_LABELS[predicted_class_id]
                        if predicted_class_id < len(CIFAR100_FINE_LABELS)
                        else f"Class {predicted_class_id}"
                    )
                    model_type_name = "CIFAR-100 (Custom Trained)"
                    total_classes = 100
                    superclass = None
                    if predicted_class_id in CIFAR100_FINE_TO_COARSE:
                        superclass_id = CIFAR100_FINE_TO_COARSE[predicted_class_id]
                        superclass = CIFAR100_COARSE_LABELS[superclass_id]

                confidence = float(predictions[0][predicted_class_id])

                # Get all class probabilities
                if self.current_model_type == "custom":
                    custom_labels = [
                        "airplane",
                        "bus",
                        "cat",
                        "chair",
                        "cinema",
                        "deer",
                        "dog",
                        "flower",
                        "fruit",
                        "horse",
                        "library",
                        "truck",
                    ]
                    all_predictions = [
                        {
                            "class_id": i,
                            "class_name": custom_labels[i]
                            if i < len(custom_labels)
                            else f"Class {i}",
                            "probability": float(predictions[0][i]),
                        }
                        for i in range(len(predictions[0]))
                    ]
                else:
                    all_predictions = [
                        {
                            "class_id": i,
                            "class_name": CIFAR100_FINE_LABELS[i]
                            if i < len(CIFAR100_FINE_LABELS)
                            else f"Class {i}",
                            "probability": float(predictions[0][i]),
                        }
                        for i in range(len(predictions[0]))
                    ]

                # Sort by probability (descending)
                all_predictions.sort(key=lambda x: x["probability"], reverse=True)

                return {
                    "success": True,
                    "predicted_class": predicted_class.replace("_", " ").title()
                    if isinstance(predicted_class, str)
                    else predicted_class,
                    "predicted_class_id": predicted_class_id,
                    "confidence": confidence,
                    "confidence_percent": f"{confidence * 100:.2f}%",
                    "superclass": superclass.replace("_", " ").title()
                    if superclass
                    else None,
                    "all_predictions": all_predictions[:10],
                    "top_5": all_predictions[:5],
                    "model_type": model_type_name,
                    "total_classes": total_classes,
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
        is_imagenet = self.current_model_type == "imagenet"

        info = {
            "model_loaded": self.model_loaded,
            "model_path": "Pre-trained (ImageNet)"
            if is_imagenet
            else CIFAR100_MODEL_PATH,
            "model_type": "MobileNetV2 (ImageNet)"
            if is_imagenet
            else "MobileNetV2 (CIFAR-100)",
            "current_model": self.current_model_type,
            "model_config": self.model_config,
        }

        if self.model_loaded and self.model is not None:
            info["total_params"] = self.model.count_params()
            info["input_shape"] = "(224, 224, 3)"
            info["num_classes"] = 1000 if is_imagenet else 100
            info["dataset"] = "ImageNet" if is_imagenet else "CIFAR-100"

            if is_imagenet:
                info["description"] = (
                    "Pre-trained on 1.2M images, can classify 1000 different objects"
                )
                info["accuracy"] = "~71% Top-1 (ImageNet benchmark)"
                info["categories_include"] = [
                    "Animals (dogs, cats, birds, fish, insects...)",
                    "Vehicles (cars, planes, boats, bikes...)",
                    "Food (fruits, vegetables, dishes...)",
                    "Objects (furniture, electronics, tools...)",
                    "Nature (plants, landscapes...)",
                    "And many more!",
                ]
            else:
                info["description"] = (
                    "Custom trained on CIFAR-100, 100 common object classes"
                )
                info["accuracy"] = "64.23% Top-1 (Your trained model)"
                info["categories_include"] = [
                    "Animals (dog, cat, bear, lion, elephant...)",
                    "Vehicles (bicycle, bus, motorcycle, train...)",
                    "Nature (forest, mountain, sea, cloud...)",
                    "Food (apple, orange, mushroom, sweet pepper...)",
                    "Household items (bed, chair, table, lamp...)",
                    "And 95 more classes!",
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


def switch_model(model_type):
    """
    Switch the global classifier to a different model type

    Args:
        model_type: "imagenet" or "cifar100"

    Returns:
        dict: Result with success status and model info
    """
    global _classifier_instance

    if model_type not in AVAILABLE_MODELS:
        return {
            "success": False,
            "error": f"Unknown model type: {model_type}. Available: {list(AVAILABLE_MODELS.keys())}",
        }

    # If we have an existing classifier, switch it
    if _classifier_instance is not None:
        success = _classifier_instance.switch_model(model_type)
    else:
        # Create new classifier with the requested model type
        _classifier_instance = ImageClassifier(model_type=model_type)
        success = _classifier_instance.model_loaded

    if success:
        return {
            "success": True,
            "model_type": model_type,
            "model_info": _classifier_instance.get_model_info(),
        }
    else:
        return {"success": False, "error": f"Failed to load {model_type} model"}


def get_available_models():
    """
    Get list of available models for switching

    Returns:
        dict: Available models with their configurations
    """
    return AVAILABLE_MODELS


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
