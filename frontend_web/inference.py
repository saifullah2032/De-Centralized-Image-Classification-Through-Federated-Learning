"""
Model Inference Module
Handles image classification predictions using the trained global model
"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # Must be set before importing Keras

import numpy as np
from PIL import Image
import keras

from backend_fl.config import (
    MODEL_PATH,
    LABELS,
    INPUT_SHAPE,
    NUM_CLASSES,
    DATASET,
    CIFAR100_COARSE_LABELS,
    CIFAR100_FINE_TO_COARSE,
)


class ImageClassifier:
    """
    Image classifier for CIFAR-10 using the trained federated model
    """

    def __init__(self, model_path=MODEL_PATH):
        """
        Initialize the classifier

        Args:
            model_path: Path to the trained model file (.h5)
        """
        self.model_path = model_path
        self.model = None
        self.model_loaded = False

        # Try to load model
        self.load_model()

    def load_model(self):
        """Load the trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                print(f"Loading model from {self.model_path}...")
                self.model = keras.models.load_model(self.model_path)
                self.model_loaded = True
                print("[OK] Model loaded successfully")
            except Exception as e:
                print(f"[X] Error loading model: {e}")
                self.model_loaded = False
        else:
            print(f"[!] Model not found at {self.model_path}")
            print("  Train the model first using the FL server and clients")
            self.model_loaded = False

    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image as NumPy array with shape (1, 32, 32, 3)
        """
        try:
            # Open and convert to RGB
            img = Image.open(image_path).convert("RGB")

            # Resize to CIFAR-10 dimensions
            img = img.resize((INPUT_SHAPE[0], INPUT_SHAPE[1]))

            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)

            # Normalize pixel values to [0, 1]
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            return img_array

        except Exception as e:
            print(f"[X] Error preprocessing image: {e}")
            raise

    def predict(self, image_path):
        """
        Predict the class of an image

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing:
                - success: Whether prediction succeeded
                - predicted_class: Name of the predicted class
                - predicted_class_id: ID of the predicted class
                - confidence: Confidence score (0-1)
                - all_predictions: List of all class probabilities
                - error: Error message (if failed)
        """
        if not self.model_loaded:
            return {
                "success": False,
                "error": "Model not loaded. Please train the model first.",
            }

        try:
            # Preprocess image
            img_array = self.preprocess_image(image_path)

            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)

            # Get predicted class
            predicted_class_id = int(np.argmax(predictions[0]))
            predicted_class = LABELS[predicted_class_id]
            confidence = float(predictions[0][predicted_class_id])

            # Get superclass for CIFAR-100
            superclass = None
            if DATASET == "CIFAR100" and predicted_class_id in CIFAR100_FINE_TO_COARSE:
                superclass_id = CIFAR100_FINE_TO_COARSE[predicted_class_id]
                superclass = CIFAR100_COARSE_LABELS[superclass_id]

            # Get all class probabilities
            all_predictions = [
                {
                    "class_id": i,
                    "class_name": LABELS[i],
                    "probability": float(predictions[0][i]),
                }
                for i in range(len(LABELS))
            ]

            # Sort by probability (descending)
            all_predictions.sort(key=lambda x: x["probability"], reverse=True)

            return {
                "success": True,
                "predicted_class": predicted_class,
                "predicted_class_id": predicted_class_id,
                "confidence": confidence,
                "superclass": superclass,  # CIFAR-100 superclass (or None for CIFAR-10)
                "all_predictions": all_predictions,
                "top_5": all_predictions[:5],
            }

        except Exception as e:
            print(f"[X] Prediction error: {e}")
            return {"success": False, "error": str(e)}

    def get_model_info(self):
        """
        Get information about the loaded model

        Returns:
            Dictionary containing model information
        """
        info = {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
        }

        if self.model_loaded:
            info["total_params"] = self.model.count_params()
            info["input_shape"] = str(INPUT_SHAPE)
            info["num_classes"] = NUM_CLASSES
            info["dataset"] = DATASET
            info["classes"] = LABELS

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


if __name__ == "__main__":
    """Test inference module"""
    print("Testing inference module...")

    # Create classifier
    classifier = ImageClassifier()

    # Get model info
    info = classifier.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    if classifier.model_loaded:
        print("\n[OK] Inference module test passed!")
        print("  Note: To test prediction, run with an actual image file")
    else:
        print("\n[!] Model not loaded (expected if not trained yet)")
        print("  Run FL training first to generate the model")
