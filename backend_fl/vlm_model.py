"""
VLM Model Module for Decentralized Multimodal Visual Assistant
Supports BLIP (lightweight) for image captioning and description generation
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image


class VLMModel:
    """
    Vision-Language Model wrapper with LoRA support for federated learning.

    Uses lightweight BLIP model for CPU-friendly inference.
    """

    SUPPORTED_MODELS = {
        "blip": "Salesforce/blip-image-captioning-base",
    }

    def __init__(
        self,
        model_name: str = "blip",
        device: str = None,
        lora_config: Dict = None,
    ):
        """
        Initialize VLM model.

        Args:
            model_name: Name of the VLM model to use
            device: Device to load model on ('cuda' or 'cpu')
            lora_config: LoRA configuration dictionary
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lora_config = lora_config or self._default_lora_config()
        self.model = None
        self.processor = None
        self.lora_adapter = None
        self.is_lora_merged = False

    def _default_lora_config(self) -> Dict:
        """Default LoRA configuration."""
        return {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": [
                "query",
                "value",
                "key",
                "dense",
            ],  # BLIP attention modules
            "bias": "none",
            "task_type": "SEQ_CLS",
        }

    def load_base_model(self, model_id: str = None):
        """
        Load the base VLM model (frozen).

        Args:
            model_id: HuggingFace model ID (uses default if None)
        """
        if model_id is None:
            model_id = self.SUPPORTED_MODELS.get(
                self.model_name, self.SUPPORTED_MODELS["blip"]
            )

        print(f"Loading base VLM model: {self.model_name} ({model_id})")
        print(f"Device: {self.device}")

        try:
            if self.model_name == "blip":
                self._load_blip(model_id)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

            print(f"✓ Base model loaded successfully")

        except Exception as e:
            print(f"Error loading base model: {e}")
            raise

    def _load_blip(self, model_id: str):
        """Load lightweight BLIP model."""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration

            print(f"Loading BLIP from {model_id}...")
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            )
            self.model.to(self.device)
            self.model.eval()

        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

    def _load_blip2(self, model_id: str):
        """Load BLIP-2 model."""
        try:
            from transformers import Blip2Processor, Blip2Model

            print(f"Loading BLIP-2 from {model_id}...")
            self.processor = Blip2Processor.from_pretrained(model_id)
            self.model = Blip2Model.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            )
            self.model.to(self.device)
            self.model.eval()

        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

    def _load_paligemma(self, model_id: str):
        """Load PaliGemma model."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map=self.device,
            )
            self.model.eval()

        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

    def setup_lora(self):
        """Set up LoRA adapters using PEFT."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("Please install PEFT: pip install peft")

        print(f"Setting up LoRA with config: {self.lora_config}")

        lora_config = LoraConfig(
            r=self.lora_config.get("r", 8),
            lora_alpha=self.lora_config.get("lora_alpha", 16),
            lora_dropout=self.lora_config.get("lora_dropout", 0.05),
            target_modules=self.lora_config.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            bias=self.lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.lora_adapter = self.model

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"✓ LoRA initialized")
        print(
            f"  Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)"
        )
        print(f"  Total parameters: {total_params:,}")

    def load_lora_weights(self, adapter_path: str):
        """
        Load LoRA weights from a file.

        Args:
            adapter_path: Path to saved LoRA weights
        """
        if not os.path.exists(adapter_path):
            print(f"Warning: LoRA weights not found at {adapter_path}")
            return

        print(f"Loading LoRA weights from: {adapter_path}")

        state_dict = torch.load(adapter_path, map_location=self.device)

        if self.model is not None and hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✓ LoRA weights loaded")
        else:
            print(
                "Warning: Model not initialized. LoRA weights will be applied when model is loaded."
            )

    def save_lora_weights(self, adapter_path: str):
        """
        Save LoRA weights to a file.

        Args:
            adapter_path: Path to save LoRA weights
        """
        if self.model is None:
            raise ValueError("No model loaded")

        os.makedirs(os.path.dirname(adapter_path), exist_ok=True)

        if hasattr(self.model, "base_model"):
            state_dict = self.model.base_model.state_dict()
        else:
            state_dict = self.model.state_dict()

        torch.save(state_dict, adapter_path)
        print(f"✓ LoRA weights saved to: {adapter_path}")

    def get_trainable_parameters(self) -> List[np.ndarray]:
        """
        Get only trainable LoRA parameters for federated learning.

        Returns:
            List of NumPy arrays containing trainable parameters
        """
        if self.model is None:
            raise ValueError("No model loaded")

        trainable_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param.detach().cpu().numpy())

        return trainable_params

    def set_trainable_parameters(self, parameters: List[np.ndarray]):
        """
        Set trainable parameters from federated server.

        Args:
            parameters: List of NumPy arrays from server
        """
        if self.model is None:
            raise ValueError("No model loaded")

        param_iter = iter(parameters)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_data = next(param_iter)
                param.data = torch.from_numpy(param_data).to(param.device)

    def generate_description(
        self,
        image: Image.Image,
        prompt: str = None,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate a 5-point structured description for an image.
        Uses lightweight BLIP for captioning, then maps to 5-point structure.

        Args:
            image: PIL Image to describe
            prompt: Optional custom prompt (uses default if None)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text description
        """
        if self.model is None:
            raise ValueError("No model loaded")

        try:
            if self.model_name == "blip":
                return self._generate_blip(image, max_new_tokens)
            else:
                raise ValueError(f"Unsupported model for generation: {self.model_name}")

        except Exception as e:
            print(f"Error generating description: {e}")
            return f"Error: {str(e)}"

    def _generate_blip(
        self,
        image: Image.Image,
        max_new_tokens: int,
    ) -> str:
        """Generate description using lightweight BLIP."""
        self.model.eval()

        try:
            detailed_parts = []

            # Generate class/category - BLIP format
            inputs = self.processor(image, "a photo of", return_tensors="pt").to(
                self.device, torch.float32
            )
            with torch.no_grad():
                caption_ids = self.model.generate(
                    **inputs, max_new_tokens=30, num_beams=5
                )
                caption = self.processor.decode(
                    caption_ids[0], skip_special_tokens=True
                )
            detailed_parts.append(f"1. Class: {caption}")

            # Generate detailed description
            inputs = self.processor(image, "This is", return_tensors="pt").to(
                self.device, torch.float32
            )
            with torch.no_grad():
                desc_ids = self.model.generate(**inputs, max_new_tokens=50, num_beams=5)
                desc = self.processor.decode(desc_ids[0], skip_special_tokens=True)
            detailed_parts.append(f"2. What it is: {desc}")

            # Generate purpose (use different prompt pattern)
            inputs = self.processor(image, "It is used for", return_tensors="pt").to(
                self.device, torch.float32
            )
            with torch.no_grad():
                purpose_ids = self.model.generate(
                    **inputs, max_new_tokens=30, num_beams=5
                )
                purpose = self.processor.decode(
                    purpose_ids[0], skip_special_tokens=True
                )
            detailed_parts.append(f"3. Purpose: {purpose}")

            # Generate origin/setting
            inputs = self.processor(
                image, "It is typically found in", return_tensors="pt"
            ).to(self.device, torch.float32)
            with torch.no_grad():
                origin_ids = self.model.generate(
                    **inputs, max_new_tokens=30, num_beams=5
                )
                origin = self.processor.decode(origin_ids[0], skip_special_tokens=True)
            detailed_parts.append(f"4. Origin: {origin}")

            # Generate specific name
            inputs = self.processor(
                image, "The specific name is", return_tensors="pt"
            ).to(self.device, torch.float32)
            with torch.no_grad():
                name_ids = self.model.generate(**inputs, max_new_tokens=30, num_beams=5)
                name = self.processor.decode(name_ids[0], skip_special_tokens=True)
            detailed_parts.append(f"5. Name: {name}")

            return "\n".join(detailed_parts)

        except Exception as e:
            print(f"BLIP generation error: {e}")
            # Fallback to simple caption
            try:
                inputs = self.processor(image, "a photo of", return_tensors="pt").to(
                    self.device, torch.float32
                )
                with torch.no_grad():
                    caption_ids = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens
                    )
                    caption = self.processor.decode(
                        caption_ids[0], skip_special_tokens=True
                    )
                return f"1. Class: {caption}\n2. What it is: {caption}\n3. Purpose: Unknown\n4. Origin: Unknown\n5. Name: {caption}"
            except Exception as e2:
                return f"Error generating description: {str(e2)}"

    def parse_five_point_description(self, text: str) -> Dict[str, str]:
        """
        Parse generated text into structured 5-point format.

        Args:
            text: Generated text from VLM

        Returns:
            Dictionary with keys: 1. Class, 2. What it is, etc.
        """
        result = {}

        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            for i in range(1, 6):
                prefix = f"{i}."
                if (
                    line.lower().startswith(prefix.lower())
                    or line.lower().startswith(f"{i}. class")
                    or line.lower().startswith(f"{i}. what")
                ):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        result[key] = value
                    break
                elif line[0].isdigit() and "." in line[:3]:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        result[key] = value
                    break

        return result

    def merge_lora_weights(self):
        """Merge LoRA weights into base model for inference."""
        if hasattr(self.model, "merge_weights"):
            self.model.merge_weights()
            self.is_lora_merged = True
            print("✓ LoRA weights merged into base model")

    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✓ Model unloaded")


def load_vlm_model(
    model_name: str = "moondream2",
    device: str = None,
    lora_config: Dict = None,
) -> VLMModel:
    """
    Convenience function to load a VLM model.

    Args:
        model_name: Name of the VLM model
        device: Device to load on
        lora_config: LoRA configuration

    Returns:
        VLMModel instance
    """
    vlm = VLMModel(model_name=model_name, device=device, lora_config=lora_config)
    vlm.load_base_model()
    vlm.setup_lora()
    return vlm


def get_lora_weight_size(lora_config: Dict) -> int:
    """
    Calculate approximate size of LoRA weights in bytes.

    Args:
        lora_config: LoRA configuration

    Returns:
        Approximate size in bytes
    """
    r = lora_config.get("r", 8)
    alpha = lora_config.get("lora_alpha", 16)

    hidden_dim = 2048
    num_layers = 12

    lora_a_params = r * hidden_dim * num_layers * 2
    lora_b_params = hidden_dim * r * num_layers * 2

    total_params = lora_a_params + lora_b_params
    bytes_per_param = 4

    return total_params * bytes_per_param


class VLMInference:
    """Inference wrapper for VLM model in web service."""

    def __init__(self, model_path: str = None):
        """
        Initialize VLM for inference.

        Args:
            model_path: Path to LoRA weights (optional)
        """
        self.vlm_model = None
        self.model_path = model_path
        self.model_loaded = False

    def load(self, model_name: str = "moondream2"):
        """Load VLM model."""
        if self.model_loaded:
            print("Model already loaded")
            return

        print(f"Loading VLM model: {model_name}")
        self.vlm_model = load_vlm_model(model_name=model_name)

        if self.model_path and os.path.exists(self.model_path):
            self.vlm_model.load_lora_weights(self.model_path)

        self.model_loaded = True
        print("✓ VLM model loaded for inference")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Generate 5-point description for image.

        Args:
            image: PIL Image

        Returns:
            Dictionary with description and metadata
        """
        if not self.model_loaded:
            self.load()

        prompt = self.vlm_model._default_five_point_prompt()

        raw_output = self.vlm_model.generate_description(
            image=image,
            prompt=prompt,
            max_new_tokens=512,
        )

        parsed = self.vlm_model.parse_five_point_description(raw_output)

        return {
            "raw_output": raw_output,
            "structured": parsed,
            "success": True,
        }

    def unload(self):
        """Unload model."""
        if self.vlm_model:
            self.vlm_model.unload_model()
        self.model_loaded = False
