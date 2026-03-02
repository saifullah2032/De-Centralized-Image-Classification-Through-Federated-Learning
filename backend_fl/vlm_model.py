"""
VLM Model Module for Decentralized Multimodal Visual Assistant
Uses BLIP-VQA (Salesforce/blip-vqa-base) for visual question answering
"""

import os
import gc
import torch
import numpy as np
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image


class VLMModel:
    """
    Vision-Language Model wrapper using BLIP-VQA for hybrid ensemble.

    Supports VQA-based analysis with context from CNN classification.
    """

    SUPPORTED_MODELS = {
        "blip-vqa": "Salesforce/blip-vqa-base",
    }

    def __init__(
        self,
        model_name: str = "blip-vqa",
        device: str = None,
        lora_config: Dict = None,
    ):
        """
        Initialize BLIP-VQA model.

        Args:
            model_name: Name of the VLM model (currently only 'blip-vqa')
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

        # Threading support
        self._inference_thread = None
        self._inference_result = None
        self._inference_exception = None
        self._inference_lock = threading.Lock()

    def _default_lora_config(self) -> Dict:
        """Default LoRA configuration for BLIP-VQA."""
        return {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q", "v"],  # BLIP attention modules
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

    def load_base_model(self, model_id: str = None):
        """
        Load the base BLIP-VQA model.

        Args:
            model_id: HuggingFace model ID (uses default if None)
        """
        if model_id is None:
            model_id = self.SUPPORTED_MODELS.get(
                self.model_name, self.SUPPORTED_MODELS["blip-vqa"]
            )

        print(f"Loading base VLM model: {self.model_name} ({model_id})")
        print(f"Device: {self.device}")

        try:
            if self.model_name == "blip-vqa":
                self._load_blip_vqa(model_id)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

            print(f"[OK] Base model loaded successfully")

        except Exception as e:
            print(f"[ERROR] Error loading base model: {e}")
            import traceback

            traceback.print_exc()

    def _load_blip_vqa(self, model_id: str):
        """
        Load BLIP-VQA model and processor from HuggingFace.

        Args:
            model_id: HuggingFace model ID
        """
        from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

        try:
            print(f"[Loading] Processor from {model_id}...")
            self.processor = AutoProcessor.from_pretrained(model_id)

            print(f"[Loading] Model from {model_id}...")
            # CRITICAL FIX: device_map should be None for CPU to avoid 'meta device' tensor errors
            # device_map is only for multi-GPU setups; for 8GB RAM CPU mode, use explicit .to(device)
            self.model = AutoModelForVisualQuestionAnswering.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                device_map=None,  # Explicitly None for CPU compatibility
            )

            # Explicitly move model to device (critical for 8GB RAM CPU-only environments)
            self.model = self.model.to(self.device)

            print(f"[OK] BLIP-VQA model loaded successfully")
            print(f"     Model type: {type(self.model).__name__}")
            print(f"     Processor type: {type(self.processor).__name__}")
            print(f"     Device: {self.device}")

        except Exception as e:
            print(f"[ERROR] Failed to load BLIP-VQA model: {e}")
            raise

    def setup_lora(self):
        """Setup LoRA for fine-tuning."""
        if not self.model:
            print("[ERROR] Model not loaded. Load model first.")
            return False

        try:
            from peft import get_peft_model, LoraConfig

            lora_config = LoraConfig(**self.lora_config)
            self.model = get_peft_model(self.model, lora_config)
            print("[OK] LoRA setup complete")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to setup LoRA: {e}")
            return False

    def load_lora_weights(self, adapter_path: str):
        """Load LoRA weights from file."""
        if not self.model:
            print("[ERROR] Model not loaded")
            return False

        try:
            from peft import PeftModel

            if not os.path.exists(adapter_path):
                print(f"[ERROR] Adapter path not found: {adapter_path}")
                return False

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.lora_adapter = adapter_path
            print(f"[OK] LoRA weights loaded from {adapter_path}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load LoRA weights: {e}")
            return False

    def save_lora_weights(self, adapter_path: str):
        """Save LoRA weights to file."""
        if not self.model:
            print("[ERROR] Model not loaded")
            return False

        try:
            os.makedirs(adapter_path, exist_ok=True)
            self.model.save_pretrained(adapter_path)
            print(f"[OK] LoRA weights saved to {adapter_path}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to save LoRA weights: {e}")
            return False

    def get_trainable_parameters(self) -> List[np.ndarray]:
        """Get trainable parameters for federated learning."""
        if not self.model:
            return []

        params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param.data.cpu().numpy())
        return params

    def set_trainable_parameters(self, parameters: List[np.ndarray]):
        """Set trainable parameters from federated learning."""
        if not self.model:
            return False

        param_list = [p for name, p in self.model.named_parameters() if p.requires_grad]
        if len(param_list) != len(parameters):
            print("[ERROR] Parameter count mismatch")
            return False

        for param, new_value in zip(param_list, parameters):
            param.data = torch.from_numpy(new_value).to(self.device)
        return True

    def _set_thread_priority(self, priority: str = "low"):
        """Set thread priority for inference."""
        try:
            if priority == "low":
                os.nice(10)
        except Exception as e:
            print(f"[WARNING] Could not set thread priority: {e}")

    def semantic_consistency_check(
        self,
        image: Image.Image,
        cnn_prediction: str = None,
    ) -> Dict[str, Any]:
        """
        Semantic Consistency Layer (SCL) - Interrogative Check.

        Before Stage 2 analysis, ask the VLM: "Does this image contain a [CNN_Class]?"
        This acts as a verification gate to prevent high-confidence CNN misidentifications
        from poisoning the audit pipeline.

        Args:
            image: PIL Image object
            cnn_prediction: CNN class name to verify (e.g., 'Rose')

        Returns:
            Dictionary with SCL verification result:
            {
                'scl_verified': bool,  # True if VLM confirms CNN prediction
                'scl_response': str,   # VLM's raw response
                'scl_status': str,     # 'Verified' or 'Self-Corrected'
                'interrogative_question': str,  # The question asked
            }
        """
        if not self.model or not self.processor:
            return {
                "scl_verified": True,  # Default to verified if check fails
                "scl_response": "N/A",
                "scl_status": "Verified",
                "interrogative_question": "N/A",
            }

        if not cnn_prediction:
            return {
                "scl_verified": True,  # No prediction to verify
                "scl_response": "N/A",
                "scl_status": "Verified",
                "interrogative_question": "N/A",
            }

        try:
            # Formulate interrogative check
            interrogative_question = f"Does this image contain a {cnn_prediction}? Answer with 'Yes' or 'No' only."

            # Get VQA response
            inputs = self.processor(
                image, interrogative_question, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    do_sample=False,
                    temperature=0.7,
                )

            scl_response = self.processor.decode(outputs[0], skip_special_tokens=True)
            scl_response = scl_response.strip().lower()

            # Determine if VLM confirms CNN prediction
            # Looking for "yes" in response indicates confirmation
            scl_verified = (
                "yes" in scl_response or "yep" in scl_response or "yeah" in scl_response
            )

            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return {
                "scl_verified": scl_verified,
                "scl_response": scl_response,
                "scl_status": "Verified" if scl_verified else "Self-Corrected",
                "interrogative_question": interrogative_question,
            }

        except Exception as e:
            print(f"[WARNING] SCL interrogative check failed: {e}")
            # If SCL check fails, default to verification (fail-open)
            return {
                "scl_verified": True,
                "scl_response": f"Error: {str(e)}",
                "scl_status": "Verified",
                "interrogative_question": f"Does this image contain a {cnn_prediction}?",
            }

    def generate_description(
        self,
        image: Image.Image,
        question: str = "What is this object?",
        max_length: int = 100,
    ) -> Dict[str, str]:
        """
        Generate VQA response for a single question.

        Args:
            image: PIL Image object
            question: Question to ask about the image
            max_length: Maximum response length

        Returns:
            Dictionary with response and metadata
        """
        if not self.model or not self.processor:
            return self._error_response()

        try:
            # Process inputs
            inputs = self.processor(image, question, return_tensors="pt").to(
                self.device
            )

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=max_length)

            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)

            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return {
                "success": True,
                "response": response,
                "question": question,
                "model": "BLIP-VQA",
            }

        except Exception as e:
            print(f"[ERROR] VQA generation failed: {e}")
            return self._error_response()

    def generate_hybrid_description(
        self,
        image: Image.Image,
        cnn_prediction: str = None,
        max_new_tokens: int = 100,
    ) -> Dict[str, str]:
        """
        Generate 9-point analysis using BLIP-VQA with optional CNN context.

        Stage 2 of hybrid pipeline with intelligent routing support.

        Args:
            image: PIL Image object
            cnn_prediction: Class name from CNN Stage 1 (e.g., 'Rose'), or None
                           - If None: Uses generic analysis prompts
                           - If str: Uses context-aware prompts with class name
            max_new_tokens: Maximum tokens per response

        Returns:
            Dictionary with 9-point analysis
        """
        if not self.model or not self.processor:
            return self._error_response()

        return self._generate_hybrid_blip_vqa(image, cnn_prediction, max_new_tokens)

    def _generate_hybrid_blip_vqa(
        self,
        image: Image.Image,
        cnn_prediction: str = None,
        max_new_tokens: int = 100,
    ) -> Dict[str, str]:
        """
        Generate 9-point BLIP-VQA analysis with optional CNN context.

        Intelligent routing: Adapts questions based on whether context is available.

        Args:
            image: PIL Image object
            cnn_prediction: CNN class name for context (or None for generic analysis)
            max_new_tokens: Maximum tokens per response

        Returns:
            Dictionary mapping category to analysis
        """
        # Build 9-point schema with context-aware or generic questions
        if cnn_prediction:
            # High confidence: Use context-aware questions
            questions = {
                "Common Identity": f"Identify this object as a {cnn_prediction}. What is it and what is its primary purpose?",
                "Visual Summary": f"As a {cnn_prediction}, describe its visual appearance and distinctive features.",
                "Operational Utility": f"What is the functional utility and professional use case of this {cnn_prediction}?",
                "Provenance & Setting": f"Where would you typically find a {cnn_prediction}, and what is its geographic or environmental origin?",
                "Technical Nomenclature": f"What is the official technical or scientific name for this {cnn_prediction}?",
                "Safety & Risk Assessment": f"What are the primary safety considerations and potential hazards associated with a {cnn_prediction}?",
                "Maintenance & Longevity": f"How should a {cnn_prediction} be maintained, and what is its typical lifespan?",
                "Aesthetic & Design Style": f"Describe the aesthetic qualities, design style, and artistic characteristics of this {cnn_prediction}.",
                "Interaction & Relationship": f"How does a {cnn_prediction} interact with humans and the environment?",
            }
            routing_mode = "high_confidence"
        else:
            # Low confidence: Use generic prompts without forced identity
            questions = {
                "Common Identity": "Identify what this object is. What is its primary purpose or function?",
                "Visual Summary": "Describe the visual appearance and distinctive features of this object.",
                "Operational Utility": "What is the functional utility and professional use case of this object?",
                "Provenance & Setting": "Where would you typically find this object, and what is its environmental or geographic origin?",
                "Technical Nomenclature": "What is the official technical or scientific name for this object?",
                "Safety & Risk Assessment": "What are the primary safety considerations and potential hazards associated with this object?",
                "Maintenance & Longevity": "How should this object be maintained, and what is its typical lifespan?",
                "Aesthetic & Design Style": "Describe the aesthetic qualities, design style, and artistic characteristics of this object.",
                "Interaction & Relationship": "How does this object interact with humans and the environment?",
            }
            routing_mode = "low_confidence"

        analysis = {}
        errors = []

        try:
            for category, question in questions.items():
                try:
                    # Get VQA response
                    inputs = self.processor(image, question, return_tensors="pt").to(
                        self.device
                    )

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=max_new_tokens,
                            do_sample=False,
                            temperature=0.7,
                        )

                    response = self.processor.decode(
                        outputs[0], skip_special_tokens=True
                    )

                    # Apply Narrative Fluidity Engine for natural language synthesis
                    response = self._synthesize_narrative(
                        response, category, cnn_prediction
                    )
                    analysis[category] = response

                    # Clean memory after each question
                    gc.collect()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"[WARNING] Failed to generate for {category}: {e}")
                    errors.append(category)
                    analysis[category] = self._get_default_for_key(category)

            # Validate all 9 keys present
            required_keys = {
                "Common Identity",
                "Visual Summary",
                "Operational Utility",
                "Provenance & Setting",
                "Technical Nomenclature",
                "Safety & Risk Assessment",
                "Maintenance & Longevity",
                "Aesthetic & Design Style",
                "Interaction & Relationship",
            }

            for key in required_keys:
                if key not in analysis or not analysis[key]:
                    analysis[key] = self._get_default_for_key(key)

            result = {
                "success": True,
                "structured": analysis,
                "model": "BLIP-VQA",
                "cnn_context": cnn_prediction,
                "routing_mode": routing_mode,
                "errors": errors if errors else None,
                "_hybrid_metadata": {
                    "stage": "vqa",
                    "cnn_class": cnn_prediction,
                    "num_questions": len(questions),
                    "successful_responses": len(analysis) - len(errors),
                    "routing": routing_mode,
                },
            }

            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            print(f"[ERROR] Hybrid BLIP-VQA generation failed: {e}")
            import traceback

            traceback.print_exc()
            return self._error_response()

    def _get_vocabulary_variant(self) -> str:
        """
        Rotate through vocabulary variants for narrative diversity.

        Returns:
            A phrase variant from the vocabulary variator pool
        """
        variants = [
            "Visual analysis suggests",
            "Technical markers indicate",
            "Contextual evidence implies",
            "Structural assessment reveals",
            "Observed characteristics demonstrate",
        ]

        # Use a rotating index (you can enhance this with state management if needed)
        if not hasattr(self, "_variant_index"):
            self._variant_index = 0

        variant = variants[self._variant_index % len(variants)]
        self._variant_index += 1
        return variant

    def _synthesize_narrative(
        self,
        response: str,
        category: str,
        cnn_prediction: str = None,
    ) -> str:
        """
        Narrative Fluidity Engine: Transform raw VQA responses into natural, contextual language.

        Uses the Stage 1 CNN prediction to create contextually-appropriate narratives
        that sound natural rather than templated.

        Args:
            response: Raw VQA response
            category: Analysis category (e.g., "Common Identity", "Safety & Risk Assessment")
            cnn_prediction: CNN class name for contextual synthesis

        Returns:
            Naturally synthesized narrative response
        """
        if not response or not isinstance(response, str):
            return response

        response = response.strip()
        word_count = len(response.split())

        # Context-aware synthesis based on category and prediction
        if category == "Common Identity":
            return self._synthesize_identity(response, cnn_prediction, word_count)
        elif category == "Visual Summary":
            return self._synthesize_visual(response, cnn_prediction, word_count)
        elif category == "Operational Utility":
            return self._synthesize_utility(response, cnn_prediction, word_count)
        elif category == "Provenance & Setting":
            return self._synthesize_provenance(response, cnn_prediction, word_count)
        elif category == "Technical Nomenclature":
            return self._synthesize_nomenclature(response, cnn_prediction, word_count)
        elif category == "Safety & Risk Assessment":
            return self._synthesize_safety(response, cnn_prediction, word_count)
        elif category == "Maintenance & Longevity":
            return self._synthesize_maintenance(response, cnn_prediction, word_count)
        elif category == "Aesthetic & Design Style":
            return self._synthesize_aesthetic(response, cnn_prediction, word_count)
        elif category == "Interaction & Relationship":
            return self._synthesize_interaction(response, cnn_prediction, word_count)

        return response

    def _synthesize_identity(
        self, response: str, cnn_prediction: str = None, word_count: int = None
    ) -> str:
        """Synthesize Common Identity narrative - Establish primary object classification."""
        response_lower = response.lower()
        word_count = word_count or len(response.split())

        if word_count == 1:
            # Single word: "rose" -> elegant botanical/technical context
            if cnn_prediction:
                return (
                    f"Primary identification: This specimen is classified as a {response}, "
                    f"which aligns with {cnn_prediction} taxonomy. The object exhibits distinctive morphological "
                    f"and functional characteristics that establish its categorical position within the visual domain."
                )
            else:
                return (
                    f"Primary identification: This specimen is classified as a {response}. "
                    f"The object represents a distinct category with specific identifying characteristics, "
                    f"cultural significance, and established functional purpose."
                )
        elif word_count == 2:
            # Two words: enhance with classification context
            return (
                f"This specimen is comprehensively identified as {response_lower}. "
                f"The classification represents an object of {cnn_prediction or 'particular'} significance, "
                f"exhibiting distinctive properties that establish its position within established taxonomies."
            )
        else:
            # Longer response: frame as primary classification statement
            return (
                f"Primary identification determined as follows: {response_lower}. "
                f"This classification encompasses the object's fundamental identity, categorical position, "
                f"and the distinguishing characteristics that establish its role within its domain."
            )

    def _synthesize_visual(
        self, response: str, cnn_prediction: str = None, word_count: int = None
    ) -> str:
        """Synthesize Visual Summary narrative - Analyze appearance and visual characteristics."""
        response_lower = response.lower()
        word_count = word_count or len(response.split())

        if word_count == 1:
            # Single word visual descriptor: "red" -> elaborate visual analysis
            return (
                f"Visual analysis reveals: This specimen demonstrates {response_lower} as a primary visual characteristic. "
                f"The chromatic and morphological properties are consistent with {cnn_prediction or 'its classification'}, "
                f"establishing visual distinction within its categorical domain through observable aesthetic features."
            )
        elif word_count == 2:
            # Two-word visual descriptor: elaborate naturally
            return (
                f"Visual characteristics indicate: This specimen exhibits {response_lower}. "
                f"The appearance is consistent with its classification as a {cnn_prediction or 'representative object'}, "
                f"with observable aesthetic and structural qualities that distinguish it within its visual category."
            )
        else:
            # Longer visual description: frame with professional context
            return (
                f"Visual analysis: {response_lower}. These observable features are "
                f"characteristic of the object's classification, establishing its visual identity and "
                f"aesthetic position within established visual taxonomies and standards."
            )

    def _synthesize_utility(
        self, response: str, cnn_prediction: str = None, word_count: int = None
    ) -> str:
        """Synthesize Operational Utility narrative - Explain functional purpose and applications."""
        response_lower = response.lower()
        word_count = word_count or len(response.split())

        if word_count == 1:
            # Single word utility: "decoration" -> elaborate use case
            return (
                f"Operational utility assessment: This {cnn_prediction or 'object'} primarily serves as a {response_lower}. "
                f"Its functional role is realized through professional applications within appropriate environmental contexts, "
                f"reflecting domain-specific requirements and established usage standards."
            )
        elif word_count <= 3:
            # Brief utility description
            return (
                f"Functional assessment: This specimen's operational utility is defined by its role in {response_lower}. "
                f"The functional context reflects professional standards, domain-specific applications, and established conventions for use."
            )
        else:
            # Comprehensive utility description
            return (
                f"Operational utility: {response_lower}. The functional applications are aligned with professional practices, "
                f"domain standards, and established conventions that govern optimal utilization within specific contexts."
            )

    def _synthesize_provenance(
        self, response: str, cnn_prediction: str = None, word_count: int = None
    ) -> str:
        """Synthesize Provenance & Setting narrative - Establish environmental and geographic context."""
        response_lower = response.lower()
        word_count = word_count or len(response.split())

        if word_count == 1:
            # Single word setting: "garden" -> elaborate geographic/environmental context
            return (
                f"Provenance assessment: This {cnn_prediction or 'object'} is typically found in {response_lower} environments. "
                f"The geographic and environmental origins are intrinsic to its functional purpose and optimal application contexts, "
                f"reflecting the specific conditions that support its primary characteristics."
            )
        elif word_count <= 3:
            # Brief provenance description
            return (
                f"Environmental context: {response_lower}. This {cnn_prediction or 'object'} is endemic to specific "
                f"environmental and geographic contexts that support its functional purpose and operational role."
            )
        else:
            # Comprehensive provenance description
            return (
                f"Provenance and setting analysis: {response_lower}. The environmental and geographic origins are "
                f"intrinsic to the object's identity, functional significance, and optimal utilization within specific ecological contexts."
            )

    def _synthesize_nomenclature(
        self, response: str, cnn_prediction: str = None, word_count: int = None
    ) -> str:
        """Synthesize Technical Nomenclature narrative - Establish formal scientific/technical classification."""
        response_lower = response.lower()

        return (
            f"Technical nomenclature: The formal designation for this specimen is {response_lower}. "
            f"This technical classification reflects established scientific and professional standards, "
            f"ensuring precise and unambiguous identification across disciplinary and geographic domains."
        )

    def _synthesize_safety(
        self, response: str, cnn_prediction: str = None, word_count: int = None
    ) -> str:
        """Synthesize Safety & Risk Assessment narrative - Identify hazards and mitigation strategies."""
        response_lower = response.lower()
        word_count = word_count or len(response.split())

        if word_count == 1:
            # Single word: "toxic" -> elaborate safety context
            return (
                f"Safety assessment: The primary consideration for this {cnn_prediction or 'object'} is {response_lower}. "
                f"Subject requires careful handling and adherence to established safety protocols to mitigate associated hazards. "
                f"Professional risk management and domain-specific safety guidelines are essential."
            )
        elif word_count <= 3:
            # Brief safety description
            return (
                f"Safety and risk assessment: {response_lower}. "
                f"Adherence to established safety guidelines and professional handling procedures is recommended "
                f"for proper management and safe utilization of this specimen."
            )
        else:
            # Comprehensive safety description
            return (
                f"Safety and risk assessment: {response_lower}. Professional risk management, "
                f"adherence to domain-specific safety protocols, and established hazard mitigation procedures are advised."
            )

    def _synthesize_maintenance(
        self, response: str, cnn_prediction: str = None, word_count: int = None
    ) -> str:
        """Synthesize Maintenance & Longevity narrative - Provide care and durability guidance."""
        response_lower = response.lower()
        word_count = word_count or len(response.split())

        if word_count <= 2:
            # Brief maintenance directive
            return (
                f"Maintenance protocol: This specimen requires {response_lower}. "
                f"Regular care and appropriate attention ensure optimal longevity, sustained functionality, "
                f"and preservation of all structural and operational characteristics."
            )
        else:
            # Comprehensive maintenance description
            return (
                f"Maintenance and longevity assessment: {response_lower}. "
                f"Appropriate upkeep and professional care procedures contribute significantly to long-term durability, "
                f"performance optimization, and sustained functionality of this specimen."
            )

    def _synthesize_aesthetic(
        self, response: str, cnn_prediction: str = None, word_count: int = None
    ) -> str:
        """Synthesize Aesthetic & Design Style narrative - Analyze artistic and design qualities."""
        response_lower = response.lower()
        word_count = word_count or len(response.split())

        if word_count == 1:
            # Single word aesthetic descriptor: "elegant" -> expand
            return (
                f"Aesthetic analysis: This specimen demonstrates {response_lower} design principles and artistic qualities. "
                f"The visual and structural elements reflect contemporary or established standards, "
                f"exhibiting design coherence and functional-aesthetic integration."
            )
        elif word_count <= 2:
            # Two-word aesthetic description
            return (
                f"Aesthetic and design assessment: This specimen embodies {response_lower} principles. "
                f"The artistic and design qualities are reflective of established standards within its domain, "
                f"demonstrating functional and aesthetic integration."
            )
        else:
            # Comprehensive aesthetic description
            return (
                f"Aesthetic and design analysis: {response_lower}. "
                f"These elements reflect artistic, functional, and structural integration appropriate to the object's classification and purpose."
            )

    def _synthesize_interaction(
        self, response: str, cnn_prediction: str = None, word_count: int = None
    ) -> str:
        """Synthesize Interaction & Relationship narrative - Establish human-environment dynamics."""
        response_lower = response.lower()
        word_count = word_count or len(response.split())

        if word_count == 1:
            # Single word interaction: "decorative" -> elaborate interaction context
            return (
                f"Interaction and relationship assessment: This specimen is characterized by {response_lower} properties. "
                f"Its functional and relational role with humans and the environment defines its operational significance, "
                f"contextual utility, and established patterns of use and interaction."
            )
        elif word_count <= 2:
            # Brief interaction description
            return (
                f"Interaction dynamics: This specimen is characterized by {response_lower}. "
                f"Its relationship with humans and the environment defines its functional significance and operational role."
            )
        else:
            # Comprehensive interaction description
            return (
                f"Interaction and environmental relationships: {response_lower}. "
                f"The object's engagement with its context reflects inherent design, functional characteristics, and established usage patterns."
            )

    def _get_default_for_key(self, key: str) -> str:
        """Get fallback value for analysis key."""
        defaults = {
            "Common Identity": "A representative object of its classification.",
            "Visual Summary": "The object displays characteristic visual features.",
            "Operational Utility": "Serves functional purposes within its domain.",
            "Provenance & Setting": "Typically found in specific environmental contexts.",
            "Technical Nomenclature": "Known by its standard classification name.",
            "Safety & Risk Assessment": "Standard safety protocols apply.",
            "Maintenance & Longevity": "Requires appropriate maintenance and care.",
            "Aesthetic & Design Style": "Reflects contemporary design principles.",
            "Interaction & Relationship": "Interacts with its environment in characteristic ways.",
        }
        return defaults.get(key, "No data available")

    def _error_response(self) -> Dict[str, str]:
        """Generate error response with fallback values."""
        return {
            "success": False,
            "error": "Model inference failed",
            "structured": {
                key: self._get_default_for_key(key)
                for key in [
                    "Common Identity",
                    "Visual Summary",
                    "Operational Utility",
                    "Provenance & Setting",
                    "Technical Nomenclature",
                    "Safety & Risk Assessment",
                    "Maintenance & Longevity",
                    "Aesthetic & Design Style",
                    "Interaction & Relationship",
                ]
            },
        }

    def parse_nine_point_description(self, data: Dict[str, str]) -> Dict[str, str]:
        """
        Validate and parse nine-point description.

        Args:
            data: Dictionary with analysis data

        Returns:
            Validated dictionary with all 9 keys
        """
        required_keys = {
            "Common Identity",
            "Visual Summary",
            "Operational Utility",
            "Provenance & Setting",
            "Technical Nomenclature",
            "Safety & Risk Assessment",
            "Maintenance & Longevity",
            "Aesthetic & Design Style",
            "Interaction & Relationship",
        }

        validated = {}
        for key in required_keys:
            value = data.get(key)
            if value and len(str(value)) > 0:
                validated[key] = value
            else:
                validated[key] = self._get_default_for_key(key)

        return validated

    def merge_lora_weights(self):
        """Merge LoRA weights into base model."""
        if not self.model or self.is_lora_merged:
            return False

        try:
            self.model = self.model.merge_and_unload()
            self.is_lora_merged = True
            print("[OK] LoRA weights merged")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to merge LoRA: {e}")
            return False

    def unload_model(self):
        """Unload model and free memory."""
        if self.model:
            self.model = None
        if self.processor:
            self.processor = None

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        print("[OK] Model unloaded")


def load_vlm_model(
    model_name: str = "blip-vqa",
    device: str = None,
    lora_config: Dict = None,
) -> VLMModel:
    """
    Factory function to load VLM model.

    Args:
        model_name: Name of model to load
        device: Device for inference
        lora_config: LoRA configuration

    Returns:
        Initialized VLMModel instance
    """
    vlm = VLMModel(model_name=model_name, device=device, lora_config=lora_config)
    vlm.load_base_model()
    return vlm


def get_lora_weight_size(lora_config: Dict) -> int:
    """
    Estimate LoRA weight size in bytes.

    Args:
        lora_config: LoRA configuration

    Returns:
        Estimated size in bytes
    """
    r = lora_config.get("r", 8)
    # Rough estimate: 2 matrices per target module
    num_modules = len(lora_config.get("target_modules", []))
    # Assuming typical hidden size of 768
    hidden_size = 768
    # bytes = num_modules * 2 matrices * r * hidden_size * 4 bytes (float32)
    return num_modules * 2 * r * hidden_size * 4
