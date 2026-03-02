"""
Simplified Moondream2 loader that works with both CPU and CUDA.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_utils import PreTrainedModel


def patch_moondream_loading():
    """Patch Moondream2's missing all_tied_weights_keys attribute."""
    original_getattr = PreTrainedModel.__getattribute__
    
    def patched_getattr(self, name):
        if name == "all_tied_weights_keys":
            if not hasattr(self, "_all_tied_weights_keys"):
                self._all_tied_weights_keys = {}
            return self._all_tied_weights_keys
        return original_getattr(self, name)
    
    PreTrainedModel.__getattribute__ = patched_getattr


def load_moondream2(model_id: str, device: str = "cpu"):
    """Load Moondream2 with optional 4-bit quantization.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load on ('cuda' or 'cpu')
        
    Returns:
        tuple: (model, processor)
    """
    print(f"Loading Moondream2 from {model_id}...")
    
    # Load tokenizer/processor
    processor = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    
    # Patch before loading
    patch_moondream_loading()
    
    # Check if we should use quantization
    use_quantization = torch.cuda.is_available()
    
    if use_quantization:
        print("CUDA available - using 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("OK Model loaded with 4-bit quantization (estimated 1.2GB RAM)")
    else:
        print("No CUDA - loading full-precision on CPU")
        print("(This may be slower but will work on 8GB+ systems)")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        print("OK Model loaded (full precision)")
    
    model.eval()
    return model, processor


if __name__ == "__main__":
    # Test loading
    try:
        model, processor = load_moondream2("vikhyatk/moondream2")
        print("\nSuccess! Model loaded.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
