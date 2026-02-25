# Project Pivot: Decentralized Multimodal Visual Assistant with FedLoRA

**Project Status**: Complete Architecture Redesign  
**New Objective**: Build a privacy-preserving, federated multimodal VLM system for structured image description generation

---

## Executive Summary

This document outlines the complete transformation from CNN-based image classification to a **Decentralized Multimodal Visual Assistant** powered by Vision-Language Models (VLM) with Federated Low-Rank Adaptation (FedLoRA).

### Available Model Options (User Can Choose)

Users can now choose between two model types:

| Option | Model Type | Task | Description |
|--------|-----------|------|-------------|
| **VLM (NEW)** | Moondream2/BLIP/PaliGemma | 5-Point Description | Generate structured textual descriptions |
| **CNN (Retained)** | MobileNetV2 (CIFAR-100/Custom) | Image Classification | Standard classification into categories |

### Key Changes
- **New VLM Model**: CNN (MobileNetV2) → Vision-Language Model (Moondream2/BLIP/PaliGemma)
- **New Task**: Image Classification → Structured 5-Point Textual Description
- **New Training**: Standard Federated Learning → FedLoRA (LoRA adapters only)
- **Retained**: CIFAR-100 and Custom trained models remain available for classification
- **Dataset for VLM**: CIFAR-100 → Custom JSON/Image pairs with structured captions

---

## Phase 1: Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEDERATED MULTIMODAL SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                   │
│  │   Client 0  │   │   Client 1  │   │   Client N  │                   │
│  │              │   │              │   │              │               │
│  │ ┌──────────┐ │   │ ┌──────────┐ │   │ ┌──────────┐ │               │
│  │ │  Images   │ │   │ │  Images   │ │   │ │  Images   │ │               │
│  │ │  + JSON   │ │   │ │  + JSON   │ │   │ │  + JSON   │ │               │
│  │ └──────────┘ │   │ └──────────┘ │   │ └──────────┘ │               │
│  │              │   │              │   │              │               │
│  │ ┌──────────┐ │   │ ┌──────────┐ │   │ ┌──────────┐ │               │
│  │ │ Base VLM │ │   │ │ Base VLM │ │   │ │ Base VLM │ │               │
│  │ │ (Frozen) │ │   │ │ (Frozen) │ │   │ │ (Frozen) │ │               │
│  │ └──────────┘ │   │ └──────────┘ │   │ └──────────┘ │               │
│  │              │   │              │   │              │               │
│  │ ┌──────────┐ │   │ ┌──────────┐ │   │ ┌──────────┐ │               │
│  │ │ LoRA     │ │   │ │ LoRA     │ │   │ │ LoRA     │ │               │
│  │ │ Adapter  │ │   │ │ Adapter  │ │   │ │ Adapter  │ │               │
│  │ │ (Train)  │ │   │ │ (Train)  │ │   │ │ (Train)  │ │               │
│  │ └──────────┘ │   │ └──────────┘ │   │ └──────────┘ │               │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘               │
│         │                  │                  │                         │
│         └──────────────────┼──────────────────┘                         │
│                            │                                            │
│                    ┌───────▼───────┐                                     │
│                    │   Flower     │                                     │
│                    │   Server     │                                     │
│                    │              │                                     │
│                    │ ┌──────────┐ │                                     │
│                    │ │ FedLoRA  │ │                                     │
│                    │ │ Aggregate│ │                                     │
│                    │ └──────────┘ │                                     │
│                    └───────┬──────┘                                     │
│                            │                                            │
│                    ┌───────▼──────┐                                     │
│                    │    Flask     │                                     │
│                    │  Web UI      │                                     │
│                    │              │                                     │
│                    │ /predict     │                                     │
│                    │ (VLM + 5pt)  │                                     │
│                    └──────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow

```
Image Upload → Hidden Prompt Injection → VLM Inference → 5-Point Description

Hidden Prompt Template:
"Analyze this image and provide a structured description with exactly 5 points:
1. Class: [main object category]
2. What it is: [detailed description]
3. Purpose: [function/use]
4. Origin: [setting/context]
5. Name: [specific name if applicable]"
```

---

## Phase 2: Model Architecture

### 2.1 VLM Selection

| Model | Size | Pros | Cons |
|-------|------|------|------|
| **Moondream2** | ~1.7B | Lightweight, fast inference, good captioning | Limited complex reasoning |
| **BLIP-2** | ~2.3B | Strong vision-language alignment | Larger, slower |
| **PaliGemma** | ~3B | Google quality, flexible | Heavy for edge devices |

**Recommended**: Moondream2 for federated deployment (smaller LoRA footprint)

### 2.2 LoRA Configuration

```python
LORA_CONFIG = {
    "r": 8,                      # Rank (low-rank adaptation)
    "lora_alpha": 16,            # LoRA scaling factor
    "lora_dropout": 0.05,        # Dropout for LoRA layers
    "target_modules": [          # Modules to apply LoRA
        "q_proj", "k_proj", 
        "v_proj", "o_proj"
    ],
    "bias": "none",              # No bias training
    "task_type": "CAUSAL_LM"     # Language generation task
}
```

### 2.3 FedLoRA Algorithm

```
FedLoRA Round t:
1. Server broadcasts global LoRA weights W_loRA(t)
2. Each client k:
   a. Loads base VLM (frozen)
   b. Loads/adapts LoRA adapter with W_loRA(t)
   c. Trains on local (image, 5-point JSON) pairs
   d. Computes gradient updates ΔW_loRA(k)
   e. Sends only ΔW_loRA(k) to server (NOT base model)
3. Server aggregates: W_loRA(t+1) = Σ(n_k/n) × (W_loRA(t) + ΔW_loRA(k))
4. Repeat for T rounds
```

**Key Advantage**: Only ~10-50 MB LoRA weights transmitted (vs ~7GB full model)

---

## Phase 3: Dataset Specification

### 3.1 Custom JSON/Image Format

```json
{
  "image_id": "img_001",
  "image_path": "client_data/images/img_001.jpg",
  "caption": {
    "1. Class": "domestic animal",
    "2. What it is": "A golden retriever dog sitting in a grassy field",
    "3. Purpose": "Companion animal, often used as service dog",
    "4. Origin": "Originally bred in Scotland for retrieving game",
    "5. Name": "Golden Retriever"
  }
}
```

### 3.2 Data Partitioning

- **Non-IID Distribution**: Dirichlet (α=0.5) for heterogeneous client data
- **Local Split**: 80% train, 20% test per client
- **Minimum Samples**: 50 images per client for meaningful training

---

## Phase 4: Implementation Plan

### 4.1 Directory Structure

```
Image Classification/
├── backend_fl/
│   ├── __init__.py
│   ├── config.py                 # Updated for VLM config
│   ├── vlm_model.py              # NEW: VLM + LoRA wrapper
│   ├── fl_client.py              # UPDATED: FedLoRA client
│   ├── fl_server.py              # UPDATED: FedLoRA server
│   ├── strategies.py              # UPDATED: FedLoRA strategy
│   ├── data_utils.py             # UPDATED: JSON/image loader
│   └── vlm_dataset.py            # NEW: VLM dataset class
│
├── frontend_web/
│   ├── app.py                    # UPDATED: VLM inference endpoint
│   ├── inference.py              # UPDATED: VLM inference logic
│   ├── templates/
│   │   ├── index.html            # UPDATED: Model selection UI
│   │   └── predict.html          # UPDATED: 5-point description display
│   └── static/
│       └── css/
│           └── style.css         # UPDATED: Beautiful description styling
│
├── models/
│   ├── global_lora_weights/      # NEW: LoRA adapter storage
│   │   ├── round_1/
│   │   ├── round_2/
│   │   └── ...
│   ├── global_model.h5           # RETAINED: Trained CNN model (optional)
│   └── cifar100_mobilenetv2_best.h5  # RETAINED: CIFAR-100 model
│
├── data/
│   └── sample_vlm_data/          # NEW: Sample JSON/image data
│       └── annotations.json
│
├── run_server.py                 # UPDATED: Start FedLoRA server
├── run_client.py                 # UPDATED: Start FedLoRA client
├── run_web.py                    # UPDATED: Start Flask web
├── Plan.md                       # This document
└── requirements.txt              # UPDATED: Add VLM dependencies
```

### 4.2 Core Components

#### 4.2.1 VLM Model Wrapper (`vlm_model.py`)

```python
# Core Functions:
- load_vlm_model(model_name="moondream2")     # Load base VLM
- load_lora_adapter(adapter_path)             # Load LoRA weights
- save_lora_adapter(adapter_path)             # Save LoRA weights
- get_trainable_parameters()                   # Extract LoRA params only
- merge_lora_to_base()                        # Optional: merge for inference
- generate_description(image, prompt)          # Generate 5-point description
```

#### 4.2.2 FedLoRA Client (`fl_client.py`)

```python
# Core Functions:
- VLMClient.get_parameters()                  # Get LoRA weights only
- VLMClient.fit(parameters, config)          # Train LoRA locally
- VLMClient.evaluate(parameters, config)     # Evaluate locally
- start_client()                             # Main entry point
```

#### 4.2.3 FedLoRA Server (`fl_server.py`)

```python
# Core Functions:
- FedLoRAStrategy.aggregate_fit()            # Aggregate LoRA weights
- start_server()                             # Main entry point
```

### 4.3 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Upload image, get 5-point description |
| `/status` | GET | Check model/VLoRA status |
| `/model-selection` | GET/POST | Choose VLM or trained CNN model |

---

## Phase 5: User Interface

### 5.1 Model Selection Screen

Users can now choose between:
1. **VLM Model (New)**: Multimodal Visual Assistant with 5-point descriptions
2. **Trained CNN Model**: Existing CIFAR-100/custom models

### 5.2 Description Display

```
┌─────────────────────────────────────────────────────────────────┐
│                    UPLOADED IMAGE                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                                                         │  │
│  │                    [Image Preview]                      │  │
│  │                                                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                      5-POINT DESCRIPTION                        │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  1. CLASS                                                        │
│    └─► Domestic Animal                                          │
│                                                                 │
│  2. WHAT IT IS                                                  │
│    └─► A golden retriever dog sitting in a grassy field        │
│                                                                 │
│  3. PURPOSE                                                     │
│    └─► Companion animal, often used as service dog             │
│                                                                 │
│  4. ORIGIN                                                      │
│    └─► Originally bred in Scotland for retrieving game          │
│                                                                 │
│  5. NAME                                                        │
│    └─► Golden Retriever                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 6: Dependencies

### 6.1 Updated Requirements

```txt
# Federated Learning
flwr>=1.7.0

# VLM & LoRA
transformers>=4.36.0
peft>=0.8.0
accelerate>=0.25.0
torch>=2.1.0

# Image Processing
Pillow>=10.0.0
torchvision>=0.16.0

# Web Framework
Flask>=3.0.0
Flask-CORS>=4.0.0

# Utilities
python-dotenv>=1.0.0
numpy>=1.24.0

# Optional: For Moondream2
einops>=0.7.0
```

---

## Phase 7: Migration Guide

### 7.1 What Was Removed

- ❌ MobileNetV2 architecture
- ❌ Custom CNN model
- ❌ CIFAR-10/CIFAR-100 classification head
- ❌ Standard FedAvg (full model weights)

### 7.2 What Was Added

- ✅ VLM base model (Moondream2/BLIP/PaliGemma)
- ✅ LoRA adapters (trainable parameters only)
- ✅ FedLoRA aggregation strategy
- ✅ JSON/image dataset loader
- ✅ 5-point structured description generation

### 7.3 What Was Retained

- ✅ Flower (flwr) for federated orchestration
- ✅ Flask web interface
- ✅ Non-IID data partitioning
- ✅ Model selection option (VLM vs. trained CNN)

---

## Phase 8: Implementation Priorities

### Priority 1: Core Infrastructure
1. [ ] Update `requirements.txt` with VLM dependencies
2. [ ] Create `vlm_model.py` with VLM + LoRA wrapper
3. [ ] Update `fl_client.py` for FedLoRA
4. [ ] Update `fl_server.py` for LoRA aggregation

### Priority 2: Web Interface
5. [ ] Update Flask endpoints for VLM inference
6. [ ] Add model selection (VLM vs. trained CNN)
7. [ ] Style 5-point description display

### Priority 3: Data Pipeline
8. [ ] Create `vlm_dataset.py` for JSON/image loading
9. [ ] Prepare sample data for testing

### Priority 4: Testing
10. [ ] Test VLM inference locally
11. [ ] Test FedLoRA training simulation
12. [ ] Verify LoRA weight aggregation

---

## Appendix A: LoRA Weight Format

```python
# LoRA weights are stored as state_dict
lora_weights = {
    "lora_A.weight": np.array,  # [r, hidden_dim]
    "lora_B.weight": np.array,  # [hidden_dim, r]
    # ... for each target module
}

# Total size: ~10-50 MB depending on rank and model size
```

## Appendix B: Privacy Guarantee

**What is transmitted:**
- ✅ LoRA adapter weights (10-50 MB)
- ✅ Training metrics (loss, sample count)
- ❌ Raw images
- ❌ Image captions/descriptions
- ❌ Base VLM parameters (frozen)

---

**Document Version**: 2.0 (Project Pivot)  
**Last Updated**: 2026-02-25  
**Status**: Ready for Implementation
