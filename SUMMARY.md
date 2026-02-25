# Decentralized Multimodal Visual Assistant - Project Summary

## Project Overview

**Project Name:** DecentralizedAI - Federated Multimodal Visual Assistant  
**Type:** Privacy-Preserving Distributed Machine Learning System with Vision-Language Model  
**Status:** Production-Ready | VLM + CIFAR-100 Classification

---

## What Does This System Do?

This system now supports **two modes**:

### 1. VLM Mode (New - Multimodal Visual Assistant)
- Takes any uploaded image
- Generates a **structured 5-point description**:
  1. **Class** - Main object category
  2. **What it is** - Detailed description  
  3. **Purpose** - Function/use
  4. **Origin** - Setting/context
  5. **Name** - Specific name/type
- Uses lightweight BLIP model (`Salesforce/blip-image-captioning-base`)
- Works on 8GB RAM systems
- Supports FedLoRA training (federated LoRA adapters)

### 2. CNN Mode (Classic - Image Classification)
- Classifies images into **CIFAR-100 categories** (100 classes)
- Uses trained MobileNetV2 model
- Returns predicted class with confidence score

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              FEDERATED MULTIMODAL SYSTEM                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │   Client 0  │   │   Client 1  │   │   Client N  │      │
│  │              │   │              │   │              │      │
│  │ Base VLM    │   │ Base VLM    │   │ Base VLM    │      │
│  │ (Frozen)    │   │ (Frozen)    │   │ (Frozen)    │      │
│  │              │   │              │   │              │      │
│  │ LoRA        │   │ LoRA        │   │ LoRA        │      │
│  │ Adapter     │   │ Adapter     │   │ Adapter     │      │
│  │ (Trainable) │   │ (Trainable) │   │ (Trainable) │      │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘      │
│         │                  │                  │                │
│         └──────────────────┼──────────────────┘                │
│                            │                                 │
│                    ┌───────▼───────┐                         │
│                    │   Flower     │                         │
│                    │   Server     │                         │
│                    │  (FedLoRA)   │                         │
│                    └───────┬──────┘                         │
│                            │                                 │
│                    ┌───────▼──────┐                         │
│                    │    Flask     │                         │
│                    │  Web UI      │                         │
│                    │  /predict    │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Options (User Selectable)

| Model | Type | Output | Use Case |
|-------|------|--------|----------|
| **CIFAR-100** | CNN | 100-class classification | Standard image classification |
| **VLM** | BLIP | 5-point descriptions | Multimodal visual assistant |

---

## Key Features

✅ **Privacy-Preserving** - Raw data stays on devices  
✅ **FedLoRA** - Only LoRA adapter weights transmitted (~10-50MB vs 7GB full model)  
✅ **Dual Mode** - Choose between classification or VLM descriptions  
✅ **Lightweight** - Works on 8GB RAM laptops  
✅ **Web Interface** - Easy image upload and prediction  
✅ **Federated Training** - Train across multiple clients  

---

## Technologies

| Component | Technology |
|-----------|------------|
| VLM | Salesforce/blip-image-captioning-base |
| LoRA | PEFT (Parameter-Efficient Fine-Tuning) |
| FL Framework | Flower (flwr) |
| Web | Flask |
| ML | PyTorch, Keras |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start web server
python run_web.py

# Access http://localhost:5000
# Login: admin / admin123

# Choose model:
# - CIFAR-100: Standard classification
# - VLM: Generate 5-point descriptions
```

---

## Training (Optional FedLoRA)

```bash
# Terminal 1: Start server
python run_server.py --num-rounds 5

# Terminal 2-3: Start clients
python run_client.py --client-id 0
python run_client.py --client-id 1
```

---

## Performance

- **VLM Inference**: ~5-15 seconds per image (CPU)
- **LoRA Weights**: ~10-50 MB per round
- **Model Size**: BLIP ~400MB base + ~10MB LoRA

---

**Last Updated:** February 2026  
**Version:** 2.0 (Project Pivot)
