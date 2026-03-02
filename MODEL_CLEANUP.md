# Model Files Cleanup & Archive Plan

## Summary

This document identifies redundant and legacy model files that can be safely archived or removed after the Final Production Refactor to Nuclear Truth Protocol with ImageNet-1K MobileNetV2.

## Key Finding: No Active .h5 Files Needed

**The new production system does NOT require stored .h5 model files:**

- **MobileNetV2 (ImageNet-1K)**: Loaded directly from Keras/TensorFlow Hub at runtime
- **BLIP-VQA Model**: Loaded from HuggingFace transformers at runtime
- **No stored weights needed for inference**

## Legacy Model Files (Can Be Archived)

### CIFAR-100 Trained Models (LEGACY - DO NOT USE)

These files trained on CIFAR-100 (100 classes) and are no longer used in the production system:

1. **cifar100_mobilenetv2.h5** (~200 MB)
   - Status: LEGACY
   - Training Dataset: CIFAR-100
   - Architecture: MobileNetV2
   - Action: **ARCHIVE**
   - Reason: System now uses ImageNet-1K via Keras Hub

2. **cifar100_mobilenetv2_best.h5** (~200 MB)
   - Status: LEGACY
   - Training Dataset: CIFAR-100
   - Architecture: MobileNetV2
   - Action: **ARCHIVE**
   - Reason: Superseded by ImageNet-1K MobileNetV2

### Federated Learning Training Artifacts (LEGACY)

These files are from previous federated learning training rounds and are historical/experimental:

3. **global_model.h5** (~200 MB)
   - Status: LEGACY
   - Purpose: Initial federated learning model
   - Rounds Completed: Unknown
   - Action: **ARCHIVE**
   - Reason: FL training now uses direct Keras models; these are historical records

4. **global_model_round_1.h5 through global_model_round_20.h5** (20 files, ~4 GB total)
   - Status: LEGACY
   - Purpose: FL training checkpoints across 20 rounds
   - Accuracy Progression: Tracked in model_history.json
   - Action: **ARCHIVE** (keep only model_history.json for metadata)
   - Reason: These are historical training artifacts; production uses live inference

5. **custom_model_best.h5** (~200 MB)
   - Status: LEGACY
   - Purpose: Experimental custom model
   - Action: **ARCHIVE**
   - Reason: Not part of production pipeline

6. **custom_model_round1.h5, custom_model_round2.h5** (2 files, ~400 MB)
   - Status: LEGACY
   - Purpose: Experimental training rounds
   - Action: **ARCHIVE**
   - Reason: Experimental only; not production

### Total Legacy Model Files to Archive

- **Count**: ~25 .h5 files
- **Total Size**: ~5.5 GB
- **Action**: Archive to `/archive/legacy_models/` or cloud storage
- **Retention**: Recommended for historical analysis only

## Required Files (DO NOT DELETE)

### 1. Metadata & Configuration

- **model_history.json** - KEEP
  - Contains training accuracy/loss history
  - Used for admin dashboard metrics
  - Size: ~5 KB
  - Critical for: Historical performance tracking

- **model_round_*_metadata.json** (multiple files) - KEEP
  - Contains round-specific metadata
  - Used for federated learning analysis
  - Size: ~50 KB total
  - Critical for: FL round tracking and analysis

- **.gitkeep** - KEEP
  - Git repository marker
  - Ensures models/ directory is tracked

### 2. Runtime Models (Loaded on Demand)

**These are NOT stored as .h5 files but loaded at runtime:**

- **MobileNetV2 ImageNet-1K**
  - Source: tensorflow.keras.applications.MobileNetV2
  - Loading: `tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet')`
  - Size: ~100 MB (cached by Keras at runtime)
  - Action: No file storage needed

- **BLIP-VQA Model**
  - Source: Salesforce/blip-vqa-base from HuggingFace
  - Loading: From transformers library
  - Size: ~350 MB (cached by transformers at runtime)
  - Action: No file storage needed

## Recommended Archive Strategy

### Option 1: Local Archive (For Repository Cleanup)

```bash
# Create archive directory
mkdir -p archive/legacy_models
mkdir -p archive/fl_training_history

# Move legacy files
move models/cifar100_mobilenetv2*.h5 archive/legacy_models/
move models/custom_model*.h5 archive/legacy_models/
move models/global_model*.h5 archive/legacy_models/

# Keep only metadata
# models/model_history.json (STAY)
# models/model_round_*_metadata.json (STAY)
```

### Option 2: Cloud Archive (For Long-term Storage)

```bash
# Compress legacy models
tar -czf legacy_models_YYYYMMDD.tar.gz archive/legacy_models/

# Upload to cloud storage (S3, GCS, Azure Blob, etc.)
aws s3 cp legacy_models_YYYYMMDD.tar.gz s3://your-bucket/archives/
```

### Option 3: Delete (If Confident in Production System)

```bash
# Only after confirming production inference works correctly
rm models/cifar100_mobilenetv2*.h5
rm models/custom_model*.h5
rm models/global_model_round_*.h5
```

## Post-Refactor Model Loading Flow

### Production Inference Path (New)

```
1. User uploads image
   ↓
2. inference.py: predict_hybrid(image_path)
   ↓
3. Stage 1: Load MobileNetV2 from Keras
   tf.keras.applications.MobileNetV2(weights='imagenet')
   ↓
4. Stage 1.5: Nuclear Truth Protocol SCL
   Check confidence threshold
   If confidence < 50% → Mandatory VLM discovery
   ↓
5. Stage 2/3: Load BLIP-VQA from HuggingFace
   from transformers import AutoModelForVQA
   ↓
6. Return results (no saved model files needed)
```

**Key**: Models are loaded from official sources (Keras Hub, HuggingFace) at runtime.

## Data Directory Status

### Location: `/data/train/`

- **Content**: ImageNet training data (organized by class folder)
- **Purpose**: Reference/development only (NOT used in production inference)
- **Action**: Can be removed if disk space is needed
- **Reason**: Production inference doesn't require local training data

### Location: `/data/val/`

- **Content**: ImageNet validation data (if exists)
- **Purpose**: Historical validation/testing
- **Action**: Can be removed if disk space is needed
- **Reason**: Keras models come pre-trained; no local validation needed

## Impact Analysis

### Removing Legacy .h5 Files

**What breaks**: Nothing
- No production code reads these files
- inference.py loads models from Keras/HuggingFace
- app.py calls inference.py methods

**What improves**:
- ~5.5 GB freed in /models/ directory
- Faster git operations (fewer large files)
- Cleaner repository structure

## Implementation Checklist

- [ ] Review this document with team
- [ ] Confirm production inference works without .h5 files
- [ ] Choose archive strategy (local/cloud/delete)
- [ ] Create archive directory structure
- [ ] Move/compress legacy model files
- [ ] Update .gitignore to exclude archive/
- [ ] Verify inference still works
- [ ] Test production prediction on sample images
- [ ] Document any changes in version control
- [ ] Update deployment documentation

## Testing After Cleanup

```bash
# 1. Test standard inference
python -c "
from frontend_web.inference import get_classifier
clf = get_classifier(mode='standard')
result = clf.predict('test_image.jpg')
print('Standard inference:', result['predicted_class'])
"

# 2. Test ensemble (hybrid) inference
python -c "
from frontend_web.inference import get_classifier
clf = get_classifier(mode='ensemble')
result = clf.predict_hybrid('test_image.jpg')
print('Ensemble inference:', result['predicted_class'])
"

# 3. Verify no .h5 loading errors in app.py
python run_web.py
# Navigate to http://localhost:5000/predict
# Upload test image
# Verify prediction completes successfully
```

## Archive Manifest

Use this template to document what was archived:

```
Archive Date: YYYY-MM-DD
Archived By: [Name]
Method: [local|cloud|delete]
Files Count: 25
Total Size: 5.5 GB

Legacy Model Files:
- cifar100_mobilenetv2.h5 (200 MB) - CIFAR-100 model, removed
- cifar100_mobilenetv2_best.h5 (200 MB) - CIFAR-100 best, removed
- custom_model_best.h5 (200 MB) - Custom experimental, removed
- custom_model_round1.h5, round2.h5 (400 MB) - Custom training, removed
- global_model.h5 (200 MB) - FL initial model, removed
- global_model_round_1-20.h5 (4 GB) - FL training checkpoints, removed

Retained Files:
- model_history.json (5 KB) - FL training history
- model_round_*_metadata.json (50 KB) - FL metadata
- .gitkeep - Repository marker

Verification: ✓ Production inference tested and working
```

---

**Docu
