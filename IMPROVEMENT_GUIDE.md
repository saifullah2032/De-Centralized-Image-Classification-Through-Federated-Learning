# Model Improvement Guide

## Current Status: 53.86% Accuracy After 10 Rounds

### Analysis of Training Results

#### Accuracy Progression:
```
Round 1-2:   10.00%  - Random baseline
Round 3:     33.56%  - Breakthrough (+23.56%)
Round 4:     38.24%  - Steady improvement (+4.68%)
Round 5:     43.22%  - Continuing (+4.98%)
Round 6:     44.44%  - Slowing (+1.22%)
Round 7:     47.51%  - Recovery (+3.07%)
Round 8:     52.19%  - Good progress (+4.68%)
Round 9:     50.91%  - Slight regression (-1.28%)
Round 10:    53.86%  - Final improvement (+2.95%)
```

#### Key Observations:
- **Improvement rate: 4.39% per round** (average)
- **Estimated rounds to 85%: ~17 total** (need ~7 more rounds)
- **Plateauing**: Accuracy growth is slowing down
- **Model capacity**: Current model (871K params) may be limited

---

## Improvements Implemented

### 1. ✅ Enhanced Model Architecture

**File: `backend_fl/model_enhanced.py`**

#### New Architectures Available:
- **Enhanced MobileNetV2** (default) - Larger capacity with alpha=1.0
  - Parameters: ~2.3M (vs 871K baseline)
  - Deeper classification head (3 dense layers vs 2)
  - Better regularization (L2 + BatchNorm)
  
- **Custom CNN** - Optimized for CIFAR-10
  - Parameters: ~1.5M
  - 3 convolutional blocks
  - Designed specifically for 32x32 images
  
- **ResNet50V2** - High capacity (for best accuracy)
  - Parameters: ~23M
  - Deep residual network
  
- **EfficientNetB0** - Balanced performance
  - Parameters: ~4M
  - Compound scaling

#### Improvements in Enhanced MobileNetV2:
```python
# OLD (871K params, alpha=0.5):
- GlobalAveragePooling2D
- Dropout(0.2)
- Dense(128, relu)
- BatchNormalization
- Dropout(0.5)
- Dense(10, softmax)

# NEW (2.3M params, alpha=1.0):
- GlobalAveragePooling2D
- BatchNormalization
- Dropout(0.3)
- Dense(256, relu) + L2 regularization
- BatchNormalization
- Dropout(0.4)
- Dense(128, relu) + L2 regularization
- BatchNormalization
- Dropout(0.3)
- Dense(10, softmax)
```

### 2. ✅ Data Augmentation Module

**File: `backend_fl/data_augmentation.py`**

#### Standard Augmentation (Keras layers):
- Random horizontal flip
- Random rotation (±15°)
- Random zoom (±10%)
- Random translation (±10%)
- Random brightness (±20%)
- Random contrast (±20%)

#### Advanced Techniques:
- **MixUp**: Linearly interpolate between training examples
- **CutMix**: Cut and paste patches between images

### 3. ✅ Optimized Hyperparameters

**File: `backend_fl/config.py`**

#### Changes:
```python
# OLD:
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_ALPHA = 0.5

# NEW:
LOCAL_EPOCHS = 5          # More training per round
BATCH_SIZE = 64           # Better GPU utilization
LEARNING_RATE = 0.0005    # More stable convergence
MODEL_ALPHA = 1.0         # Full model capacity
```

### 4. ✅ Updated Model Implementation

**File: `backend_fl/model.py`**

The default `get_model()` now uses the enhanced architecture:
- Larger capacity (alpha=1.0)
- Better regularization
- More layers in classification head

---

## How to Use Improvements

### Option 1: Run 20-Round Training (Recommended)

**Expected Results:**
- Round 17-20: ~65-75% accuracy
- Training time: ~3-4 hours
- Best for reaching 85% target

```bash
# Terminal 1: Server
python run_server.py --num-rounds 20 --min-clients 2

# Terminal 2: Client 0
python run_client.py --client-id 0 --num-clients 2

# Terminal 3: Client 1
python run_client.py --client-id 1 --num-clients 2
```

### Option 2: Continue from Round 10 (Resume Training)

The system automatically loads the last checkpoint (`models/global_model.h5`), so just start training again and it will improve from 53.86%:

```bash
# Same commands as Option 1 - will continue from current model
python run_server.py --num-rounds 10 --min-clients 2
```

### Option 3: Fresh Training with Enhanced Model

Delete old checkpoints and start from scratch with the new architecture:

```bash
# Backup old models
mkdir models_backup
mv models/*.h5 models_backup/

# Start fresh 20-round training
python run_server.py --num-rounds 20 --min-clients 2
```

### Option 4: Try Different Architectures

To use specific architectures, modify `backend_fl/model.py`:

```python
# For Custom CNN (fast training, good accuracy):
from backend_fl.model_enhanced import get_model as get_enhanced_model

def get_model(pretrained=None, alpha=None):
    return get_enhanced_model(architecture="custom_cnn", pretrained=False)

# For ResNet50 (highest accuracy, slower):
def get_model(pretrained=None, alpha=None):
    return get_enhanced_model(architecture="resnet50", pretrained=False)
```

---

## Expected Improvements

### Baseline vs Enhanced Model:

| Metric | Baseline (10 rounds) | Enhanced (20 rounds) |
|--------|---------------------|---------------------|
| **Model Size** | 871K params | 2.3M params |
| **Accuracy (R10)** | 53.86% | ~65-70% |
| **Accuracy (R20)** | N/A | ~75-85% |
| **Loss (R10)** | 1.915 | ~1.2-1.5 |
| **Training Time/Round** | ~10 min | ~12-15 min |

### Why These Improvements Work:

1. **Larger Model Capacity (2.6x more params)**
   - Can learn more complex patterns
   - Better feature extraction
   - Reduced underfitting

2. **Better Hyperparameters**
   - More local epochs → Better convergence
   - Lower learning rate → Stable training
   - Larger batch size → Better gradient estimates

3. **Data Augmentation**
   - Prevents overfitting
   - Improves generalization
   - Effective data size increases

4. **Enhanced Regularization**
   - L2 weight decay
   - Multiple BatchNorm layers
   - Progressive dropout

---

## Quick Comparison: All Architectures

| Architecture | Params | Expected Acc (20R) | Training Speed | Use Case |
|--------------|--------|-------------------|----------------|----------|
| **Baseline MobileNet** | 871K | ~55-60% | Fast | Edge devices |
| **Enhanced MobileNet** | 2.3M | ~75-85% | Medium | Balanced |
| **Custom CNN** | 1.5M | ~70-80% | Fast | CIFAR-10 optimized |
| **EfficientNetB0** | 4M | ~80-88% | Medium | High accuracy |
| **ResNet50V2** | 23M | ~85-92% | Slow | Best accuracy |

---

## Testing the Improvements

After training completes, test the improved model:

```bash
# 1. Generate visualizations
python visualize_training.py

# 2. Start web interface
python run_web.py

# 3. Visit http://localhost:5000
# Login: admin / admin123
# Upload test images and see improved predictions!
```

---

## Monitoring Training

Watch for these indicators of successful training:

### ✅ Good Signs:
- Steady accuracy increase (>3% per round)
- Loss decreasing consistently
- Validation accuracy close to training accuracy
- No sudden drops in performance

### ⚠️ Warning Signs:
- Accuracy plateauing early (<60% by round 15)
- Loss increasing or oscillating
- Large gap between train/validation accuracy (overfitting)
- Clients timing out or disconnecting

---

## Troubleshooting

### Issue: Accuracy still plateaus around 60-70%

**Solutions:**
1. Train for 30 rounds instead of 20
2. Reduce learning rate: `LEARNING_RATE=0.0002`
3. Try ResNet50V2 architecture
4. Increase to 3-4 clients (more diverse data)

### Issue: Training too slow

**Solutions:**
1. Use Custom CNN (faster than MobileNet)
2. Reduce batch size to 32
3. Reduce local epochs to 3
4. Train with fewer rounds (15 instead of 20)

### Issue: Out of memory errors

**Solutions:**
1. Reduce batch size: `BATCH_SIZE=32`
2. Use Enhanced MobileNet instead of ResNet50
3. Close other applications
4. Reduce model alpha: `MODEL_ALPHA=0.75`

---

## Next Steps

1. ✅ **Immediate**: Run 20-round training with enhanced model
2. ⏳ **After 20 rounds**: Evaluate results and visualize improvements
3. ⏳ **If needed**: Train additional 10 rounds to reach 85% target
4. ⏳ **Advanced**: Try ensemble of multiple architectures
5. ⏳ **Production**: Deploy best model to web interface

---

## Commands Summary

```bash
# Activate environment
.\venv\Scripts\Activate.ps1

# 20-round enhanced training
python run_server.py --num-rounds 20 --min-clients 2   # Terminal 1
python run_client.py --client-id 0 --num-clients 2     # Terminal 2
python run_client.py --client-id 1 --num-clients 2     # Terminal 3

# Visualize results
python visualize_training.py

# Test web interface
python run_web.py
```

---

## Expected Timeline

- **Rounds 11-15** (50 min): 54% → 65% accuracy
- **Rounds 16-20** (50 min): 65% → 75% accuracy
- **Total**: ~1.5-2 hours to reach 75%+ accuracy

**Target: 75-85% accuracy by Round 20** 🎯
