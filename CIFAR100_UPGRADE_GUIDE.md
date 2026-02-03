# 🚀 CIFAR-100 UPGRADE GUIDE

## ✅ UPGRADE COMPLETE! System Now Supports 100 Classes

---

## 🎯 What Changed?

Your federated learning system has been upgraded from **CIFAR-10 (10 classes)** to **CIFAR-100 (100 classes)**!

### Before (CIFAR-10):
- **10 object categories**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **6,000 images per class**
- **Simple classification**

### After (CIFAR-100):
- **100 fine-grained categories**: orchids, babies, beds, dolphins, mushrooms, and 95 more!
- **20 superclasses**: aquatic_mammals, flowers, people, household_furniture, etc.
- **500 images per class**
- **More sophisticated and realistic**

---

## 📊 CIFAR-100 Dataset Overview

### Dataset Statistics:
| Property | Value |
|----------|-------|
| **Total Images** | 60,000 (same as CIFAR-10) |
| **Training Set** | 50,000 images |
| **Test Set** | 10,000 images |
| **Image Size** | 32×32 pixels (RGB) |
| **Fine Classes** | 100 specific categories |
| **Coarse Classes** | 20 superclass groups |
| **Images per Fine Class** | 500 (vs 6,000 in CIFAR-10) |

### The 20 Superclasses (Coarse Labels):

1. **aquatic_mammals** (5 classes)
   - beaver, dolphin, otter, seal, whale

2. **fish** (5 classes)
   - aquarium_fish, flatfish, ray, shark, trout

3. **flowers** (5 classes)
   - orchid, poppy, rose, sunflower, tulip

4. **food_containers** (5 classes)
   - bottle, bowl, can, cup, plate

5. **fruit_and_vegetables** (5 classes)
   - apple, mushroom, orange, pear, sweet_pepper

6. **household_electrical_devices** (5 classes)
   - clock, keyboard, lamp, telephone, television

7. **household_furniture** (5 classes)
   - bed, chair, couch, table, wardrobe

8. **insects** (5 classes)
   - bee, beetle, butterfly, caterpillar, cockroach

9. **large_carnivores** (5 classes)
   - bear, leopard, lion, tiger, wolf

10. **large_man-made_outdoor_things** (5 classes)
    - bridge, bus, house, road, skyscraper

11. **large_natural_outdoor_scenes** (5 classes)
    - cloud, forest, mountain, plain, sea

12. **large_omnivores_and_herbivores** (5 classes)
    - camel, cattle, chimpanzee, elephant, kangaroo

13. **medium_mammals** (5 classes)
    - fox, possum, raccoon, skunk, squirrel

14. **non-insect_invertebrates** (5 classes)
    - crab, lobster, snail, spider, worm

15. **people** (5 classes)
    - baby, boy, girl, man, woman

16. **reptiles** (5 classes)
    - crocodile, dinosaur, lizard, snake, turtle

17. **small_mammals** (5 classes)
    - hamster, mouse, rabbit, shrew, squirrel

18. **trees** (5 classes)
    - maple_tree, oak_tree, palm_tree, pine_tree, willow_tree

19. **vehicles_1** (5 classes)
    - bicycle, bus, motorcycle, pickup_truck, train

20. **vehicles_2** (5 classes)
    - lawn_mower, rocket, streetcar, tank, tractor

### All 100 Fine Classes (Alphabetical):

```
apple, aquarium_fish, baby, bear, beaver, bed, bee, beetle, bicycle, bottle,
bowl, boy, bridge, bus, butterfly, camel, can, castle, caterpillar, cattle,
chair, chimpanzee, clock, cloud, cockroach, couch, crab, crocodile, cup, dinosaur,
dolphin, elephant, flatfish, forest, fox, girl, hamster, house, kangaroo, keyboard,
lamp, lawn_mower, leopard, lion, lizard, lobster, man, maple_tree, motorcycle, mountain,
mouse, mushroom, oak_tree, orange, orchid, otter, palm_tree, pear, pickup_truck, pine_tree,
plain, plate, poppy, porcupine, possum, rabbit, raccoon, ray, road, rocket,
rose, sea, seal, shark, shrew, skunk, skyscraper, snail, snake, spider,
squirrel, streetcar, sunflower, sweet_pepper, table, tank, telephone, television, tiger, tractor,
train, trout, tulip, turtle, wardrobe, whale, willow_tree, wolf, woman, worm
```

---

## 🔧 Technical Changes Made

### 1. Configuration Updates (`backend_fl/config.py`)

```python
# NEW: Dataset selection
DATASET = "CIFAR100"  # Changed from CIFAR10

# Automatic class count
NUM_CLASSES = 100  # Was 10

# Added all 100 fine labels
CIFAR100_FINE_LABELS = ["apple", "aquarium_fish", ..., "worm"]

# Added 20 superclass labels  
CIFAR100_COARSE_LABELS = ["aquatic_mammals", "fish", ..., "vehicles_2"]

# Added fine-to-coarse mapping
CIFAR100_FINE_TO_COARSE = {0: 4, 1: 1, ...}  # Maps each class to superclass
```

### 2. Data Loading Updates (`backend_fl/data_utils.py`)

```python
# NEW: CIFAR-100 loader
def load_cifar100(normalize=True, label_mode='fine'):
    """Load CIFAR-100 with 100 fine classes"""
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
    # ... preprocessing ...

# NEW: Unified dataset loader
def load_dataset(dataset_name=None):
    """Automatically loads CIFAR-10 or CIFAR-100 based on config"""
    if dataset_name == "CIFAR100":
        return load_cifar100()
    else:
        return load_cifar10()
```

### 3. Model Architecture (No Changes Needed!)

The model automatically adapts:
```python
# Model output layer uses NUM_CLASSES from config
Dense(NUM_CLASSES, activation="softmax")  # Now outputs 100 instead of 10
```

**Model Parameters:**
- Input: 32×32×3 (unchanged)
- Architecture: Enhanced MobileNetV2 (unchanged)
- Output: 100 classes (was 10)
- Total parameters: ~2.3M (unchanged)

### 4. Web Interface Updates (`frontend_web/inference.py`)

```python
# Now supports superclass prediction
result = {
    "predicted_class": "orchid",           # Fine class
    "superclass": "flowers",               # Superclass (NEW!)
    "confidence": 0.87,
    "top_5": [...]                         # Top 5 predictions
}
```

---

## 🎯 What Can It Now Classify?

### Nature (20 classes):
- **Flowers**: orchid, poppy, rose, sunflower, tulip
- **Trees**: maple, oak, palm, pine, willow
- **Scenes**: cloud, forest, mountain, plain, sea

### Animals (35 classes):
- **Aquatic**: beaver, dolphin, otter, seal, whale
- **Fish**: aquarium_fish, flatfish, ray, shark, trout
- **Carnivores**: bear, leopard, lion, tiger, wolf
- **Herbivores**: camel, cattle, chimpanzee, elephant, kangaroo
- **Medium Mammals**: fox, possum, raccoon, skunk, squirrel
- **Small Mammals**: hamster, mouse, rabbit, shrew
- **Insects**: bee, beetle, butterfly, caterpillar, cockroach
- **Reptiles**: crocodile, dinosaur, lizard, snake, turtle
- **Invertebrates**: crab, lobster, snail, spider, worm

### People (5 classes):
- baby, boy, girl, man, woman

### Household (15 classes):
- **Furniture**: bed, chair, couch, table, wardrobe
- **Electronics**: clock, keyboard, lamp, telephone, television
- **Containers**: bottle, bowl, can, cup, plate

### Food (5 classes):
- apple, mushroom, orange, pear, sweet_pepper

### Vehicles (10 classes):
- bicycle, bus, motorcycle, pickup_truck, train
- lawn_mower, rocket, streetcar, tank, tractor

### Man-Made Structures (5 classes):
- bridge, castle, house, road, skyscraper

---

## 🚀 How to Use CIFAR-100

### Quick Start:

```bash
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Train with CIFAR-100 (3 terminals)
python run_server.py --num-rounds 20 --min-clients 2
python run_client.py --client-id 0 --num-clients 2
python run_client.py --client-id 1 --num-clients 2

# 3. Test predictions
python run_web.py
# Visit: http://localhost:5000
```

### Configuration Options:

**To switch back to CIFAR-10:**
```python
# In backend_fl/config.py, change:
DATASET = "CIFAR10"  # Instead of CIFAR100
```

**To use superclasses (20 classes instead of 100):**
```python
# In backend_fl/data_utils.py, modify load_cifar100():
cifar100.load_data(label_mode='coarse')  # Instead of 'fine'
```

---

## 📈 Expected Performance

### CIFAR-10 vs CIFAR-100 Performance:

| Metric | CIFAR-10 | CIFAR-100 | Difference |
|--------|----------|-----------|------------|
| **Classes** | 10 | 100 | 10x more |
| **Images/Class** | 6,000 | 500 | 12x less data per class |
| **Difficulty** | Easy | Hard | More challenging |
| **Expected Accuracy (20R)** | 75-85% | 40-55% | Lower due to complexity |
| **Expected Accuracy (50R)** | 85-90% | 55-70% | Requires more training |
| **Training Time/Round** | ~12 min | ~13-15 min | Slightly slower |

### Why is CIFAR-100 Harder?

1. **Less data per class**: 500 images vs 6,000
2. **More fine-grained**: Distinguishing "rose" vs "orchid" vs "tulip" is harder than "bird" vs "dog"
3. **Similar classes**: Many visually similar categories (e.g., different trees, different mammals)
4. **10x more classes**: Much larger output space to learn

### Realistic Accuracy Targets:

```
CIFAR-100 Training Progression:
Round 5:    ~15-20%  (Random baseline is 1%)
Round 10:   ~25-35%  (Learning patterns)
Round 20:   ~40-55%  (Good performance) ← Target
Round 50:   ~55-70%  (Excellent for FL)
Round 100:  ~70-75%  (Near state-of-the-art for federated)
```

**Note:** 50-60% accuracy on CIFAR-100 is considered **very good** for federated learning!

---

## 🎨 Example Predictions

### Before (CIFAR-10):
```
Input: [Image of flower]
Prediction: bird (generic, often wrong)
Confidence: 45%
```

### After (CIFAR-100):
```
Input: [Image of flower]
Prediction: orchid (specific!)
Superclass: flowers
Confidence: 78%

Top 5:
1. orchid (78%)
2. rose (12%)
3. tulip (5%)
4. poppy (3%)
5. sunflower (1%)
```

---

## ⚙️ Advanced Features

### Hierarchical Prediction:

With CIFAR-100, you get **two-level prediction**:

```python
result = classifier.predict("image.jpg")

# Fine-grained class
print(result['predicted_class'])  # "dolphin"

# Superclass grouping
print(result['superclass'])  # "aquatic_mammals"
```

This is useful when:
- Fine prediction has low confidence → Use superclass
- Application needs broad categories → Use superclass
- Debugging model performance → Compare fine vs coarse accuracy

### Confusion Analysis:

CIFAR-100 allows studying **semantic confusion**:
- Does the model confuse "boy" and "girl"? (same superclass)
- Does it confuse "rose" and "orchid"? (same superclass)
- Does it confuse "dolphin" and "shark"? (different superclasses)

---

## 📊 Training Recommendations

### For Best Results:

1. **Train Longer**: 20-30 rounds minimum (vs 10-15 for CIFAR-10)
2. **Use More Clients**: 3-5 clients helps with diverse data
3. **Lower Learning Rate**: 0.0003 instead of 0.0005
4. **More Local Epochs**: 7-10 instead of 5
5. **Data Augmentation**: Essential for small per-class data

### Optimal Hyperparameters for CIFAR-100:

```python
# In backend_fl/config.py
LOCAL_EPOCHS = 7      # More training per round
BATCH_SIZE = 64       # Keep same
LEARNING_RATE = 0.0003  # Lower for stability
MODEL_ALPHA = 1.0     # Full capacity needed
```

### Training Timeline:

```
Recommended: 30 rounds

Rounds 1-10:    ~20-30% accuracy (2 hours)
Rounds 11-20:   ~35-45% accuracy (2 hours)
Rounds 21-30:   ~50-60% accuracy (2 hours)
Total:          ~50-60% accuracy (6 hours) ✓
```

---

## 🔍 Testing the Upgrade

### Verify CIFAR-100 is Active:

```bash
# Check configuration
python -c "from backend_fl.config import DATASET, NUM_CLASSES, LABELS; print(f'Dataset: {DATASET}'); print(f'Classes: {NUM_CLASSES}'); print(f'First 10: {LABELS[:10]}')"

# Expected output:
# Dataset: CIFAR100
# Classes: 100
# First 10: ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', ...]
```

### Test Data Loading:

```bash
python -c "from backend_fl.data_utils import load_dataset; X_train, y_train, X_test, y_test = load_dataset(); print(f'Loaded: {X_train.shape[0]} training images'); print(f'Classes: {y_train.shape[1]}')"

# Expected output:
# Loading CIFAR-100 dataset (label_mode=fine)...
# Loaded: 50000 training images
# Classes: 100
```

### Test Model Creation:

```bash
python -c "from backend_fl.model import get_model; model = get_model(); print(f'Output layer: {model.layers[-1].output_shape}')"

# Expected output:
# Output layer: (None, 100)
```

---

## 📝 Code Changes Summary

### Files Modified:
1. ✅ `backend_fl/config.py` - Added CIFAR-100 labels and configuration
2. ✅ `backend_fl/data_utils.py` - Added CIFAR-100 loader
3. ✅ `frontend_web/inference.py` - Added superclass support
4. ✅ `backend_fl/fl_client.py` - Updated to use unified loader
5. ✅ `backend_fl/model.py` - Already uses NUM_CLASSES (no changes needed!)

### Files Unchanged:
- `backend_fl/fl_server.py` - Works with any number of classes
- `backend_fl/strategies.py` - Class-agnostic
- `run_server.py`, `run_client.py`, `run_web.py` - No changes needed

---

## 🎯 Quick Comparison

| Feature | CIFAR-10 | CIFAR-100 |
|---------|----------|-----------|
| **Classes** | 10 | 100 |
| **Superclasses** | No | 20 |
| **Examples** | Generic (bird, car) | Specific (orchid, pickup_truck) |
| **Difficulty** | ⭐⭐ Easy | ⭐⭐⭐⭐ Hard |
| **Data per Class** | 6,000 | 500 |
| **Training Time** | 10-15 rounds | 20-30 rounds |
| **Target Accuracy** | 75-85% | 50-60% |
| **Use Case** | Simple classification | Real-world application |
| **Code Changes** | N/A | Minimal |

---

## 🚀 Next Steps

### Immediate:

1. **Verify the upgrade works:**
   ```bash
   python -c "from backend_fl.config import DATASET, NUM_CLASSES; print(f'{DATASET}: {NUM_CLASSES} classes')"
   ```

2. **Start training:**
   ```bash
   python run_server.py --num-rounds 20 --min-clients 2
   ```

3. **Monitor progress:**
   ```bash
   python visualize_training.py
   ```

### After Training:

1. **Test predictions:**
   - Upload images of flowers, people, furniture, animals
   - Check both fine class and superclass predictions
   - Compare confidence scores

2. **Analyze performance:**
   - Which superclasses perform best?
   - Which classes are most confused?
   - How does accuracy compare to CIFAR-10?

3. **Further improvements:**
   - Train for 30-50 rounds
   - Add more clients (3-5)
   - Use learning rate scheduling
   - Enable advanced data augmentation

---

## 💡 Pro Tips

1. **Don't expect 85% accuracy**: CIFAR-100 is 10x harder; 50-60% is excellent!
2. **Use superclasses for debugging**: If fine accuracy is low, check coarse accuracy
3. **Train longer**: 20 rounds minimum, 30-50 for best results
4. **More data helps**: Use 3-5 clients instead of 2
5. **Be patient**: Each round takes 13-15 minutes with 100 classes

---

## 📖 Resources

### What is CIFAR-100?
- Official: https://www.cs.toronto.edu/~kriz/cifar.html
- Paper: "Learning Multiple Layers of Features from Tiny Images" (Alex Krizhevsky, 2009)

### Why Use CIFAR-100?
- ✅ More realistic than CIFAR-10
- ✅ Tests fine-grained classification
- ✅ Widely used benchmark
- ✅ Same image size (32×32)
- ✅ Minimal code changes needed

---

## 🎉 Summary

✅ **Upgrade Complete!**

Your system now supports:
- **100 fine-grained classes** (vs 10)
- **20 superclass categories** for hierarchical prediction
- **More realistic scenarios** (flowers, people, furniture, specific animals)
- **Minimal code changes** (configuration-driven)
- **Same training process** (just train longer)

**Start training now to see your model classify orchids, dolphins, and wardrobes!** 🚀

---

**Quick Start Command:**
```bash
python run_server.py --num-rounds 20 --min-clients 2
```

**Expected Result:** 50-60% accuracy after 20 rounds (excellent for federated CIFAR-100!)
