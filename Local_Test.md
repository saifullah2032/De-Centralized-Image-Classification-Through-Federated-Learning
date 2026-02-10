# Local Testing Guide - Windows System

Complete guide for setting up and testing the Federated Learning Image Classification project on a Windows system.

**NEW: Now using Pre-trained ImageNet model with 1000 classes for accurate real-world image classification!**

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Testing Web Interface](#testing-web-interface)
- [Using ImageNet Predictions (NEW)](#using-imagenet-predictions-new)
- [Training Custom Model (Optional)](#training-custom-model-optional)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## What's New: ImageNet Integration

The project now uses a **pre-trained MobileNetV2 model** trained on ImageNet:

| Feature | Before (CIFAR-100) | Now (ImageNet) |
|---------|-------------------|----------------|
| Classes | 100 | **1000** |
| Input Size | 32x32 pixels | **224x224 pixels** |
| Training Required | Yes | **No (Pre-trained)** |
| Real-world Accuracy | ~27% | **~70%+** |
| Categories | Limited | Animals, vehicles, food, objects, nature, and more! |

**No training required** - the model is ready to use immediately!

---

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11 (64-bit)
- **Python Version**: 3.9, 3.10, or 3.11 (Python 3.12+ not supported yet)
- **RAM**: Minimum 8 GB (16 GB recommended for training)
- **Storage**: At least 20 GB free space
- **Network**: Stable internet connection for downloading dependencies

### Required Software

1. **Python 3.9-3.11**
   - Download from: https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"
   - Verify installation:
     ```powershell
     python --version
     ```

2. **Git** (if cloning from repository)
   - Download from: https://git-scm.com/download/win
   - Verify installation:
     ```powershell
     git --version
     ```

3. **PowerShell** (comes with Windows)
   - Open PowerShell as Administrator

---

## Installation

### Step 1: Clone or Download the Project

**Option A: Clone from GitHub**
```powershell
cd C:\Users\rayan\Downloads
git clone https://github.com/saifullah2032/De-Centralized-Image-Classification-Through-Federated-Learning.git
cd "De-Centralized-Image-Classification-Through-Federated-Learning"
```

**Option B: Use Existing Directory**
```powershell
cd "C:\Users\rayan\Downloads\Image CLassification"
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**Note**: If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

This will install:
- **Flower 1.7.0+** - Federated Learning framework
- **Keras 3.0+** - Deep learning framework (with JAX backend)
- **JAX** - High-performance numerical computing
- **Flask 3.0+** - Web framework
- **NumPy, scikit-learn, Pillow, matplotlib** - Data processing libraries

**Installation Time**: Approximately 5-10 minutes depending on your internet speed.

### Step 4: Verify Installation

```powershell
# Test if all modules are installed correctly
python -c "import flwr, keras, flask, jax; print('✓ All dependencies installed successfully!')"
```

If you see the success message, you're ready to proceed!

---

## Testing Web Interface

The web interface allows you to interact with the Federated Learning system through a modern, user-friendly dashboard.

### Step 1: Start the Web Application

```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Start the web server
python run_web.py
```

**Expected Output**:
```
======================================================================
FEDERATED LEARNING COMMAND CENTER
======================================================================
  Host:     0.0.0.0
  Port:     5000
  Login:    admin / admin123 (default credentials)
  URL:      http://localhost:5000
======================================================================

 * Serving Flask app 'frontend_web.app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.x.x:5000
Press CTRL+C to quit
```

### Step 2: Access the Web Interface

1. Open your web browser (Chrome, Firefox, or Edge)
2. Navigate to: **http://localhost:5000**
3. You should see the landing page with a modern dark theme

### Step 3: Login to the System

**Default Credentials**:

| Role | Username | Password |
|------|----------|----------|
| Admin | `admin` | `admin123` |
| Client | `client` | `client123` |

**Steps**:
1. Click "Login" in the navigation bar
2. Enter username: `admin`
3. Enter password: `admin123`
4. Click "Sign In"

### Step 4: Explore the Interface

#### Landing Page Features
- Project overview and description
- Key features showcase
- Modern dark theme with gradient effects
- Responsive design (works on mobile, tablet, desktop)

#### Admin Dashboard (`/admin/dashboard`)
- Real-time training metrics
- Model accuracy and loss charts
- Live event stream from FL server
- Client participation monitoring
- Training progress visualization

#### Prediction Page (`/predict`)
- Drag & drop image upload
- Real-time image classification using **ImageNet pre-trained model**
- **1000 class categories** (animals, vehicles, food, objects, nature, etc.)
- Confidence scores for top predictions
- No training required - works immediately!
- Interactive chart showing prediction probabilities

**The model can classify**: dogs, cats, birds, cars, planes, food, furniture, electronics, plants, and 990+ more categories!

### Step 5: Test Responsive Design

1. **Desktop View**: Full width, horizontal navigation
2. **Tablet View** (< 992px): Hamburger menu icon appears
3. **Mobile View** (< 768px): Optimized for touch, larger buttons

**To Test**: Resize your browser window or use Chrome DevTools (F12) > Toggle Device Toolbar (Ctrl+Shift+M)

### Step 6: Test Interactive Features

1. **Theme Toggle**: Click sun/moon icon in navbar to switch dark/light theme
2. **Scroll Animations**: Scroll down to see fade-in effects
3. **Button Ripples**: Click any button to see Material Design ripple effect
4. **Card Tilt**: Hover over cards to see 3D tilt effect
5. **Mobile Menu**: On mobile view, click hamburger icon to open sidebar menu

### Step 7: Stop the Web Server

Press `Ctrl+C` in the PowerShell terminal to stop the server.

---

## Using ImageNet Predictions (NEW)

The system now uses a **pre-trained MobileNetV2 model** that can classify images into **1000 different categories** without any training required!

### Step 1: Start the Web Application

```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Start the web server
python run_web.py
```

### Step 2: Access the Prediction Page

1. Open browser: http://localhost:5000
2. Login as admin (`admin` / `admin123`)
3. Click "Predict" in navbar

### Step 3: Upload an Image

1. Drag & drop any image onto the upload area, OR
2. Click to browse and select an image
3. Supported formats: JPG, PNG, JPEG

### Step 4: View Results

The model will classify your image and show:
- **Top Prediction**: Most likely class with confidence percentage
- **Top 5 Predictions**: Alternative classifications
- **Confidence Chart**: Visual representation of probabilities

### What Can It Classify?

The ImageNet model recognizes **1000 categories** including:

| Category | Examples |
|----------|----------|
| **Animals** | Dogs (120+ breeds), cats, birds, fish, insects, bears, elephants |
| **Vehicles** | Cars, planes, boats, bikes, trains, motorcycles |
| **Food** | Fruits, vegetables, dishes, drinks, desserts |
| **Objects** | Furniture, electronics, tools, clothing, sports equipment |
| **Nature** | Plants, flowers, landscapes, trees |
| **Buildings** | Houses, churches, castles, bridges |

### Example Predictions

**Dog Image** → "Golden Retriever" (95.2% confidence)
**Car Image** → "Sports Car" (87.5% confidence)
**Food Image** → "Pizza" (92.1% confidence)

### Model Details

- **Architecture**: MobileNetV2
- **Training Data**: ImageNet (1.2 million images)
- **Input Size**: 224x224 pixels (auto-resized)
- **Output**: 1000 class probabilities
- **Model Size**: ~14 MB (downloaded automatically on first use)

---

## Training Custom Model (Optional)

> **Note**: Training is **optional** since the project now uses pre-trained ImageNet model. Only train if you want to experiment with Federated Learning on CIFAR datasets.

To use the Federated Learning training feature with CIFAR datasets, you need to run a server and multiple client nodes.

### Understanding the Training Process

**Federated Learning Architecture**:
```
┌────────────────────────────────────────────────────────┐
│              Federated Learning Training                │
├────────────────────────────────────────────────────────┤
│                                                          │
│  Client 0          Client 1          Client 2          │
│  ┌──────┐         ┌──────┐         ┌──────┐           │
│  │ Data │         │ Data │         │ Data │           │
│  │10,000│         │10,000│         │10,000│           │
│  │images│         │images│         │images│           │
│  └──┬───┘         └──┬───┘         └──┬───┘           │
│     │                │                │                │
│     │ Train Local    │ Train Local    │ Train Local   │
│     ↓                ↓                ↓                │
│  Weights          Weights          Weights            │
│  (3-4 MB)         (3-4 MB)         (3-4 MB)           │
│     │                │                │                │
│     └────────────────┼────────────────┘                │
│                      ↓                                 │
│              ┌──────────────┐                          │
│              │  FL Server   │                          │
│              │              │                          │
│              │  Aggregate   │                          │
│              │  (FedAvg)    │                          │
│              │              │                          │
│              │  Global Model│                          │
│              └──────────────┘                          │
│                                                          │
└────────────────────────────────────────────────────────┘
```

### Training Configuration

**Default Settings** (in `backend_fl/config.py`):
- **Dataset**: ImageNet (1000 classes) - Pre-trained, no training needed
- **Alternative**: CIFAR-100 (100 classes) or CIFAR-10 (10 classes) for FL training
- **Number of Rounds**: 10 (configurable)
- **Number of Clients**: 5 (can run with minimum 2)
- **Local Epochs**: 5 per round
- **Batch Size**: 64
- **Learning Rate**: 0.0005
- **Model**: Enhanced MobileNetV2 (2.3M parameters)

> **To switch to CIFAR mode for training**, set `DATASET=CIFAR100` or `DATASET=CIFAR10` in your `.env` file.

### Quick Training (Automated) - RECOMMENDED

**Windows:**
```powershell
# Activate virtual environment first
.\venv\Scripts\Activate.ps1

# Run automated start script
.\start_all.bat
```

This will automatically open multiple terminals and start:
- 1 FL Server (port 8080)
- 5 FL Clients (client IDs 0-4)
- 1 Web Interface (port 5000)

**Training Time**: Approximately 2-4 hours for 10 rounds with 5 clients.

### Manual Training (Step-by-Step)

For more control, you can manually start each component.

#### Step 1: Open Multiple PowerShell Terminals

You need **at least 3 terminals** (1 server + 2 clients):
- Press `Win + R`, type `powershell`, press Enter
- Repeat to open 3 separate PowerShell windows

#### Step 2: Navigate to Project Directory (in each terminal)

```powershell
cd "C:\Users\rayan\Downloads\Image CLassification"
.\venv\Scripts\Activate.ps1
```

#### Step 3: Start FL Server (Terminal 1)

```powershell
# Terminal 1: FL Server
python run_server.py --num-rounds 10 --min-clients 2
```

**Expected Output**:
```
Configuration loaded successfully!
  - Dataset: CIFAR100 (100 classes)
  - FL Server: 0.0.0.0:8080
  - Web Server: 0.0.0.0:5000
  - Training: 10 rounds, 5 clients, 5 local epochs
  - Non-IID Alpha: 0.5

======================================================================
FEDERATED LEARNING SERVER
======================================================================
  Server Address:    0.0.0.0:8080
  Min Clients:       2
  Training Rounds:   10
  Strategy:          FedAvg with custom evaluation
======================================================================

INFO :      Starting Flower server, config: num_rounds=10, no round_timeout
```

**Keep this terminal running!**

#### Step 4: Start FL Clients (Terminals 2, 3, ...)

**Terminal 2 - Client 0:**
```powershell
# Terminal 2: Client 0
python run_client.py --client-id 0 --num-clients 2
```

**Terminal 3 - Client 1:**
```powershell
# Terminal 3: Client 1
python run_client.py --client-id 1 --num-clients 2
```

**Expected Output** (per client):
```
Configuration loaded successfully!
  - Dataset: CIFAR100 (100 classes)
  - FL Server: 0.0.0.0:8080
  - Web Server: 0.0.0.0:5000
  - Training: 10 rounds, 5 clients, 5 local epochs
  - Non-IID Alpha: 0.5

======================================================================
FEDERATED LEARNING CLIENT
======================================================================
  Client ID:         0
  Total Clients:     2
  Server Address:    localhost:8080
  Dataset:           CIFAR100 (100 classes)
  Local Epochs:      5
  Batch Size:        64
======================================================================

Downloading data from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
...
Partitioning dataset using Dirichlet distribution (alpha=0.5)...
Client data loaded: 25000 training samples, 5000 test samples

Connecting to server at localhost:8080...
INFO :      Opened insecure gRPC connection (no certificates were passed)
```

**Keep these terminals running!**

#### Step 5: Monitor Training Progress

**Option A: Watch Terminal Output**
- Server terminal shows round progress
- Client terminals show local training epochs

**Option B: Use Admin Dashboard**
1. Open browser: http://localhost:5000
2. Login as admin (`admin` / `admin123`)
3. Click "Admin Dashboard"
4. View real-time metrics and event stream

#### Step 6: Wait for Training to Complete

**Training Progress Example**:
```
Round 1/10:
  - Client 0: Training 5 epochs...
  - Client 1: Training 5 epochs...
  - Server: Aggregating weights...
  - Global accuracy: 45.2%

Round 2/10:
  - Client 0: Training 5 epochs...
  - Client 1: Training 5 epochs...
  - Server: Aggregating weights...
  - Global accuracy: 62.8%

...

Round 10/10:
  - Client 0: Training 5 epochs...
  - Client 1: Training 5 epochs...
  - Server: Aggregating weights...
  - Global accuracy: 78.5%

Training complete! Model saved to: models/global_model.h5
```

**Expected Time**:
- 2 clients: ~10-15 minutes per round (2-3 hours total for 10 rounds)
- 5 clients: ~15-20 minutes per round (3-4 hours total for 10 rounds)

**Expected Accuracy**:
- **CIFAR-10**: 75-85% after 10 rounds
- **CIFAR-100**: 50-65% after 10 rounds (100 classes is much harder)

### Step 7: Verify Model Training

After training completes, verify the model was saved:

```powershell
# Check if model exists
dir models

# You should see:
# - global_model.h5 (trained model, ~10-20 MB)
# - model_history.json (training metrics)
```

### Step 8: Visualize Training Results (Optional)

```powershell
# Generate training visualization
python visualize_training.py
```

This will create charts showing:
- Accuracy over rounds
- Loss over rounds
- Training progress visualization

### Step 9: Test Model Predictions

1. Start web interface: `python run_web.py`
2. Open browser: http://localhost:5000
3. Login as admin
4. Click "Predict" in navbar
5. Upload any image (will be automatically resized to 224x224 for ImageNet)
6. View predicted class and confidence scores

**Test Images**:
- Use any real-world image (photos from your phone, internet images, etc.)
- The ImageNet model can classify 1000 different categories
- No need for specific test images - it works with any photo!

---

## Troubleshooting

### Issue 1: "Python not found"

**Problem**: Command `python` is not recognized.

**Solution**:
1. Verify Python is installed: Open "Add or Remove Programs", search for Python
2. Add Python to PATH:
   - Windows Key + R → `sysdm.cpl` → Advanced → Environment Variables
   - Edit "Path" → Add: `C:\Users\<username>\AppData\Local\Programs\Python\Python311`
3. Restart PowerShell

### Issue 2: "Execution Policy Error"

**Problem**: Cannot run `.\venv\Scripts\Activate.ps1`

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 3: "Module 'flwr' not found"

**Problem**: Import error when running scripts.

**Solution**:
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue 4: "Port 5000 already in use"

**Problem**: Web server won't start.

**Solution**:
```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process (replace <PID> with actual process ID)
taskkill /PID <PID> /F

# Or change port in .env file
# Edit .env: WEB_PORT=5001
```

### Issue 5: "Port 8080 already in use"

**Problem**: FL server won't start.

**Solution**:
```powershell
# Find process using port 8080
netstat -ano | findstr :8080

# Kill the process
taskkill /PID <PID> /F

# Or change port in .env file
# Edit .env: FL_SERVER_PORT=8081
```

### Issue 6: "Connection refused" when starting clients

**Problem**: Clients cannot connect to server.

**Solution**:
1. Make sure FL server is running first
2. Check server terminal for errors
3. Verify firewall isn't blocking port 8080
4. Try restarting the server

### Issue 7: "Out of Memory" during training

**Problem**: System runs out of RAM during training.

**Solution**:
1. Reduce batch size in `.env`:
   ```
   BATCH_SIZE=32  # Or even 16
   ```
2. Use fewer clients (minimum 2)
3. Close other applications
4. Restart your computer

### Issue 8: "Model not loaded" in prediction page

**Problem**: No model file found.

**Solution**:
1. The ImageNet model is downloaded automatically on first use
2. Ensure you have internet connection for initial download (~14 MB)
3. Check if Keras cache directory has the model weights
4. If using CIFAR mode, train a model first (see "Training Custom Model" section)

### Issue 9: Training is too slow

**Problem**: Training takes longer than expected.

**Solution**:
1. Reduce number of rounds:
   ```powershell
   python run_server.py --num-rounds 5 --min-clients 2
   ```
2. Use fewer local epochs in `.env`:
   ```
   LOCAL_EPOCHS=3
   ```
3. Increase batch size (if you have enough RAM):
   ```
   BATCH_SIZE=128
   ```

### Issue 10: "JAX/Keras backend error"

**Problem**: Keras cannot use JAX backend.

**Solution**:
```powershell
# Reinstall JAX
pip uninstall jax jaxlib
pip install jax[cpu]

# Verify installation
python -c "import jax; print(jax.devices())"
```

### Issue 11: Browser shows old cached version

**Problem**: UI changes don't appear.

**Solution**:
- Hard reload: `Ctrl + Shift + R` (Chrome/Edge) or `Ctrl + F5` (Firefox)
- Clear browser cache
- Try incognito/private mode

---

## Next Steps

### 1. Improve Model Accuracy

**Option A: Train for More Rounds**
```powershell
python run_server.py --num-rounds 20 --min-clients 2
```

**Option B: Use More Clients**
```powershell
# Start 5 clients instead of 2
python run_client.py --client-id 0 --num-clients 5
python run_client.py --client-id 1 --num-clients 5
python run_client.py --client-id 2 --num-clients 5
python run_client.py --client-id 3 --num-clients 5
python run_client.py --client-id 4 --num-clients 5
```

**Option C: Adjust Hyperparameters**

Edit `.env` file:
```env
NUM_ROUNDS=20
LOCAL_EPOCHS=5
BATCH_SIZE=64
LEARNING_RATE=0.0005
```

### 2. Switch Between ImageNet and CIFAR Modes

Edit `.env` file:
```env
# For ImageNet (1000 classes, pre-trained, recommended)
DATASET=IMAGENET

# For CIFAR-10 (10 classes, requires training)
DATASET=CIFAR10

# For CIFAR-100 (100 classes, requires training)
DATASET=CIFAR100
```

**Note**: ImageNet mode uses pre-trained weights and doesn't require training. CIFAR modes require Federated Learning training.

### 3. Test with Your Own Images

1. Prepare any images (photos, screenshots, internet images)
2. Navigate to http://localhost:5000/predict
3. Drag & drop your image
4. View predictions with confidence scores

**The ImageNet model can classify**:
- Animals (dogs, cats, birds, fish, insects, wildlife)
- Vehicles (cars, planes, boats, motorcycles, bikes)
- Food (fruits, vegetables, dishes, drinks)
- Objects (furniture, electronics, clothing, tools)
- Nature (plants, flowers, trees, landscapes)
- And 950+ more categories!

### 4. Deploy to Multiple Machines

To run clients on different computers:

1. **On Server Machine**:
   ```powershell
   python run_server.py --num-rounds 10 --min-clients 2
   ```

2. **On Client Machines**:
   - Edit `.env` file:
     ```env
     FL_CLIENT_SERVER_ADDRESS=<server_ip>:8080
     ```
   - Run client:
     ```powershell
     python run_client.py --client-id 0 --num-clients 5
     ```

3. **Network Requirements**:
   - Ensure firewall allows port 8080
   - All machines must be on same network (or use port forwarding)

### 5. Monitor System Resources

**Windows Task Manager**:
- Press `Ctrl + Shift + Esc`
- Monitor CPU, RAM, and Network usage during training

**Expected Resource Usage**:
- **CPU**: 50-80% during training epochs
- **RAM**: 4-8 GB per client
- **Network**: ~50-100 MB per round (model weights)

### 6. Production Deployment

For production use:

1. **Change Default Credentials**:
   - Edit `.env`:
     ```env
     ADMIN_PASSWORD=<strong_password>
     FLASK_SECRET_KEY=<random_secret_key>
     ```

2. **Use Production Server**:
   ```powershell
   pip install waitress
   waitress-serve --port=5000 frontend_web.app:app
   ```

3. **Enable HTTPS**:
   - Use reverse proxy (Nginx, Apache)
   - Add SSL certificates

4. **Set Up Monitoring**:
   - Use logging
   - Set up alerts
   - Monitor model performance

---

## Additional Resources

### Project Documentation

- **README.md** - Project overview and general setup
- **prd.md** - Product Requirements Document
- **NAVIGATION_MAP.md** - Project structure and navigation
- **JAVASCRIPT_ENHANCEMENTS.md** - UI/UX features documentation

### Configuration Files

- **.env** - Environment variables (customize here)
- **requirements.txt** - Python dependencies
- **backend_fl/config.py** - Training configuration constants

### Useful Commands

```powershell
# Check Python version
python --version

# List installed packages
pip list

# Update a package
pip install --upgrade <package_name>

# Check disk space
dir models
dir logs

# View training logs
type logs\training.log

# View model history
type models\model_history.json
```

### Getting Help

- **GitHub Issues**: https://github.com/saifullah2032/De-Centralized-Image-Classification-Through-Federated-Learning/issues
- **Flower Documentation**: https://flower.ai/docs/
- **Keras Documentation**: https://keras.io/
- **Flask Documentation**: https://flask.palletsprojects.com/

---

## Summary Checklist

### Initial Setup
- [ ] Python 3.9-3.11 installed
- [ ] Project downloaded/cloned
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Installation verified

### Web Interface Testing
- [ ] Web server started successfully
- [ ] Browser opened at http://localhost:5000
- [ ] Logged in as admin
- [ ] Explored landing page
- [ ] Tested responsive design
- [ ] Tested interactive features

### Model Training
- [ ] FL server started
- [ ] At least 2 FL clients started
- [ ] Training completed successfully
- [ ] Model saved to `models/global_model.h5`
- [ ] Training visualized (optional)

### Prediction Testing
- [ ] Model loaded successfully
- [ ] Test image uploaded
- [ ] Predictions displayed correctly
- [ ] Confidence scores visible

---

## Contact & Support

For questions or issues:
1. Check [Troubleshooting](#troubleshooting) section first
2. Review project documentation files
3. Open an issue on GitHub
4. Contact project maintainers

---

**Congratulations!** You've successfully set up and tested the Federated Learning Image Classification system on your Windows machine!

The system now uses **pre-trained ImageNet model** with **1000 classes** for accurate real-world image classification - no training required!

*Last Updated: February 10, 2026*
