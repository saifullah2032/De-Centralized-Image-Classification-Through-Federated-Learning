# Quick Start Guide

## Prerequisites

- Python 3.10+ 
- Windows/Linux/macOS
- 8GB RAM minimum (for VLM)
- Git

---

## Setup

### 1. Clone & Install

```bash
# Clone repository
git clone https://github.com/saifullah2032/De-Centralized-Image-Classification-Through-Federated-Learning
cd "De-Centralized-Image-Classification-Through-Federated-Learning"

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\Activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install VLM Dependencies

```bash
pip install torch torchvision transformers peft accelerate einops
pip install flask-sqlalchemy flask-login
```

---

## Running the Project

### Start Web Server

```bash
python run_web.py
```

Access: **http://localhost:5000**

Login credentials:
- Username: `admin`
- Password: `admin123`

### Using the Web Interface

1. **Login** with admin credentials
2. Go to **Predict** page
3. **Select Prediction Mode**:
   - **Standard Mode**: Fast MobileNetV2 (ImageNet-1K) classification
   - **Ensemble Mode**: Triple-Layer with Nuclear Truth Protocol (MobileNetV2 + SCL + BLIP-VQA)
4. **Upload Image** and get predictions
5. **Toggle Theme**: Click the theme toggle button (top-right) to switch between light mode (Vibrant Coral) and dark mode (Neon Cyan)

---

## Model Options

| Model | Description | Features |
|-------|-------------|----------|
| Standard | Fast MobileNetV2 (ImageNet-1K) classification with confidence scores | ~5-10 seconds |
| Ensemble | Triple-Layer with Nuclear Truth Protocol: CNN → SCL → VLM Analysis | ~30-40 seconds |

---

## UI/UX Features

### Light Mode (Default)
- **Primary Color**: Vibrant Ocean Coral (#72edf1)
- **Typography**: Fredoka font for playful aesthetic
- **Borders**: Crisp black borders with comic book-style shadows
- **Background**: Clean white with subtle animated ocean dots

### Dark Mode
- **Primary Color**: Neon Cyan (#00f3ff)
- **Background**: Abyssal midnight blue (#0a0e27)
- **Borders**: Crisp white for maximum contrast
- **Effects**: Smooth neon glow shadows for depth

### Animations
- Smooth 200-300ms transitions with cubic-bezier easing
- 2px hover lift on interactive elements
- Zero layout shift for optimal performance
- Animated ocean background with swimming fish on home page

---

## Optional: Federated Training

### Start Server

```bash
python run_server.py --num-rounds 5 --min-clients 2
```

### Start Clients (separate terminals)

```bash
python run_client.py --client-id 0
python run_client.py --client-id 1
```

---

## Troubleshooting

### VLM not loading?
```bash
pip install torch torchvision transformers --upgrade
```

### Port already in use?
```bash
# Kill existing process
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Import errors?
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
Image Classification/
├── backend_fl/                    # FL core (server, client, VLM)
│   ├── vlm_model.py              # VQA model (Salesforce/blip-vqa-base)
│   ├── server.py                 # Federated Learning server
│   └── client.py                 # Federated Learning client
├── frontend_web/                 # Flask web app
│   ├── static/
│   │   ├── css/custom.css        # Professional UI styling (light & dark mode)
│   │   ├── js/ocean-animations.js # Ocean background & fish animations
│   │   └── svg/                  # Fish SVG assets
│   ├── templates/                # HTML pages (responsive design)
│   └── app.py                    # Flask application
├── models/                       # Saved models (VLM only - MobileNetV2 ImageNet-1K loads from Keras)
├── uploads/                      # Uploaded images for inference
├── test_images/                  # Sample images for testing
├── design-system/                # Design system documentation
├── Plan.md                       # Architecture & technical docs
├── SUMMARY.md                    # Project summary
├── UI_POLISH_FINAL_REPORT.md    # UI/UX enhancements documentation
└── run_*.py                      # Entry points
```

---

## Support

- Check logs in terminal output
- See **Plan.md** for architecture details
- See **SUMMARY.md** for full project info
- See **UI_POLISH_FINAL_REPORT.md** for UI/UX implementation details
- See **design-system/** for design system documentation
