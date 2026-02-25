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
git clone <repo-url>
cd "Image Classification"

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
3. **Select Model**:
   - **CIFAR-100**: Standard image classification (100 classes)
   - **VLM**: Multimodal visual assistant (5-point descriptions)
4. **Upload Image** and get predictions

---

## Model Options

| Model | Description |
|-------|-------------|
| CIFAR-100 | Classifies images into 100 categories |
| VLM | Generates 5-point structured descriptions |

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
├── backend_fl/          # FL core (server, client, VLM)
├── frontend_web/       # Flask web app
├── models/            # Saved models
├── uploads/           # Uploaded images
├── Plan.md            # Architecture docs
├── SUMMARY.md         # Project summary
└── run_*.py          # Entry points
```

---

## Support

- Check logs in terminal output
- See Plan.md for architecture details
- See SUMMARY.md for full project info
