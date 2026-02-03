# Product Requirements Document (PRD)
# Decentralized Image Classification with Federated Learning

**Project Name**: DecentralizedAI - Federated Learning Command Center  
**Version**: 1.0  
**Date**: February 2026  
**Status**: Development Phase  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Vision & Goals](#2-product-vision--goals)
3. [Problem Statement](#3-problem-statement)
4. [Target Users](#4-target-users)
5. [Technical Requirements](#5-technical-requirements)
6. [Functional Requirements](#6-functional-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [System Architecture](#8-system-architecture)
9. [User Stories](#9-user-stories)
10. [Success Metrics](#10-success-metrics)
11. [Development Phases & Task Breakdown](#11-development-phases--task-breakdown)
12. [Privacy & Compliance Requirements](#12-privacy--compliance-requirements)
13. [Testing Requirements](#13-testing-requirements)
14. [Deployment Strategy](#14-deployment-strategy)
15. [Timeline & Milestones](#15-timeline--milestones)
16. [Risk Assessment](#16-risk-assessment)
17. [Future Enhancements](#17-future-enhancements)

---

## 1. Executive Summary

### 1.1 Product Overview

DecentralizedAI is a production-grade, privacy-preserving distributed machine learning system that enables collaborative model training across multiple edge devices without centralizing raw data. The system implements Federated Learning using the FedAvg algorithm to train image classification models on CIFAR-10 dataset while ensuring data never leaves local devices.

### 1.2 Key Features

- **Privacy-First Architecture**: Raw data remains on edge devices; only model weights (3-4 MB) are transmitted
- **Non-IID Data Handling**: Realistic heterogeneous data distributions using Dirichlet partitioning (α=0.5)
- **FedAvg Aggregation**: Weighted averaging of client models to synthesize global model
- **Web Command Center**: Dark-themed Flask dashboard with role-based access control (RBAC)
- **Real-Time Monitoring**: Server-Sent Events (SSE) for live training visualization
- **Privacy Verification**: Network traffic analysis to confirm no raw data transmission
- **Compliance Ready**: GDPR and HIPAA compliant architecture

### 1.3 Research Foundation

Based on the research paper "Decentralized Image Classification with Federated Learning" by Dr. Sri Hari Nallamala et al., demonstrating:
- 86% accuracy on Non-IID CIFAR-10 (competitive with centralized training at ~92%)
- Successful mitigation of Client Drift phenomenon
- Zero raw data transmission verified via network analysis

---

## 2. Product Vision & Goals

### 2.1 Vision Statement

Build a production-ready federated learning platform that demonstrates privacy-preserving machine learning can achieve competitive accuracy while maintaining regulatory compliance and user privacy.

### 2.2 Primary Goals

1. **Privacy**: Ensure 100% data isolation - no raw data leaves edge devices
2. **Performance**: Achieve 85%+ accuracy on CIFAR-10 within 10 training rounds
3. **Usability**: Provide intuitive web interface for both admins and clients
4. **Compliance**: Meet GDPR and HIPAA requirements for healthcare/sensitive data
5. **Scalability**: Support 5-50 distributed clients with minimal communication overhead

### 2.3 Success Criteria

- ✅ Global model accuracy ≥ 85% after 10 rounds
- ✅ Inference latency < 500ms per image
- ✅ Zero PII/raw data detected in network traffic (Wireshark verification)
- ✅ Web UI loads predictions in < 2 seconds
- ✅ System handles 5 concurrent clients without degradation
- ✅ Test coverage ≥ 80%

---

## 3. Problem Statement

### 3.1 Current Challenges

**Problem**: Traditional centralized machine learning creates critical vulnerabilities:

1. **Privacy Risks**: Aggregating sensitive data creates "honeypot" targets for attackers
2. **Regulatory Barriers**: GDPR/HIPAA prohibit cross-jurisdictional data sharing
3. **Communication Overhead**: High-resolution image uploads cause bandwidth strain
4. **Data Silos**: Organizations cannot collaborate due to competitive/compliance constraints

### 3.2 Proposed Solution

**Federated Learning** inverts the computation model:
- **Before**: Data → Central Server → Model
- **After**: Model → Edge Devices → Weights → Central Server

**Key Principle**: "Data never leaves the device; only model updates are shared."

---

## 4. Target Users

### 4.1 Primary Users

| User Role | Description | Key Needs |
|-----------|-------------|-----------|
| **System Administrator** | Oversees federated training | Real-time monitoring, control over training parameters, audit logs |
| **Data Scientist** | Analyzes model performance | Training metrics, model versioning, experimentation tools |
| **Client Node Operator** | Manages local training | Local performance monitoring, privacy verification |
| **End User** | Performs inference | Simple image upload, fast predictions, confidence scores |

### 4.2 User Personas

**Persona 1: Dr. Sarah Chen - Hospital IT Administrator**
- **Role**: Admin at Hospital A
- **Goal**: Train collaborative diagnosis model without sharing patient X-rays
- **Pain Points**: HIPAA compliance, data breach concerns
- **Needs**: Real-time training dashboard, privacy audit reports

**Persona 2: Alex Kumar - ML Engineer**
- **Role**: Client node operator
- **Goal**: Monitor local training performance
- **Needs**: Client-side metrics, model updates, transparent process

---

## 5. Technical Requirements

### 5.1 Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **FL Framework** | Flower | 1.7.0 | Federated learning orchestration |
| **ML Framework** | TensorFlow/Keras | 2.13.0 | Model training and inference |
| **Web Framework** | Flask | 3.0.0 | Web UI and REST API |
| **Model Architecture** | MobileNetV2 | - | Lightweight CNN for edge devices |
| **Dataset** | CIFAR-10 | - | 60,000 32×32 images, 10 classes |
| **Programming Language** | Python | 3.9-3.11 | Core implementation |
| **Frontend** | HTML/CSS/JavaScript | - | User interface |

### 5.2 System Requirements

**Server Requirements**:
- CPU: 4+ cores (Intel/AMD)
- RAM: 8 GB minimum
- Storage: 20 GB (for models + dataset)
- Network: 100 Mbps LAN

**Client Requirements**:
- CPU: 2+ cores
- RAM: 2 GB minimum
- Storage: 5 GB
- Network: Stable connection to server

---

## 6. Functional Requirements

### 6.1 Federated Learning Core

#### FR-1: Model Architecture
**Priority**: P0 (Critical)  
**Description**: Implement shared MobileNetV2 architecture  
**Acceptance Criteria**:
- Model accepts input shape (32, 32, 3)
- Outputs 10-class probability distribution
- Model size ≤ 5 MB
- Reproducible architecture across all nodes

#### FR-2: Data Partitioning
**Priority**: P0 (Critical)  
**Description**: Non-IID data partitioning using Dirichlet distribution  
**Acceptance Criteria**:
- Partition CIFAR-10 across N clients (default N=5)
- Each client has heterogeneous class distribution (α=0.5)
- Total samples = 50,000 (training set)
- Each client receives ≥ 100 samples
- Validation: Coefficient of Variation > 0.3

#### FR-3: Federated Averaging (FedAvg)
**Priority**: P0 (Critical)  
**Description**: Implement FedAvg aggregation strategy  
**Acceptance Criteria**:
- Weighted averaging: w_global = Σ(nk/n × wk)
- Aggregation completes in ≤ 15 seconds per round
- Supports 2-50 clients
- Handles client failures gracefully

#### FR-4: Model Persistence
**Priority**: P0 (Critical)  
**Description**: Save global model after each round  
**Acceptance Criteria**:
- Save to `models/global_model.h5`
- Version tracking: `model_round_1.h5`, `model_round_2.h5`, etc.
- Metadata includes: round number, accuracy, loss, timestamp
- Enable model rollback to previous rounds

### 6.2 Command Center (Web Interface)

#### FR-5: Authentication & RBAC
**Priority**: P0 (Critical)  
**Description**: Role-based access control  
**Acceptance Criteria**:
- Roles: Admin, Client, Public
- Login page with username/password
- Session management (JWT tokens, 1-hour expiration)
- Admin-only routes: `/admin/dashboard`, `/admin/events`
- Client routes: `/predict`, `/status`
- Public routes: `/`, `/docs`

#### FR-6: Admin Dashboard
**Priority**: P0 (Critical)  
**Description**: Real-time training monitoring  
**Acceptance Criteria**:
- Display: Current round, global accuracy, global loss
- Live chart: Accuracy/Loss vs Round (updates every 5 seconds)
- Client participation status (5 cards showing active/idle)
- Aggregation metrics: Time, communication overhead
- SSE log stream: Real-time training events

#### FR-7: Image Upload & Inference
**Priority**: P0 (Critical)  
**Description**: Prediction interface for end users  
**Acceptance Criteria**:
- Drag-and-drop image upload
- Support formats: JPG, PNG
- File size limit: 5 MB
- Inference time: < 100ms
- Display: Predicted class, confidence %, top-5 predictions
- Horizontal bar chart showing all 10 class probabilities

#### FR-8: Real-Time Monitoring (SSE)
**Priority**: P1 (High)  
**Description**: Server-Sent Events for live updates  
**Acceptance Criteria**:
- Endpoint: `/admin/events`
- Events: `AGGREGATION_START`, `AGGREGATION_COMPLETE`, `CLIENT_UPDATE`
- Data includes: timestamp, round, message, metrics
- Auto-reconnect on disconnect
- Log persistence to `logs/training.log`

#### FR-9: Model Status API
**Priority**: P1 (High)  
**Description**: REST API for system status  
**Acceptance Criteria**:
- Endpoint: `GET /status`
- Returns: `{model_loaded, model_version, accuracy, current_round, total_rounds}`
- Response time: < 50ms
- Cached for 5 seconds

#### FR-10: Training Metrics API
**Priority**: P1 (High)  
**Description**: Historical training data  
**Acceptance Criteria**:
- Endpoint: `GET /metrics`
- Returns: JSON with rounds, accuracies, losses, aggregation times
- Used for chart rendering
- Stored in `models/model_history.json`

### 6.3 Privacy & Compliance

#### FR-11: Privacy Verification
**Priority**: P0 (Critical)  
**Description**: Network traffic analysis  
**Acceptance Criteria**:
- Wireshark PCAP capture during training
- Verify: 0 raw image bytes detected
- Only Protocol Buffers (model weights) transmitted
- Generate privacy report: `/privacy-report`

#### FR-12: Compliance Reporting
**Priority**: P1 (High)  
**Description**: GDPR/HIPAA audit reports  
**Acceptance Criteria**:
- Report includes: TLS verification, data minimization, access logs
- Downloadable as PDF
- Auto-generated after each training session
- Includes timestamp, auditor role, findings

---

## 7. Non-Functional Requirements

### 7.1 Performance

| Metric | Requirement |
|--------|-------------|
| **Model Accuracy** | ≥ 85% (CIFAR-10, after 10 rounds) |
| **Inference Latency** | < 500ms per image |
| **Aggregation Time** | ≤ 15 seconds per round |
| **Web UI Load Time** | < 2 seconds (initial load) |
| **API Response Time** | < 100ms (95th percentile) |

### 7.2 Scalability

- Support 5-50 concurrent clients
- Handle 1000 inference requests/hour
- Database can store 1000+ training rounds
- Disk usage: < 50 GB for 100 model versions

### 7.3 Reliability

- Uptime: 99.5% during training sessions
- Graceful handling of client disconnections
- Auto-recovery from server crashes (within 5 minutes)
- Data integrity: Model weights verified via checksums

### 7.4 Security

- TLS 1.3 encryption for all client-server communication
- AES-256 encryption for model weights at rest
- Input validation on all file uploads (prevent malicious files)
- Rate limiting: 100 requests/minute per IP
- SQL injection prevention (parameterized queries)
- XSS protection (content security policy)

### 7.5 Usability

- Dark theme for 24/7 monitoring (low eye strain)
- Responsive design (desktop, tablet, mobile)
- Error messages: User-friendly, actionable
- Documentation: Inline help, tooltips
- Accessibility: WCAG 2.1 Level AA compliance

### 7.6 Maintainability

- Code coverage: ≥ 80%
- Modular architecture (separation of concerns)
- Consistent naming conventions (PEP 8)
- Inline documentation (docstrings)
- Version control (Git)

---

## 8. System Architecture

### 8.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Federated Learning System                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│   ┌──────────────┐         ┌──────────────┐                │
│   │   Client 0   │         │   Client 1   │                │
│   │  (Hospital A)│         │  (Hospital B)│                │
│   │              │         │              │                │
│   │ Local Data:  │         │ Local Data:  │                │
│   │ 10,000 images│         │ 10,000 images│                │
│   │              │         │              │                │
│   │ Train Locally│         │ Train Locally│                │
│   │      ↓       │         │      ↓       │                │
│   │  Weights (3MB)──────┐  │  Weights (3MB)─────┐          │
│   └──────────────┘      │  └──────────────┘     │          │
│                         │                        │          │
│                         ↓                        ↓          │
│                   ┌────────────────────────────────┐        │
│                   │   Federated Server (FedAvg)   │        │
│                   │                                │        │
│                   │  • Aggregate weights           │        │
│                   │  • w_global = Σ(nk/n × wk)    │        │
│                   │  • Evaluate on test set        │        │
│                   │  • Save global model           │        │
│                   │                                │        │
│                   │  models/global_model.h5        │        │
│                   └────────────────────────────────┘        │
│                              │                              │
│                              ↓                              │
│                   ┌────────────────────────────────┐        │
│                   │   Web Interface (Flask)        │        │
│                   │                                │        │
│                   │  • Admin Dashboard             │        │
│                   │  • Inference UI                │        │
│                   │  • Real-time monitoring (SSE)  │        │
│                   │  • Privacy reports             │        │
│                   │                                │        │
│                   │  http://localhost:5000         │        │
│                   └────────────────────────────────┘        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Component Diagram

```
backend_fl/
├── config.py          → Configuration constants
├── model.py           → MobileNetV2 architecture
├── data_utils.py      → CIFAR-10 loading, Non-IID partitioning
├── fl_server.py       → Flower server (aggregator)
├── fl_client.py       → Flower client (local trainer)
└── strategies.py      → Custom FedAvg with model saving

frontend_web/
├── app.py             → Flask application
├── auth.py            → RBAC & authentication
├── inference.py       → Model loading & prediction
├── templates/         → HTML pages
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── training_monitor.html
│   └── admin_panel.html
└── static/
    ├── css/style.css
    └── js/
        ├── app.js
        ├── sse_client.js
        └── charts.js
```

### 8.3 Data Flow

**Training Flow**:
1. Server broadcasts initial weights to all clients
2. Each client trains locally for E epochs (3-5)
3. Clients send updated weights back to server
4. Server aggregates: w_global = Σ(nk/n × wk)
5. Server evaluates on test set
6. Server saves global model
7. Repeat for R rounds (10)

**Inference Flow**:
1. User uploads image via `/predict`
2. Server preprocesses: resize to 32×32, normalize
3. Load global model from `models/global_model.h5`
4. Run inference: predictions = model.predict(image)
5. Return JSON: {class, confidence, all_predictions}
6. Frontend displays results with bar chart

---

## 9. User Stories

### 9.1 Admin User Stories

**US-1**: As an admin, I want to see real-time training progress so I can monitor model convergence.  
**Acceptance Criteria**:
- Dashboard updates every 5 seconds
- Shows current round, accuracy, loss
- Displays client participation status

**US-2**: As an admin, I want to download privacy audit reports so I can verify GDPR compliance.  
**Acceptance Criteria**:
- Report includes network traffic analysis
- Downloadable as PDF
- Contains timestamp and findings

**US-3**: As an admin, I want to pause training so I can investigate anomalies.  
**Acceptance Criteria**:
- Pause button on dashboard
- Training resumes from same round
- Clients wait for resume signal

### 9.2 Client User Stories

**US-4**: As a client operator, I want to monitor local training so I can ensure my node is contributing.  
**Acceptance Criteria**:
- View local accuracy, loss
- See number of samples used
- Estimate time to completion

**US-5**: As a client operator, I want to verify my data never leaves my device so I can trust the system.  
**Acceptance Criteria**:
- Access privacy verification report
- See network traffic analysis
- Confirm only weights transmitted

### 9.3 End User Stories

**US-6**: As an end user, I want to upload an image and get predictions so I can classify images.  
**Acceptance Criteria**:
- Drag-and-drop upload
- Prediction displayed in < 2 seconds
- Shows confidence score

**US-7**: As an end user, I want to see top-5 predictions so I can understand model uncertainty.  
**Acceptance Criteria**:
- Display top-5 classes with probabilities
- Bar chart visualization
- Color-coded by confidence

---

## 10. Success Metrics

### 10.1 Key Performance Indicators (KPIs)

| KPI | Target | Measurement |
|-----|--------|-------------|
| **Global Model Accuracy** | ≥ 85% | Test set evaluation after round 10 |
| **Inference Latency** | < 500ms | Average time from upload to result |
| **Training Rounds to Convergence** | ≤ 10 | Rounds until accuracy plateaus |
| **Communication Overhead** | < 300 MB/round | Total data transmitted per round |
| **Privacy Verification** | 0 PII packets | Wireshark analysis |
| **Web UI Load Time** | < 2 seconds | Time to interactive (TTI) |
| **System Uptime** | ≥ 99.5% | During training sessions |

### 10.2 Expected Results (from Research Paper)

| Round | Loss | Accuracy | Notes |
|-------|------|----------|-------|
| 1 | 2.30 | 45% | Random initialization |
| 2 | 1.85 | 62.5% | Learning general features |
| 3 | 1.23 | 71.3% | Convergence accelerating |
| 4 | 0.78 | 78.9% | Near plateau |
| 5 | 0.42 | 86.2% | Client Drift (+0.01 loss, but +7% accuracy) |

---

## 11. Development Phases & Task Breakdown

### Phase 1: Environment Setup & Project Architecture
**Duration**: 1 day  
**Priority**: P0 (Critical)

#### Task 1.1: Create Project Structure
- [ ] Create directory structure (backend_fl/, frontend_web/, tests/, docs/, models/, logs/)
- [ ] Initialize Git repository
- [ ] Create `.gitignore` (exclude venv, *.pyc, *.h5, logs/)
- [ ] Create README.md skeleton

#### Task 1.2: Initialize Virtual Environment
- [ ] Run `python -m venv venv`
- [ ] Activate environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux)
- [ ] Verify Python version (3.9-3.11)

#### Task 1.3: Install Dependencies
- [ ] Create `requirements.txt` with all dependencies
- [ ] Run `pip install -r requirements.txt`
- [ ] Verify installations: `pip list`
- [ ] Test imports: `import flwr, tensorflow, flask`

#### Task 1.4: Environment Configuration
- [ ] Create `.env.example` with template variables
- [ ] Create `.env` with actual configuration
- [ ] Set environment variables: `FLASK_SECRET_KEY`, `FL_SERVER_ADDRESS`

**Deliverables**: Functional development environment, all dependencies installed

---

### Phase 2: Shared Components Implementation
**Duration**: 2 days  
**Priority**: P0 (Critical)

#### Task 2.1: Backend Configuration (`backend_fl/config.py`)
- [ ] Define constants: `INPUT_SHAPE = (32, 32, 3)`
- [ ] Set `NUM_CLASSES = 10`
- [ ] Configure `NUM_ROUNDS = 10`, `LOCAL_EPOCHS = 3`, `BATCH_SIZE = 32`
- [ ] Set `LEARNING_RATE = 0.001`
- [ ] Define `CIFAR10_LABELS = ['airplane', 'automobile', ...]`
- [ ] Add Dirichlet alpha parameter: `ALPHA = 0.5`

#### Task 2.2: Shared Model Architecture (`backend_fl/model.py`)
- [ ] Implement `get_model()` function
- [ ] Load MobileNetV2 base (pre-trained on ImageNet optional)
- [ ] Add custom classification head:
  - Global average pooling
  - Dense(128, activation='relu')
  - Dropout(0.5)
  - Dense(10, activation='softmax')
- [ ] Compile model: optimizer=Adam(lr=0.001), loss=categorical_crossentropy
- [ ] Verify model summary: total params ~3-4M
- [ ] Test model instantiation: `model = get_model(); assert model is not None`

#### Task 2.3: CIFAR-10 Data Loading (`backend_fl/data_utils.py`)
- [ ] Implement `load_cifar10()` function
- [ ] Download dataset via `tensorflow.keras.datasets.cifar10.load_data()`
- [ ] Normalize pixel values: `X = X / 255.0`
- [ ] Convert labels to categorical: `y = to_categorical(y, 10)`
- [ ] Return: `(X_train, y_train, X_test, y_test)`
- [ ] Verify shapes: `assert X_train.shape == (50000, 32, 32, 3)`

#### Task 2.4: Non-IID Data Partitioning (`backend_fl/data_utils.py`)
- [ ] Implement `partition_data_non_iid(num_clients=5, alpha=0.5)`
- [ ] For each class c ∈ {0, 1, ..., 9}:
  - Draw proportions from Dirichlet(alpha)
  - Assign sample indices to clients based on proportions
- [ ] Validate: Each client receives ≥ 100 samples
- [ ] Log class distribution for each client
- [ ] Return: `List[Dict]` with partition indices
- [ ] Test: Verify coefficient of variation > 0.3

#### Task 2.5: Client Data Access Functions
- [ ] Implement `get_client_data(client_id, partition_dict)`
- [ ] Return: `(X_train_local, y_train_local, X_test_local, y_test_local)`
- [ ] Implement `get_test_set()` for server-side validation
- [ ] Test: Verify no data overlap between clients

**Deliverables**: Shared model, CIFAR-10 loader, Non-IID partitioner

---

### Phase 3: Federated Learning Core Implementation
**Duration**: 3 days  
**Priority**: P0 (Critical)

#### Task 3.1: Custom FedAvg Strategy (`backend_fl/strategies.py`)
- [ ] Create `SaveModelStrategy` class extending `flwr.server.strategy.FedAvg`
- [ ] Implement `aggregate_fit(server_round, results, failures)`:
  - [ ] Call parent `aggregate_fit()` for weight averaging
  - [ ] Convert aggregated weights to TensorFlow model
  - [ ] Evaluate on server test set
  - [ ] Save to `models/global_model.h5`
  - [ ] Log metrics to `logs/training.log`
  - [ ] Update `models/model_history.json`
- [ ] Implement `aggregate_evaluate(server_round, results, failures)`:
  - [ ] Compute average accuracy across clients
  - [ ] Return averaged loss and accuracy
- [ ] Set strategy properties:
  - [ ] `min_available_clients = 2`
  - [ ] `min_fit_clients = 2`
  - [ ] `fraction_fit = 1.0`
- [ ] Test: Mock aggregation with dummy weights

#### Task 3.2: Federated Server (`backend_fl/fl_server.py`)
- [ ] Load configuration from `config.py`
- [ ] Load test dataset via `get_test_set()`
- [ ] Initialize `SaveModelStrategy`
- [ ] Configure server parameters:
  - [ ] `num_rounds = 10`
  - [ ] `min_available_clients = 2`
- [ ] Start Flower server: `flwr.server.start_server()`
- [ ] Listen on `0.0.0.0:8080`
- [ ] Log: "Flower server started at 0.0.0.0:8080"
- [ ] For each round:
  - [ ] Collect weights from clients
  - [ ] Aggregate using FedAvg
  - [ ] Evaluate on test set
  - [ ] Save global model
  - [ ] Log: "Round X completed. Accuracy: Y%"
- [ ] After all rounds: Save final model + metrics

#### Task 3.3: Federated Client (`backend_fl/fl_client.py`)
- [ ] Create `CIFARClient` class extending `flwr.client.NumPyClient`
- [ ] Implement `__init__(client_id, num_clients)`:
  - [ ] Load partition via `get_client_data(client_id)`
  - [ ] Initialize model via `get_model()`
  - [ ] Store local data: `(X_train_local, y_train_local, X_test_local, y_test_local)`
- [ ] Implement `get_parameters(config)`:
  - [ ] Return `model.get_weights()` as NumPy arrays
- [ ] Implement `fit(parameters, config)`:
  - [ ] Set model weights from server
  - [ ] Train locally for `LOCAL_EPOCHS` (3-5):
    - Shuffle local training data
    - Train with `batch_size=32`
    - Update model weights locally
  - [ ] Compute: loss, accuracy on local data
  - [ ] Return: `(updated_weights, num_samples, metrics)`
- [ ] Implement `evaluate(parameters, config)`:
  - [ ] Set model weights from server
  - [ ] Evaluate on local test set
  - [ ] Return: `(loss, accuracy)`
- [ ] Add CLI argument parsing: `--client-id`, `--num-clients`
- [ ] Test: Start client, verify connection to server

#### Task 3.4: Client Drift Mitigation (Optional Enhancement)
- [ ] Track accuracy across rounds in `SaveModelStrategy`
- [ ] Detect drift: If `|accuracy[t] - accuracy[t-1]| > 0.05`, log warning
- [ ] Implement adaptive local epochs: Reduce if drift detected
- [ ] Test: Verify drift detection with intentionally skewed data

**Deliverables**: Functional FL server + clients, model aggregation working

---

### Phase 4: Command Center (Web Interface)
**Duration**: 2 days  
**Priority**: P0 (Critical)

#### Task 4.1: Flask Application Setup (`frontend_web/app.py`)
- [ ] Initialize Flask app: `app = Flask(__name__)`
- [ ] Configure secret key: `app.config['SECRET_KEY']`
- [ ] Set upload folder: `app.config['UPLOAD_FOLDER'] = 'uploads/'`
- [ ] Set max file size: `app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024` (5 MB)
- [ ] Enable CORS: `CORS(app)`

#### Task 4.2: Authentication & RBAC (`frontend_web/auth.py`)
- [ ] Initialize Flask-Login: `login_manager = LoginManager()`
- [ ] Create `User` class with roles: admin, client, public
- [ ] Implement password hashing: `generate_password_hash()`, `check_password_hash()`
- [ ] Create default users: admin, client
- [ ] Implement `@app.route('/login', methods=['GET', 'POST'])`
- [ ] Implement `@app.route('/logout')`
- [ ] Create decorators: `@admin_required`, `@login_required`
- [ ] Test: Login as admin, verify redirection to `/admin/dashboard`

#### Task 4.3: Model Inference (`frontend_web/inference.py`)
- [ ] Create `ImageClassifier` class
- [ ] Implement `__init__(model_path='models/global_model.h5')`:
  - [ ] Load model architecture from `model.py`
  - [ ] Load weights from `model_path`
  - [ ] Compile model
- [ ] Implement `preprocess_image(image_path)`:
  - [ ] Open image, resize to 32×32, convert to RGB
  - [ ] Normalize: `pixel_values / 255.0`
  - [ ] Return: `(1, 32, 32, 3)` NumPy array
- [ ] Implement `predict(image_path)`:
  - [ ] Preprocess image
  - [ ] Run: `predictions = model.predict(image_array)`
  - [ ] Get top prediction: `class_idx = np.argmax(predictions[0])`
  - [ ] Return: `{class, confidence, all_predictions}`
- [ ] Add error handling: Model not found, invalid image format
- [ ] Test: Predict on test image, verify output format

#### Task 4.4: Flask Routes - Core Endpoints
- [ ] `GET /` → Landing page (index.html)
- [ ] `GET /login` → Login page
- [ ] `POST /login` → Authenticate user
- [ ] `GET /logout` → Logout user
- [ ] `GET /predict` → Prediction UI (predict.html)
- [ ] `POST /predict` → Handle image upload, return prediction
  - [ ] Validate file type (JPG, PNG)
  - [ ] Validate file size (≤ 5 MB)
  - [ ] Save to uploads/
  - [ ] Call `ImageClassifier.predict()`
  - [ ] Delete uploaded file
  - [ ] Return JSON: `{success, prediction, confidence, all_predictions}`
- [ ] `GET /status` → Model status
  - [ ] Return: `{model_loaded, model_version, accuracy, current_round}`
- [ ] `GET /metrics` → Training metrics
  - [ ] Load from `models/model_history.json`
  - [ ] Return: `{rounds, accuracies, losses}`
- [ ] Test: Upload image via Postman, verify response

#### Task 4.5: Admin Dashboard Routes
- [ ] `GET /admin/dashboard` → Admin dashboard (training_monitor.html)
  - [ ] Require admin role
  - [ ] Display: Current round, accuracy, loss
  - [ ] Client participation status
- [ ] `GET /admin/events` → Server-Sent Events (SSE) endpoint
  - [ ] Stream training events: `yield f"data: {json.dumps(event)}\n\n"`
  - [ ] Events: `AGGREGATION_START`, `AGGREGATION_COMPLETE`, `CLIENT_UPDATE`
- [ ] `GET /admin/model-history` → Download model versions
  - [ ] List all `models/model_round_*.h5` files
  - [ ] Enable download
- [ ] `GET /privacy-report` → Privacy audit report
  - [ ] Load from `logs/privacy_report.json`
  - [ ] Render as HTML or downloadable PDF
- [ ] Test: Login as admin, verify dashboard loads

#### Task 4.6: HTML Templates
- [ ] `base.html` → Dark-themed base template
  - [ ] Navbar: Logo, title, user menu
  - [ ] Sidebar: Dashboard, Predict, Metrics, Logout
  - [ ] Footer: Copyright, links
- [ ] `index.html` → Landing page
  - [ ] Project overview, login button
- [ ] `login.html` → Login form
  - [ ] Username/password fields, submit button
- [ ] `predict.html` → Image upload UI
  - [ ] Drag-and-drop zone, file input
  - [ ] Preview image
  - [ ] Results section: Predicted class, confidence, bar chart
- [ ] `training_monitor.html` → Admin dashboard
  - [ ] Current round, accuracy, loss cards
  - [ ] Line chart: Accuracy/Loss vs Round
  - [ ] Client participation status
  - [ ] Live log stream (SSE)
  - [ ] Controls: Pause, Resume, Export
- [ ] Test: Render all templates, verify dark theme

#### Task 4.7: Frontend JavaScript
- [ ] `app.js` → Image upload handler
  - [ ] Validate file type, size
  - [ ] Show preview
  - [ ] Send to `/predict` endpoint via Fetch API
  - [ ] Display prediction results
  - [ ] Render bar chart (Chart.js)
  - [ ] Poll `/status` every 5 seconds
- [ ] `sse_client.js` → Server-Sent Events listener
  - [ ] Connect to `/admin/events`
  - [ ] On message: Append to log, update charts
  - [ ] On error: Reconnect
- [ ] `charts.js` → Chart rendering
  - [ ] Line chart: Accuracy/Loss vs Round
  - [ ] Bar chart: Class probabilities
  - [ ] Update in real-time
- [ ] Test: Upload image, verify prediction displayed

#### Task 4.8: CSS Styling
- [ ] `style.css` → Dark theme
  - [ ] Define CSS variables: `--bg-primary`, `--accent`, `--success`
  - [ ] Navbar: Dark background, sticky position
  - [ ] Cards: Dark background, border on hover
  - [ ] Confidence bar: Gradient fill
  - [ ] Responsive design: Mobile-friendly
- [ ] Test: Verify dark theme on all pages

**Deliverables**: Functional web UI with login, prediction, admin dashboard

---

### Phase 5: Testing & Validation
**Duration**: 2 days  
**Priority**: P0 (Critical)

#### Task 5.1: Unit Tests
- [ ] `tests/test_model.py`
  - [ ] Test model instantiation
  - [ ] Verify input/output shapes
  - [ ] Check weight count
- [ ] `tests/test_data_utils.py`
  - [ ] Test CIFAR-10 loading
  - [ ] Verify Non-IID partitioning correctness
  - [ ] Assert no data leakage between partitions
- [ ] `tests/test_fl_client.py`
  - [ ] Mock Flower server communication
  - [ ] Test `get_parameters()` returns correct shape
  - [ ] Test `fit()` updates weights
- [ ] `tests/test_inference.py`
  - [ ] Test image preprocessing
  - [ ] Test prediction output format
  - [ ] Verify confidence scores sum to 1.0
- [ ] Run: `pytest tests/ -v --cov=backend_fl --cov=frontend_web`
- [ ] Target: ≥ 80% code coverage

#### Task 5.2: Integration Tests
- [ ] Test end-to-end FL training:
  - [ ] Start server → clients connect → training completes → model saved
- [ ] Test model persistence:
  - [ ] Train → Save → Load → Verify weights identical
- [ ] Test web inference:
  - [ ] Upload image → Process → Prediction displayed → Metrics logged
- [ ] Test RBAC:
  - [ ] Login as client → Try accessing `/admin/dashboard` → Verify 403 Forbidden

**Deliverables**: ≥ 80% test coverage, all tests passing

---

### Phase 5.5: Privacy Verification
**Duration**: 1 day  
**Priority**: P0 (Critical)

#### Task 5.5.1: Wireshark Network Traffic Analysis
- [ ] Install Wireshark
- [ ] Start packet capture on loopback interface (`lo` or `127.0.0.1`)
- [ ] Filter: `tcp.port == 8080`
- [ ] Start FL server + clients
- [ ] Let training run for 2 rounds
- [ ] Stop packet capture
- [ ] Save as `logs/network_traffic.pcap`

#### Task 5.5.2: Analyze PCAP for Raw Data
- [ ] Open PCAP in Wireshark
- [ ] Search for JPEG/PNG headers: `FF D8 FF` (JPEG), `89 50 4E 47` (PNG)
- [ ] Search for pixel data patterns
- [ ] Verify: 0 image bytes detected
- [ ] Confirm: Only Protocol Buffers (model weights) transmitted

#### Task 5.5.3: Generate Privacy Report
- [ ] Create `logs/privacy_report.json`:
  - [ ] `data_isolation_verified: true`
  - [ ] `protocol_buffers_detected: 5234`
  - [ ] `image_data_transmitted: 0`
  - [ ] `wireshark_analysis_date: 2026-02-01`
  - [ ] `conclusion: "NO RAW IMAGE DATA DETECTED"`
- [ ] Implement `/privacy-report` route to display report
- [ ] Test: Access report, verify findings

**Deliverables**: Privacy verification report, PCAP analysis

---

### Phase 6: Experimental Validation
**Duration**: 1 day  
**Priority**: P1 (High)

#### Task 6.1: Proof-of-Intelligence Test
- [ ] Prepare DEER image (not in CIFAR-10 training set)
- [ ] Save to `tests/data/deer_test.jpg`
- [ ] Run federated training (5 rounds)
- [ ] Load global model: `keras.models.load_model('models/global_model_r5.h5')`
- [ ] Preprocess DEER image: resize to 32×32, normalize
- [ ] Run inference: `predictions = model.predict(deer_image)`
- [ ] Verify: Predicted class == "DEER" (class_id = 4)
- [ ] Verify: Confidence ≥ 50%
- [ ] Document results in `logs/proof_of_intelligence.txt`
- [ ] Test: Run `python tests/test_proof_of_intelligence.py`

#### Task 6.2: Non-IID Distribution Validation
- [ ] Implement `tests/test_non_iid_validation.py`
- [ ] Load CIFAR-10 and partition
- [ ] Analyze each client's class distribution
- [ ] Calculate coefficient of variation
- [ ] Verify: CV > 0.3 (True Non-IID)
- [ ] Visualize distribution per client (matplotlib)
- [ ] Save plots to `logs/non_iid_distributions.png`

#### Task 6.3: Client Drift Analysis
- [ ] Track round-by-round metrics: loss, accuracy
- [ ] Identify rounds where loss increases but accuracy improves
- [ ] Document Client Drift phenomenon
- [ ] Create table: Round | Loss | Accuracy | Loss Δ | Drift Score
- [ ] Save analysis to `logs/client_drift_analysis.md`

**Deliverables**: Proof-of-Intelligence passed, Non-IID verified, Client Drift documented

---

### Phase 7: Execution & Deployment
**Duration**: 1 day  
**Priority**: P1 (High)

#### Task 7.1: Local Simulation
- [ ] Create `run_server.py`:
  - [ ] Import `fl_server.py`
  - [ ] Start server: `python run_server.py`
- [ ] Create `run_clients.py`:
  - [ ] Accept `--client-id` argument
  - [ ] Start client: `python run_clients.py --client-id 0`
- [ ] Create `run_web.py`:
  - [ ] Import `app.py`
  - [ ] Start Flask: `python run_web.py`
- [ ] Test: Start server, 5 clients, web UI → Verify training completes

#### Task 7.2: Docker Containerization (Optional)
- [ ] Create `Dockerfile` for server
- [ ] Create `Dockerfile` for client
- [ ] Create `Dockerfile` for web UI
- [ ] Create `docker-compose.yml`:
  - [ ] Services: fl-server, fl-client-0 to fl-client-4, web-ui
- [ ] Test: `docker-compose up` → Verify training

#### Task 7.3: Multi-Machine Deployment (Optional)
- [ ] Configure server to listen on `0.0.0.0:8080`
- [ ] Update client config to connect to server IP
- [ ] Deploy server on cloud VM (AWS/GCP/Azure)
- [ ] Deploy clients on separate machines
- [ ] Test: Cross-machine training

**Deliverables**: Functional local deployment, Docker setup (optional)

---

### Phase 8: Performance Optimization (Optional)
**Duration**: 1 day  
**Priority**: P2 (Low)

#### Task 8.1: Model Quantization
- [ ] Convert model to TensorFlow Lite
- [ ] Apply INT8 quantization
- [ ] Reduce model size: 3.4 MB → 0.9 MB
- [ ] Verify accuracy drop ≤ 2%

#### Task 8.2: Gradient Compression
- [ ] Implement top-K gradient selection
- [ ] Send only top 1% important gradients
- [ ] Measure communication reduction

**Deliverables**: Optimized model, reduced communication overhead

---

### Phase 9: Documentation & Compliance
**Duration**: 2 days  
**Priority**: P1 (High)

#### Task 9.1: Documentation Files
- [ ] `README.md`:
  - [ ] Project overview, quick start guide
  - [ ] Installation instructions
  - [ ] Usage examples
  - [ ] Results & benchmarks
- [ ] `ARCHITECTURE.md`:
  - [ ] System design, data flow diagrams
  - [ ] Privacy guarantees
  - [ ] Scalability considerations
- [ ] `API.md`:
  - [ ] REST API specification
  - [ ] Request/response examples
  - [ ] Error codes
- [ ] `DEPLOYMENT.md`:
  - [ ] Prerequisites, setup instructions
  - [ ] Configuration options
  - [ ] Troubleshooting guide

#### Task 9.2: HIPAA Compliance Guide
- [ ] Document authentication mechanisms
- [ ] Explain encryption (TLS 1.3, AES-256)
- [ ] Audit logging implementation
- [ ] Data minimization strategy
- [ ] Backup & recovery procedures
- [ ] Create checklist: `docs/HIPAA_COMPLIANCE.md`

#### Task 9.3: GDPR Compliance Audit
- [ ] Verify data minimization
- [ ] Test "Right to be Forgotten" (remove client, retrain)
- [ ] Review access logs for PII
- [ ] Generate audit report: `docs/GDPR_AUDIT.md`
- [ ] Include: Findings, recommendations, approval

**Deliverables**: Complete documentation, compliance guides

---

### Phase 10: Security & Privacy
**Duration**: 1 day  
**Priority**: P1 (High)

#### Task 10.1: TLS Encryption
- [ ] Generate SSL certificate: `openssl req -x509 -newkey rsa:4096 ...`
- [ ] Configure Flower server with TLS
- [ ] Test: `openssl s_client -connect localhost:8080 -tls1_3`

#### Task 10.2: Input Validation
- [ ] Validate file uploads: Check MIME type, magic bytes
- [ ] Sanitize filenames: `secure_filename()`
- [ ] Prevent directory traversal attacks
- [ ] Limit file size: 5 MB

#### Task 10.3: Differential Privacy (Optional)
- [ ] Implement Gaussian noise addition to gradients
- [ ] Set privacy budget: ε = 1.0
- [ ] Clip gradients: C = 1.0
- [ ] Test: Verify privacy guarantee

**Deliverables**: TLS enabled, input validation implemented, DP (optional)

---

## 12. Privacy & Compliance Requirements

### 12.1 HIPAA Compliance

**Requirement**: Safeguard patient health information (PHI)

**Implementation**:
- **Authentication**: Role-based access control (Admin/Client)
- **Encryption in Transit**: TLS 1.3 for all client-server communication
- **Encryption at Rest**: AES-256 for model weights
- **Audit Logging**: All access logged with timestamp, user, action
- **Data Minimization**: Raw patient data never leaves facility
- **Backup & Recovery**: Model versioning, off-site encrypted backups

**Verification**:
- Wireshark analysis confirms 0 PHI transmitted
- Audit logs reviewed quarterly
- Penetration testing annually

### 12.2 GDPR Compliance

**Requirement**: Protect personal data of EU citizens

**Implementation**:
- **Data Minimization**: Only model weights transmitted
- **Storage Limitation**: Data kept locally, not centralized
- **Right to be Forgotten**: Can retrain without specific client's data
- **Data Subject Consent**: Clients explicitly agree to training
- **Breach Notification**: Notification within 72 hours (minimal risk: weights alone reveal no personal info)

**Verification**:
- PCAP analysis: 0 PII packets detected
- Test "Right to be Forgotten": Remove client, retrain, verify model converges

---

## 13. Testing Requirements

### 13.1 Test Coverage

- **Unit Tests**: ≥ 80% code coverage
- **Integration Tests**: All major workflows (training, inference, RBAC)
- **End-to-End Tests**: Full system simulation
- **Performance Tests**: Latency, throughput, scalability
- **Security Tests**: Input validation, XSS, SQL injection
- **Privacy Tests**: Network traffic analysis

### 13.2 Test Environments

- **Local**: Single machine, 5 simulated clients
- **Staging**: Docker containers, isolated network
- **Production**: Multi-machine deployment, cloud VMs

---

## 14. Deployment Strategy

### 14.1 Local Deployment

1. Clone repository
2. Create virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Start server: `python run_server.py`
5. Start clients (5 terminals): `python run_clients.py --client-id 0-4`
6. Start web UI: `python run_web.py`
7. Access: `http://localhost:5000`

### 14.2 Docker Deployment

1. Build images: `docker-compose build`
2. Start services: `docker-compose up`
3. Access: `http://localhost:5000`

### 14.3 Cloud Deployment

1. Provision VM (AWS EC2, GCP Compute Engine)
2. Install dependencies
3. Configure firewall: Open ports 8080, 5000
4. Start services via systemd or supervisord
5. Set up SSL certificate (Let's Encrypt)
6. Access: `https://your-domain.com`

---

## 15. Timeline & Milestones

| Phase | Duration | Start Date | End Date | Milestone |
|-------|----------|------------|----------|-----------|
| **Phase 1**: Environment Setup | 1 day | Day 1 | Day 1 | Dev environment ready |
| **Phase 2**: Shared Components | 2 days | Day 2 | Day 3 | Model + data utils complete |
| **Phase 3**: FL Core | 3 days | Day 4 | Day 6 | Server + clients functional |
| **Phase 4**: Web Interface | 2 days | Day 7 | Day 8 | Web UI live |
| **Phase 5**: Testing | 2 days | Day 9 | Day 10 | ≥ 80% coverage |
| **Phase 5.5**: Privacy Verification | 1 day | Day 11 | Day 11 | Privacy report generated |
| **Phase 6**: Experimental Validation | 1 day | Day 12 | Day 12 | Proof-of-Intelligence passed |
| **Phase 7**: Deployment | 1 day | Day 13 | Day 13 | Local simulation working |
| **Phase 8**: Optimization (Optional) | 1 day | Day 14 | Day 14 | Model quantized |
| **Phase 9**: Documentation | 2 days | Day 15 | Day 16 | All docs complete |
| **Phase 10**: Security | 1 day | Day 17 | Day 17 | TLS enabled |
| **Total** | **17 days** | Day 1 | Day 17 | **Production-ready** |

**Accelerated Timeline** (with 3-person team): 5-7 days

---

## 16. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Client disconnections during training** | High | Medium | Implement timeout handling, min_available_clients=2 |
| **Model accuracy below 85%** | Medium | High | Tune hyperparameters, increase rounds to 15 |
| **Privacy verification fails** | Low | Critical | Audit code for data leakage, re-run Wireshark analysis |
| **Web UI performance degradation** | Medium | Medium | Optimize chart rendering, implement caching |
| **GDPR compliance issues** | Low | High | Consult legal team, conduct third-party audit |
| **Network bandwidth constraints** | Medium | Medium | Implement gradient compression, reduce communication frequency |

---

## 17. Future Enhancements

### 17.1 Advanced FL Algorithms
- **FedProx**: Add proximal term to handle Non-IID data better
- **Scaffold**: Client-drift correction via control variates
- **Moon**: Momentum-based optimization

### 17.2 Multi-Task Learning
- Shared feature extraction layer
- Personalized classification heads per client

### 17.3 Real-World Integration
- **Healthcare**: Medical image classification (X-rays, MRIs)
- **Finance**: Fraud detection without sharing transaction data
- **IoT**: Edge device model training
- **Mobile Apps**: iOS/Android inference

### 17.4 Blockchain Integration
- Immutable training records
- Incentive mechanisms (reward clients for participation)

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Federated Learning** | Distributed ML paradigm where data never leaves edge devices |
| **FedAvg** | Federated Averaging algorithm for aggregating client models |
| **Non-IID** | Non-identically distributed data (heterogeneous class distributions) |
| **Client Drift** | Phenomenon where clients optimize for local data, causing global model divergence |
| **Dirichlet Distribution** | Probability distribution used to create Non-IID data partitions |
| **RBAC** | Role-Based Access Control (Admin, Client, Public) |
| **SSE** | Server-Sent Events (real-time unidirectional communication) |
| **PCAP** | Packet Capture file format (used by Wireshark) |

---

## Appendix B: References

1. **Communication-Efficient Learning of Deep Networks from Decentralized Data** (FedAvg) - https://arxiv.org/abs/1602.05629
2. **Federated Learning with Non-IID Data** - https://arxiv.org/abs/1909.06335
3. **MobileNetV2: Inverted Residuals and Linear Bottlenecks** - https://arxiv.org/abs/1801.04381
4. **Flower: A Friendly Federated Learning Framework** - https://flower.ai/
5. **GDPR - General Data Protection Regulation** - https://gdpr-info.eu/
6. **HIPAA - Health Insurance Portability and Accountability Act** - https://www.hhs.gov/hipaa/

---

**Document Status**: Ready for Implementation  
**Next Steps**: Begin Phase 1 - Environment Setup  
**Contact**: [Your Name/Team]  
**Last Updated**: February 2026
