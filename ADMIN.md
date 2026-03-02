# ADMIN.md - Federated Learning System Administrator Guide

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Admin Dashboard Interface](#admin-dashboard-interface)
4. [Real-Time Monitoring](#real-time-monitoring)
5. [Triple-Layer Architecture Overview](#triple-layer-architecture-overview)
6. [System Health & Status](#system-health--status)
7. [Federated Learning Coordination](#federated-learning-coordination)
8. [Model Management](#model-management)
9. [Performance Metrics](#performance-metrics)
10. [Live Log Monitoring](#live-log-monitoring)
11. [System Architecture Visualization](#system-architecture-visualization)
12. [Training Round Management](#training-round-management)
13. [User & Node Management](#user--node-management)
14. [System Configuration](#system-configuration)
15. [Troubleshooting](#troubleshooting)
16. [Best Practices](#best-practices)
17. [Advanced Monitoring](#advanced-monitoring)
18. [Getting Help](#getting-help)

---

## Overview

The **Admin Dashboard** is the nerve center of the DecentralizedAI federated learning system. As an administrator, you have complete visibility into the Triple-Layer Consistency Pipeline, real-time performance metrics, federated learning coordination, and system health status.

### Key Responsibilities

- **Monitor System Performance**: Track inference times, accuracy, and pipeline efficiency
- **Oversee Federated Learning**: Coordinate training rounds, manage model updates, and synchronize across edge nodes
- **Manage Models**: Ensure MobileNetV2, BLIP-VQA, and SCL models are properly loaded and operational
- **System Health**: Keep watch on resource usage, temperature, and error rates
- **Live Logging**: Review real-time events, training progress, and anomalies
- **Configuration Management**: Adjust system parameters for optimal performance

### System Architecture at a Glance

The system implements a **Triple-Layer Consistency Pipeline** with autonomous error correction:

```
Input Image (ImageNet-1K)
         ↓
Stage 1: CNN Classification (MobileNetV2)
    ├─ Predicts class + confidence
    └─ ~0.1-0.5s inference time
         ↓
Stage 1.5: Semantic Consistency Layer (NEW)
    ├─ Verifies CNN prediction with BLIP-VQA
    ├─ Detects inconsistencies autonomously
    ├─ Routes to Stage 2 or triggers self-correction
    └─ ~3-8s verification time
         ↓
Stage 2/3: VLM Analysis + Narrative Synthesis
    ├─ Generates 9-point professional analysis
    ├─ Uses corrected identity (if SCL intervened)
    ├─ Synthesizes narrative report
    └─ ~5-15s analysis time
         ↓
Output: Master Audit Report (9-point summary + PDF)
```

---

## Getting Started

### Admin Login

1. Navigate to `http://localhost:5000/login`
2. Enter credentials:
   - **Username**: `admin`
   - **Password**: `admin123` (default - change in production!)
3. Click "Log In"
4. You'll be automatically redirected to `/admin/dashboard`

### First Steps Checklist

- [ ] Verify all models are loaded (check System Health Status widget)
- [ ] Review system resources (RAM, CPU, disk space)
- [ ] Check recent training logs in the Live Log panel
- [ ] Verify federated learning connectivity (if enabled)
- [ ] Review inference time metrics for performance baseline

### Accessing the Admin Dashboard

The admin dashboard is available at: `http://localhost:5000/admin/dashboard`

**Only accessible to users with admin role** - the system enforces this via the `@admin_required` decorator on the backend (`frontend_web/auth.py:18`).

---

## Admin Dashboard Interface

The admin dashboard is organized into several key sections:

### Dashboard Header

```
⚙️  Triple-Layer Consistency Pipeline Monitor
Real-time metrics for ImageNet-1K MobileNetV2 → SCL → BLIP-VQA Triple-Layer Architecture
```

The header identifies you as monitoring the complete federated learning pipeline with real-time data flowing from edge nodes.

### Layout Structure

The dashboard uses a responsive grid layout with the following sections:

```
┌─────────────────────────────────────────────────────┐
│ DASHBOARD HEADER                                     │
├─────────────────────────────────────────────────────┤
│ [STATS ROW - 5 Metric Widgets]                      │
├─────────────────────────────────────────────────────┤
│ ┌──────────────────────────┬──────────────────────┐ │
│ │ PERFORMANCE METRICS      │   LIVE LOG PANEL     │ │
│ │ (Chart with 2 datasets)  │   (Terminal-style)   │ │
│ │                          │                      │ │
│ └──────────────────────────┴──────────────────────┘ │
├─────────────────────────────────────────────────────┤
│ SYSTEM ARCHITECTURE (4-Stage Pipeline Display)      │
├─────────────────────────────────────────────────────┤
│ SYSTEM HEALTH STATUS (3 Model Health Cards)        │
└─────────────────────────────────────────────────────┘
```

### Stat Widgets (Top Row)

The top row displays 5 critical metrics in real-time:

#### 1. **Avg. Inference Time (Hybrid)** 🚀
- **Color**: Coral accent
- **Shows**: Average time for complete Triple-Layer pipeline
- **Includes**: Stage 1 CNN + Stage 1.5 SCL + Stage 2/3 VLM
- **Target**: 9-25 seconds on CPU (8GB RAM)
- **Typical Values**: 45-55ms during monitoring
- **Impact**: Lower is better; watch for degradation over time

#### 2. **Ensemble Mode Usage Rate** 📊
- **Color**: Success/green accent
- **Shows**: Percentage of predictions using Triple-Layer Hybrid mode
- **Typical Values**: 75-95% (depending on client preferences)
- **Importance**: Higher usage means more comprehensive analysis
- **Monitors**: System adoption of full pipeline vs. CNN-only mode

#### 3. **SCL Self-Correction Rate** 🛡️
- **Color**: Warning/orange accent
- **Shows**: Percentage of images where SCL detected inconsistency and triggered self-correction
- **Typical Values**: 5-15% (varies by image type and CNN confidence)
- **Interpretation**:
  - Low rate (< 5%): CNN performing well, good data quality
  - Medium rate (5-15%): Normal operation, consistent detections
  - High rate (> 20%): May indicate training data shift or model drift
- **Purpose**: Autonomous error correction - prevents bad predictions from reaching clients

#### 4. **FL Round Status** 🔄
- **Color**: Warning/orange accent
- **Shows**: Current federated learning training round number
- **Format**: "Round X" where X = total completed rounds
- **Updates**: When FL coordinator completes a synchronization cycle
- **Importance**: Tracks federated learning progress across nodes

#### 5. **System Health** ❤️
- **Color**: Dynamic (green = optimal, red = issues)
- **Shows**: Overall system operational status with animated pulse indicator
- **Statuses**:
  - 🟢 **Optimal**: All systems operational, no errors
  - 🟡 **Degraded**: Minor issues detected, investigation recommended
  - 🔴 **Critical**: System issues, immediate action required
- **Monitoring**: Covers model loading, inference capability, resource availability

---

## Real-Time Monitoring

### Charts Section

The **Triple-Layer Pipeline Performance Metrics** chart displays two key metrics:

#### Dataset 1: Hybrid Inference Time (ms) 📈

- **Line**: Solid cyan/blue color (`#00d4ff`)
- **Points**: Circle markers with black borders
- **Shows**: Milliseconds for complete pipeline execution
- **Y-Axis**: Left side (milliseconds)
- **Interpretation**:
  - Stable line = consistent performance
  - Rising trend = potential degradation
  - Spikes = temporary resource contention
  
**What to Watch For**:
- Sustained increase above 55ms baseline
- Regular spikes during peak hours
- Sudden drops (may indicate model skipping stages)

#### Dataset 2: Ensemble Mode Usage (%) 📊

- **Line**: Dashed blue color with no points (`#0066cc`)
- **Shows**: What percentage of predictions used Triple-Layer mode
- **Y-Axis**: Right side (0-100%)
- **Interpretation**:
  - 75-95%: Good adoption of full pipeline
  - < 50%: Clients prefer faster CNN-only mode
  - Rising trend: Increasing confidence in pipeline quality

**Chart Controls**:
- **Hover**: Show exact values for any data point
- **Legend**: Click legend items to toggle visibility
- **Interaction**: View specific time periods by hovering

#### Chart Refresh Rate

The chart updates **every 5 seconds** with latest metrics from the backend. The data comes from:
- Flask route: `/api/metrics`
- Returns JSON with historical data points
- Automatic updates via JavaScript polling

---

## Triple-Layer Architecture Overview

### Stage 1: CNN Classifier 🧠

**Component Display**:
```
[Stage 1: CNN Classifier]
MobileNetV2
ImageNet-1K trained
```

**Technical Details**:
- **Model**: MobileNetV2
- **Training Data**: ImageNet-1K (100 image classes)
- **Performance**: ~88% accuracy on test set
- **Inference Time**: 100-500ms
- **Memory**: ~100MB model weight
- **Purpose**: Fast initial classification of input image

**Admin Monitoring**:
- Status badge shows "Ready" when loaded
- Model file size displayed (~100MB)
- Load status indicator in System Health section
- Part of baseline comparison

### Stage 1.5: Semantic Consistency Layer with Nuclear Truth Protocol ⚠️

**Component Display**:
```
[Stage 1.5: Nuclear Truth SCL]
BLIP-VQA + Truth Discovery
Consistency verification + Mandatory VLM Discovery
```
*(Note: This section has orange background to highlight the critical verification stage)*

**Technical Details**:
- **Model**: BLIP-VQA with Nuclear Truth Protocol SCL
- **Purpose**: Verify CNN prediction and mandate VLM discovery for low-confidence predictions
- **Verification Method**: 
  - If CNN confidence ≥ 50%: Ask "What is this image of?" and compare to CNN class
  - If CNN confidence < 50%: Mandatory truth discovery - query "Identify the main object in 2 technical words" and OVERWRITE predicted_class
- **Inference Time**: 3-8 seconds
- **Autonomous**: Operates without human intervention
- **Absolute Threshold**: 50% confidence (NO EXCEPTIONS)
- **Routing Decision**: 
  - If verified (≥50% confidence match): Route to Stage 2/3 with confidence boost
  - If low confidence (<50%): Trigger mandatory truth discovery, use VLM's answer as canonical identity

**How Nuclear Truth Protocol Works**:

1. **CNN produces prediction** (e.g., "Palace" with 20% confidence)
2. **Nuclear Truth Check**:
   - Is confidence ≥ 50%? No → TRIGGER MANDATORY TRUTH DISCOVERY
3. **Mandatory VLM Query**: "Identify the main object in 2 technical words"
   - VLM responds: "Garden statue"
4. **OVERWRITE Decision**:
   - Original CNN prediction: DISCARDED
   - Canonical identity: "Garden statue" (VLM correction)
   - Status: "Self-Corrected"
5. **All Stage 2/3 prompts** use corrected class name (e.g., "What is maintenance protocol for garden statue?")
6. **Results displayed** with Nuclear Truth badge + corrected identity

**Admin Monitoring**:
- Track "SCL Self-Correction Rate" in stats widget (% of images below 50% threshold)
- View corrected identities in analysis results
- Monitor correction patterns in live logs
- Analyze confidence distribution and threshold impacts
- Engineering Synthesis Language used in all prompts

**Why Nuclear Truth Matters**:
- Prevents bad CNN predictions (< 50% confidence) from being used
- Guarantees absolute identification accuracy through mandatory VLM verification
- Provides deterministic logic: confidence < 50% = automatic override
- Generates valuable correction data for federated learning
- Improves overall system reliability and user trust
- Professional engineering synthesis language ensures quality output

### Stage 2/3: Vision-Language Model Analysis 🎯

**Component Display**:
```
[Stage 3: VLM]
BLIP-VQA
Salesforce/blip-vqa-base
```

**Technical Details**:
- **Model**: BLIP-VQA (Vision-Language Model)
- **Source**: Salesforce/blip-vqa-base (open-source)
- **Inference Time**: 5-15 seconds
- **Memory**: ~350MB model weight
- **Input**: Original image + CNN class (or SCL-corrected class)
- **Output**: 9-point professional analysis

**Analysis Categories Generated**:
1. Common Identity - What is the primary subject?
2. Visual Summary - Key visual elements
3. Operational Utility - What is its purpose/use?
4. Provenance & Setting - Where/when was this taken?
5. Technical Nomenclature - Technical classification
6. Safety & Risk Assessment - Potential hazards
7. Maintenance & Longevity - Care requirements
8. Aesthetic & Design Style - Visual style/design
9. Interaction & Relationship - Context and relationships

**Synthesis Engine**:
- Combines VQA answers into narrative report
- Professional language generation
- Structured format for PDF export
- CSV export with all categories

### Federated Learning Component 📡

**Component Display**:
```
[Federated Learning]
LoRA Adapters
Encrypted synchronization
```

**Overview**:
- Distributed training across edge nodes
- LoRA (Low-Rank Adaptation) for efficient updates
- Encrypted gradient exchange
- Coordinated by admin system

---

## System Health & Status

The **System Health Status** section displays real-time status of critical components:

### 1. BLIP-VQA Model Status 📄

```
BLIP-VQA Model
Status Badge: Ready (green)
├─ Model file: ~350MB
└─ Status: ✓ Loaded
```

**What This Means**:
- ✅ **Green "Ready"**: Model is loaded in memory and ready for inference
- ⚠️ **Yellow "Loading"**: Model is being loaded, brief delay expected
- ❌ **Red "Failed"**: Model failed to load, VLM analysis unavailable

**Admin Actions If Red**:
1. Check server logs for model loading errors
2. Verify disk space for model weights
3. Restart Flask server with `python run_web.py`
4. Check GPU/memory availability

**Performance Impact**:
- If BLIP-VQA fails: Triple-Layer mode becomes unavailable
- Clients can still use CNN-only classification
- VLM-based analysis features disabled

### 2. MobileNetV2 Model Status 🔧

```
MobileNetV2 Model
Status Badge: Ready (green)
├─ Model file: ~100MB
└─ Status: ✓ Loaded
```

**What This Means**:
- ✅ **Green "Ready"**: CNN model loaded, Stage 1 operational
- ⚠️ **Yellow "Loading"**: Initial load in progress
- ❌ **Red "Failed"**: CNN unavailable, no predictions possible

**Admin Actions If Red**:
1. Check PyTorch installation
2. Verify ImageNet-1K weights are downloaded
3. Check system has minimum 2GB free RAM
4. Restart Flask server

**Critical Note**:
- If MobileNetV2 fails, the entire system is non-functional
- This is the foundational Stage 1 model
- Should be first item checked in any troubleshooting

### 3. System Resources 🖥️

```
System Resources
Status Badge: Optimal (green)
├─ RAM Used: X.X MB / 8GB
└─ Inference Speed: X ms avg
```

**RAM Usage Monitoring**:
- **Green (< 6GB used)**: Excellent, room for operations
- **Yellow (6-7GB used)**: Good, but approaching limit
- **Red (> 7.5GB used)**: Concerning, may cause slowdowns
- **Critical (> 7.8GB used)**: System at risk of OOM errors

**Inference Speed Baseline**:
- **45-50ms**: Excellent performance
- **50-60ms**: Good performance, normal variations
- **60-80ms**: Acceptable, watch for degradation
- **> 80ms**: Investigate for system issues

**Typical Resource Allocation**:
- BLIP-VQA Model: ~350MB (loaded once)
- MobileNetV2 Model: ~100MB (loaded once)
- Flask Framework + Database: ~200-300MB
- Operating System: ~500MB-1GB
- Inference Working Memory: ~2-3GB per request

**What to Do If Resources Are Low**:
1. Restart Flask server to free memory
2. Close unnecessary applications on server
3. Check for memory leaks in logs
4. Consider increasing system RAM
5. Reduce concurrent users temporarily

---

## Federated Learning Coordination

### Overview

Federated Learning (FL) enables collaborative model training across distributed edge nodes without centralizing sensitive data.

### How It Works

```
Round N: Coordinator initiates
         ↓
Edge Nodes: Download current model weights
         ↓
Edge Nodes: Train locally on private data
         ↓
Edge Nodes: Compute local gradients
         ↓
Edge Nodes: Send encrypted updates to server
         ↓
Server: Aggregate updates from all nodes
         ↓
Server: Update global model with aggregated gradients
         ↓
Round N+1: Repeat
```

### Admin's Role in FL

**Monitoring**:
- Track FL round status (displayed in stats widget as "Round X")
- Review training progress in performance metrics chart
- Monitor node participation rates
- Check for stragglers or disconnected nodes

**Coordination**:
- Initiate training rounds via `/api/metrics` endpoint
- Monitor model convergence
- Approve model updates before deployment
- Handle node synchronization issues

**Configuration**:
- Set aggregation strategy (FedAvg, weighted average)
- Define privacy parameters (differential privacy epsilon)
- Configure gradient clipping thresholds
- Set node participation requirements

### Monitoring FL Progress

**Key Metrics for FL**:
1. **Round Number**: Total completed rounds (shown in "FL Round Status")
2. **Accuracy Trend**: From performance metrics chart
3. **Loss Trend**: From performance metrics chart
4. **Node Participation**: Count of active nodes contributing
5. **Synchronization Time**: Time to complete aggregation

**Expected Behavior**:
- Accuracy should gradually increase or plateau
- Loss should decrease over rounds
- Each round typically takes 5-15 minutes
- All nodes should participate consistently

**Warning Signs**:
- Accuracy plateaus or decreases (model degradation)
- Some nodes missing from multiple rounds (connectivity issue)
- Synchronization time increasing (aggregation bottleneck)
- High variance in node gradients (data heterogeneity)

### LoRA Adapters in FL

The system uses **LoRA (Low-Rank Adaptation)** for efficient federated learning:

**Benefits**:
- Reduces bandwidth by 99% (only adapters sent, not full model)
- Faster convergence
- Lower computational overhead on edge nodes
- Maintains model quality

**What Gets Synchronized**:
- Only low-rank adapter weights (~1MB vs 350MB for full model)
- Gradient updates for LoRA matrices
- Metadata (round number, node ID, timestamp)
- Encryption wrappers and signatures

**Admin Monitoring**:
- Adapter sync status in live logs
- Bandwidth usage in system resources
- Convergence speed vs full model fine-tuning

---

## Model Management

### Model Status Dashboard

The "System Health Status" section provides a unified view of all critical models:

**Models Displayed**:
1. **BLIP-VQA Model** (350MB) - Stage 2/3 VLM
2. **MobileNetV2 Model** (100MB) - Stage 1 CNN
3. **SCL Module** (embedded in BLIP-VQA) - Stage 1.5
4. **LoRA Adapters** (< 1MB) - Federated Learning

### Model Loading Process

When the Flask server starts (`python run_web.py`):

1. **MobileNetV2 loads first** (100MB, ~2 seconds)
   - ImageNet-1K weights downloaded from PyTorch hub
   - Moved to CPU/GPU based on availability
   - Validated with test batch

2. **BLIP-VQA loads second** (350MB, ~5-10 seconds)
   - Downloaded from Hugging Face hub
   - Model quantization for memory efficiency
   - Feature extractor and decoder initialized

3. **LoRA Adapters initialized** (< 1MB, instant)
   - Configuration loaded from disk
   - Frozen with base model
   - Ready for federated learning

4. **SCL module activates** (no additional weight)
   - Uses BLIP-VQA's interrogative prompting
   - Verification logic initialized
   - Routing rules loaded

### Model Troubleshooting

**BLIP-VQA Not Loading**:
```
Error: "Model loading failed for Salesforce/blip-vqa-base"

Solution:
1. Check internet connection (needed for first download)
2. Verify disk space (need 500MB+ free)
3. Check GPU drivers if using CUDA
4. Try: pip install --upgrade transformers
5. Restart Flask: python run_web.py
```

**MobileNetV2 Not Loading**:
```
Error: "Failed to load ImageNet-1K weights"

Solution:
1. Check PyTorch installation: pip show torch
2. Verify ImageNet-1K is properly set up
3. Check /tmp or cache directory isn't full
4. Try: torch.hub.set_dir('/new/cache/path')
5. Restart Flask
```

**Inference Very Slow**:
```
Possible Causes:
1. Model on CPU instead of GPU
2. Not enough RAM available
3. Disk swapping occurring
4. Concurrent requests overloading server

Solutions:
1. Check device in logs: "Model on: cpu/cuda"
2. Close other applications
3. Increase system RAM
4. Add request queuing/rate limiting
```

### Model Updates

**Federated Learning Updates**:
- New versions created after each FL round
- Previous version kept as backup
- Rollback capability available
- Version control through git

**Manual Model Updates**:
1. Download new model weights
2. Validate on test set
3. Backup current version
4. Replace in model directory
5. Restart Flask server
6. Verify in dashboard

---

## Performance Metrics

### Understanding the Metrics

The performance metrics chart tracks two critical dimensions:

#### 1. Hybrid Inference Time (Left Y-Axis)

**What It Measures**: Total time to complete entire Triple-Layer pipeline

**Components**:
- Stage 1 CNN: 100-500ms
- Stage 1.5 SCL: 3-8 seconds (only if needed)
- Stage 2/3 VLM: 5-15 seconds
- **Total**: 9-25 seconds typical

**Factors Affecting Performance**:
- **Image size**: Larger images = longer processing
- **SCL trigger rate**: More corrections = longer pipeline
- **System load**: CPU contention increases time
- **Memory pressure**: Swapping to disk causes slowdowns
- **Model precision**: FP16 vs FP32 affects speed

**Optimization Targets**:
- **Best Case**: < 12 seconds (fast hardware, small images)
- **Target Case**: 15-20 seconds (balanced, medium images)
- **Acceptable**: < 25 seconds (slower hardware)
- **Concerning**: > 30 seconds (investigate)

#### 2. Ensemble Mode Usage (Right Y-Axis)

**What It Measures**: Percentage of predictions using full Triple-Layer pipeline

**Why This Matters**:
- Higher = more comprehensive analysis available to clients
- Lower = clients preferring faster CNN-only mode
- Trend shows system adoption

**Expected Values**:
- Initial deployment: 40-60% adoption
- Mature deployment: 75-95% adoption
- Ideal: 80-90% (balance between speed and quality)

**Improving Adoption**:
1. Optimize inference time (make Hybrid faster)
2. Show ROI of comprehensive analysis (PDFs, reports)
3. Educate clients on SCL benefits
4. Offer priority queuing for Hybrid requests
5. Create comparison reports showing Hybrid vs CNN-only

### Interpreting Performance Trends

**Stable Trend** ✅
```
Inference Time: Consistent 15-20s
Ensemble Usage: Steady 80%
→ System performing well, no action needed
```

**Rising Inference Time** ⚠️
```
Inference Time: Increasing from 15s to 25s
→ Investigate causes:
  1. More SCL corrections being triggered?
  2. System memory pressure?
  3. Model precision changed?
  4. Increased concurrent requests?
```

**Dropping Ensemble Usage** ⚠️
```
Ensemble Usage: Decreasing from 80% to 50%
→ Causes:
  1. Inference time increased (clients switching to faster CNN)
  2. Error reports for Hybrid mode
  3. VLM model issues
→ Optimize performance to recover adoption
```

**Sudden Spike** 🚨
```
Inference Time: Normal at 18s, then spikes to 40s
→ Immediate investigation:
  1. Check system resources (RAM, CPU)
  2. Review live logs for errors
  3. Restart Flask if necessary
  4. Check for concurrent heavy operations
```

### Performance Benchmarking

**Establish Your Baseline**:
1. Let system run for 1 hour after startup
2. Record average inference time and ensemble usage
3. Note system hardware specs
4. This becomes your "normal" reference

**Monitor for Degradation**:
- Any sustained increase > 20% from baseline = investigate
- Track daily/weekly averages
- Create performance reports for stakeholders
- Set automated alerts if > 150% of baseline

---

## Live Log Monitoring

### Live Log Panel

The **Live Log** section displays real-time system events as they occur:

```
┌──────────────────────────────────────────────────┐
│ ⚙️ Live Log                                       │
│ [🟢 Live] (Connecting... / Live / Connection Lost)│
├──────────────────────────────────────────────────┤
│ 14:32:45 Round 5 aggregation started             │
│ 14:32:46 Node 1 connected (gradient received)    │
│ 14:32:47 Node 2 connected (gradient received)    │
│ 14:32:48 Node 3 connected (gradient received)    │
│ 14:32:50 All nodes synchronized. Aggregating...  │
│ 14:32:55 Model updated. Round 5 complete!       │
│ 14:33:01 User request: Triple-Layer analysis    │
│ 14:33:15 Stage 1 CNN: cat (92% confidence)      │
│ 14:33:18 Stage 1.5 SCL: Verified ✓              │
│ 14:33:32 Stage 3 Analysis: Complete             │
│ 14:33:33 PDF exported successfully              │
└──────────────────────────────────────────────────┘
```

### Log Entries

Each log entry contains:
- **Timestamp**: HH:MM:SS when event occurred
- **Event Type**: Category of the event
- **Message**: Description of what happened

### How Live Logs Work

The live logs use **Server-Sent Events (SSE)** for real-time streaming:

1. **Connection**: Dashboard opens persistent connection to `/admin/events`
2. **Streaming**: Server sends events as they occur
3. **Display**: JavaScript appends new entries to the log panel
4. **Scrolling**: Panel auto-scrolls to show latest entries
5. **Limit**: Only last 50 entries kept in memory

### Understanding Log Events

**Federated Learning Events**:
```
Round X aggregation started
    → FL coordinator initiates new round
    
Node N connected (gradient received)
    → Edge node participated, uploaded gradients
    
All nodes synchronized. Aggregating...
    → All nodes have reported, aggregation in progress
    
Model updated. Round X complete!
    → Round finished, new global model deployed
```

**Prediction Events**:
```
User request: [Mode]
    → Client initiated analysis
    
Stage 1 CNN: [class] ([conf]% confidence)
    → CNN classification result
    
Stage 1.5 SCL: Verified ✓
    → SCL confirmed CNN prediction (no correction)
    
Stage 1.5 SCL: Self-Corrected ⚠️
    → SCL detected inconsistency, corrected identity
    
Stage 3 Analysis: Complete
    → VLM finished 9-point analysis
    
[Output type] exported successfully
    → PDF/CSV export completed
```

**Error Events**:
```
ERROR: [Component] failed
    → Model or service failed
    
WARNING: [Issue detected]
    → Non-fatal issue requiring attention
    
NETWORK: Node N disconnected
    → FL node lost connection, may retry
```

### Using Live Logs for Monitoring

**Real-Time Feedback**:
- Watch for error messages immediately
- Track request processing time (start to export complete)
- Monitor FL round progress
- Identify problematic patterns

**Debugging**:
- Trace a specific client request through all stages
- Find timestamp of issue occurrence
- Cross-reference with performance metrics chart
- Identify resource contention

**Capacity Planning**:
- Count requests per minute
- Measure average request duration
- Identify peak usage patterns
- Plan for growth

### Connection Status Indicators

**Connection Status Badges**:

🟢 **Live**: Connected and receiving events
- Dashboard is actively monitoring
- Real-time updates flowing
- Normal operation

🟡 **Connecting...**: Initial connection being established
- Dashboard is setting up SSE connection
- Brief delay before events appear
- Usually resolves within 1-2 seconds

🔴 **Connection Lost**: SSE connection failed
- Dashboard will attempt reconnect every 5 seconds
- Server logs still generate events (just not displayed)
- Refresh page to manually reconnect
- Check browser console for errors

### Troubleshooting Connection Issues

**If you see "Connecting..." for > 10 seconds**:
1. Check browser console (F12 → Console tab)
2. Verify backend server is running
3. Check firewall isn't blocking SSE connections
4. Try refreshing the page
5. Check server logs for `/admin/events` errors

**If you see "Connection Lost"**:
1. Verify Flask server is still running
2. Check network connectivity
3. Look for server errors in logs
4. Restart Flask: `python run_web.py`
5. Close and reopen admin dashboard

---

## System Architecture Visualization

The **Triple-Layer Consistency Pipeline Architecture** section provides a visual overview of the 4-stage system:

### Layout

```
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Stage 1: CNN     │ Stage 1.5: SCL   │ Stage 3: VLM     │ Federated Learn. │
│ Classifier       │ Verification     │ Analysis         │ LoRA Adapters    │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ MobileNetV2      │ Interrogative    │ BLIP-VQA         │ Encrypted sync.  │
│ ImageNet-1K trained│ Verification     │ Salesforce/blip. │ Network wide     │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

Each component includes:
- **Icon**: Visual indicator of stage function
- **Title**: Stage name and function
- **Model**: Which model is used
- **Detail**: Technical specification or status

### Stage-by-Stage Breakdown

**Stage 1 - CNN Classifier** (Blue)
```
┌─────────────────────────────────────┐
│ 🧊 Stage 1: CNN Classifier          │
├─────────────────────────────────────┤
│ MobileNetV2                         │
│ ImageNet-1K trained                   │
└─────────────────────────────────────┘
```
- Quick classification
- Foundation for entire pipeline
- Provides initial hypothesis

**Stage 1.5 - SCL Verification** (Orange)
```
┌─────────────────────────────────────┐
│ 🛡️ Stage 1.5: SCL                  │
├─────────────────────────────────────┤
│ Interrogative                       │
│ Consistency verification            │
└─────────────────────────────────────┘
```
- Autonomous verification (orange background highlights importance)
- Detects errors automatically
- Routes based on consistency

**Stage 3 - VLM Analysis** (Blue)
```
┌─────────────────────────────────────┐
│ ✨ Stage 3: VLM                     │
├─────────────────────────────────────┤
│ BLIP-VQA                            │
│ Salesforce/blip-vqa-base            │
└─────────────────────────────────────┘
```
- Comprehensive analysis
- 9-point discovery framework
- Professional reporting

**Federated Learning** (Blue)
```
┌─────────────────────────────────────┐
│ 🌐 Federated Learning               │
├─────────────────────────────────────┤
│ LoRA Adapters                       │
│ Encrypted synchronization           │
└─────────────────────────────────────┘
```
- Distributed collaborative learning
- Privacy-preserving training
- Network-wide model improvement

---

## Training Round Management

### Understanding FL Rounds

A **training round** is one complete cycle of federated learning:

```
Round X Timeline:
├─ T+0s: Coordinator signals start
├─ T+5s: Nodes download current model
├─ T+60s: Nodes train locally (varies)
├─ T+120s: Nodes send encrypted updates
├─ T+150s: Server aggregates gradients
├─ T+155s: New global model ready
└─ Total: ~155 seconds (≈2.5 minutes per round)
```

### Monitoring Rounds

**Round Status Widget** shows:
- Current round number (e.g., "Round 5")
- Updates every time aggregation completes
- Indicates system is actively learning

**Performance Chart** shows:
- Accuracy trend across rounds (left axis)
- Loss trend across rounds (left axis)
- Should show improvement over time

**Expected Behavior**:
- Accuracy increases 1-5% per round (initially)
- Loss decreases 10-20% per round (initially)
- Plateau after 20-50 rounds (convergence)
- Slower improvement at higher accuracy

### When to Intervene

**Stop Training If**:
- Accuracy starts decreasing for 3+ rounds
- Loss increases significantly
- Model diverging (not converging)
- Hardware failure detected
- Memory issues detected

**Adjust Training If**:
- Convergence too slow → increase learning rate or node count
- Oscillating accuracy → decrease learning rate
- Nodes not participating → check connectivity
- System unstable → reduce batch sizes

### Round Completion Checklist

After each FL round completes:

- [ ] Check all expected nodes participated
- [ ] Verify accuracy/loss improved
- [ ] Confirm new model loaded
- [ ] Monitor inference performance (may improve)
- [ ] Check for any errors in logs
- [ ] Review system resources (memory, CPU)

---

## User & Node Management

### User Roles

The system supports two user types:

**Admin Users** (You)
```
├─ Access: /admin/dashboard
├─ View: System metrics, real-time logs, FL progress
├─ Can: Monitor system, manage configuration
├─ Cannot: Run predictions directly (use /predict for that)
└─ Default: admin / admin123
```

**Client Users**
```
├─ Access: /client/dashboard, /predict
├─ View: Edge node status, analysis results
├─ Can: Run predictions, download reports, join FL
├─ Cannot: Access admin dashboard or system metrics
└─ Default: client / client123
```

### User Management

**Viewing Active Users**:
- Check database directly (users table)
- Count from recent prediction logs
- Monitor concurrent sessions in logs

**Adding Users** (Backend only):
```python
from frontend_web.models import User
new_user = User(username="newadmin", role="admin")
new_user.set_password("password123")
db.session.add(new_user)
db.session.commit()
```

**Changing Passwords**:
1. Access database: `sqlite3 data/app.db`
2. Update user password (requires hash)
3. Or modify directly in code and restart

**Deleting Users**:
- Remove from users table in database
- Session data automatically cleaned up after timeout

### Edge Nodes (Federated Learning)

**Edge Node** = A client device participating in federated learning

**Node Registration**:
1. Client logs in with account
2. Configures edge node settings
3. Joins FL round automatically
4. Appears in aggregation

**Node Status**:
- **Connected**: Actively participating
- **Training**: Currently training locally
- **Syncing**: Uploading gradients
- **Idle**: Waiting for next round
- **Disconnected**: Lost connection (will retry)

**Monitoring Nodes**:
- Count nodes in FL round from logs
- Track which nodes are consistently participating
- Identify stragglers or problematic nodes

**Node Issues**:
- **Node drops out**: May rejoin in next round
- **Network delay**: Slow uploads increase round time
- **Insufficient data**: Node produces weak gradients
- **Hardware failure**: Node becomes unavailable

---

## System Configuration

### Key Configuration Files

**Flask Configuration** (`frontend_web/config.py`):
```python
DEBUG = False              # Production: always False
SECRET_KEY = 'your-secret' # Change in production!
DATABASE_PATH = 'data/app.db'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB file upload limit
```

**Model Configuration** (`backend_fl/config.json`):
```json
{
  "cnn_model": "mobilenetv2",
  "cnn_dataset": "cifar100",
  "vlm_model": "Salesforce/blip-vqa-base",
  "scl_mode": "interrogative",
  "device": "cpu",
  "precision": "fp32"
}
```

**Flask Server** (`run_web.py`):
```python
app.run(host='0.0.0.0', port=5000, debug=False)
```

### Important Settings to Change

**Before Production Deployment**:

1. **Change SECRET_KEY**:
   ```python
   import secrets
   secrets.token_hex(32)  # Generate new key
   ```

2. **Change Default Passwords**:
   - Update admin user password in database
   - Send secure credentials to admins
   - Remove default client account

3. **Enable HTTPS**:
   - Get SSL certificate
   - Configure Flask for HTTPS
   - Update allowed hosts

4. **Database Security**:
   - Use PostgreSQL instead of SQLite
   - Enable database backups
   - Restrict database access

5. **API Authentication**:
   - Implement API keys for `/api/metrics`
   - Add rate limiting
   - Enable CORS properly

### Optimization Settings

**For Better Performance**:

1. **Model Precision**:
   ```python
   # In vlm_model.py
   model = AutoModelForVQA.from_pretrained(..., torch_dtype=torch.float16)
   ```

2. **Batch Processing**:
   - Enable request batching
   - Queue predictions during peak hours

3. **Caching**:
   - Cache model outputs
   - Cache frequently requested results

4. **Resource Limits**:
   - Set max concurrent requests
   - Set timeout for long predictions
   - Monitor and kill stuck requests

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Dashboard Shows "Connection Lost"

**Symptoms**:
- Red indicator next to Live Log
- No new log entries appearing
- Dashboard doesn't refresh

**Root Causes**:
1. Flask server crashed or stopped
2. Network connectivity issue
3. Browser security policy blocking SSE
4. Firewall blocking connections

**Solutions**:
```bash
# Check if server is running
lsof -i :5000  # On Linux/Mac
netstat -ano | findstr :5000  # On Windows

# Restart server if needed
cd C:\Users\rayan\Downloads\Image\ CLassification
python run_web.py

# Clear browser cache and refresh
# Ctrl+Shift+Delete → Clear all → Refresh page

# Check browser console for errors
# F12 → Console tab → Look for CSP errors
```

#### Issue 2: Inference Time Exceeds 25 Seconds

**Symptoms**:
- "Avg. Inference Time" widget shows 30-40s+
- Client complaints about slow predictions
- Ensemble usage dropping

**Root Causes**:
1. System out of RAM (disk swapping)
2. Model on CPU instead of GPU
3. Too many concurrent predictions
4. Large input images

**Diagnosis**:
```bash
# Check RAM usage
free -h  # Linux
Get-WmiObject Win32_OperatingSystem  # PowerShell

# Check CPU usage
top  # Linux
tasklist /v  # Windows

# Check model device in logs
# Should say "Model on: cpu" or "Model on: cuda"
```

**Solutions**:
1. **Close other applications** on server
2. **Restart Flask** to free memory
3. **Reduce image preprocessing** if applicable
4. **Add more RAM** to system (upgrade)
5. **Use GPU** if available (configure CUDA)

#### Issue 3: Models Won't Load

**Symptoms**:
- Dashboard shows "Failed" status for models
- Predictions error out
- Server logs full of loading errors

**Root Causes**:
1. Disk space full
2. Internet down (can't download models)
3. Corrupted model cache
4. Wrong Python/PyTorch version

**Solutions**:
```bash
# Check disk space
df -h  # Linux
wmic logicaldisk get name,size,freespace  # Windows

# Clear model cache
rm -rf ~/.cache/huggingface/  # Linux
rmdir %USERPROFILE%\.cache\huggingface /s  # Windows

# Upgrade PyTorch
pip install --upgrade torch transformers

# Download models manually with internet
python -c "from transformers import AutoModelForVQA; AutoModelForVQA.from_pretrained('Salesforce/blip-vqa-base')"

# Restart server
python run_web.py
```

#### Issue 4: FL Nodes Not Participating

**Symptoms**:
- "FL Round Status" stays at same number for long time
- Logs show only 1-2 nodes, missing others
- Round aggregation fails or takes very long

**Root Causes**:
1. Nodes disconnected from network
2. Node crashes or restarts
3. Server not accepting connections
4. Firewall blocking node connections

**Solutions**:
```bash
# Check server connectivity
netstat -tln | grep 5000  # Check listening

# Monitor network for disconnects
tcpdump -i any -n 'port 5000'  # Real-time connections

# Check server logs for connection errors
tail -f server.log  # Watch live logs

# Restart FL coordinator
python run_web.py  # Restarts all systems
```

#### Issue 5: SCL Correction Rate Unusually High/Low

**Symptoms**:
- "SCL Self-Correction Rate" > 30% (too high)
- "SCL Self-Correction Rate" < 2% (too low)
- Inconsistent from day to day

**Root Causes**:
1. CNN model degradation (over-fitting, drift)
2. VLM model differences from CNN training data
3. Data distribution changes
4. SCL sensitivity incorrectly configured

**Solutions**:
1. **High correction rate (> 30%)**:
   - Review CNN accuracy on test set
   - Check for training data drift
   - Consider retraining CNN
   - Analyze which classes trigger corrections

2. **Low correction rate (< 2%)**:
   - May indicate perfect CNN performance (unlikely)
   - Check if SCL is actually running
   - Verify BLIP-VQA model is loaded
   - Test with edge cases manually

### Debugging Techniques

**Enable Verbose Logging**:
```python
# In frontend_web/app.py
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add detailed logging to routes
logger.debug(f"Request received: {request.json}")
logger.debug(f"Prediction: {result}")
```

**Check System Logs**:
```bash
# Flask server logs (terminal output)
# Should show all requests and model loads

# Database queries
sqlite3 data/app.db .log stdout

# System resource usage (real-time)
watch -n 1 free -h  # RAM every second
watch -n 1 'nvidia-smi'  # GPU every second
```

**Test Components Individually**:
```python
# Test CNN directly
from backend_fl.cnn_model import load_cnn_model
model = load_cnn_model()
output = model(test_image)
print(f"CNN prediction: {output}")

# Test BLIP-VQA
from backend_fl.vlm_model import load_vlm_model
model = load_vlm_model()
answer = model.vqa(image, question="What is this?")
print(f"VQA answer: {answer}")

# Test SCL verification
from backend_fl.scl_verification import verify_prediction
verified = verify_prediction(image, cnn_class, cnn_confidence)
print(f"SCL verification: {verified}")
```

---

## Best Practices

### Daily Admin Checklist

**Morning (Start of Day)**:
- [ ] Check admin dashboard loads properly
- [ ] Verify all models show "Ready" status
- [ ] Review overnight logs for errors
- [ ] Check system resources (RAM, disk)
- [ ] Confirm database backup completed

**During Business Hours**:
- [ ] Monitor inference times (should stay consistent)
- [ ] Watch ensemble usage rate (should be 75-95%)
- [ ] Check error rate in logs (should be < 1%)
- [ ] Monitor SCL correction rate (5-15% normal)
- [ ] Track user count and active sessions

**End of Day**:
- [ ] Review performance metrics for trends
- [ ] Document any issues encountered
- [ ] Back up database and logs
- [ ] Verify overnight backups are scheduled
- [ ] Review federated learning progress

### Performance Optimization

**For Faster Inference**:
1. Use smaller input images (128x128 instead of 512x512)
2. Enable model quantization (FP16 instead of FP32)
3. Use GPU if available (CUDA-capable GPU)
4. Batch requests during peak hours
5. Cache repeated predictions

**For Better Accuracy**:
1. Monitor SCL correction patterns
2. Retrain CNN if drift detected
3. Improve federated learning rounds
4. Collect user feedback on results
5. A/B test model updates before full deployment

**For Reliability**:
1. Implement health checks every 30 seconds
2. Set up automated alerts for failures
3. Create runbooks for common failures
4. Test disaster recovery procedures
5. Maintain system redundancy

### Data Privacy & Security

**Federated Learning Privacy**:
- Gradients encrypted before transmission
- No raw data sent to central server
- Differential privacy can be applied
- Verify LoRA adapter encryption working

**Admin Panel Security**:
- Use strong admin password (20+ characters)
- Change default credentials immediately
- Enable HTTPS in production
- Restrict admin access by IP if possible
- Log all admin actions

**User Data**:
- Implement data retention policies
- Purge old prediction history
- Anonymize user analytics
- Respect privacy requests
- Encrypt database at rest

### Capacity Planning

**Monitor These Metrics**:
- Predictions per hour (should be tracked in logs)
- Average inference time (should be < 25 seconds)
- System resource usage (RAM, CPU, disk)
- Error rate (should be < 1%)
- User count growth

**When to Upgrade**:
- **RAM**: If consistently > 7GB used → add more
- **CPU**: If load average > 2.0 → faster processor
- **Disk**: If < 20% free → expand storage
- **Bandwidth**: If users report slowness → check network
- **Models**: If inference slow → consider GPU

---

## Advanced Monitoring

### Setting Up Alerts

**Alert Conditions** (examples):
```
IF inference_time > 30 seconds THEN send_alert
IF system_memory > 7.5 GB THEN send_alert
IF error_rate > 5% in 5 minutes THEN send_alert
IF gpu_memory > 80% THEN send_alert
IF no_nodes_for 10 minutes THEN send_alert
```

**Alert Channels**:
- Email notifications
- Slack/Discord webhooks
- SMS for critical issues
- Dashboard notifications
- Log file alerts

### Metrics to Export

**For External Monitoring** (Prometheus, Grafana, etc.):

```
# Inference metrics
inference_time_ms{mode="hybrid"} 18.5
inference_count_total{mode="cnn"} 142
inference_errors_total 2

# Model metrics
model_load_success{model="blip-vqa"} 1
model_load_success{model="mobilenetv2"} 1
model_memory_bytes{model="blip-vqa"} 367001600

# FL metrics
fl_round_number 5
fl_nodes_participating 3
fl_aggregation_time_seconds 15.2

# System metrics
system_memory_bytes 8589934592
system_memory_used_bytes 6442450944
system_cpu_count 8
system_cpu_percent 45.2
```

### Creating Performance Reports

**Weekly Report Should Include**:
1. Total predictions processed
2. Average inference time
3. Ensemble mode adoption rate
4. Error rate (with breakdown)
5. FL round progress
6. Resource utilization
7. Notable events/issues
8. Performance trends

**Monthly Report Should Include**:
1. System uptime percentage
2. Performance trends (improving/degrading)
3. User growth metrics
4. Model accuracy improvements
5. Recommendations for optimization
6. Capacity planning analysis
7. Cost analysis
8. ROI on system investment

---

## Getting Help

### Viewing Server Logs

**Real-Time Log Monitoring**:
```bash
# On Linux/Mac
tail -f server.log

# On Windows PowerShell
Get-Content server.log -Wait

# Watch with search filter
tail -f server.log | grep ERROR
```

**Log Locations**:
```
Flask output: Terminal/console where you ran python run_web.py
Application logs: logs/app.log (if configured)
Database logs: logs/database.log (if enabled)
System logs: System event viewer (Windows) or /var/log (Linux)
```

### Accessing the Database Directly

**View Current Data**:
```bash
# Open database
sqlite3 data/app.db

# View tables
.tables

# View users
SELECT id, username, role FROM user;

# View recent predictions
SELECT id, username, created_at, model_used FROM predictions ORDER BY created_at DESC LIMIT 10;

# Count statistics
SELECT COUNT(*) as total_predictions FROM predictions;
```

### Testing Components

**Test CNN Model**:
```python
python -c "
from backend_fl.cnn_model import load_cnn_model
model = load_cnn_model()
print('CNN Model loaded successfully')
"
```

**Test BLIP-VQA Model**:
```python
python -c "
from backend_fl.vlm_model import load_vlm_model
model = load_vlm_model()
print('BLIP-VQA Model loaded successfully')
"
```

**Test Full Pipeline**:
```bash
# Start Flask server
python run_web.py

# In another terminal, test an endpoint
curl -X POST http://localhost:5000/api/metrics -H "Content-Type: application/json"
```

### Useful Commands

**Check Port is Available**:
```bash
# Linux/Mac
lsof -i :5000

# Windows
netstat -ano | findstr :5000
```

**Restart Services**:
```bash
# Kill Flask server
# Find PID from above commands, then:
kill -9 PID  # Linux/Mac
taskkill /PID PID /F  # Windows

# Restart
python run_web.py
```

**View System Info**:
```bash
# Linux/Mac
uname -a
free -h
df -h

# Windows
systeminfo
wmic os get totalvisiblememorysize,freephysicalmemory
diskpart
```

### Support Resources

**For Technical Issues**:
1. Check troubleshooting section above
2. Review live logs in dashboard
3. Check application logs in /logs directory
4. Enable debug logging for detailed info
5. Test components individually

**For Hardware Issues**:
1. Check system resources
2. Verify adequate RAM/disk space
3. Check network connectivity
4. Verify GPU drivers if using GPU
5. Test with minimal load (single request)

**For Model Issues**:
1. Verify models are properly downloaded
2. Check model cache directory
3. Verify PyTorch/transformers versions
4. Test model loading directly (see Testing Components)
5. Try clearing cache and redownloading

**For Federated Learning Issues**:
1. Verify nodes have network connectivity
2. Check firewall allows port 5000
3. Monitor aggregation logs
4. Verify all nodes have same model version
5. Check gradient synchronization in logs

---

## Quick Reference

### Key Routes

| Route | Purpose | Access |
|-------|---------|--------|
| `/admin/dashboard` | Main monitoring dashboard | Admin only |
| `/admin/events` | Live event stream (SSE) | Admin only |
| `/api/metrics` | Performance metrics data | Admin only |
| `/api/status` | System health status | Admin only |
| `/predict` | Client prediction interface | Client + Admin |
| `/results` | Prediction results display | Client + Admin |
| `/login` | Authentication | All |
| `/logout` | End session | All |

### Important Files

| File | Purpose | Edit? |
|------|---------|-------|
| `frontend_web/app.py` | Flask application | Careful |
| `frontend_web/auth.py` | Authentication logic | Careful |
| `frontend_web/models.py` | Database models | Careful |
| `backend_fl/vlm_model.py` | VLM and SCL logic | Careful |
| `backend_fl/cnn_model.py` | CNN inference | Careful |
| `data/app.db` | SQLite database | Backup first |
| `config.json` | System configuration | Backup first |

### Default Credentials

```
Admin:
  Username: admin
  Password: admin123

Client:
  Username: client
  Password: client123

⚠️ CHANGE THESE IN PRODUCTION!
```

### System Requirements

```
Minimum:
- 8GB RAM (4GB minimum for models)
- 500MB free disk space
- CPU: Intel i5 or equivalent
- Network: 10Mbps for FL (varies by participation)

Recommended:
- 16GB+ RAM
- 1GB free disk space
- GPU: NVIDIA with CUDA support (optional, 2x speedup)
- Network: 100Mbps+ for FL

Tested On:
- Windows 10/11 with 8GB RAM, CPU inference
- Linux (Ubuntu 20.04+)
- macOS 12+
```

---

## Conclusion

You now have comprehensive knowledge to effectively administer the DecentralizedAI federated learning system. The Triple-Layer Consistency Pipeline with autonomous error correction provides robust image analysis with built-in quality control.

**Key Takeaways**:

1. **Monitor First**: Always check the dashboard before responding to issues
2. **Understand Pipeline**: Know how Stage 1→1.5→2/3 flows and why each stage matters
3. **Watch Metrics**: Inference time and ensemble usage tell you system health
4. **Trust Logs**: Live logs show exactly what's happening in real-time
5. **Plan Ahead**: Track capacity needs before hitting limits
6. **Secure System**: Change defaults, use HTTPS, protect credentials

**Regular Maintenance**:
- Daily: Check dashboard, review logs, verify models loaded
- Weekly: Review performance trends, backup database
- Monthly: Plan capacity, create reports, update documentation

For questions or issues not covered here, refer to the troubleshooting section, check application logs, or consult the live dashboard logs for specific error messages.

---

**Document Version**: 1.0  
**Last Updated**: March 4, 2025  
**System**: DecentralizedAI with Triple-Layer Consistency Pipeline
