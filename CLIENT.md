# Client User Guide - Edge Discovery Node

**Version:** 3.2 (Triple-Layer Consistency Pipeline)  
**Status:** Production-Ready  
**Last Updated:** March 4, 2026

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Client Interface Components](#client-interface-components)
- [Triple-Layer Prediction Analysis](#triple-layer-prediction-analysis)
- [Features & Capabilities](#features--capabilities)
- [Privacy & Security](#privacy--security)
- [Federated Learning Participation](#federated-learning-participation)
- [Troubleshooting](#troubleshooting)

---

## Overview

The **Client Node** is a privacy-preserving edge discovery interface designed for individual users, researchers, and organizations to perform advanced image analysis using the Triple-Layer Consistency Pipeline without compromising data privacy.

### Key Characteristics

- **Privacy-First**: Images are processed locally; no raw data transmission
- **Advanced AI**: Access to Triple-Layer architecture (CNN + SCL + VLM) for professional-grade analysis
- **User-Friendly**: Intuitive interface for image upload and insight discovery
- **Collaborative**: Optional participation in federated learning to improve shared models
- **Real-Time**: Instant 9-point analysis with autonomous error correction

### Target Users

- Researchers performing image classification and analysis
- Organizations needing privacy-preserving AI analysis
- Academic institutions exploring federated learning
- Privacy-conscious users wanting advanced AI without data collection

---

## Getting Started

### 1. Access the Client Interface

Navigate to: **http://localhost:5000**

### 2. Authentication

**Default Client Credentials** (for development/testing):
- **Username:** `client`
- **Password:** `client123`

**For Production:**
- Create your own account via the registration page
- Use strong passwords
- Keep credentials secure

### 3. Login Flow

1. Visit the homepage
2. Click "Get Started" or navigate to `/login`
3. Enter username and password
4. Click "Sign In"
5. You'll be redirected to the prediction interface

---

## Client Interface Components

### A. Navigation Bar

Located at the top of every page, provides quick access to:

| Component | Function | Availability |
|-----------|----------|---------------|
| **Home** | Return to homepage | Always |
| **Discover Insights** | Go to prediction page | After login |
| **My Node** | Access client dashboard | Client users only |
| **Recent History** | View past predictions | After login |
| **Account Menu** | Profile settings, logout | After login |

### B. Prediction Page Interface

#### 1. **Mode Selection Section** (Top of Page)

Three prediction modes available:

##### **ImageNet-1K Classification with Nuclear Truth Protocol**
- **Icon:** 🏷️ (Classification tag)
- **Description:** Standard CNN-only classification using MobileNetV2
- **Time:** ~0.1-0.5 seconds
- **Output:** Single class prediction with confidence percentage
- **Use Case:** Quick identification without detailed analysis
- **Color:** Light blue accent

##### **Vision-Language Model (VLM)**
- **Icon:** ✨ (Sparkles/magic)
- **Description:** Pure multimodal analysis using BLIP-VQA
- **Time:** ~5-15 seconds
- **Output:** 5-point discovery analysis
- **Use Case:** Detailed insights without CNN guidance
- **Color:** Purple/magenta accent

##### **Triple-Layer Hybrid Ensemble** ⭐ (RECOMMENDED)
- **Icon:** 🧠 (Brain - intelligence)
- **Description:** Full Triple-Layer pipeline with autonomous error correction
- **Time:** ~9-25 seconds
- **Output:** 9-point comprehensive analysis with SCL verification
- **Features:**
  - CNN rapid classification (Stage 1)
  - Semantic Consistency Layer verification (Stage 1.5)
  - VLM 9-point analysis (Stage 2/3)
  - Automated error correction
  - Professional PDF export
- **Use Case:** Professional intelligence reports, maximum accuracy
- **Color:** Cyan/neon blue (highlighted)

**Selection Method:**
1. Click on the desired mode card
2. Card highlights and becomes active
3. Confirm mode before uploading image

#### 2. **Image Upload Section**

**Drag & Drop Area:**
- Large interactive zone in the center
- Visual feedback on hover
- Supports drag-and-drop functionality

**Upload Methods:**
1. **Drag & Drop**: Drag image file directly onto the box
2. **Click to Browse**: Click upload area to open file dialog
3. **Paste from Clipboard**: Press Ctrl+V to paste image

**Supported Formats:**
- `.jpg`, `.jpeg` - JPEG images
- `.png` - PNG images (recommended for quality)
- `.gif` - GIF images (uses first frame)
- `.bmp` - Bitmap images
- `.webp` - WebP modern format

**File Constraints:**
- **Maximum Size:** 50 MB
- **Recommended Size:** 100 KB - 5 MB (optimal performance)
- **Image Dimensions:** 224x224 to 512x512 recommended
- **Minimum Dimension:** 64x64 pixels

**Upload Feedback:**
- ✓ Green checkmark: File accepted
- ⚠️ Yellow warning: File size warning
- ❌ Red error: File rejected (format, size, or integrity)

#### 3. **Analysis Controls**

**Analyze Button**
- Large prominent button below upload area
- Text: "Analyze Image" or "Predict & Synthesize"
- Becomes active after image selection
- Disabled until valid image uploaded

**Recent Images Carousel** (Optional)
- Shows thumbnails of recent predictions
- Quick re-run previous analyses
- Saves upload time for batch operations

#### 4. **Processing Indicator**

**During Analysis:**
- Animated loading spinner with rotating animation
- "Analyzing image with Triple-Layer Pipeline..."
- Progress message shows current stage:
  - "Stage 1: CNN Classification..."
  - "Stage 1.5: SCL Verification..."
  - "Stage 2/3: VLM Analysis..."
  - "Generating Report..."
- Estimated time remaining (if known)

**Processing Time Expectations:**
- ImageNet-1K: 0.5 - 1 second
- VLM: 5 - 15 seconds
- **Hybrid (Recommended): 9 - 25 seconds**

---

### C. Results Page Interface

After analysis completes, you'll see the **Results Page** with multiple components:

#### 1. **Master Audit Report (10-Card Navigation System)**

A professional 10-card flashcard interface showing all analysis results:

**Cards 1-9: Discovery Categories**

| Card | Category | Content |
|------|----------|---------|
| 1 | **Common Identity** | What the object is; primary classification and purpose |
| 2 | **Visual Summary** | Physical appearance, colors, textures, visual characteristics |
| 3 | **Operational Utility** | Functional purpose, practical applications, use cases |
| 4 | **Provenance & Setting** | Origin, environment, geographic context, typical locations |
| 5 | **Technical Nomenclature** | Scientific names, classification, technical terminology |
| 6 | **Safety & Risk Assessment** | Potential hazards, safety considerations, handling precautions |
| 7 | **Maintenance & Longevity** | Care requirements, durability, preservation methods |
| 8 | **Aesthetic & Design** | Artistic qualities, design principles, visual appeal |
| 9 | **Interaction & Relationships** | Human-object interactions, social context, cultural significance |

**Card 10: Master Audit Report**
- Complete 9-point analysis grid summary
- SCL verification badge
- Corrected identification (if SCL detected error)
- Professional conclusion
- PDF download option

**Navigation Controls:**

- **Navigation Dots** (Bottom): 10 circular dots representing each card
  - Click any dot to jump to that card
  - Active dot highlighted in cyan
  - Hover shows card name tooltip

- **Arrow Buttons** (Sides):
  - Left arrow ← : Previous card
  - Right arrow → : Next card
  - Disabled at boundaries

- **Keyboard Navigation:**
  - Left Arrow Key: Previous card
  - Right Arrow Key: Next card
  - Home Key: First card
  - End Key: Last card
  - Number Keys (1-9): Jump to specific card

#### 2. **Stage 1: CNN Classification Result**

**Display Box** (Top-right corner):
- **Bold Cyan Border:** Neon 2px border for emphasis
- **Content:**
  - 🔍 **Predicted Class:** Object identification (e.g., "Rose", "Cat", "Laptop")
  - 📊 **Confidence:** Percentage (0-100%) with confidence meter
  - 🎯 **Routing Mode:** "Standard" or "High-Confidence Context"

**Appearance:**
- Background: Subtle gradient (Deep Ocean theme)
- Text: Large, bold, readable
- Update dynamically as analysis progresses

#### 3. **Stage 1.5: SCL Verification Badge** ⭐ NEW

**Prominent Status Display** (Middle section):
- **Color-Coded Badge:**
  - 🟢 **Green (Verified):** SCL confirmed CNN prediction is accurate
  - 🟠 **Orange (Self-Corrected):** SCL detected discrepancy; VLM provided correction

**Content:**
- **SCL Status:** "Verified" or "Self-Corrected"
- **Interrogative Question:** The specific question BLIP-VQA used to verify
- **VLM Response:** The Vision-Language Model's answer to the question
- **Explanation:** Brief description of what SCL determined

**Example (Verified):**
```
Status: ✓ Verified (Green)
Question: "Is this object a Rose?"
Response: "Yes, this is clearly a rose with distinct petal structure and stem"
Explanation: CNN prediction confirmed through semantic analysis
```

**Example (Self-Corrected):**
```
Status: ⚠ Self-Corrected (Orange)
Question: "Is this object a car?"
Response: "This appears to be a red bus, not a car"
Explanation: CNN predicted 'car' but VLM identified 'bus'. Using corrected identification.
```

#### 4. **Analysis Cards Grid** (Hybrid Mode)

Below the navigation dots, displays current card content:

**Card Layout:**
- Card Title: Large, bold, cyan-colored
- Card Content: Full professional narrative (2-3 paragraphs)
- Subtitle: Smaller descriptive text above title
- Border: Bottom accent line in cyan

**Text Styling:**
- Font: Georgia Serif (professional, elegant)
- Size: 1rem (readable)
- Color: Primary text color (#0f172a dark theme)
- Line Height: 1.6 (comfortable reading)

**Content Example (Common Identity card):**
```
Common Identity

A botanical specimen displaying characteristics of cultivated Rosa species with 
structured petal arrangement and established vascular framework. The specimen exhibits 
the distinctive morphological attributes consistent with ornamental rose varieties, 
featuring prominent sepals and a composite flower structure typical of Rosaceae family 
classification.
```

#### 5. **Confidence & Status Indicators**

**CNN Confidence Meter:**
- Horizontal bar graph
- 0% (Red) ← → 100% (Green)
- Percentage displayed in center
- Dynamic updates during analysis

**SCL Status Indicator:**
- Icon + Text badge
- Updates after Stage 1.5 completes
- Shows "Checking..." during verification

**Processing Progress:**
- Multi-stage progress indicator
- Shows which stage is active
- Visual completion markers for finished stages

#### 6. **PDF Export Section**

**"Generate Professional Report" Button:**
- Location: Top-right of Master Audit Report (Card 10)
- Styling: Large cyan button with shadow
- Text: "Download PDF Report"
- Icon: 📄 Document symbol

**PDF Report Features:**
- **Format:** A4 Portrait
- **Style:** Deep Ocean aesthetic with neon cyan accents
- **Content:**
  1. Premium header with report title
  2. Timestamp and document ID
  3. Stage 1 CNN identification
  4. Stage 1.5 SCL verification badge
  5. Interrogative check details
  6. 9-point analysis grid
  7. Professional summary conclusion
  8. System intelligence metadata
  9. Footer with document ID format: `HIA-YYYYMMDD-HHMMSS`

**File Output:**
- Filename: `Ensemble_Intelligence_Report_YYYYMMDD_HHMMSS.pdf`
- Size: 500-800 KB (optimized for email)
- Quality: JPEG 0.98 quality, 2x canvas scale

#### 7. **CSV Data Export**

**"Download CSV Audit"** Button:
- Location: Near PDF download button
- Text: "Export as CSV"
- Icon: 📊 Spreadsheet symbol

**CSV Content:**
1. Technical header with timestamp
2. Stage 1 CNN results
3. Stage 1.5 SCL verification data
4. Stage 3 VLM 9-point synthesis (rows)
5. Deep-data export (column format with all 9 categories)
6. System intelligence metadata

**Use Cases:**
- Data analysis and research
- Integration with external tools
- Archival and record-keeping
- Batch processing

#### 8. **Recent Discovery History**

**Gallery View** (Separate page or sidebar):
- Thumbnail grid of recent predictions
- Shows last 20 predictions
- Displays:
  - Image thumbnail
  - CNN prediction
  - Confidence percentage
  - Timestamp
  - SCL status (badge)

**History Interaction:**
- Click thumbnail to view full results
- Hover shows quick info tooltip
- Delete button to remove from history (optional)
- Batch export option

---

### D. Client Dashboard (Node Monitoring)

Accessible via "My Node" or `/client/dashboard`

**Dashboard Components:**

#### 1. **Edge Node Configuration Panel**

**Discovery Batch Selector:**
- Dropdown menu: "Batch 0 - Batch 9"
- Each batch represents 500 images for processing
- Selection isolated to local filesystem
- Privacy notice: "Zero data transmission"

**Control Buttons:**
- 🔵 **Begin Discovery:** Start local processing
- 🔴 **Halt Processing:** Stop current batch
- Visual state indication: Disabled when not applicable

**Privacy Architecture Badge:**
- Locked icon (🔒)
- Text: "Privacy Guard Active"
- Subtitle: "Local data never leaves this node"
- Color: Green (success status)

#### 2. **Metrics & Progress Cards**

Three key metric displays:

| Metric | Shows | Unit |
|--------|-------|------|
| **Processing State** | Current status (Idle/Processing/Complete) | Text |
| **Insights Generated** | Number of analysis results completed | Count |
| **Images Processed** | Total images analyzed in session | Count |

#### 3. **Discovery Processing Progress Panel**

**Progress Bar:**
- Visual percentage indicator (0-100%)
- Cyan color gradient
- Real-time update during processing
- Percentage text badge

**Edge Processing Log:**
- Terminal-style interface
- Monospace font (`JetBrains Mono` or `Courier New`)
- Black background with colored text
- Auto-scrolling output
- Shows:
  - Process startup messages
  - Model loading status
  - Image processing logs
  - Error messages (red text)
  - Success confirmations (green text)

**Log Entry Format:**
```
[HH:MM:SS] Starting edge discovery node 0...
[HH:MM:SS] Loading BLIP-VQA model into local memory...
[HH:MM:SS] Processing image 1/500 (0%)...
[HH:MM:SS] SCL verification in progress...
[HH:MM:SS] Analysis complete ✓
[HH:MM:SS] Processing image 2/500 (0.4%)...
```

---

## Triple-Layer Prediction Analysis

### Understanding the Analysis Pipeline

#### **Stage 1: CNN Classification (MobileNetV2)**
- **Time:** 0.1-0.5 seconds
- **Input:** Image file
- **Process:** 
  - Loads MobileNetV2 model (ImageNet-1K trained)
  - Converts image to 224x224 resolution
  - Runs through neural network layers
  - Outputs class prediction and confidence
- **Output:** 
  - Predicted class (100 ImageNet-1K categories)
  - Confidence score (0-100%)
  - Routing mode determination

#### **Stage 1.5: Semantic Consistency Layer (SCL)** ⭐ NEW
- **Time:** 3-8 seconds
- **Purpose:** Autonomous error detection and correction
- **Process:**
  1. Generate interrogative question based on CNN confidence
  2. Ask BLIP-VQA: "Is this a [CNN prediction]?"
  3. Analyze response to check consistency
  4. Compare CNN understanding vs VLM understanding
  5. Determine if correction is needed

- **Output:** 
  - SCL Status (Verified or Self-Corrected)
  - Interrogative question
  - VLM response
  - Correction flag (if applicable)
  - Routing decision for Stage 3

**How SCL Works:**

```
High Confidence (>80%)
CNN predicts: "Rose"
├─ Interrogative: "Is this object a Rose?"
├─ VLM Response: "Yes, this is clearly a rose with distinct petal structure"
└─ Result: ✓ VERIFIED (Green) → Use context-aware prompts in Stage 3

Low Confidence (<50%)
CNN predicts: "Car"
├─ Interrogative: "Is this object a car?"
├─ VLM Response: "This appears to be a red bus, not a car"
└─ Result: ⚠ SELF-CORRECTED (Orange) → Use generic discovery prompts in Stage 3
```

#### **Stage 2/3: VLM Analysis & Narrative Synthesis**
- **Time:** 5-15 seconds (combined)
- **Input:** Image + CNN prediction + SCL routing decision
- **Process:**
  - Generate 9 category-specific questions
  - Ask BLIP-VQA each question
  - Receive raw responses
  - Transform into professional narratives
  - Apply context-awareness based on SCL status

- **Output:** 9-point comprehensive analysis

**Context-Aware Routing:**

- **If SCL Verified (Green):**
  - Use class-specific prompts with CNN prediction
  - Example: "A Rose is a botanical specimen. Describe its..."
  - VLM has CNN context to guide analysis

- **If SCL Self-Corrected (Orange):**
  - Use generic discovery prompts without CNN prediction
  - Example: "What is this object? Describe its..."
  - VLM analyzes fresh without CNN bias

### The 9-Point Analysis

Each analysis category represents a different dimension of understanding:

#### **1. Common Identity**
What is this object? What is it called?
- Primary classification
- Object name and type
- Categorical placement
- Distinctive characteristics

#### **2. Visual Summary**
What does it look like?
- Color and appearance
- Visual features
- Texture and surface quality
- Overall aesthetic impression

#### **3. Operational Utility**
What is it used for?
- Functional purpose
- Practical applications
- Use cases and roles
- Operational context

#### **4. Provenance & Setting**
Where does it come from?
- Origin and background
- Geographic context
- Environmental associations
- Historical context

#### **5. Technical Nomenclature**
What is its technical classification?
- Scientific names
- Industry designations
- Classification systems
- Technical terminology

#### **6. Safety & Risk Assessment**
What dangers or precautions exist?
- Potential hazards
- Safety considerations
- Handling requirements
- Risk factors

#### **7. Maintenance & Longevity**
How is it cared for and preserved?
- Care requirements
- Durability factors
- Maintenance procedures
- Preservation methods

#### **8. Aesthetic & Design**
What makes it visually interesting?
- Design qualities
- Artistic elements
- Aesthetic principles
- Style characteristics

#### **9. Interaction & Relationships**
How does it relate to humans and environments?
- Human interactions
- Social context
- Cultural significance
- Ecological relationships

---

## Features & Capabilities

### A. Image Analysis

**Supported Analysis Types:**

| Type | Features | Time |
|------|----------|------|
| **Quick Classification** | Class + confidence | <1s |
| **VLM Discovery** | 5-point analysis | 5-15s |
| **Professional Analysis** | 9-point + PDF + SCL | 9-25s |

**Image Input Flexibility:**
- File upload (drag & drop)
- Clipboard paste (Ctrl+V)
- File browser selection
- URL input (some setups)

### B. Analysis Export

**PDF Professional Reports:**
- Deep Ocean aesthetic
- Georgia Serif typography
- Neon cyan accents
- 9-point analysis grid
- SCL verification badge
- Ready for sharing/archival

**CSV Technical Export:**
- Row-based 9-point synthesis
- Column-based deep-data format
- System metadata
- Timestamp included
- Suitable for data analysis

**Recent History:**
- Last 20 predictions stored
- Quick re-analysis capability
- Batch operations possible

### C. Performance Monitoring

**Real-Time Metrics:**
- Inference time tracking
- Stage completion times
- SCL verification time
- Memory usage (optional)

**Historical Tracking:**
- Performance trends
- Model effectiveness metrics
- SCL correction rate
- User activity patterns

### D. Privacy Controls

**Data Handling:**
- All processing is local
- No cloud transmission of images
- Only LoRA adapter weights synced (encrypted)
- Session data cleared on logout
- User consent for federated learning (optional)

**Security Features:**
- HTTPS encryption (recommended)
- Session timeout (default: 24 hours)
- Password hashing (bcrypt)
- CSRF protection on forms

---

## Privacy & Security

### Local Processing Guarantee

**Images Never Leave Your System:**
- All image analysis happens on your local device
- Raw image data is never transmitted to servers
- Only anonymized prediction results are stored
- Session-based temporary processing

### Federated Learning Participation (Optional)

**What Gets Shared:**
- Only LoRA adapter weight updates
- Encrypted with TLS 1.3
- Compressed for efficiency (~10-50MB per round)
- Model improvement parameters only

**What Stays Private:**
- Raw training images
- Session data
- Personal information
- Prediction history (unless explicitly shared)

**How to Participate:**
1. Opt-in via privacy settings
2. Complete local training rounds
3. Submit LoRA weights to federation server
4. Contribute to global model improvement

**How to Opt-Out:**
- Keep federated learning disabled (default)
- No training rounds will be initiated
- Standard prediction analysis unaffected
- Privacy-first by default

### Data Deletion

**Automatic Deletion:**
- Recent history cleared on logout
- Session cache cleared after 24 hours
- Temporary files cleaned up
- Browser cache not used for sensitive data

**Manual Deletion:**
- Clear history button on client dashboard
- Remove individual predictions from history
- Logout to end session
- Clear browser cookies if needed

### Privacy Protocol

For detailed privacy information, visit `/privacy-protocol`

**Key Points:**
- GDPR compliant architecture
- No third-party data sharing
- User control over all data
- Transparent processing pipeline
- Right to delete all data

---

## Federated Learning Participation

### What is Federated Learning?

A collaborative training approach where:
- Multiple clients train models locally
- Only improved parameters are shared
- Central server aggregates improvements
- Global model improves without centralizing data

### How Client Participation Works

#### **1. Opt-In Process**

1. Navigate to **Settings** → **Privacy & Learning**
2. Toggle **"Join Federated Learning Network"**
3. Accept terms and conditions
4. Confirm selection

#### **2. Training Round Participation**

When a new federated learning round begins:

1. **Notification:** You'll see "New FL Round Available"
2. **Configuration:** Automatic setup on your node
3. **Local Training:**
   - Models train on your local data
   - No data leaves your system
   - Runs in background (can be paused)
   - Time: 30 minutes - 2 hours per round

4. **Weight Upload:**
   - Improved LoRA weights encrypted
   - Uploaded to federation server
   - Takes 5-10 minutes
   - Size: 10-50MB

5. **Global Aggregation:**
   - Server combines all client weights
   - Creates improved global model
   - Distributes back to all clients
   - Takes 15-30 minutes

#### **3. Benefits of Participation**

- 📊 Access to improved global models
- 🤝 Contribute to collective intelligence
- 🔍 Track your impact on model improvement
- 📈 Receive performance analytics
- 🎁 Potential incentives (depends on deployment)

#### **4. Round Monitoring**

**On Client Dashboard:**
- Current round status
- Training progress percentage
- Estimated time remaining
- Weight upload status
- Global model version

**Notifications:**
- Round start
- Training completion
- Upload status
- Round completion
- New model available

### Storage & System Requirements

**Minimum Requirements for FL Participation:**
- **RAM:** 8 GB (for model training)
- **Disk:** 10 GB free space
- **Network:** Stable internet (10 Mbps+)
- **Processor:** Multi-core CPU or GPU (optional)

**Storage Usage:**
- Model files: ~450 MB (BLIP-VQA + MobileNetV2)
- LoRA weights: ~50 MB per round
- Training cache: ~200 MB (temporary, cleaned up)
- Total baseline: ~700 MB

---

## Troubleshooting

### Common Issues & Solutions

#### **Issue: "Image Upload Failed"**

**Symptoms:** Red error message after trying to upload

**Causes & Solutions:**
1. **File Too Large**
   - Solution: Compress image or resize to <5MB
   - Use online compressors: https://tinypng.com

2. **Unsupported Format**
   - Solution: Convert to JPG, PNG, or GIF
   - Use online converters: https://convertio.co

3. **Corrupted File**
   - Solution: Re-save the image in your photo editor
   - Try a different image to verify

4. **Browser Issues**
   - Solution: Clear browser cache (Ctrl+Shift+Del)
   - Try different browser (Chrome, Firefox, Edge)
   - Check browser console for errors (F12)

#### **Issue: Analysis Takes Very Long (>30 seconds)**

**Symptoms:** Spinner keeps spinning, analysis doesn't complete

**Causes & Solutions:**
1. **System Resources Low**
   - Close other applications
   - Check RAM usage (Task Manager on Windows)
   - Restart browser/computer if memory critical

2. **Slow Internet**
   - Only relevant for optional FL sync
   - Local analysis shouldn't require internet
   - Check internet speed: https://speedtest.net

3. **Model Loading Delay**
   - First run may take longer (models loading into memory)
   - Subsequent analyses should be faster
   - Check status in browser console

4. **Browser Performance**
   - Update browser to latest version
   - Disable heavy extensions
   - Try incognito/private mode

#### **Issue: "Login Failed" Error**

**Symptoms:** Can't log in despite correct credentials

**Causes & Solutions:**
1. **Wrong Credentials**
   - Verify username: `client` (default)
   - Verify password: `client123` (default)
   - Check Caps Lock isn't on

2. **Account Issues**
   - Account may be disabled
   - Try registering new account
   - Contact admin if persistent

3. **Session Expired**
   - Sessions expire after 24 hours
   - Log in again
   - Clear cookies if problems continue

4. **Database Issues**
   - Restart the Flask server
   - Check `frontend_web/app.py` running successfully
   - Check database connection

#### **Issue: "CSV/PDF Export Doesn't Work"**

**Symptoms:** Download button clicked but nothing happens

**Causes & Solutions:**
1. **Browser Download Settings**
   - Check browser download folder
   - Adjust download location in settings
   - Allow downloads for localhost

2. **File Too Large**
   - Try CSV instead of PDF (smaller)
   - Check available disk space
   - Clear downloads folder

3. **Browser Blocking**
   - Check browser console for errors (F12)
   - Allow popups/downloads for localhost
   - Try different browser

4. **Session Data Missing**
   - Re-run prediction analysis
   - Ensure you've completed analysis before export
   - Check recent history has results

#### **Issue: "Image Not Processing (Stuck on Loading)"**

**Symptoms:** Loading spinner appears but never finishes

**Causes & Solutions:**
1. **Model Not Loaded**
   - Check backend logs for errors
   - Verify BLIP-VQA model downloaded (~350MB)
   - Verify MobileNetV2 available (~100MB)

2. **GPU Memory Issues** (if using GPU)
   - Reduce image size
   - Close other GPU-dependent apps
   - Switch to CPU-only mode

3. **Network Timeout** (for optional features)
   - Check internet connection
   - Try again in a few moments
   - Report persistent issues to admin

4. **Restart Required**
   - Restart Flask server: `python run_web.py`
   - Clear browser cache
   - Logout and log back in

#### **Issue: "SCL Badge Not Showing"**

**Symptoms:** SCL verification status doesn't appear on results

**Causes & Solutions:**
1. **Using Wrong Mode**
   - Ensure using "Triple-Layer Hybrid Ensemble" mode
   - Not ImageNet-1K or VLM-only modes
   - Verify mode selection before upload

2. **Analysis Incomplete**
   - Wait for all stages to complete
   - Check for error messages
   - Verify results page fully loaded

3. **Display Bug**
   - Refresh page (F5 or Ctrl+R)
   - Clear browser cache
   - Try different browser

#### **Issue: "CSV Doesn't Contain All 9 Points"**

**Symptoms:** CSV export missing some analysis categories

**Causes & Solutions:**
1. **Using Single-Mode Prediction**
   - SCL & 9-point only in Hybrid mode
   - Use Triple-Layer mode for full export
   - Other modes have fewer categories

2. **Incomplete Analysis**
   - Ensure all stages completed (10 cards visible)
   - Check for error messages
   - Re-run prediction if needed

3. **Export Generated During Analysis**
   - Wait for "Complete" status
   - Don't interrupt analysis
   - Export after all cards populated

#### **Issue: "Corrected Identity Not Showing"**

**Symptoms:** SCL shows "Self-Corrected" but corrected class name not visible

**Causes & Solutions:**
1. **Check Master Audit Report**
   - Go to Card 10 (Master Audit Report)
   - Look for "VLM-Corrected" identifier
   - Verify SCL status is orange (Self-Corrected)

2. **View PDF Report**
   - Download PDF export
   - Check Stage 1.5 SCL badge section
   - PDF shows full corrected identity

3. **Common Identity Card**
   - Check Card 1 (Common Identity)
   - Text should reflect corrected classification
   - Description starts with corrected object name

4. **Display Issue**
   - Refresh page
   - Navigate away and back to results
   - Check browser console for JavaScript errors

---

### Getting Help

**Resources:**
- 📖 Read this guide: CLIENT.md
- 📚 Check README.md for general system information
- 🔐 Review PRIVACY.md for privacy details
- 📊 See SUMMARY.md for technical architecture

**Common References:**
- Default credentials: `client` / `client123`
- Server: `http://localhost:5000`
- Client Dashboard: `http://localhost:5000/client/dashboard`
- Privacy Policy: `http://localhost:5000/privacy-protocol`

**Reporting Issues:**
1. Reproduce the issue
2. Note the exact error message
3. Check browser console (F12)
4. Include system information:
   - Browser and version
   - Operating system
   - RAM available
   - Network connection type
5. Report to admin or development team

---

### Best Practices

**For Optimal Performance:**

1. **Image Selection**
   - Use clear, well-lit images
   - 224x224 - 512x512 resolution preferred
   - Avoid extremely cluttered backgrounds
   - Single object focus improves accuracy

2. **Timing**
   - Use during off-peak hours if federated learning active
   - Avoid running many analyses in parallel
   - Allow models to warm up (first run slower)

3. **Data Management**
   - Regularly clear history if storage limited
   - Backup important PDF exports
   - Archive CSV data for research

4. **Privacy**
   - Keep strong passwords
   - Logout after use
   - Don't leave device unattended during analysis
   - Review privacy settings periodically

5. **Federated Learning**
   - Keep computer on during training rounds (if participating)
   - Use stable internet connection
   - Don't interrupt uploads
   - Monitor performance improvements

---

**Last Updated:** March 4, 2026  
**Version:** 3.2 (Triple-Layer Consistency Pipeline)  
**Status:** Production-Ready

For updates and detailed technical information, see:
- [SUMMARY.md](SUMMARY.md) - Technical architecture
- [README.md](README.md) - System overview
- [ADMIN.md](ADMIN.md) - Administrator guide
