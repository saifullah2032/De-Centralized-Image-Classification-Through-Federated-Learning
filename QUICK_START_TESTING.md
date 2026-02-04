# 🚀 QUICK START GUIDE - Testing the Federated Learning UI

## Prerequisites
- Python 3.8+ with all dependencies installed
- Trained model exists at `models/global_model.h5` ✅ (Already present)

---

## 🎯 Testing the Complete System

### **Step 1: Start the Web Server**
```bash
cd "C:\Users\rayan\Downloads\Image CLassification"
python frontend_web/app.py
```

Expected output:
```
======================================================================
FEDERATED LEARNING COMMAND CENTER
======================================================================
  Host:     0.0.0.0
  Port:     5000
  Database: SQLite (persistent)
  URL:      http://localhost:5000
======================================================================
  Default Credentials:
    Admin:  admin / admin123
    Client: client / client123
======================================================================
```

---

### **Step 2: Open Browser**
Navigate to: **http://localhost:5000**

---

### **Step 3: Test User Flows**

#### 🔐 **Flow A: Admin Login → Image Analysis**

1. **Landing Page**
   - Observe: Hero section with fade-in animation
   - Observe: Feature cards lift on hover
   - Click: **"Get Started"** button

2. **Login Page**
   - Username: `admin`
   - Password: `admin123`
   - OR click **"Admin"** quick login button
   - Click: **"Authenticate"**

3. **Redirected to Analysis Page**
   - Observe: "Global Intelligence Active" green banner
   - Observe: Model Context card shows MobileNetV2, JAX backend
   
4. **Upload Test Image**
   - Option 1: Drag & drop any image file
   - Option 2: Click drop zone to browse
   - Supported: `.png`, `.jpg`, `.jpeg`
   
   > **First Upload Note**: Model loading takes ~20-30 seconds (JAX compilation). This is normal!
   > Subsequent predictions will be instant.

5. **View Results**
   - Top prediction with confidence percentage
   - Progress bar showing probability density
   - Model statistics (rounds completed, accuracy)
   - Click: **"New Analysis"** to test another image
   - Click: **"Return Home"** to go back

6. **Access Admin Dashboard**
   - From navbar: Click **"Dashboard"**
   - Observe: Current Round, Global Accuracy, Status cards
   - Observe: Training Progress chart (Chart.js with live data)
   - Observe: Live Log terminal (if FL training is active)

---

#### 👤 **Flow B: Client Registration → Node Monitor**

1. **Landing Page**
   - Click: **"Join Network"** (purple button in navbar)

2. **Registration Page**
   - Username: `testclient`
   - Password: `test123`
   - Confirm: `test123`
   - Role: Select **"CLIENT"** (default)
   - Click: **"Provision Node"**

3. **Login with New Credentials**
   - Use credentials created above
   - Click: **"Authenticate"**

4. **Client Dashboard**
   - From navbar: Click **"Node Status"**
   - Observe: Node Configuration panel with dataset partition selector
   - Click: **"Initialize Node"** to simulate training
   - Observe: Progress bar animates
   - Observe: Terminal logs appear in real-time
   - Wait for completion: Status changes to "Synchronized"

5. **Test Analysis**
   - From navbar: Click **"Analysis"**
   - Upload an image (same as admin flow)
   - Results page identical for all roles

---

#### 🔒 **Flow C: Privacy Verification**

1. **Login as any user**
2. **From navbar: Click "Privacy"**
3. **Two Scenarios**:
   
   **If report exists** (`logs/privacy_report.json`):
   - View: Verification status table
   - View: Protocol Buffers detected count
   - View: Raw image data leaked (should be 0 bytes)
   
   **If report doesn't exist**:
   - View: Warning banner "No Report Available"
   - View: Generation Protocol instructions
   - See commands: `tshark`, `python scripts/verify_privacy.py`

---

## 🎨 Animation Testing Checklist

### **Landing Page (index.html)**
- [ ] Hero title fades in smoothly
- [ ] Description slides from left
- [ ] Network icon pulses continuously
- [ ] Feature cards lift 6px on hover
- [ ] Buttons lift 2px on hover with shadow

### **Login/Register Pages**
- [ ] Icon glows with pulse animation
- [ ] Form fields focus with indigo border
- [ ] Quick login buttons highlight on hover
- [ ] Role selection buttons show active state

### **Predict Page**
- [ ] Drop zone scales up on drag-over
- [ ] Preview image appears with smooth transition
- [ ] Submit button disabled if no model (with visual feedback)

### **Results Page**
- [ ] Success checkmark glows on entry
- [ ] Confidence bar animates from 0% to final value
- [ ] Cards stagger in (1st, 2nd, 3rd with delays)

### **Dashboard Pages**
- [ ] Stat cards stagger in (4 cards with 0.1s delay each)
- [ ] Chart animates line drawing (Chart.js native)
- [ ] Terminal logs scroll smoothly

---

## 🖼️ Sample Images for Testing

Since the model is trained on CIFAR-100, best results with:
- Animals: cats, dogs, bears, elephants
- Vehicles: cars, trucks, bicycles, trains
- Nature: trees, flowers, clouds
- Objects: bottles, chairs, keyboards

Upload any image - the model will resize to 32x32 automatically.

---

## 📊 Expected Results

### **Model Performance**
Based on `model_history.json`:
- Trained Rounds: **20**
- Expected Accuracy: **~30-40%** (CIFAR-100 is challenging, 100 classes)
- Confidence scores will vary (10-50% typical for top prediction)

### **Response Times**
- **First prediction**: 20-30 seconds (model loading + JAX compilation)
- **Subsequent predictions**: < 2 seconds
- **Page navigation**: Instant
- **Dashboard updates**: Real-time (SSE stream every 2s)

---

## 🐛 Troubleshooting

### **Issue: Prediction takes too long**
**Cause**: JAX JIT compilation on first run  
**Solution**: Wait 30 seconds, then subsequent predictions are instant

### **Issue: "Model not loaded" error**
**Cause**: `models/global_model.h5` doesn't exist  
**Solution**: Run FL training first (already done in your case ✅)

### **Issue: Upload fails with "Invalid file type"**
**Cause**: Non-image file uploaded  
**Solution**: Use `.png`, `.jpg`, or `.jpeg` files only

### **Issue: Dashboard shows "No training history"**
**Cause**: `models/model_history.json` missing  
**Solution**: Already exists in your setup ✅

### **Issue: Privacy report empty**
**Cause**: `logs/privacy_report.json` not generated  
**Solution**: Run `python scripts/verify_privacy.py` after FL training with packet capture

---

## ✅ Success Indicators

You'll know the system is working when:
1. ✅ Login redirects based on role (admin → dashboard, client → analysis)
2. ✅ Image upload shows preview before submission
3. ✅ Results page displays predicted class + confidence
4. ✅ Dashboard chart renders with historical data
5. ✅ All navigation links work without 404 errors
6. ✅ Animations are smooth (no janky movements)
7. ✅ Mobile view adapts (test at 375px width)

---

## 🎉 Demo Script (5 Minutes)

**"Watch me classify an image using a privacy-preserving federated learning model!"**

1. **[0:00-0:30]** Show landing page, explain federated learning concept
2. **[0:30-1:00]** Login as admin, navigate to Analysis
3. **[1:00-1:30]** Upload cat image, explain local preprocessing
4. **[1:30-3:00]** Wait for prediction (mention JAX compilation if first time)
5. **[3:00-3:30]** Show results: "Cat - 45% confidence"
6. **[3:30-4:00]** Navigate to Admin Dashboard, show training curve
7. **[4:00-4:30]** Open Privacy Report, explain zero data leakage
8. **[4:30-5:00]** Summarize: "Secure, distributed, production-ready"

---

## 📞 Quick Reference

| Action | URL |
|--------|-----|
| Home | http://localhost:5000/ |
| Login | http://localhost:5000/login |
| Register | http://localhost:5000/register |
| Analysis | http://localhost:5000/predict |
| Admin Dashboard | http://localhost:5000/admin/dashboard |
| Client Dashboard | http://localhost:5000/client/dashboard |
| Privacy Report | http://localhost:5000/privacy-report |

---

## 🔥 Pro Tips

1. **Keep browser console open** (F12) to see real-time request logs
2. **Test on mobile** by resizing browser to 375px width
3. **Try multiple images** - second prediction is much faster!
4. **Watch SSE logs** in Network tab when on Admin Dashboard
5. **Use demo credentials** for quick testing (buttons on login page)

---

**Ready to test? Start the server and open http://localhost:5000!** 🚀
