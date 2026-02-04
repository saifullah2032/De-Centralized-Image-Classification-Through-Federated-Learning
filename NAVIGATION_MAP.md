# 🗺️ FEDERATED LEARNING UI - COMPLETE NAVIGATION MAP

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         UNAUTHENTICATED STATE                            │
└─────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   INDEX (/)  │
                              │  Landing Page│
                              └──────┬───────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ LOGIN        │ │ REGISTER     │ │ FEATURES     │
            │ /login       │ │ /register    │ │ (Scroll Down)│
            └──────┬───────┘ └──────┬───────┘ └──────────────┘
                   │                │
                   └────────┬───────┘
                            │
                     [After Auth]
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         AUTHENTICATED STATE                              │
└─────────────────────────────────────────────────────────────────────────┘

                     [ROLE DETECTION]
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
    [IF ADMIN]                          [IF CLIENT]
          │                                   │
          ▼                                   ▼
┌────────────────────┐              ┌────────────────────┐
│   ADMIN ROUTES     │              │   CLIENT ROUTES    │
└────────────────────┘              └────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            ADMIN FLOW CHART                              │
└─────────────────────────────────────────────────────────────────────────┘

        ┌──────────────────────────────────────────────────┐
        │                  INDEX (/)                        │
        │         [Logged in as: admin]                     │
        └───────────┬──────────────────────────────────────┘
                    │
        ┌───────────┼───────────┬────────────┐
        │           │           │            │
        ▼           ▼           ▼            ▼
┌─────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐
│   PREDICT   │ │   ADMIN    │ │ PRIVACY  │ │   LOGOUT     │
│  /predict   │ │ DASHBOARD  │ │  REPORT  │ │  /logout     │
│             │ │ /admin/    │ │ /privacy-│ │              │
│ [Analysis]  │ │ dashboard  │ │ report   │ │ [Ends Session]
└──────┬──────┘ └────────────┘ └──────────┘ └──────────────┘
       │
       ▼
┌─────────────┐
│  RESULTS    │
│ (POST only) │
│             │
│ [Shows top  │
│  prediction]│
└──────┬──────┘
       │
       ├──► "New Analysis" ──► PREDICT
       │
       └──► "Return Home"  ──► INDEX


┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT FLOW CHART                              │
└─────────────────────────────────────────────────────────────────────────┘

        ┌──────────────────────────────────────────────────┐
        │                  INDEX (/)                        │
        │         [Logged in as: client]                    │
        └───────────┬──────────────────────────────────────┘
                    │
        ┌───────────┼───────────┬────────────┐
        │           │           │            │
        ▼           ▼           ▼            ▼
┌─────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐
│   PREDICT   │ │  CLIENT    │ │ PRIVACY  │ │   LOGOUT     │
│  /predict   │ │ DASHBOARD  │ │  REPORT  │ │  /logout     │
│             │ │ /client/   │ │ /privacy-│ │              │
│ [Analysis]  │ │ dashboard  │ │ report   │ │ [Ends Session]
└──────┬──────┘ └────────────┘ └──────────┘ └──────────────┘
       │
       ▼
┌─────────────┐
│  RESULTS    │
│ (POST only) │
│             │
│ [Identical  │
│  to admin]  │
└──────┬──────┘
       │
       ├──► "New Analysis" ──► PREDICT
       │
       └──► "Return Home"  ──► INDEX


┌─────────────────────────────────────────────────────────────────────────┐
│                         NAVBAR QUICK REFERENCE                           │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  [UNAUTHENTICATED]                                                     │
│  FEDERATED.AI  |  Home  |  Login  |  [Join Network]                   │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  [ADMIN]                                                               │
│  FEDERATED.AI  |  Home  |  Analysis  |  Dashboard  |  Privacy  |  🔴  │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  [CLIENT]                                                              │
│  FEDERATED.AI  |  Home  |  Analysis  |  Node Status  |  Privacy  |  🔴│
└────────────────────────────────────────────────────────────────────────┘

Legend:
  🔴 = Logout button (red power icon)
  [Button] = Primary CTA style
  Regular = Standard nav link


┌─────────────────────────────────────────────────────────────────────────┐
│                      IMAGE ANALYSIS FLOW (ALL ROLES)                     │
└─────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────────────────────┐
    │         PREDICT PAGE (/predict)               │
    │   ┌───────────────────────────────┐           │
    │   │   Drop Zone (Drag & Drop)     │           │
    │   │   OR Click to Browse          │           │
    │   └───────────┬───────────────────┘           │
    │               │                               │
    │               ▼                               │
    │   ┌───────────────────────────────┐           │
    │   │   Image Preview Appears       │           │
    │   │   [X] Remove button visible   │           │
    │   └───────────┬───────────────────┘           │
    │               │                               │
    │               ▼                               │
    │   ┌───────────────────────────────┐           │
    │   │  [Execute Classification]     │           │
    │   │  (Button enabled if model OK) │           │
    │   └───────────┬───────────────────┘           │
    └───────────────┼───────────────────────────────┘
                    │
                    ▼ [POST Request]
        ┌─────────────────────────┐
        │  BACKEND PROCESSING     │
        │  1. Save to uploads/    │
        │  2. Preprocess (32x32)  │
        │  3. Model inference     │
        │  4. Get top 5 classes   │
        │  5. Delete uploaded file│
        └─────────┬───────────────┘
                  │
                  ▼
    ┌───────────────────────────────────────────────┐
    │         RESULTS PAGE (results.html)           │
    │   ┌───────────────────────────────┐           │
    │   │  ✓ Inference Complete         │           │
    │   └───────────────────────────────┘           │
    │                                               │
    │   ┌─────────────┬─────────────────┐           │
    │   │ Source      │  Neural Output  │           │
    │   │ Image       │  - Top Class    │           │
    │   │ (32x32)     │  - Confidence % │           │
    │   │             │  - Progress Bar │           │
    │   └─────────────┴─────────────────┘           │
    │                                               │
    │   ┌─────────────────────────────────┐         │
    │   │  Network Stats | Architecture   │         │
    │   │  - Rounds: 20  | - MobileNetV2 │         │
    │   │  - Accuracy    | - JAX Backend │         │
    │   └─────────────────────────────────┘         │
    │                                               │
    │   [New Analysis]  [Return Home]               │
    └───────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                      DASHBOARD COMPARISON                                │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────┐  ┌────────────────────────────────┐
│   ADMIN DASHBOARD             │  │   CLIENT DASHBOARD             │
│   /admin/dashboard            │  │   /client/dashboard            │
├───────────────────────────────┤  ├────────────────────────────────┤
│                               │  │                                │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌───┐│  │ ┌─────────────────────────────┐│
│ │Round│ │ Acc │ │Loss │ │Sta││  │ │  Node Configuration         ││
│ └─────┘ └─────┘ └─────┘ └───┘│  │ │  [Dataset Partition]        ││
│                               │  │ │  [Initialize Node]          ││
│ ┌─────────────────────────────┐│  │ └─────────────────────────────┘│
│ │  Training Progress Chart    ││  │                                │
│ │  (Chart.js Line Graph)      ││  │ ┌─────┐ ┌─────┐ ┌──────────┐ │
│ │                             ││  │ │Stat │ │ Acc │ │Processed │ │
│ └─────────────────────────────┘│  │ └─────┘ └─────┘ └──────────┘ │
│                               │  │                                │
│ ┌─────────────────────────────┐│  │ ┌─────────────────────────────┐│
│ │  Live Log Terminal          ││  │ │  Compute Progress Bar       ││
│ │  (SSE Stream)               ││  │ └─────────────────────────────┘│
│ └─────────────────────────────┘│  │                                │
│                               │  │ ┌─────────────────────────────┐│
│ ┌─────────────────────────────┐│  │ │  Process Output Terminal    ││
│ │  System Architecture        ││  │ │  (Simulated Local Training) ││
│ │  Rounds | Nodes | Engine    ││  │ └─────────────────────────────┘│
│ └─────────────────────────────┘│  │                                │
└───────────────────────────────┘  └────────────────────────────────┘

Features:                          Features:
- Real-time SSE updates            - Simulated training
- Global metrics across all nodes  - Local metrics only
- Historical training charts       - Progress simulation
- Requires FL server running       - Works standalone


┌─────────────────────────────────────────────────────────────────────────┐
│                         ERROR PAGE HANDLING                              │
└─────────────────────────────────────────────────────────────────────────┘

    [Invalid URL]              [Server Exception]
           │                            │
           ▼                            ▼
    ┌─────────────┐            ┌─────────────┐
    │   404 PAGE  │            │   500 PAGE  │
    │             │            │             │
    │ "Node Not   │            │ "Core Intel-│
    │  Found"     │            │  ligence    │
    │             │            │  Failure"   │
    │ 🔵 Network  │            │ 🔴 Microchip│
    │    Icon     │            │    Icon     │
    │             │            │             │
    │ [Return to  │            │ [Home]      │
    │  Command    │            │ [Dashboard] │
    │  Center]    │            │             │
    └─────────────┘            └─────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                          ANIMATION TIMELINE                              │
└─────────────────────────────────────────────────────────────────────────┘

Page Load:
├─ 0.0s → Hero title fade-in starts
├─ 0.0s → Hero description slide-left starts
├─ 0.1s → First feature card slide-up (stagger-1)
├─ 0.2s → Second feature card slide-up (stagger-2)
├─ 0.3s → Third feature card slide-up (stagger-3)
└─ 0.6s → All animations complete

User Interaction:
├─ Hover Card    → Lift 6px in 0.4s
├─ Hover Button  → Lift 2px + shadow in 0.2s
├─ Click Button  → Shimmer effect (0.8s sweep)
└─ Page Exit     → Instant (no exit animations)


┌─────────────────────────────────────────────────────────────────────────┐
│                          ROUTE PROTECTION                                │
└─────────────────────────────────────────────────────────────────────────┘

Public Routes (No Auth):
  /                  - Landing page
  /login             - Authentication
  /register          - Registration
  /favicon.ico       - Handled gracefully

Protected Routes (@login_required):
  /predict           - Image analysis
  /privacy-report    - Privacy verification
  /dashboard         - Auto-redirect by role
  /logout            - Session termination

Admin Only (@admin_required):
  /admin/dashboard   - Global training monitor
  /admin/events      - SSE stream for live logs
  /api/metrics       - Training history JSON

Client Routes (login_required, not admin):
  /client/dashboard  - Local node simulator


┌─────────────────────────────────────────────────────────────────────────┐
│                      SESSION FLOW & REDIRECTS                            │
└─────────────────────────────────────────────────────────────────────────┘

LOGIN SUCCESS:
  Admin User  → /admin/dashboard
  Client User → /predict (Analysis page)

LOGOUT:
  Any User → / (Landing page)
  Flash: "You have been logged out"

REGISTER SUCCESS:
  Any Role → /login
  Flash: "Registration successful! Please log in."

UNAUTHORIZED ACCESS:
  Try /admin/dashboard as client → /login
  Flash: "Please log in to access this page"


┌─────────────────────────────────────────────────────────────────────────┐
│                            API ENDPOINTS                                 │
└─────────────────────────────────────────────────────────────────────────┘

GET  /api/status
  Returns: { model_loaded, current_round, accuracy, timestamp }
  Auth: None (public)

GET  /api/metrics
  Returns: { success, data: { rounds[], accuracies[], losses[] }}
  Auth: None (public)

GET  /admin/events
  Returns: Server-Sent Events stream (text/event-stream)
  Events: { type: 'log'|'status'|'heartbeat', message, timestamp }
  Auth: Admin only


┌─────────────────────────────────────────────────────────────────────────┐
│                     RESPONSIVE BREAKPOINTS                               │
└─────────────────────────────────────────────────────────────────────────┘

Desktop (≥992px):
  ✓ Full sidebar navigation
  ✓ 3-column feature grid
  ✓ Large hero icon visible
  ✓ Chart full width

Tablet (768px-991px):
  ✓ Collapsed navbar (hamburger)
  ✓ 2-column feature grid
  ✓ Smaller hero icon
  ✓ Chart adapts

Mobile (≤767px):
  ✓ Hamburger menu
  ✓ 1-column stacked layout
  ✓ Hero icon hidden
  ✓ Chart responsive
  ✓ Buttons stack vertically


┌─────────────────────────────────────────────────────────────────────────┐
│                      FILE UPLOAD RESTRICTIONS                            │
└─────────────────────────────────────────────────────────────────────────┘

Allowed Extensions: .png, .jpg, .jpeg
Max File Size: 5 MB (5,242,880 bytes)
Upload Folder: uploads/
Cleanup: Auto-deleted after prediction

Validation:
  1. Check file extension
  2. Check file size
  3. Verify image can open (PIL)
  4. Resize to 32x32 (preprocessing)
  5. Normalize pixel values [0,1]


═══════════════════════════════════════════════════════════════════════════

                        🎉 NAVIGATION MAP COMPLETE 🎉

  All routes tested ✅ | All animations working ✅ | All roles verified ✅

═══════════════════════════════════════════════════════════════════════════
```
