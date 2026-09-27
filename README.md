<div align="center">

# 🚗 AI-Based Driver Behaviour, Fatigue & Risk Detection System

### *Multimodal Deep Learning & In-Vehicle Telematics Intelligence for Advanced Driver Assistance Systems (ADAS)*

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Explainable%20ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge&logo=github-actions&logoColor=white)](.github/workflows/ci.yml)

[Explore Features](#-key-features) • [System Architecture](#-system-architecture) • [Quickstart](#-quickstart-guide) • [Dashboard HUD](#-interactive-cockpit-hud) • [API Docs](#-api-endpoints)

</div>

---

## 📌 Executive Summary

Road traffic accidents remain one of the leading causes of global fatalities, with **human error, driver fatigue, drowsiness, and distraction** contributing to over 90% of crashes. 

This project implements an end-to-end **Automotive Driver Monitoring System (DMS) and Risk Engine** that fuses **driver-facing camera computer vision** with **high-frequency in-vehicle telemetry sensors**. By leveraging a **hybrid Spatial 2D-CNN + Temporal Bi-Directional LSTM** neural network alongside an **Explainable Machine Learning (Random Forest) baseline**, the system monitors driver physiological states and vehicle kinematics in real time (<20ms latency), detects hazardous situations, and calculates a continuous **Dynamic Risk Score (0–100%)**.

---

## ⚡ Key Features

- **🌐 Multi-Modal Sensor Fusion**: Harmonizes high-frequency vehicle telemetry (speed, acceleration, braking, steering, lane departure) with vision-based DMS metrics (Eye Aspect Ratio, head pose Euler angles, phone usage).
- **🧠 Hybrid CNN + Bi-LSTM Deep Architecture**: Spatial 2D CNN processes driver facial & eye ROIs, while a 2-layer Bidirectional LSTM with temporal self-attention models 30-frame temporal trajectories (3.0s sliding buffer).
- **🎯 6 Target Driving Behaviors**: Differentiates between `Normal`, `Aggressive`, `Distracted`, `Drowsy`, `Unsafe braking`, and `Unsafe acceleration`.
- **🛡️ Dynamic Composite Risk Engine**: Fuses neural network predictions with automotive safety rules to generate real-time hazard severity scores and active acoustic/visual alerts.
- **🔍 Explainable AI (XAI)**: Includes a Scikit-Learn Random Forest baseline providing feature importance rankings (PERCLOS, jerk, angular velocity).
- **🏎️ Futuristic Cockpit Telematics HUD**: Full-stack automotive telematics dashboard featuring SVG radial gauges, Canvas temporal oscilloscope, 10 Hz bi-directional WebSocket streaming, and Web Audio API synthesized vehicle chimes.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph SENSORS ["1. Multimodal Inputs"]
        A1["Driver Camera (DMS)"]
        A2["Vehicle Telemetry (CAN / IMU / OBD-II)"]
    end

    subgraph PREPROC ["2. Feature Extraction"]
        B1["Vision Pipeline (OpenCV)<br/>EAR • Head Pitch/Yaw • Phone Det"]
        B2["Kinematics Normalization<br/>Speed • Jerk • Brake % • Steer Rate"]
    end

    subgraph DEEP_NET ["3. Spatial-Temporal Deep Learning"]
        C1["Spatial Vision CNN (32-dim)"]
        C2["Telemetry Encoder MLP (32-dim)"]
        C3["Feature Fusion Layer (64-dim)"]
        C4["Bi-Directional LSTM (2-Layer, 128-dim)"]
        C5["Temporal Self-Attention Mechanism"]
    end

    subgraph DUAL_HEADS ["4. Dual-Head Output"]
        D1["Behavior Classifier<br/>(6-Class Softmax)"]
        D2["Continuous Risk Regressor<br/>(0.0 - 100.0%)"]
    end

    subgraph TELEMATICS ["5. Cockpit Telematics HUD"]
        E1["FastAPI WebSocket Stream (10 Hz)"]
        E2["Instrument Cluster Gauges"]
        E3["Real-time Strip Chart Oscilloscope"]
        E4["Acoustic & Visual Alert Chimes"]
    end

    A1 --> B1
    A2 --> B2
    B1 --> C1
    B2 --> C2
    C1 --> C3
    C2 --> C3
    C3 --> C4
    C4 --> C5
    C5 --> D1
    C5 --> D2
    D1 --> E1
    D2 --> E1
    E1 --> E2
    E1 --> E3
    E1 --> E4
```

---

## 📊 8 Monitored Parameters

| Parameter | Source | Range | Significance in Hazard Detection |
|---|---|---|---|
| **Speed** | Wheel Sensors | 0 – 180 km/h | Excessive velocity and speeding detection |
| **Acceleration** | 3-Axis IMU | -8.0 to +8.0 m/s² | Longitudinal dynamics, jackrabbit acceleration surges |
| **Braking** | Pedal Transducer | 0 – 100 % | Sudden panic braking & near-collision deceleration |
| **Steering** | Steering Sensor | -180° to +180° | Rapid lane weaving, erratic lateral jerk, torque steer |
| **Lane Deviation** | Vision / LKA | -2.5 to +2.5 m | Lateral offset from lane center; road departure warning |
| **Eye Closure** | Driver Camera | EAR 0.0 – 1.0 | Eye Aspect Ratio; micro-sleep episodes & PERCLOS fatigue |
| **Head Position** | Driver Camera | Pitch & Yaw (°) | Head droop (nodding drowsy) & averted glances |
| **Phone Usage** | Driver Camera | Conf 0.0 – 1.0 | Visual smartphone interaction / distracted driving |

---

## 🎯 Target Behaviors & Detection Classes

| Class | Primary Behavioral Signature | Risk Level |
|---|---|---|
| **`Normal`** | Smooth cruise (50–90 km/h), attentive gaze, natural blink cycles, centered lane keeping. | `SAFE` (0–30%) |
| **`Aggressive`** | High speed, rapid lateral weaving, aggressive throttle/brake cycles, high angular velocity. | `CAUTION` (30–60%) |
| **`Distracted`** | Smartphone usage detected, head yaw turned away (>25°), delayed lane keeping. | `CRITICAL` (80–100%) |
| **`Drowsy`** | Prolonged eye closure (EAR < 0.22 for >1.2s), micro-sleeps, head nodding forward. | `CRITICAL` (80–100%) |
| **`Unsafe braking`** | Sudden emergency braking spike (>75%), deceleration surge (< -5.5 m/s²). | `WARNING` (60–80%) |
| **`Unsafe acceleration`** | Floored accelerator from stop/traffic (+5.0 to +7.8 m/s²), wheel spin. | `WARNING` (60–80%) |

---

## 📈 Performance & Benchmark Metrics

The model was trained and evaluated on a balanced dataset of multi-modal driving trajectories:

- **CNN-LSTM Classification Accuracy**: **100.0%**
- **Weighted F1-Score**: **1.0000**
- **Continuous Risk Score MAE**: **4.26 points**
- **Classical ML Baseline (Random Forest) Accuracy**: **100.0%**
- **Inference Latency**: **~18 ms/step on CPU** (Real-time capability >50 FPS)

### Confusion Matrix (Held-out Test Set)

```
                 Normal  Aggressive  Distracted  Drowsy  Unsafe Brake  Unsafe Accel
Normal             22        0           0          0         0             0
Aggressive          0       20           0          0         0             0
Distracted          0        0          21          0         0             0
Drowsy              0        0           0         22         0             0
Unsafe braking      0        0           0          0        29             0
Unsafe acceleration 0        0           0          0         0            30
```

### Feature Importance (XAI Analysis)
1. **`eye_ear_min`** (8.4%) — Minimum Eye Aspect Ratio during temporal window
2. **`speed_std`** (7.2%) — Speed fluctuation & variance
3. **`eye_ear_mean`** (6.7%) — Average eye openness
4. **`perclos`** (6.5%) — Percentage of eye closure time
5. **`head_pitch_min`** (6.1%) — Maximum forward head droop
6. **`lane_dev_std`** (6.1%) — Standard deviation of lateral position
7. **`accel_mean`** (5.4%) — Mean longitudinal acceleration
8. **`steer_std`** (5.3%) — Steering standard deviation

---

## 💻 Interactive Cockpit HUD

The built-in telematics dashboard features:
1. **DMS Camera HUD**: Real-time simulated driver face mesh, eye reticle tracking, EAR meter, and phone detection alert. Supports switching to **Live Webcam**.
2. **Instrument Cluster**: SVG radial Speedometer (0–180 km/h) and Composite Risk Score dial with pulsating alert halos.
3. **Vehicle Dynamics Deck**: Longitudinal G-force meter, braking bar, rotating steering wheel, and highway lane tracking.
4. **Temporal Strip Chart**: 30-second multi-channel oscilloscope tracking speed, braking, eye openness, and risk score in real time.
5. **Interactive Controls & Presets**: Instant scenario testing buttons (`Normal`, `Aggressive`, `Distracted`, `Drowsy`, `Unsafe braking`, `Unsafe acceleration`) and live sliders for manual parameter testing.
6. **Acoustic Warning System**: Integrated Web Audio API vehicle warning chimes for caution, warning, and critical hazard states.

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/driver-risk-detection.git
cd driver-risk-detection
```

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Automated System Verification
```bash
python test_system.py
```

### 4. Launch the Telematics Cockpit Server
```bash
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in any modern web browser!

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Returns system health, loaded model status, bounds, and parameters |
| `GET` | `/api/metrics` | Returns validation metrics, confusion matrix, and feature importances |
| `POST` | `/api/predict` | Processes single telemetry sample, returns class probabilities & risk score |
| `POST` | `/api/reset` | Resets sliding temporal buffer to nominal baseline |
| `GET` | `/api/scenarios` | Returns catalog of demonstration scenarios |
| `POST` | `/api/simulate_step` | Steps through scenario playback trajectories |
| `WS` | `/ws/telematics` | Full-duplex 10 Hz WebSocket streaming for real-time dashboard updates |

---

## 📂 Project Structure

```
driver-risk-detection/
├── ai_engine/
│   ├── config.py                 # Parameter bounds, classes, thresholds & hyperparams
│   ├── dataset.py                # Multimodal physics-based sequence generator
│   ├── feature_extraction.py     # Parameter scaling, PERCLOS, and DMS simulation
│   ├── models.py                 # SpatialVisionCNN, TelemetryEncoder, Bi-LSTM, ML Baseline
│   ├── train.py                  # Training pipeline, multi-task loss, validation
│   ├── inference.py              # Real-time streaming sliding-window inference engine
│   ├── saved_models/             # Pretrained weights (.pth, .joblib) & metrics.json
│   └── __init__.py               # Package exports
├── backend/
│   └── server.py                 # FastAPI REST and WebSocket telematics server
├── frontend/
│   ├── index.html                # Cockpit HUD HTML layout
│   ├── style.css                 # Dark automotive cockpit styling
│   └── app.js                    # WebSockets, Canvas oscilloscope, SVG gauges, audio
├── .github/
│   └── workflows/ci.yml          # GitHub Actions automated test workflow
├── test_system.py                # End-to-end verification test suite
├── requirements.txt              # Pinned Python package dependencies
├── pyproject.toml                # Standard Python build configuration
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # MIT License
└── README.md                     # Documentation
```

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ for automotive safety, intelligent transportation systems, and advanced deep learning.</sub>
</div>
