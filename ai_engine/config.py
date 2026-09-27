"""
AI-Based Driver Behaviour, Fatigue and Risk Detection System
Configuration and Hyperparameters
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
DATA_DIR = BASE_DIR / "data"

SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Behavior Classes
CLASSES = [
    "Normal",
    "Aggressive",
    "Distracted",
    "Drowsy",
    "Unsafe braking",
    "Unsafe acceleration"
]

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Parameter Definitions & Normalization Bounds
# 8 Core Parameters:
# Vehicle Sensors: Speed, Acceleration, Braking, Steering, Lane deviation
# Camera / DMS: Eye closure (EAR), Head position (Pitch, Yaw), Phone usage
SENSOR_PARAMS = [
    "speed",             # km/h [0, 180]
    "acceleration",      # m/s^2 [-8, 8]
    "braking",           # % [0, 100]
    "steering",          # degrees [-180, 180]
    "lane_deviation",    # meters [-2.5, 2.5]
    "eye_closure",       # EAR [0.0 (closed) to 1.0 (open)]
    "head_pitch",        # degrees [-45 (nodding down) to +45]
    "head_yaw",          # degrees [-60 (turned right) to +60 (turned left)]
    "phone_usage"        # probability / flag [0.0 to 1.0]
]

NUM_TELEMETRY_FEATURES = len(SENSOR_PARAMS)

# Normalization Min-Max values for feature scaling
FEATURE_BOUNDS = {
    "speed": (0.0, 180.0),
    "acceleration": (-8.0, 8.0),
    "braking": (0.0, 100.0),
    "steering": (-180.0, 180.0),
    "lane_deviation": (-2.5, 2.5),
    "eye_closure": (0.0, 1.0),
    "head_pitch": (-45.0, 45.0),
    "head_yaw": (-60.0, 60.0),
    "phone_usage": (0.0, 1.0)
}

# Video / Camera Specifications
CAMERA_IMG_SIZE = (64, 64)       # Grayscale/RGB frame fed into Spatial CNN
CAMERA_CHANNELS = 1              # Grayscale for face/eye/dms ROIs

# Sequence & Temporal Window
SEQUENCE_LENGTH = 30             # 30 time steps (e.g. 3.0 seconds at 10 Hz)
SAMPLING_RATE_HZ = 10            # 10 samples per second

# Model Architecture Dimensions
CNN_FEATURE_DIM = 32             # Spatial feature vector from Camera CNN
TELEMETRY_EMBED_DIM = 32         # Sensor projection embedding
LSTM_HIDDEN_DIM = 64             # Bi-LSTM hidden units
LSTM_NUM_LAYERS = 2              # 2-layer Recurrent Network
DROPOUT_RATE = 0.3

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 25
RANDOM_SEED = 42

# Risk Score Configuration (Continuous boundaries)
RISK_LEVELS = {
    "SAFE": (0.0, 30.0, "#10b981", "Normal driving conditions"),
    "CAUTION": (30.0, 60.0, "#f59e0b", "Minor driving anomalies detected"),
    "WARNING": (60.0, 80.0, "#f97316", "Elevated risk - intervention recommended"),
    "CRITICAL": (80.0, 100.0, "#ef4444", "Severe danger - immediate warning triggered")
}

# Rule-based safety thresholds for risk modulation
THRESHOLDS = {
    "drowsy_ear_threshold": 0.22,           # EAR below this indicates closed eyes
    "drowsy_closed_frames": 12,             # >1.2 seconds of closed eyes = microsleep
    "head_pitch_drowsy": -22.0,             # Head drooping down
    "head_yaw_distracted": 25.0,            # Head turned away from road >25 deg
    "phone_usage_threshold": 0.50,          # Phone usage detected
    "unsafe_accel_threshold": 4.5,          # >4.5 m/s^2 aggressive acceleration
    "unsafe_brake_threshold": 65.0,         # >65% emergency brake pressure
    "lane_drift_threshold": 1.2             # >1.2m lane deviation
}
