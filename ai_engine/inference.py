"""
Real-time Streaming Inference Engine for Driver Behaviour, Fatigue and Risk Detection.
Maintains a 30-frame temporal sliding window, applies multimodal CNN-LSTM inference,
temporal exponential smoothing, rule-based verification, and active hazard alerting.
"""

import os
import json
import joblib
import numpy as np
import torch
import cv2
import base64
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from pathlib import Path

from ai_engine.config import (
    CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES,
    SEQUENCE_LENGTH, SAVED_MODELS_DIR, SENSOR_PARAMS,
    RISK_LEVELS, THRESHOLDS, CAMERA_IMG_SIZE
)
from ai_engine.models import DriverRiskCNNLSTM
from ai_engine.feature_extraction import (
    normalize_telemetry_vector,
    calculate_rule_based_risk_score,
    render_synthetic_dms_frame,
    extract_temporal_statistical_features
)


class DriverMonitoringEngine:
    """
    Production-grade Real-Time Driver Monitoring & Risk Engine.
    """
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model: Optional[DriverRiskCNNLSTM] = None
        self.ml_model = None
        self.is_loaded = False
        
        # Sliding temporal buffers (length = SEQUENCE_LENGTH)
        self.telemetry_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.camera_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        # Temporal smoothing state (EMA)
        self.smoothed_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        self.smoothed_probs[0] = 1.0  # Default Normal
        self.smoothed_risk = 15.0
        self.ema_alpha = 0.35  # Smoothing factor (higher = more responsive, lower = smoother)
        
        # Fatigue counter tracking consecutive eye closure
        self.consecutive_closed_eyes = 0
        self.consecutive_distracted_frames = 0
        
        self.load_models()
        self._initialize_buffer()

    def load_models(self):
        """Loads trained CNN-LSTM and ML baseline weights."""
        deep_path = SAVED_MODELS_DIR / "driver_cnn_lstm.pth"
        ml_path = SAVED_MODELS_DIR / "ml_baseline.joblib"
        
        if deep_path.exists():
            try:
                self.model = DriverRiskCNNLSTM().to(self.device)
                checkpoint = torch.load(deep_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                self.is_loaded = True
                print(f"[DriverMonitoringEngine] Loaded CNN-LSTM model from {deep_path} (Val F1: {checkpoint.get('val_f1', 0.0):.4f})")
            except Exception as e:
                print(f"[DriverMonitoringEngine] Error loading CNN-LSTM model: {e}")
                self.model = None
        else:
            print(f"[DriverMonitoringEngine] Checkpoint not found at {deep_path}. Run train.py first.")
            self.model = None
            
        if ml_path.exists():
            try:
                self.ml_model = joblib.load(ml_path)
                print(f"[DriverMonitoringEngine] Loaded ML baseline model from {ml_path}")
            except Exception as e:
                print(f"[DriverMonitoringEngine] Error loading ML baseline: {e}")
                self.ml_model = None

    def _initialize_buffer(self):
        """Pre-populates the buffer with 30 frames of normal baseline driving."""
        nominal_dict = {
            "speed": 60.0,
            "acceleration": 0.0,
            "braking": 0.0,
            "steering": 0.0,
            "lane_deviation": 0.0,
            "eye_closure": 0.85,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "phone_usage": 0.0
        }
        norm_vec = normalize_telemetry_vector(nominal_dict)
        frame = render_synthetic_dms_frame(eye_closure=0.85, head_pitch=0.0, head_yaw=0.0, phone_usage=0.0)
        norm_frame = (frame.astype(np.float32) / 255.0)[np.newaxis, ...]  # (1, 64, 64)
        
        for _ in range(SEQUENCE_LENGTH):
            self.telemetry_buffer.append(norm_vec)
            self.camera_buffer.append(norm_frame)

    def process_step(
        self,
        raw_telemetry: Dict[str, float],
        custom_camera_frame: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Processes a single incoming telemetry step + optional camera frame.
        Updates temporal buffer and executes multi-modal CNN-LSTM inference.
        """
        # 1. Track consecutive fatigue / distraction frames
        eye_ear = float(raw_telemetry.get("eye_closure", 0.85))
        yaw = abs(float(raw_telemetry.get("head_yaw", 0.0)))
        phone = float(raw_telemetry.get("phone_usage", 0.0))
        
        if eye_ear < THRESHOLDS["drowsy_ear_threshold"]:
            self.consecutive_closed_eyes += 1
        else:
            self.consecutive_closed_eyes = 0
            
        if phone > THRESHOLDS["phone_usage_threshold"] or yaw > THRESHOLDS["head_yaw_distracted"]:
            self.consecutive_distracted_frames += 1
        else:
            self.consecutive_distracted_frames = 0
            
        # 2. Camera Frame preparation
        if custom_camera_frame is not None:
            # Resize & convert to grayscale (64, 64)
            if len(custom_camera_frame.shape) == 3:
                gray = cv2.cvtColor(custom_camera_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = custom_camera_frame
            resized = cv2.resize(gray, CAMERA_IMG_SIZE)
            norm_frame = (resized.astype(np.float32) / 255.0)[np.newaxis, ...]  # (1, 64, 64)
            dms_render = resized
        else:
            # Render synthetic DMS frame based on driver metrics
            dms_render = render_synthetic_dms_frame(
                eye_closure=eye_ear,
                head_pitch=float(raw_telemetry.get("head_pitch", 0.0)),
                head_yaw=float(raw_telemetry.get("head_yaw", 0.0)),
                phone_usage=phone
            )
            norm_frame = (dms_render.astype(np.float32) / 255.0)[np.newaxis, ...]
            
        # 3. Normalize & push to buffers
        norm_vec = normalize_telemetry_vector(raw_telemetry)
        self.telemetry_buffer.append(norm_vec)
        self.camera_buffer.append(norm_frame)
        
        # 4. Multi-Modal CNN-LSTM Forward Pass
        if self.model is not None and self.is_loaded:
            tel_seq = torch.tensor(np.array(self.telemetry_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
            cam_seq = torch.tensor(np.array(self.camera_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                out = self.model(telemetry_seq=tel_seq, camera_seq=cam_seq)
                raw_probs = out["probabilities"].squeeze(0).cpu().numpy()
                nn_risk = float(out["risk_score"].item())
                attention = out["attention_weights"].squeeze(0).cpu().numpy().tolist()
        else:
            # Heuristic fallback if model not yet trained
            raw_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
            raw_probs[0] = 0.95
            nn_risk = 15.0
            attention = [1.0 / SEQUENCE_LENGTH] * SEQUENCE_LENGTH
            
        # 5. Rule-based Risk Score & Active Alerts
        rule_risk, alerts = calculate_rule_based_risk_score(raw_telemetry)
        
        # Severe state checks
        if self.consecutive_closed_eyes >= THRESHOLDS["drowsy_closed_frames"]:
            raw_probs[CLASS_TO_IDX["Drowsy"]] = max(raw_probs[CLASS_TO_IDX["Drowsy"]], 0.92)
            rule_risk = max(rule_risk, 88.0)
            if not any("Micro-sleep" in a for a in alerts):
                alerts.append("CRITICAL: Sustained Micro-Sleep Detected (>1.2s eyes shut)!")
                
        # 6. Temporal Exponential Smoothing
        self.smoothed_probs = (self.ema_alpha * raw_probs) + ((1.0 - self.ema_alpha) * self.smoothed_probs)
        # Normalize probabilities sum to 1.0
        self.smoothed_probs = self.smoothed_probs / np.sum(self.smoothed_probs)
        
        # Composite Fused Risk Score: 60% Neural Network + 40% Dynamic Rule Safety Engine
        instant_risk = 0.60 * nn_risk + 0.40 * rule_risk
        self.smoothed_risk = (self.ema_alpha * instant_risk) + ((1.0 - self.ema_alpha) * self.smoothed_risk)
        final_risk = float(np.clip(self.smoothed_risk, 0.0, 100.0))
        
        # 7. Class Prediction & Level Assignment
        pred_idx = int(np.argmax(self.smoothed_probs))
        pred_class = IDX_TO_CLASS[pred_idx]
        pred_confidence = float(self.smoothed_probs[pred_idx])
        
        if final_risk < 30.0:
            risk_level = "SAFE"
        elif final_risk < 60.0:
            risk_level = "CAUTION"
        elif final_risk < 80.0:
            risk_level = "WARNING"
        else:
            risk_level = "CRITICAL"
                
        # Format class probability distribution
        prob_dict = {CLASSES[i]: float(self.smoothed_probs[i]) for i in range(NUM_CLASSES)}
        
        # Encode camera frame to Base64 JPEG for browser transmission
        _, buffer = cv2.imencode(".jpg", dms_render, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        
        return {
            "prediction": {
                "class": pred_class,
                "confidence": pred_confidence,
                "probabilities": prob_dict,
                "class_index": pred_idx
            },
            "risk": {
                "score": round(final_risk, 1),
                "level": risk_level,
                "level_meta": {
                    "color": RISK_LEVELS[risk_level][2],
                    "desc": RISK_LEVELS[risk_level][3]
                },
                "alerts": alerts,
                "is_critical": (final_risk >= 80.0 or risk_level == "CRITICAL")
            },
            "telemetry": raw_telemetry,
            "temporal_attention": attention,
            "camera_frame_base64": f"data:image/jpeg;base64,{frame_b64}"
        }

    def reset(self):
        """Clears buffer and resets smoothing."""
        self.telemetry_buffer.clear()
        self.camera_buffer.clear()
        self.smoothed_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        self.smoothed_probs[0] = 1.0
        self.smoothed_risk = 15.0
        self.consecutive_closed_eyes = 0
        self.consecutive_distracted_frames = 0
        self._initialize_buffer()
