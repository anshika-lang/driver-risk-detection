"""
Deep Learning & Classical ML Models for Driver Behaviour, Fatigue and Risk Detection.
Contains:
1. SpatialVisionCNN: Feature extraction from driver camera frames.
2. TelemetryEncoder: Sensor representation network.
3. DriverRiskCNNLSTM: End-to-end multi-modal spatial-temporal neural network with dual heads.
4. ClassicalMLClassifier: Random Forest / Gradient Boosting baseline for explainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from ai_engine.config import (
    NUM_CLASSES, NUM_TELEMETRY_FEATURES, CNN_FEATURE_DIM,
    TELEMETRY_EMBED_DIM, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS,
    DROPOUT_RATE, SEQUENCE_LENGTH, CLASSES
)


class SpatialVisionCNN(nn.Module):
    """
    Lightweight 2D CNN that processes driver cabin / face frames
    to extract spatial cues (eye openness, head orientation, phone presence).
    Input: (B, 1, 64, 64) -> Output: (B, CNN_FEATURE_DIM)
    """
    def __init__(self, in_channels: int = 1, out_dim: int = CNN_FEATURE_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 64, 64)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        emb = self.fc(x)
        return emb


class TelemetryEncoder(nn.Module):
    """
    Encodes dense vehicle telemetry parameters (speed, accel, braking, steering,
    lane deviation, EAR, head pose, phone usage) at each time step.
    Input: (B, NUM_TELEMETRY_FEATURES) -> Output: (B, TELEMETRY_EMBED_DIM)
    """
    def __init__(self, in_features: int = NUM_TELEMETRY_FEATURES, out_dim: int = TELEMETRY_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 48),
            nn.LayerNorm(48),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(48, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalAttention(nn.Module):
    """
    Learned attention mechanism over the temporal sequence
    to weigh critical moments (e.g. sharp deceleration, micro-sleep duration).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_out: (B, T, hidden_dim)
        scores = self.attn(lstm_out)           # (B, T, 1)
        weights = F.softmax(scores, dim=1)     # (B, T, 1)
        context = torch.sum(lstm_out * weights, dim=1)  # (B, hidden_dim)
        return context, weights


class DriverRiskCNNLSTM(nn.Module):
    """
    End-to-end Multi-modal Architecture:
    Camera Frames + Vehicle Sensors
         ↓
    Spatial CNN + Telemetry MLP
         ↓
    Fused Embedding Sequence (T = 30)
         ↓
    Bi-Directional LSTM + Temporal Attention
         ↓
    Dual Heads:
    1. Driver Behavior Classifier (6 classes)
    2. Continuous Risk Score Regressor (0 - 100)
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        vision_dim: int = CNN_FEATURE_DIM,
        telemetry_dim: int = TELEMETRY_EMBED_DIM,
        lstm_hidden: int = LSTM_HIDDEN_DIM,
        num_lstm_layers: int = LSTM_NUM_LAYERS,
        dropout: float = DROPOUT_RATE
    ):
        super().__init__()
        self.vision_encoder = SpatialVisionCNN(in_channels=1, out_dim=vision_dim)
        self.telemetry_encoder = TelemetryEncoder(in_features=NUM_TELEMETRY_FEATURES, out_dim=telemetry_dim)
        
        fused_dim = vision_dim + telemetry_dim  # 32 + 32 = 64
        
        # Bidirectional LSTM captures forward and backward temporal context
        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        
        lstm_out_dim = lstm_hidden * 2  # Bidirectional
        self.attention = TemporalAttention(lstm_out_dim)
        
        # Head 1: Behavior Classification (6 classes)
        self.classifier_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )
        
        # Head 2: Continuous Risk Score (0.0 to 100.0)
        self.risk_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Scaled by 100 in forward
        )

    def forward(
        self,
        telemetry_seq: torch.Tensor,
        camera_seq: Optional[torch.Tensor] = None,
        precomputed_vision_emb: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        telemetry_seq: (B, T, 9)
        camera_seq: (B, T, 1, 64, 64) - Optional if precomputed_vision_emb is passed
        precomputed_vision_emb: (B, T, CNN_FEATURE_DIM)
        """
        B, T, _ = telemetry_seq.shape
        
        # 1. Encode Telemetry across all time steps
        # Flatten (B, T, 9) -> (B*T, 9)
        tel_flat = telemetry_seq.view(B * T, -1)
        tel_emb = self.telemetry_encoder(tel_flat)
        tel_emb = tel_emb.view(B, T, -1)  # (B, T, 32)
        
        # 2. Encode Vision
        if precomputed_vision_emb is not None:
            vis_emb = precomputed_vision_emb
        elif camera_seq is not None:
            cam_flat = camera_seq.view(B * T, 1, 64, 64)
            vis_emb = self.vision_encoder(cam_flat)
            vis_emb = vis_emb.view(B, T, -1)  # (B, T, 32)
        else:
            # Fallback zero visual embedding if camera is disabled
            vis_emb = torch.zeros(B, T, CNN_FEATURE_DIM, device=telemetry_seq.device)
            
        # 3. Early Multi-Modal Fusion at each time step
        fused = torch.cat([tel_emb, vis_emb], dim=-1)  # (B, T, 64)
        
        # 4. Temporal LSTM Sequence Modeling
        lstm_out, _ = self.lstm(fused)  # (B, T, 128)
        
        # 5. Temporal Attention
        context, attn_weights = self.attention(lstm_out)  # context: (B, 128)
        
        # 6. Dual Heads
        logits = self.classifier_head(context)           # (B, 6)
        probs = F.softmax(logits, dim=-1)
        
        risk_score = self.risk_head(context) * 100.0     # (B, 1) in [0, 100]
        
        return {
            "logits": logits,
            "probabilities": probs,
            "risk_score": risk_score.squeeze(-1),
            "attention_weights": attn_weights.squeeze(-1)
        }


class ClassicalMLClassifier:
    """
    Classical Machine Learning Baseline (Random Forest & Gradient Boosting)
    trained on aggregated statistical & domain-specific features.
    Provides feature importances and validation against the Deep CNN-LSTM.
    """
    def __init__(self):
        self.rf_model = RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.gb_model = HistGradientBoostingClassifier(
            max_iter=100,
            random_state=42
        )
        self.feature_names = [
            "speed_mean", "speed_std", "speed_max",
            "accel_mean", "accel_std", "accel_max", "accel_min", "jerk_mean", "jerk_max",
            "brake_mean", "brake_max", "brake_std",
            "steer_std", "steer_max_abs", "steer_rate",
            "lane_dev_mean", "lane_dev_std", "lane_dev_max",
            "eye_ear_mean", "eye_ear_min", "perclos",
            "head_pitch_mean", "head_pitch_min",
            "head_yaw_mean", "head_yaw_std", "head_yaw_max",
            "phone_mean", "phone_max", "phone_active_ratio"
        ]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.rf_model.fit(X, y)
        self.gb_model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.rf_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.rf_model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        importances = self.rf_model.feature_importances_
        sorted_pairs = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
        return {name: float(imp) for name, imp in sorted_pairs}

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        cm = confusion_matrix(y_test, preds).tolist()
        report = classification_report(y_test, preds, target_names=CLASSES, output_dict=True)
        
        return {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "confusion_matrix": cm,
            "classification_report": report,
            "feature_importance": self.get_feature_importance()
        }
