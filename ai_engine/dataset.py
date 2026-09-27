"""
Dataset module for AI-Based Driver Behaviour, Fatigue and Risk Detection.
Generates realistic multi-modal driving telemetry and visual driver monitoring sequences.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional
from pathlib import Path

from ai_engine.config import (
    CLASSES, CLASS_TO_IDX, NUM_CLASSES, SEQUENCE_LENGTH,
    FEATURE_BOUNDS, DATA_DIR, CAMERA_IMG_SIZE, NUM_TELEMETRY_FEATURES
)
from ai_engine.feature_extraction import (
    normalize_telemetry_vector,
    extract_temporal_statistical_features,
    render_synthetic_dms_frame
)


def generate_driving_sequence(
    behavior_class: str,
    seq_length: int = SEQUENCE_LENGTH,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Synthesizes a realistic 30-step driving trajectory (telemetry + camera frames)
    for a given behavior class according to automotive physics and human driver biomechanics.
    
    Returns:
    - telemetry_seq: (T, 9) normalized [0, 1]
    - camera_frames: (T, 1, 64, 64) float32 [0, 1]
    - target_risk_score: float [0, 100]
    """
    if rng is None:
        rng = np.random.RandomState()
        
    t_steps = np.linspace(0, 3.0, seq_length)
    
    # Raw parameter trajectories over T steps
    speed = np.zeros(seq_length, dtype=np.float32)
    accel = np.zeros(seq_length, dtype=np.float32)
    brake = np.zeros(seq_length, dtype=np.float32)
    steer = np.zeros(seq_length, dtype=np.float32)
    lane_dev = np.zeros(seq_length, dtype=np.float32)
    eye_ear = np.zeros(seq_length, dtype=np.float32)
    pitch = np.zeros(seq_length, dtype=np.float32)
    yaw = np.zeros(seq_length, dtype=np.float32)
    phone = np.zeros(seq_length, dtype=np.float32)
    
    base_risk = 15.0
    
    if behavior_class == "Normal":
        base_speed = rng.uniform(55.0, 85.0)
        speed[:] = base_speed + rng.normal(0, 1.2, seq_length)
        accel[:] = rng.normal(0.1, 0.4, seq_length)
        brake[:] = np.clip(rng.exponential(1.5, seq_length), 0, 15)
        steer[:] = rng.normal(0, 3.0, seq_length)
        lane_dev[:] = rng.normal(0, 0.15, seq_length)
        
        # Natural blinking (EAR mostly ~0.75-0.85, occasional dip)
        eye_ear[:] = rng.uniform(0.70, 0.88, seq_length)
        blink_idx = rng.randint(5, seq_length - 5)
        eye_ear[blink_idx:blink_idx + 2] = 0.12  # Normal quick blink
        
        pitch[:] = rng.normal(-2.0, 2.5, seq_length)
        yaw[:] = rng.normal(0.0, 3.5, seq_length)
        phone[:] = 0.0
        target_risk = float(rng.uniform(8.0, 24.0))

    elif behavior_class == "Aggressive":
        base_speed = rng.uniform(90.0, 140.0)
        speed[:] = base_speed + np.cumsum(rng.normal(0.5, 1.8, seq_length))
        # Erratic throttle and lateral maneuvering
        accel[:] = rng.uniform(1.8, 4.2, seq_length) + rng.normal(0, 0.6, seq_length)
        brake[:] = np.where(rng.rand(seq_length) > 0.7, rng.uniform(25.0, 55.0, seq_length), 0.0)
        # Frequent weaving / lane changing
        steer[:] = 28.0 * np.sin(2.5 * t_steps) + rng.normal(0, 6.0, seq_length)
        lane_dev[:] = 1.1 * np.sin(2.5 * t_steps) + rng.normal(0, 0.2, seq_length)
        
        eye_ear[:] = rng.uniform(0.75, 0.95, seq_length)  # Eyes wide open
        pitch[:] = rng.normal(1.0, 3.0, seq_length)
        yaw[:] = 15.0 * np.sin(2.5 * t_steps) + rng.normal(0, 5.0, seq_length)
        phone[:] = 0.0
        target_risk = float(rng.uniform(60.0, 78.0))

    elif behavior_class == "Distracted":
        speed[:] = rng.uniform(45.0, 75.0, seq_length) + rng.normal(0, 2.5, seq_length)
        accel[:] = rng.normal(0.0, 0.8, seq_length)
        brake[:] = np.where(rng.rand(seq_length) > 0.85, rng.uniform(15.0, 40.0, seq_length), 0.0)
        # Inattentive lane drift
        drift_direction = rng.choice([-1.0, 1.0])
        lane_dev[:] = drift_direction * (0.2 + 0.35 * t_steps) + rng.normal(0, 0.08, seq_length)
        steer[:] = rng.normal(0, 1.5, seq_length)  # Frozen steering due to distraction
        
        eye_ear[:] = rng.uniform(0.55, 0.75, seq_length)
        # Head turned down/sideways looking at phone
        is_phone = rng.rand() > 0.3
        if is_phone:
            phone[:] = rng.uniform(0.75, 1.0, seq_length)
            yaw[:] = rng.uniform(25.0, 48.0, seq_length) * drift_direction
            pitch[:] = rng.uniform(-25.0, -12.0, seq_length)
        else:
            phone[:] = 0.0
            yaw[:] = rng.uniform(30.0, 55.0, seq_length) * drift_direction
            pitch[:] = rng.normal(-5.0, 4.0, seq_length)
            
        target_risk = float(rng.uniform(68.0, 86.0))

    elif behavior_class == "Drowsy":
        speed[:] = rng.uniform(50.0, 80.0, seq_length) - (0.8 * t_steps)
        accel[:] = rng.normal(-0.2, 0.3, seq_length)
        brake[:] = 0.0
        # Wandering trajectory as consciousness fades
        lane_dev[:] = 0.15 * (t_steps ** 1.8) * rng.choice([-1.0, 1.0]) + rng.normal(0, 0.05, seq_length)
        steer[:] = rng.normal(0, 1.2, seq_length)
        
        # Prolonged eye closure (micro-sleep episode)
        onset = rng.randint(3, 10)
        eye_ear[:] = rng.uniform(0.65, 0.80, seq_length)
        eye_ear[onset:onset + 18] = rng.uniform(0.02, 0.15, min(18, seq_length - onset))
        
        # Head dropping down onto chest
        pitch[:] = -5.0 - (8.0 * t_steps) + rng.normal(0, 1.5, seq_length)
        pitch = np.clip(pitch, -40.0, 0.0)
        yaw[:] = rng.normal(0.0, 4.0, seq_length)
        phone[:] = 0.0
        target_risk = float(rng.uniform(78.0, 96.0))

    elif behavior_class == "Unsafe braking":
        # Cruising at speed then panic emergency stop
        speed[:] = 70.0
        brake_onset = rng.randint(8, 14)
        for i in range(seq_length):
            if i < brake_onset:
                speed[i] = 70.0 + rng.normal(0, 1.0)
                accel[i] = rng.normal(0.0, 0.3)
                brake[i] = 0.0
            else:
                brake_intensity = rng.uniform(75.0, 98.0)
                brake[i] = brake_intensity
                accel[i] = rng.uniform(-7.5, -5.2)  # Extreme deceleration
                speed[i] = max(0.0, speed[i - 1] - 4.5)
                
        steer[:] = rng.normal(0, 4.5, seq_length)
        lane_dev[:] = rng.normal(0, 0.25, seq_length)
        eye_ear[:] = rng.uniform(0.75, 0.95, seq_length)  # Eyes wide in surprise
        # Head thrown forward by deceleration
        pitch[:] = np.where(np.arange(seq_length) >= brake_onset, rng.uniform(-25.0, -15.0), -2.0)
        yaw[:] = rng.normal(0.0, 3.0, seq_length)
        phone[:] = 0.0
        target_risk = float(rng.uniform(82.0, 97.0))

    elif behavior_class == "Unsafe acceleration":
        # Launch / sudden flooring of accelerator
        accel_onset = rng.randint(5, 10)
        for i in range(seq_length):
            if i < accel_onset:
                speed[i] = rng.uniform(10.0, 25.0)
                accel[i] = rng.normal(0.5, 0.3)
                brake[i] = 0.0
            else:
                accel[i] = rng.uniform(5.2, 7.8)  # Dangerous acceleration surge
                speed[i] = min(170.0, speed[i - 1] + (accel[i] * 0.3 * 3.6))
                brake[i] = 0.0
                
        # Torque steer wobble
        steer[:] = np.where(
            np.arange(seq_length) >= accel_onset,
            18.0 * np.sin(3.0 * t_steps) + rng.normal(0, 3.0, seq_length),
            rng.normal(0, 2.0, seq_length)
        )
        lane_dev[:] = rng.normal(0, 0.35, seq_length)
        eye_ear[:] = rng.uniform(0.70, 0.90, seq_length)
        # Head thrown backward against headrest
        pitch[:] = np.where(np.arange(seq_length) >= accel_onset, rng.uniform(12.0, 24.0), -2.0)
        yaw[:] = rng.normal(0.0, 4.0, seq_length)
        phone[:] = 0.0
        target_risk = float(rng.uniform(72.0, 90.0))

    else:
        raise ValueError(f"Unknown class: {behavior_class}")

    # Build normalized telemetry sequence matrix (T, 9)
    telemetry_raw = np.column_stack([
        speed, accel, brake, steer, lane_dev,
        eye_ear, pitch, yaw, phone
    ])
    
    telemetry_norm = np.zeros_like(telemetry_raw, dtype=np.float32)
    camera_frames = np.zeros((seq_length, 1, CAMERA_IMG_SIZE[0], CAMERA_IMG_SIZE[1]), dtype=np.float32)
    
    for t in range(seq_length):
        raw_dict = {
            "speed": float(speed[t]),
            "acceleration": float(accel[t]),
            "braking": float(brake[t]),
            "steering": float(steer[t]),
            "lane_deviation": float(lane_dev[t]),
            "eye_closure": float(eye_ear[t]),
            "head_pitch": float(pitch[t]),
            "head_yaw": float(yaw[t]),
            "phone_usage": float(phone[t])
        }
        telemetry_norm[t] = normalize_telemetry_vector(raw_dict)
        
        # Render corresponding DMS camera frame
        frame = render_synthetic_dms_frame(
            eye_closure=eye_ear[t],
            head_pitch=pitch[t],
            head_yaw=yaw[t],
            phone_usage=phone[t]
        )
        camera_frames[t, 0] = frame.astype(np.float32) / 255.0

    return telemetry_norm, camera_frames, target_risk


class DriverDataset(Dataset):
    """
    PyTorch Dataset providing multi-modal (telemetry + camera) sequence pairs.
    """
    def __init__(
        self,
        telemetry: np.ndarray,
        camera: np.ndarray,
        labels: np.ndarray,
        risk_scores: np.ndarray
    ):
        self.telemetry = torch.tensor(telemetry, dtype=torch.float32)
        self.camera = torch.tensor(camera, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.risk_scores = torch.tensor(risk_scores, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "telemetry": self.telemetry[idx],
            "camera": self.camera[idx],
            "label": self.labels[idx],
            "risk_score": self.risk_scores[idx]
        }


def build_or_load_dataset(
    num_samples_per_class: int = 150,
    seed: int = 42,
    force_rebuild: bool = False
) -> Tuple[Dataset, Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates balanced multi-modal datasets for deep learning and ML baselines.
    Caches the generated dataset to disk (.npz) for fast restarts.
    """
    cache_file = DATA_DIR / f"driver_dataset_{num_samples_per_class}_{seed}.npz"
    
    if cache_file.exists() and not force_rebuild:
        print(f"Loading cached dataset from {cache_file}...")
        data = np.load(cache_file)
        train_tel = data["train_tel"]
        train_cam = data["train_cam"]
        train_labels = data["train_labels"]
        train_risk = data["train_risk"]
        
        val_tel = data["val_tel"]
        val_cam = data["val_cam"]
        val_labels = data["val_labels"]
        val_risk = data["val_risk"]
        
        train_ml_X = data["train_ml_X"]
        val_ml_X = data["val_ml_X"]
    else:
        print(f"Synthesizing {num_samples_per_class * NUM_CLASSES} multi-modal driving sequences...")
        rng = np.random.RandomState(seed)
        
        all_tel = []
        all_cam = []
        all_labels = []
        all_risk = []
        all_ml_feats = []
        
        for cls_name in CLASSES:
            cls_idx = CLASS_TO_IDX[cls_name]
            for _ in range(num_samples_per_class):
                tel_seq, cam_seq, risk = generate_driving_sequence(cls_name, SEQUENCE_LENGTH, rng)
                ml_feats = extract_temporal_statistical_features(tel_seq)
                
                all_tel.append(tel_seq)
                all_cam.append(cam_seq)
                all_labels.append(cls_idx)
                all_risk.append(risk)
                all_ml_feats.append(ml_feats)
                
        all_tel = np.array(all_tel, dtype=np.float32)
        all_cam = np.array(all_cam, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)
        all_risk = np.array(all_risk, dtype=np.float32)
        all_ml_feats = np.array(all_ml_feats, dtype=np.float32)
        
        # Shuffle indices
        indices = rng.permutation(len(all_labels))
        split_idx = int(0.8 * len(indices))
        
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        train_tel, val_tel = all_tel[train_idx], all_tel[val_idx]
        train_cam, val_cam = all_cam[train_idx], all_cam[val_idx]
        train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]
        train_risk, val_risk = all_risk[train_idx], all_risk[val_idx]
        train_ml_X, val_ml_X = all_ml_feats[train_idx], all_ml_feats[val_idx]
        
        np.savez_compressed(
            cache_file,
            train_tel=train_tel, train_cam=train_cam, train_labels=train_labels, train_risk=train_risk,
            val_tel=val_tel, val_cam=val_cam, val_labels=val_labels, val_risk=val_risk,
            train_ml_X=train_ml_X, val_ml_X=val_ml_X
        )
        print(f"Dataset successfully generated and saved to {cache_file}.")

    train_dataset = DriverDataset(train_tel, train_cam, train_labels, train_risk)
    val_dataset = DriverDataset(val_tel, val_cam, val_labels, val_risk)
    
    return train_dataset, val_dataset, train_ml_X, val_ml_X, train_labels, val_labels
