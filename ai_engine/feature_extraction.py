"""
Feature Extraction module for Camera + Vehicle Sensors.
Provides parameter normalization, temporal feature engineering,
visual face/eye ROI generation, and heuristic risk scoring.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any
from ai_engine.config import (
    FEATURE_BOUNDS, SENSOR_PARAMS, THRESHOLDS, CAMERA_IMG_SIZE
)

def normalize_telemetry_vector(raw_dict: Dict[str, float]) -> np.ndarray:
    """
    Normalizes a dictionary of raw sensor parameters into a [0, 1] bounded vector.
    """
    normalized = []
    for param in SENSOR_PARAMS:
        val = float(raw_dict.get(param, 0.0))
        min_val, max_val = FEATURE_BOUNDS[param]
        # Clip to min-max then normalize to [0, 1]
        clamped = np.clip(val, min_val, max_val)
        norm_val = (clamped - min_val) / (max_val - min_val + 1e-6)
        normalized.append(norm_val)
    return np.array(normalized, dtype=np.float32)


def denormalize_telemetry_vector(norm_vec: np.ndarray) -> Dict[str, float]:
    """
    Converts a normalized [0, 1] vector back to human-interpretable engineering units.
    """
    res = {}
    for idx, param in enumerate(SENSOR_PARAMS):
        min_val, max_val = FEATURE_BOUNDS[param]
        val = norm_vec[idx] * (max_val - min_val) + min_val
        res[param] = float(val)
    return res


def extract_temporal_statistical_features(sequence_window: np.ndarray) -> np.ndarray:
    """
    Extracts statistical and domain-specific aggregate features from a temporal sequence
    of shape (SEQUENCE_LENGTH, NUM_TELEMETRY_FEATURES).
    Used for classical ML algorithms (Random Forest, Gradient Boosting, SVM).
    
    Returns a 1D feature vector containing:
    - Speed: mean, std, max
    - Acceleration: mean, std, max, min, jerk_mean, jerk_max
    - Braking: mean, max, std
    - Steering: std, max_abs, rate_of_change
    - Lane deviation: mean, std, max_abs
    - Eye closure: mean, min, PERCLOS (ratio where EAR < drowsy_threshold)
    - Head pitch: mean, min (head droop)
    - Head yaw: mean, std, max_abs (glance away)
    - Phone usage: mean, max, active_frames_ratio
    """
    # sequence_window shape: (T, 9)
    # Mapping indices:
    # 0: speed, 1: accel, 2: brake, 3: steer, 4: lane_dev,
    # 5: eye_closure, 6: head_pitch, 7: head_yaw, 8: phone_usage
    feats = []
    
    speed = sequence_window[:, 0]
    accel = sequence_window[:, 1]
    brake = sequence_window[:, 2]
    steer = sequence_window[:, 3]
    lane_dev = sequence_window[:, 4]
    eye_ear = sequence_window[:, 5]
    pitch = sequence_window[:, 6]
    yaw = sequence_window[:, 7]
    phone = sequence_window[:, 8]
    
    # Speed features
    feats.extend([np.mean(speed), np.std(speed), np.max(speed)])
    
    # Acceleration & Jerk
    jerk = np.diff(accel)
    feats.extend([
        np.mean(accel), np.std(accel), np.max(accel), np.min(accel),
        np.mean(np.abs(jerk)) if len(jerk) > 0 else 0.0,
        np.max(np.abs(jerk)) if len(jerk) > 0 else 0.0
    ])
    
    # Braking features
    feats.extend([np.mean(brake), np.max(brake), np.std(brake)])
    
    # Steering dynamics
    steer_diff = np.diff(steer)
    feats.extend([
        np.std(steer), np.max(np.abs(steer)),
        np.mean(np.abs(steer_diff)) if len(steer_diff) > 0 else 0.0
    ])
    
    # Lane deviation
    feats.extend([np.mean(lane_dev), np.std(lane_dev), np.max(np.abs(lane_dev))])
    
    # Eye closure & Fatigue (PERCLOS: Percentage of Eye Closure)
    perclos = np.mean(eye_ear < THRESHOLDS["drowsy_ear_threshold"])
    feats.extend([np.mean(eye_ear), np.min(eye_ear), perclos])
    
    # Head pose (Drowsy nod and Distracted glance)
    feats.extend([
        np.mean(pitch), np.min(pitch),
        np.mean(yaw), np.std(yaw), np.max(np.abs(yaw))
    ])
    
    # Phone distraction
    phone_active = np.mean(phone > THRESHOLDS["phone_usage_threshold"])
    feats.extend([np.mean(phone), np.max(phone), phone_active])
    
    return np.array(feats, dtype=np.float32)


def render_synthetic_dms_frame(
    eye_closure: float,
    head_pitch: float,
    head_yaw: float,
    phone_usage: float,
    frame_size: Tuple[int, int] = CAMERA_IMG_SIZE
) -> np.ndarray:
    """
    Renders a realistic 64x64 grayscale Driver Monitoring System (DMS) frame
    reflecting eye state, head pose, and phone presence.
    Used for multimodal CNN-LSTM visual pipeline testing and simulation.
    """
    h, w = frame_size
    img = np.full((h, w), 35, dtype=np.uint8)  # Cabin dark background
    
    # Face center shifted by head yaw and pitch
    center_x = int(w // 2 + (head_yaw / 60.0) * (w * 0.25))
    center_y = int(h // 2 - (head_pitch / 45.0) * (h * 0.20))
    center_x = int(np.clip(center_x, 15, w - 15))
    center_y = int(np.clip(center_y, 15, h - 15))
    
    # Draw head oval
    face_radius_x = int(w * 0.28)
    face_radius_y = int(h * 0.36)
    cv2.ellipse(img, (center_x, center_y), (face_radius_x, face_radius_y), 0, 0, 360, 180, -1)
    
    # Eye positions
    eye_spacing = int(face_radius_x * 0.55)
    eye_y = int(center_y - face_radius_y * 0.15)
    left_eye_x = center_x - eye_spacing
    right_eye_x = center_x + eye_spacing
    
    # Eye openness height scaled by EAR (eye_closure in [0, 1])
    eye_h = max(1, int(face_radius_y * 0.18 * eye_closure))
    eye_w = int(face_radius_x * 0.25)
    
    # Draw eyes
    cv2.ellipse(img, (left_eye_x, eye_y), (eye_w, eye_h), 0, 0, 360, 240, -1)
    cv2.ellipse(img, (right_eye_x, eye_y), (eye_w, eye_h), 0, 0, 360, 240, -1)
    
    # Draw pupils if eyes open
    if eye_h > 2:
        cv2.circle(img, (left_eye_x, eye_y), max(1, eye_h // 2), 60, -1)
        cv2.circle(img, (right_eye_x, eye_y), max(1, eye_h // 2), 60, -1)
        
    # Draw mouth
    mouth_y = int(center_y + face_radius_y * 0.45)
    cv2.ellipse(img, (center_x, mouth_y), (int(face_radius_x * 0.35), 2), 0, 0, 360, 100, -1)
    
    # If phone usage is active, draw phone object near driver's hand/ear
    if phone_usage > 0.3:
        phone_intensity = int(140 + 100 * phone_usage)
        p_x = int(np.clip(center_x + (face_radius_x * 0.75), 10, w - 12))
        p_y = int(np.clip(center_y + 4, 10, h - 20))
        cv2.rectangle(img, (p_x, p_y), (p_x + 8, p_y + 16), phone_intensity, -1)
        # Phone screen glow
        cv2.rectangle(img, (p_x + 1, p_y + 2), (p_x + 7, p_y + 14), 255, -1)
        
    # Add subtle sensor noise for realism
    noise = np.random.normal(0, 3.0, (h, w)).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return noisy_img


def calculate_rule_based_risk_score(raw_telemetry: Dict[str, float]) -> Tuple[float, List[str]]:
    """
    Computes a deterministic expert safety risk score [0, 100] and active risk alerts
    based on the 8 parameters and domain threshold rules.
    """
    alerts = []
    risk = 10.0  # Base nominal driving risk
    
    speed = raw_telemetry.get("speed", 0.0)
    accel = raw_telemetry.get("acceleration", 0.0)
    brake = raw_telemetry.get("braking", 0.0)
    steer = abs(raw_telemetry.get("steering", 0.0))
    lane_dev = abs(raw_telemetry.get("lane_deviation", 0.0))
    eye_ear = raw_telemetry.get("eye_closure", 1.0)
    pitch = raw_telemetry.get("head_pitch", 0.0)
    yaw = abs(raw_telemetry.get("head_yaw", 0.0))
    phone = raw_telemetry.get("phone_usage", 0.0)
    
    # 1. Drowsiness / Fatigue checks
    if eye_ear < THRESHOLDS["drowsy_ear_threshold"]:
        fatigue_severity = (THRESHOLDS["drowsy_ear_threshold"] - eye_ear) / THRESHOLDS["drowsy_ear_threshold"]
        risk += 45.0 * (1.0 + fatigue_severity)
        alerts.append("DROWSINESS: Driver eyes closed (Micro-sleep hazard)")
    if pitch < THRESHOLDS["head_pitch_drowsy"]:
        risk += 25.0
        alerts.append("FATIGUE: Driver head nodding / droop detected")
        
    # 2. Distracted Driving
    if phone > THRESHOLDS["phone_usage_threshold"]:
        risk += 40.0 * phone
        alerts.append(f"DISTRACTION: Smartphone interaction ({int(phone * 100)}% conf)")
    if yaw > THRESHOLDS["head_yaw_distracted"]:
        risk += 30.0
        alerts.append(f"DISTRACTION: Gaze averted ({yaw:.1f}° off-road)")
        
    # 3. Sudden / Unsafe Braking
    if brake > THRESHOLDS["unsafe_brake_threshold"]:
        risk += 35.0 * (brake / 100.0)
        alerts.append(f"HAZARD: Severe sudden braking ({brake:.0f}% pressure)")
        
    # 4. Aggressive / Unsafe Acceleration
    if accel > THRESHOLDS["unsafe_accel_threshold"]:
        risk += 30.0 * (accel / 8.0)
        alerts.append(f"AGGRESSIVE: Excessive longitudinal acceleration ({accel:.1f} m/s²)")
    elif accel < -5.0:
        risk += 25.0
        alerts.append(f"HAZARD: High deceleration spike ({accel:.1f} m/s²)")
        
    # 5. Lane Deviation & High-Speed Instability
    if lane_dev > THRESHOLDS["lane_drift_threshold"]:
        risk += 25.0 * (lane_dev / 2.5)
        alerts.append(f"LANE DRIFT: Vehicle departing lane center ({lane_dev:.2f} m)")
        
    if speed > 130.0:
        risk += 15.0 * ((speed - 130.0) / 50.0)
        alerts.append(f"SPEEDING: High velocity ({speed:.0f} km/h)")
        
    if steer > 60.0 and speed > 60.0:
        risk += 25.0
        alerts.append("AGGRESSIVE: High steering rate at speed")
        
    risk_score = float(np.clip(risk, 0.0, 100.0))
    return risk_score, alerts
