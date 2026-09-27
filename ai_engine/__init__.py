"""
AI-Based Driver Behaviour, Fatigue and Risk Detection System
Package Initialization
"""

from ai_engine.config import (
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    SENSOR_PARAMS,
    FEATURE_BOUNDS,
    RISK_LEVELS,
    THRESHOLDS
)
from ai_engine.feature_extraction import (
    normalize_telemetry_vector,
    denormalize_telemetry_vector,
    extract_temporal_statistical_features,
    calculate_rule_based_risk_score,
    render_synthetic_dms_frame
)
from ai_engine.models import (
    SpatialVisionCNN,
    TelemetryEncoder,
    DriverRiskCNNLSTM,
    ClassicalMLClassifier
)
from ai_engine.inference import DriverMonitoringEngine

__all__ = [
    "CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "SENSOR_PARAMS",
    "FEATURE_BOUNDS",
    "RISK_LEVELS",
    "THRESHOLDS",
    "normalize_telemetry_vector",
    "denormalize_telemetry_vector",
    "extract_temporal_statistical_features",
    "calculate_rule_based_risk_score",
    "render_synthetic_dms_frame",
    "SpatialVisionCNN",
    "TelemetryEncoder",
    "DriverRiskCNNLSTM",
    "ClassicalMLClassifier",
    "DriverMonitoringEngine"
]
