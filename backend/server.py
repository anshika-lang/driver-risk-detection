"""
FastAPI Server for AI-Based Driver Behaviour, Fatigue and Risk Detection System.
Provides REST APIs and WebSocket endpoints for real-time telemetry streaming,
model inference, scenario simulations, and metrics visualization.
"""

import sys
import os
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_engine.config import (
    CLASSES, SENSOR_PARAMS, FEATURE_BOUNDS, RISK_LEVELS,
    THRESHOLDS, SAVED_MODELS_DIR, SEQUENCE_LENGTH
)
from ai_engine.inference import DriverMonitoringEngine
from ai_engine.dataset import generate_driving_sequence

# Initialize FastAPI application
app = FastAPI(
    title="AI-Based Driver Behaviour, Fatigue and Risk Detection System",
    description="Multi-modal AI DMS integrating Camera and Vehicle Telemetry using CNN + LSTM and ML baseline.",
    version="1.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Driver Monitoring Engine instance
engine = DriverMonitoringEngine()

# Paths
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)


class TelemetryInput(BaseModel):
    speed: float = 65.0
    acceleration: float = 0.0
    braking: float = 0.0
    steering: float = 0.0
    lane_deviation: float = 0.0
    eye_closure: float = 0.85
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    phone_usage: float = 0.0


class ScenarioRequest(BaseModel):
    scenario_name: str
    step_index: int = 0


@app.get("/api/status")
async def get_system_status():
    """Returns AI Engine and system status."""
    return {
        "status": "online",
        "model_loaded": engine.is_loaded,
        "device": str(engine.device),
        "classes": CLASSES,
        "parameters": SENSOR_PARAMS,
        "bounds": FEATURE_BOUNDS,
        "thresholds": THRESHOLDS,
        "risk_levels": RISK_LEVELS,
        "sequence_length": SEQUENCE_LENGTH
    }


@app.get("/api/metrics")
async def get_model_metrics():
    """Returns trained model performance metrics, confusion matrices, and feature importance."""
    metrics_path = SAVED_MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return data
    else:
        # Fallback benchmark metrics if training is in progress
        return {
            "status": "pending_or_default",
            "message": "Metrics will be available once training finishes.",
            "classes": CLASSES
        }


@app.post("/api/predict")
async def predict_telemetry(telemetry: TelemetryInput):
    """
    Ingests a single telemetry step, updates sliding window,
    and runs real-time CNN-LSTM inference.
    """
    raw_dict = telemetry.model_dump()
    result = engine.process_step(raw_dict)
    return result


@app.post("/api/reset")
async def reset_engine():
    """Resets temporal sliding buffer and smoothing state."""
    engine.reset()
    return {"status": "success", "message": "Buffer reset to nominal baseline."}


@app.get("/api/scenarios")
async def get_scenarios():
    """List of available demonstration scenarios."""
    scenarios = [
        {
            "id": "Normal",
            "name": "Normal Highway Cruise",
            "description": "Steady highway speed (70 km/h), attentive gaze, gentle lane keeping.",
            "expected_class": "Normal",
            "icon": "shield-check",
            "badge_color": "#10b981"
        },
        {
            "id": "Aggressive",
            "name": "Aggressive Weaving & High Acceleration",
            "description": "High speed (120+ km/h), rapid lane weaving, erratic steering, harsh throttle.",
            "expected_class": "Aggressive",
            "icon": "zap",
            "badge_color": "#f59e0b"
        },
        {
            "id": "Distracted",
            "name": "Smartphone Texting & Averted Gaze",
            "description": "Active phone usage, driver head turned 40° away from windshield, slow lane drift.",
            "expected_class": "Distracted",
            "icon": "smartphone",
            "badge_color": "#f97316"
        },
        {
            "id": "Drowsy",
            "name": "Driver Fatigue & Micro-Sleep Episode",
            "description": "Prolonged eye closure (>1.5s), head nodding down, progressive vehicle drifting.",
            "expected_class": "Drowsy",
            "icon": "moon",
            "badge_color": "#ef4444"
        },
        {
            "id": "Unsafe braking",
            "name": "Sudden Panic Emergency Braking",
            "description": "Sudden 95% brake pressure spike, severe -7 m/s² deceleration, driver forward inertia.",
            "expected_class": "Unsafe braking",
            "icon": "alert-octagon",
            "badge_color": "#ef4444"
        },
        {
            "id": "Unsafe acceleration",
            "name": "Excessive Throttle & Jackrabbit Launch",
            "description": "Floored accelerator (+6.5 m/s²), wheel spin, torque steer wobble.",
            "expected_class": "Unsafe acceleration",
            "icon": "trending-up",
            "badge_color": "#f59e0b"
        }
    ]
    return scenarios


# Cache pre-generated scenario sequences for seamless streaming
SCENARIO_CACHE = {}

def get_or_create_scenario_sequence(scenario_name: str):
    if scenario_name not in SCENARIO_CACHE:
        rng = np.random.RandomState(42)
        tel_seq, _, _ = generate_driving_sequence(scenario_name, seq_length=120, rng=rng)
        # Denormalize sequence for realistic playback
        steps = []
        for t in range(len(tel_seq)):
            norm_vec = tel_seq[t]
            # Convert back to raw
            step_dict = {}
            for i, p in enumerate(SENSOR_PARAMS):
                min_v, max_v = FEATURE_BOUNDS[p]
                step_dict[p] = float(norm_vec[i] * (max_v - min_v) + min_v)
            steps.append(step_dict)
        SCENARIO_CACHE[scenario_name] = steps
    return SCENARIO_CACHE[scenario_name]


@app.post("/api/simulate_step")
async def simulate_step(req: ScenarioRequest):
    """Returns next step in scenario and model's prediction."""
    steps = get_or_create_scenario_sequence(req.scenario_name)
    idx = req.step_index % len(steps)
    raw_step = steps[idx]
    result = engine.process_step(raw_step)
    result["step_index"] = idx
    result["total_steps"] = len(steps)
    return result


@app.websocket("/ws/telematics")
async def websocket_telematics_stream(websocket: WebSocket):
    """
    Real-time bi-directional WebSocket streaming:
    - Client can send manual control changes or command scenario playback.
    - Server streams predictions, DMS frames, and telemetry at 10 Hz.
    """
    await websocket.accept()
    
    current_mode = "manual"  # 'manual' or 'scenario'
    active_scenario = "Normal"
    scenario_step_idx = 0
    is_paused = False
    manual_telemetry = {
        "speed": 65.0,
        "acceleration": 0.0,
        "braking": 0.0,
        "steering": 0.0,
        "lane_deviation": 0.0,
        "eye_closure": 0.85,
        "head_pitch": 0.0,
        "head_yaw": 0.0,
        "phone_usage": 0.0
    }
    
    async def receiver():
        nonlocal current_mode, active_scenario, scenario_step_idx, manual_telemetry, is_paused
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                cmd = msg.get("cmd")
                
                if cmd == "set_mode":
                    current_mode = msg.get("mode", "manual")
                    if "scenario" in msg:
                        active_scenario = msg.get("scenario")
                        scenario_step_idx = 0
                elif cmd == "update_telemetry":
                    manual_telemetry.update(msg.get("telemetry", {}))
                elif cmd == "set_scenario":
                    active_scenario = msg.get("scenario", "Normal")
                    current_mode = "scenario"
                    scenario_step_idx = 0
                elif cmd == "toggle_pause":
                    is_paused = msg.get("paused", not is_paused)
                elif cmd == "reset":
                    engine.reset()
                    scenario_step_idx = 0
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[WebSocket Receiver Error] {e}")

    receiver_task = asyncio.create_task(receiver())
    
    try:
        while True:
            if not is_paused:
                if current_mode == "scenario":
                    steps = get_or_create_scenario_sequence(active_scenario)
                    step_data = steps[scenario_step_idx % len(steps)]
                    scenario_step_idx += 1
                    output = engine.process_step(step_data)
                    output["scenario"] = active_scenario
                    output["step_index"] = scenario_step_idx
                else:
                    # In manual mode, add slight realistic micro-jitter
                    jittered = dict(manual_telemetry)
                    jittered["speed"] += float(np.random.normal(0, 0.15))
                    output = engine.process_step(jittered)
                    output["mode"] = "manual"
                    
                await websocket.send_json(output)
                
            await asyncio.sleep(0.1)  # 10 Hz streaming
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Stream Error] {e}")
    finally:
        receiver_task.cancel()


# Mount frontend static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"status": "Frontend loading..."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
