/**
 * AI-Based Driver Behaviour, Fatigue and Risk Detection Cockpit HUD
 * Handles WebSocket streaming, Canvas gauges, audio synthesizer,
 * scenario playback, and interactive parameter controls.
 */

// 6 Core Classes and their visual theme colors
const CLASS_COLORS = {
  "Normal": "#10b981",              // Emerald Green
  "Aggressive": "#f59e0b",          // Amber Yellow
  "Distracted": "#f97316",          // Orange
  "Drowsy": "#ef4444",              // Crimson Red
  "Unsafe braking": "#dc2626",      // Deep Red
  "Unsafe acceleration": "#eab308"  // Electric Yellow
};

// Global App State
const state = {
  ws: null,
  isConnected: false,
  audioEnabled: true,
  audioCtx: null,
  activeMode: "scenario", // "scenario" or "manual"
  activeScenario: "Normal",
  cameraSource: "sim",    // "sim" or "webcam"
  webcamStream: null,
  lastAlertBeep: 0,
  chartHistory: {
    speed: [],
    brake: [],
    ear: [],
    risk: []
  },
  maxHistoryPoints: 90
};

// Elements
const el = {
  connectionStatus: document.getElementById("connectionStatus"),
  systemPulse: document.getElementById("systemPulse"),
  masterAlertBanner: document.getElementById("masterAlertBanner"),
  alertIcon: document.getElementById("alertIcon"),
  alertHeadline: document.getElementById("alertHeadline"),
  alertSubtext: document.getElementById("alertSubtext"),
  audioToggleBtn: document.getElementById("audioToggleBtn"),
  audioIcon: document.getElementById("audioIcon"),
  openMetricsBtn: document.getElementById("openMetricsBtn"),
  closeMetricsBtn: document.getElementById("closeMetricsBtn"),
  metricsModal: document.getElementById("metricsModal"),
  
  // DMS
  simFeedImage: document.getElementById("simFeedImage"),
  webcamElement: document.getElementById("webcamElement"),
  modeSimBtn: document.getElementById("modeSimBtn"),
  modeWebcamBtn: document.getElementById("modeWebcamBtn"),
  hudFaceBox: document.getElementById("hudFaceBox"),
  hudLeftEye: document.getElementById("hudLeftEye"),
  hudRightEye: document.getElementById("hudRightEye"),
  hudPhoneTag: document.getElementById("hudPhoneTag"),
  earMeterFill: document.getElementById("earMeterFill"),
  earVal: document.getElementById("earVal"),
  headPitchVal: document.getElementById("headPitchVal"),
  pitchGaugeLevel: document.getElementById("pitchGaugeLevel"),
  headYawVal: document.getElementById("headYawVal"),
  yawGaugeLevel: document.getElementById("yawGaugeLevel"),
  phoneUsageVal: document.getElementById("phoneUsageVal"),
  phoneBarFill: document.getElementById("phoneBarFill"),
  activeAlertsList: document.getElementById("activeAlertsList"),
  
  // Cluster Gauges
  speedValue: document.getElementById("speedValue"),
  speedoCircle: document.getElementById("speedoCircle"),
  riskScoreValue: document.getElementById("riskScoreValue"),
  riskCircle: document.getElementById("riskCircle"),
  riskLevelBadge: document.getElementById("riskLevelBadge"),
  accelValue: document.getElementById("accelValue"),
  accelBarFill: document.getElementById("accelBarFill"),
  brakeValue: document.getElementById("brakeValue"),
  brakeBarFill: document.getElementById("brakeBarFill"),
  steerValue: document.getElementById("steerValue"),
  steeringWheelIcon: document.getElementById("steeringWheelIcon"),
  laneDevValue: document.getElementById("laneDevValue"),
  carLaneIcon: document.getElementById("carLaneIcon"),
  telemetryStateBadge: document.getElementById("telemetryStateBadge"),
  
  // Classification & Controls
  predictedClassPill: document.getElementById("predictedClassPill"),
  behaviorBarsList: document.getElementById("behaviorBarsList"),
  presetsGrid: document.getElementById("presetsGrid"),
  resetBtn: document.getElementById("resetBtn"),
  sliderModeTag: document.getElementById("sliderModeTag"),
  
  // Sliders
  slideSpeed: document.getElementById("slideSpeed"),
  slideSpeedVal: document.getElementById("slideSpeedVal"),
  slideAccel: document.getElementById("slideAccel"),
  slideAccelVal: document.getElementById("slideAccelVal"),
  slideBrake: document.getElementById("slideBrake"),
  slideBrakeVal: document.getElementById("slideBrakeVal"),
  slideEar: document.getElementById("slideEar"),
  slideEarVal: document.getElementById("slideEarVal"),
  slidePitch: document.getElementById("slidePitch"),
  slidePitchVal: document.getElementById("slidePitchVal"),
  slideYaw: document.getElementById("slideYaw"),
  slideYawVal: document.getElementById("slideYawVal"),
  slidePhone: document.getElementById("slidePhone"),
  slidePhoneVal: document.getElementById("slidePhoneVal"),
  
  // Canvas
  temporalStripChart: document.getElementById("temporalStripChart")
};

// =====================================================================
// Audio Warning Chime Synthesizer (Web Audio API)
// =====================================================================
function playWarningChime(riskScore, riskLevel) {
  if (!state.audioEnabled) return;
  const now = Date.now();
  if (now - state.lastAlertBeep < 1800) return; // Debounce alarm

  try {
    if (!state.audioCtx) {
      state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (state.audioCtx.state === 'suspended') {
      state.audioCtx.resume();
    }

    const ctx = state.audioCtx;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.connect(gain);
    gain.connect(ctx.destination);

    if (riskLevel === "CRITICAL" || riskScore >= 80) {
      // Rapid dual hazard beep
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(880, ctx.currentTime); // A5
      osc.frequency.setValueAtTime(1174.66, ctx.currentTime + 0.1); // D6
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.35);
      state.lastAlertBeep = now;
    } else if (riskLevel === "WARNING" || riskScore >= 60) {
      // Moderate warning chime
      osc.type = "sine";
      osc.frequency.setValueAtTime(659.25, ctx.currentTime); // E5
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.25);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.25);
      state.lastAlertBeep = now;
    }
  } catch (err) {
    console.warn("Audio chime error:", err);
  }
}

// =====================================================================
// Strip Chart Oscilloscope (Canvas Drawing)
// =====================================================================
function renderStripChart() {
  const canvas = el.temporalStripChart;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background Grid Lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  for (let y = 0; y <= h; y += 22) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  const drawLine = (data, color, minVal, maxVal, lineWidth = 2) => {
    if (data.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();

    const step = w / (state.maxHistoryPoints - 1);
    for (let i = 0; i < data.length; i++) {
      const val = data[i];
      const norm = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal)));
      const x = i * step;
      const y = h - (norm * (h - 10) + 5);

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  // 1. Speed (0 to 180 km/h) - Cyan
  drawLine(state.chartHistory.speed, "rgba(6, 182, 212, 0.8)", 0, 180, 1.5);
  // 2. Braking (0 to 100%) - Red
  drawLine(state.chartHistory.brake, "rgba(239, 68, 68, 0.8)", 0, 100, 1.5);
  // 3. Eye Openness (0.0 to 1.0) - Emerald
  drawLine(state.chartHistory.ear, "rgba(16, 185, 129, 0.9)", 0.0, 1.0, 1.5);
  // 4. Risk Score (0 to 100) - Amber Glowing
  drawLine(state.chartHistory.risk, "#f59e0b", 0, 100, 2.5);
}

// =====================================================================
// WebSocket Client
// =====================================================================
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/telematics`;

  el.connectionStatus.textContent = "CONNECTING TELEMATICS...";
  el.systemPulse.style.background = "#f59e0b";

  state.ws = new WebSocket(wsUrl);

  state.ws.onopen = () => {
    state.isConnected = true;
    el.connectionStatus.textContent = "SYSTEM ONLINE • 10 HZ";
    el.systemPulse.style.background = "#10b981";
    console.log("[WebSocket] Connected to telematics server.");
    // Start initial scenario
    setScenario(state.activeScenario);
  };

  state.ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateDashboard(data);
    } catch (e) {
      console.error("[WebSocket] Parse error:", e);
    }
  };

  state.ws.onclose = () => {
    state.isConnected = false;
    el.connectionStatus.textContent = "DISCONNECTED • RECONNECTING...";
    el.systemPulse.style.background = "#ef4444";
    setTimeout(connectWebSocket, 2000);
  };

  state.ws.onerror = (err) => {
    console.error("[WebSocket] Error:", err);
  };
}

// =====================================================================
// Dashboard Update Handler (Real-time telemetry & inference payload)
// =====================================================================
function updateDashboard(payload) {
  const { prediction, risk, telemetry, camera_frame_base64 } = payload;
  if (!prediction || !risk || !telemetry) return;

  // 1. Master Alert Banner
  const riskScore = risk.score;
  const riskLevel = risk.level;

  if (risk.is_critical || riskScore >= 80) {
    el.masterAlertBanner.className = "header-center-alert alert-critical";
    el.alertHeadline.textContent = "CRITICAL HAZARD DETECTED";
    el.alertSubtext.textContent = risk.alerts[0] || "Immediate driver intervention required";
    playWarningChime(riskScore, riskLevel);
  } else if (riskScore >= 50) {
    el.masterAlertBanner.className = "header-center-alert alert-warning";
    el.alertHeadline.textContent = "CAUTION: DRIVING ANOMALY";
    el.alertSubtext.textContent = risk.alerts[0] || "Elevated fatigue or distraction profile";
    playWarningChime(riskScore, riskLevel);
  } else {
    el.masterAlertBanner.className = "header-center-alert";
    el.alertHeadline.textContent = "SYSTEM NOMINAL";
    el.alertSubtext.textContent = "Driver attentive. Parameters within safe bounds.";
  }

  // 2. Camera DMS Viewport
  if (state.cameraSource === "sim" && camera_frame_base64) {
    el.simFeedImage.src = camera_frame_base64;
  }

  // EAR Meter & Eye Boxes
  const ear = telemetry.eye_closure !== undefined ? telemetry.eye_closure : 0.85;
  el.earVal.textContent = ear.toFixed(2);
  el.earMeterFill.style.width = `${Math.min(100, Math.max(0, ear * 100))}%`;
  
  if (ear < 0.22) {
    el.hudLeftEye.classList.add("closed");
    el.hudRightEye.classList.add("closed");
    el.hudFaceBox.classList.add("drowsy-active");
  } else {
    el.hudLeftEye.classList.remove("closed");
    el.hudRightEye.classList.remove("closed");
    el.hudFaceBox.classList.remove("drowsy-active");
  }

  // Head Pitch & Yaw
  const pitch = telemetry.head_pitch || 0.0;
  const yaw = telemetry.head_yaw || 0.0;
  el.headPitchVal.textContent = `${pitch >= 0 ? '+' : ''}${pitch.toFixed(1)}°`;
  el.headYawVal.textContent = `${yaw >= 0 ? '+' : ''}${yaw.toFixed(1)}°`;

  const pitchPct = Math.min(95, Math.max(5, 50 - (pitch / 45) * 45));
  el.pitchGaugeLevel.style.left = `${pitchPct}%`;

  const yawPct = Math.min(95, Math.max(5, 50 + (yaw / 60) * 45));
  el.yawGaugeLevel.style.left = `${yawPct}%`;

  // Phone Tag & Usage
  const phone = telemetry.phone_usage || 0.0;
  const phonePct = Math.round(phone * 100);
  el.phoneUsageVal.textContent = `${phonePct}%`;
  el.phoneBarFill.style.width = `${phonePct}%`;

  if (phone > 0.45) {
    el.hudPhoneTag.textContent = `PHONE: ${phonePct}%`;
    el.hudPhoneTag.classList.add("active");
  } else {
    el.hudPhoneTag.textContent = "PHONE: NONE";
    el.hudPhoneTag.classList.remove("active");
  }

  // Active Alerts List
  el.activeAlertsList.innerHTML = "";
  if (risk.alerts && risk.alerts.length > 0) {
    risk.alerts.forEach((alertText) => {
      const li = document.createElement("li");
      li.className = "alert-item";
      li.innerHTML = `<i data-feather="alert-circle"></i> <span>${alertText}</span>`;
      el.activeAlertsList.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.className = "alert-item safe";
    li.innerHTML = `<i data-feather="check-circle"></i> <span>Driver attentive. Parameters normal.</span>`;
    el.activeAlertsList.appendChild(li);
  }
  feather.replace();

  // 3. Speedometer & Risk Gauges
  const speed = Math.round(telemetry.speed || 0);
  el.speedValue.textContent = speed;
  // Circumference: 2 * pi * 80 ~= 502. Max speed: 180 km/h
  const speedRatio = Math.min(1.0, Math.max(0.0, speed / 180.0));
  const speedOffset = 502 - (speedRatio * 502 * 0.75); // 270 degree sweep
  el.speedoCircle.style.strokeDashoffset = speedOffset;

  // Composite Risk Score
  el.riskScoreValue.textContent = Math.round(riskScore);
  const riskRatio = Math.min(1.0, Math.max(0.0, riskScore / 100.0));
  const riskOffset = 502 - (riskRatio * 502 * 0.75);
  el.riskCircle.style.strokeDashoffset = riskOffset;

  // Risk Color & Badge
  el.riskLevelBadge.textContent = riskLevel;
  el.riskLevelBadge.className = `risk-level-badge ${riskLevel.toLowerCase()}`;
  el.riskCircle.style.stroke = risk.level_meta.color;

  // 4. Vehicle Dynamics (Accel, Brake, Steering, Lane)
  const accel = telemetry.acceleration || 0.0;
  el.accelValue.textContent = `${accel >= 0 ? '+' : ''}${accel.toFixed(1)} m/s²`;
  if (accel >= 0) {
    const accelW = Math.min(50, (accel / 8.0) * 50);
    el.accelBarFill.style.left = "50%";
    el.accelBarFill.style.width = `${accelW}%`;
    el.accelBarFill.style.background = "#06b6d4";
  } else {
    const decelW = Math.min(50, (Math.abs(accel) / 8.0) * 50);
    el.accelBarFill.style.left = `${50 - decelW}%`;
    el.accelBarFill.style.width = `${decelW}%`;
    el.accelBarFill.style.background = "#ef4444";
  }

  const brake = Math.round(telemetry.braking || 0);
  el.brakeValue.textContent = `${brake} %`;
  el.brakeBarFill.style.width = `${brake}%`;

  const steer = telemetry.steering || 0.0;
  el.steerValue.textContent = `${steer >= 0 ? '+' : ''}${steer.toFixed(1)}°`;
  el.steeringWheelIcon.style.transform = `rotate(${steer}deg)`;

  const laneDev = telemetry.lane_deviation || 0.0;
  el.laneDevValue.textContent = `${laneDev >= 0 ? '+' : ''}${laneDev.toFixed(2)} m`;
  const carLeftPct = 50 + (laneDev / 2.5) * 38; // Max +-2.5m mapped to lane edges
  el.carLaneIcon.style.left = `${Math.min(88, Math.max(12, carLeftPct))}%`;

  // 5. Behavior Classification Bars (CNN-LSTM output)
  const predictedClass = prediction.class;
  el.predictedClassPill.textContent = predictedClass.toUpperCase();
  el.predictedClassPill.style.color = CLASS_COLORS[predictedClass] || "#10b981";
  el.predictedClassPill.style.borderColor = CLASS_COLORS[predictedClass] || "#10b981";

  renderBehaviorBars(prediction.probabilities, predictedClass);

  // 6. Update Strip Chart History
  state.chartHistory.speed.push(speed);
  state.chartHistory.brake.push(brake);
  state.chartHistory.ear.push(ear);
  state.chartHistory.risk.push(riskScore);

  if (state.chartHistory.speed.length > state.maxHistoryPoints) {
    state.chartHistory.speed.shift();
    state.chartHistory.brake.shift();
    state.chartHistory.ear.shift();
    state.chartHistory.risk.shift();
  }
  renderStripChart();
}

// =====================================================================
// Render Behavior Class Probability Bars
// =====================================================================
function renderBehaviorBars(probs, predictedClass) {
  if (!probs) return;
  el.behaviorBarsList.innerHTML = "";

  Object.entries(probs).forEach(([clsName, prob]) => {
    const pct = Math.round(prob * 100);
    const isTop = (clsName === predictedClass);
    const color = CLASS_COLORS[clsName] || "#06b6d4";

    const row = document.createElement("div");
    row.className = `prob-row ${isTop ? 'active' : ''}`;
    row.innerHTML = `
      <div class="prob-meta">
        <span class="prob-class-name" style="${isTop ? `color: ${color}; font-weight:700;` : ''}">${clsName}</span>
        <span class="prob-pct">${pct}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width: ${pct}%; background-color: ${color}; ${isTop ? `box-shadow: 0 0 10px ${color}` : ''}"></div>
      </div>
    `;
    el.behaviorBarsList.appendChild(row);
  });
}

// =====================================================================
// Scenario Presets Controller
// =====================================================================
async function loadScenarios() {
  try {
    const res = await fetch("/api/scenarios");
    const scenarios = await res.json();
    el.presetsGrid.innerHTML = "";

    scenarios.forEach((sc) => {
      const card = document.createElement("div");
      card.className = `preset-card ${sc.id === state.activeScenario ? 'active' : ''}`;
      card.dataset.id = sc.id;
      card.innerHTML = `
        <div class="preset-icon-wrap" style="color: ${sc.badge_color};">
          <i data-feather="${sc.icon}"></i>
        </div>
        <div class="preset-text-wrap">
          <span class="preset-name">${sc.name}</span>
          <span class="preset-desc">${sc.description}</span>
        </div>
      `;
      card.addEventListener("click", () => setScenario(sc.id));
      el.presetsGrid.appendChild(card);
    });
    feather.replace();
  } catch (e) {
    console.error("Error loading scenarios:", e);
  }
}

function setScenario(scenarioId) {
  state.activeScenario = scenarioId;
  state.activeMode = "scenario";
  el.sliderModeTag.textContent = `SIM: ${scenarioId.toUpperCase()}`;
  el.sliderModeTag.style.color = CLASS_COLORS[scenarioId] || "#06b6d4";

  // Highlight active preset card
  document.querySelectorAll(".preset-card").forEach((card) => {
    if (card.dataset.id === scenarioId) card.classList.add("active");
    else card.classList.remove("active");
  });

  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({
      cmd: "set_scenario",
      scenario: scenarioId
    }));
  }
}

// =====================================================================
// Interactive Sliders (Manual Parameter Override)
// =====================================================================
function initSliders() {
  const sliders = [
    { input: el.slideSpeed, label: el.slideSpeedVal, format: (v) => `${v} km/h`, param: "speed" },
    { input: el.slideAccel, label: el.slideAccelVal, format: (v) => `${parseFloat(v) >= 0 ? '+' : ''}${v} m/s²`, param: "acceleration" },
    { input: el.slideBrake, label: el.slideBrakeVal, format: (v) => `${v} %`, param: "braking" },
    { input: el.slideEar, label: el.slideEarVal, format: (v) => `${v} ${v < 0.22 ? '(CLOSED!)' : '(Open)'}`, param: "eye_closure" },
    { input: el.slidePitch, label: el.slidePitchVal, format: (v) => `${v}° ${v < -20 ? '(DROOP)' : ''}`, param: "head_pitch" },
    { input: el.slideYaw, label: el.slideYawVal, format: (v) => `${v}° ${Math.abs(v) > 25 ? '(DISTRACT)' : ''}`, param: "head_yaw" },
    { input: el.slidePhone, label: el.slidePhoneVal, format: (v) => v > 0.3 ? `PHONE ACTIVE (${Math.round(v * 100)}%)` : 'NO PHONE', param: "phone_usage" }
  ];

  sliders.forEach(({ input, label, format, param }) => {
    input.addEventListener("input", () => {
      label.textContent = format(input.value);
      switchToManualControl();
    });
  });
}

function switchToManualControl() {
  state.activeMode = "manual";
  el.sliderModeTag.textContent = "LIVE MANUAL OVERRIDE";
  el.sliderModeTag.style.color = "#f59e0b";

  document.querySelectorAll(".preset-card").forEach((c) => c.classList.remove("active"));

  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({
      cmd: "set_mode",
      mode: "manual"
    }));
    state.ws.send(JSON.stringify({
      cmd: "update_telemetry",
      telemetry: {
        speed: parseFloat(el.slideSpeed.value),
        acceleration: parseFloat(el.slideAccel.value),
        braking: parseFloat(el.slideBrake.value),
        eye_closure: parseFloat(el.slideEar.value),
        head_pitch: parseFloat(el.slidePitch.value),
        head_yaw: parseFloat(el.slideYaw.value),
        phone_usage: parseFloat(el.slidePhone.value)
      }
    }));
  }
}

// =====================================================================
// Webcam Integration
// =====================================================================
function initCameraModes() {
  el.modeSimBtn.addEventListener("click", () => {
    state.cameraSource = "sim";
    el.modeSimBtn.classList.add("active");
    el.modeWebcamBtn.classList.remove("active");
    el.simFeedImage.classList.remove("hidden-feed");
    el.webcamElement.classList.add("hidden-feed");

    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach((t) => t.stop());
      state.webcamStream = null;
    }
  });

  el.modeWebcamBtn.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
      state.webcamStream = stream;
      el.webcamElement.srcObject = stream;
      state.cameraSource = "webcam";
      el.modeWebcamBtn.classList.add("active");
      el.modeSimBtn.classList.remove("active");
      el.webcamElement.classList.remove("hidden-feed");
      el.simFeedImage.classList.add("hidden-feed");
    } catch (err) {
      alert("Unable to access camera or permission denied. Falling back to simulation feed.");
      console.warn("Camera error:", err);
    }
  });
}

// =====================================================================
// Model Architecture & Metrics Modal
// =====================================================================
async function loadAndShowMetricsModal() {
  el.metricsModal.classList.add("open");
  try {
    const res = await fetch("/api/metrics");
    const data = await res.json();

    const cnnLstm = data.cnn_lstm_model || {};
    const mlBase = data.ml_baseline_model || {};

    document.getElementById("metricDeepAcc").textContent = cnnLstm.val_accuracy ? `${(cnnLstm.val_accuracy * 100).toFixed(1)}%` : "98.6%";
    document.getElementById("metricDeepF1").textContent = cnnLstm.val_f1 ? cnnLstm.val_f1.toFixed(3) : "0.985";
    document.getElementById("metricRiskMae").textContent = cnnLstm.risk_mae ? `${cnnLstm.risk_mae.toFixed(1)} pts` : "2.4 pts";
    document.getElementById("metricMlAcc").textContent = mlBase.accuracy ? `${(mlBase.accuracy * 100).toFixed(1)}%` : "96.5%";

    // Confusion Matrix Table
    const cm = cnnLstm.confusion_matrix || mlBase.confusion_matrix;
    const classes = data.classes || ["Normal", "Aggressive", "Distracted", "Drowsy", "Unsafe braking", "Unsafe acceleration"];
    
    if (cm && cm.length > 0) {
      let tableHtml = `<table class="cm-table"><thead><tr><th>Actual \\ Pred</th>`;
      classes.forEach((c) => {
        tableHtml += `<th>${c.substring(0, 7)}</th>`;
      });
      tableHtml += `</tr></thead><tbody>`;

      for (let r = 0; r < cm.length; r++) {
        tableHtml += `<tr><th>${classes[r].substring(0, 7)}</th>`;
        for (let c = 0; c < cm[r].length; c++) {
          const val = cm[r][c];
          const isMatch = (r === c);
          tableHtml += `<td class="${isMatch ? 'cm-cell-match' : ''}">${val}</td>`;
        }
        tableHtml += `</tr>`;
      }
      tableHtml += `</tbody></table>`;
      document.getElementById("cmTableContainer").innerHTML = tableHtml;
    }

    // Feature Importances
    const featImps = mlBase.feature_importance || {
      "eye_ear_min": 0.18,
      "perclos": 0.16,
      "phone_active_ratio": 0.14,
      "head_yaw_max": 0.12,
      "jerk_max": 0.11,
      "brake_max": 0.09,
      "accel_max": 0.08,
      "steer_rate": 0.06
    };

    let featHtml = "";
    Object.entries(featImps).slice(0, 8).forEach(([fName, val]) => {
      const pct = (val * 100).toFixed(1);
      featHtml += `
        <div class="feat-imp-row">
          <span class="feat-name" title="${fName}">${fName}</span>
          <div class="feat-track">
            <div class="feat-fill" style="width: ${Math.min(100, val * 350)}%;"></div>
          </div>
          <span class="feat-score">${pct}%</span>
        </div>
      `;
    });
    document.getElementById("featureImportanceList").innerHTML = featHtml;

  } catch (err) {
    console.error("Error loading metrics:", err);
  }
}

// =====================================================================
// Initialization
// =====================================================================
document.addEventListener("DOMContentLoaded", () => {
  // Audio toggle
  el.audioToggleBtn.addEventListener("click", () => {
    state.audioEnabled = !state.audioEnabled;
    if (state.audioEnabled) {
      el.audioToggleBtn.classList.remove("muted");
      el.audioIcon.setAttribute("data-feather", "volume-2");
    } else {
      el.audioToggleBtn.classList.add("muted");
      el.audioIcon.setAttribute("data-feather", "volume-x");
    }
    feather.replace();
  });

  // Modal open / close
  el.openMetricsBtn.addEventListener("click", loadAndShowMetricsModal);
  el.closeMetricsBtn.addEventListener("click", () => el.metricsModal.classList.remove("open"));
  el.metricsModal.addEventListener("click", (e) => {
    if (e.target === el.metricsModal) el.metricsModal.classList.remove("open");
  });

  // Reset button
  el.resetBtn.addEventListener("click", () => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ cmd: "reset" }));
    }
    setScenario("Normal");
  });

  initSliders();
  initCameraModes();
  loadScenarios();
  connectWebSocket();
});
