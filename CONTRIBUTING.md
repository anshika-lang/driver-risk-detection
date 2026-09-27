# Contributing to AI Driver Behaviour & Risk Detection System

Thank you for your interest in contributing to the **AI-Based Driver Behaviour, Fatigue and Risk Detection System**! We welcome contributions to multimodal sensor fusion, deep learning architectures, computer vision pipelines, and automotive telemetry interfaces.

---

## 🛠️ Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/driver-risk-detection.git
   cd driver-risk-detection
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   Run the automated test suite:
   ```bash
   python test_system.py
   ```

---

## 🔄 Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your enhancements (ensure existing tests pass and adhere to PEP 8).
3. If you add new parameters or neural layers, update `ai_engine/config.py` and run:
   ```bash
   python -m ai_engine.train
   ```
4. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "feat: add lateral jerk calculation to telemetry encoder"
   ```
5. Push to your fork and submit a Pull Request.

---

## 📋 Code of Conduct
- Be respectful, constructive, and collaborative.
- Document any new API endpoints, mathematical models, or sensor parameters.
