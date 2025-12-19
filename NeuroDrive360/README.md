# NeuroDrive 360 – Agentic AI for Vehicle Health

An academic-style reference implementation for predictive maintenance using agentic AI, combining interpretable workflows with production-ready APIs and a Streamlit research dashboard.

## Problem Statement
Modern vehicles emit continuous telematics streams (speed, engine temperature, vibration, battery voltage, mileage). Manual triage cannot scale to early fault detection across fleets. NeuroDrive 360 automates validation, diagnosis, and maintenance recommendation while preserving inspectability of intermediate reasoning steps.

## Agentic AI Architecture (DataAgent → DiagnosisAgent → SchedulerAgent)
- **DataAgent:** Validates and normalizes telematics payloads to ensure schema compliance and plausible ranges.
- **DiagnosisAgent:** Applies supervised and unsupervised models to generate fault probability and anomaly score.
- **SchedulerAgent:** Converts model outputs into categorical risk levels (LOW / MEDIUM / HIGH) and prescribes maintenance actions.
- **Orchestration:** Implemented with LangGraph (`MaintenanceWorkflow`), exposed through FastAPI endpoints `/health`, `/diagnosis`, and `/diagnosis/debug` (intermediate state streaming for research). Streamlit consumes these endpoints for interactive analysis.

## Machine Learning Models & Evaluation
- **Fault probability:** Gradient-boosted trees (`xgboost_classifier.pkl`) trained on labeled telematics to estimate supervised risk.
- **Anomaly score:** Isolation Forest (`isolation_forest.pkl`) with companion scaler (`isolation_forest_scaler.pkl`) to detect distributional shifts.
- **Risk mapping:** Combines fault probability and anomaly score into discrete risk levels driving downstream recommendations.
- **Validation checks:** Workflow enforces presence of required fields and completeness of returned agent state to prevent partial or inconsistent outputs.

## Demo Scenarios (LOW / MEDIUM / HIGH)
- **LOW:** Near-nominal readings (e.g., engine temperature ≈ 90°C, vibration ≈ 3 g, voltage ≈ 12.8 V, mileage ≈ 50k) yield low fault probability and anomaly scores.
- **MEDIUM:** Moderate deviations (e.g., engine temperature ≈ 110°C or vibration ≈ 7 g) elevate risk and trigger precautionary actions.
- **HIGH:** Severe anomalies (e.g., vibration > 12 g, voltage < 11 V, elevated temperature) produce high risk and urgent maintenance recommendations.

## How to Run Locally
1. Use Python 3.10.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the FastAPI backend: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
4. Launch the Streamlit dashboard: `streamlit run app.py`
5. Optional: set `BACKEND_URL` to point the dashboard at a remote API; defaults to `http://localhost:8000` for local development.

## HuggingFace Deployment
- Streamlit entrypoint: `app.py` at the repository root imports and runs `streamlit_app.dashboard.main`.
- Configure the environment variable `BACKEND_URL` in the Space to target the deployed FastAPI service.
- Ensure required model artifacts in `models/` are present in the backend environment.

## Author Information
Buddarraju Lalit
B.Tech 3rd year


