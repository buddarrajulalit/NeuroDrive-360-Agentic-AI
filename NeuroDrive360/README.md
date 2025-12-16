# NeuroDrive 360

Research-oriented automotive predictive maintenance system combining agentic AI workflows with ML-based risk assessment and a Streamlit dashboard.

## Problem Statement
Vehicle fleets generate high-volume telematics data that must be assessed continuously for early fault detection. Manual triage is slow and inconsistent. NeuroDrive 360 automates diagnosis by validating sensor inputs, estimating failure risk, and recommending maintenance actions with transparent, inspectable reasoning.

## Agentic AI Architecture
- **LangGraph workflow:** `DataAgent → DiagnosisAgent → SchedulerAgent` executed by `MaintenanceWorkflow`.
- **DataAgent:** Validates and cleans telematics payloads (speed, engine temperature, vibration, battery voltage, mileage).
- **DiagnosisAgent:** Applies supervised and unsupervised models to estimate fault probability and anomaly score.
- **SchedulerAgent:** Maps risk to recommended maintenance actions (LOW / MEDIUM / HIGH) and returns a consolidated agent state.
- **APIs:** FastAPI service exposes `/health`, `/diagnosis`, and `/diagnosis/debug` (intermediate state streaming for research).
- **UI:** Streamlit dashboard consumes the API, renders metrics, and highlights risk levels with contextual recommendations.

## Technologies Used
- FastAPI, Uvicorn (backend service)
- Streamlit (dashboard; Hugging Face Spaces entrypoint `app.py`)
- LangGraph (agentic workflow orchestration)
- XGBoost, Scikit-learn, Joblib (ML models and persistence)
- Pandas, NumPy (data processing)

## ML Models & Evaluation
- **Fault probability:** Gradient-boosted trees (`xgboost_classifier.pkl`) trained on labeled telematics for supervised risk estimation.
- **Anomaly score:** Isolation Forest (`isolation_forest.pkl`) with fitted scaler (`isolation_forest_scaler.pkl`) for unsupervised anomaly detection.
- **Risk mapping:** Combines fault probability and anomaly score into categorical risk levels (LOW / MEDIUM / HIGH) used by the scheduler.
- **Validation:** Workflow enforces presence of required fields and checks completeness of returned agent state to ensure reliable responses.

## Demo Scenarios (LOW / MED / HIGH)
- **LOW:** Nominal inputs (e.g., speed ≈ 90 km/h, engine temp ≈ 90°C, vibration ≈ 3 g, voltage ≈ 12.8 V, mileage ≈ 50k) → low fault probability and anomaly score; green indicator.
- **MED:** Mild deviations (e.g., higher engine temp ≈ 110°C or vibration ≈ 7 g) → moderate risk; amber indicator with precautionary action.
- **HIGH:** Severe anomalies (e.g., vibration > 12 g, voltage < 11 V, elevated temperature) → high fault probability/anomaly; red indicator and urgent maintenance recommendation.

## How to Run Locally
1. **Environment:** Python 3.10 recommended.
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Start backend:** `uvicorn app.main:app --host 0.0.0.0 --port 8000`
4. **Launch dashboard:** `streamlit run app.py`
5. **Configure backend URL (optional):** set `BACKEND_URL` to target a remote API; defaults to `http://localhost:8000` for local runs.

## Deployment (HuggingFace Spaces)
- Spaces Streamlit app entrypoint is `app.py`, which imports and runs `streamlit_app.dashboard.main`.
- Set `BACKEND_URL` secret/variable to point the dashboard to the deployed FastAPI service.
- Ensure model artifacts in `models/` are available to the backend container.

## Author
NeuroDrive 360 Engineering Team

