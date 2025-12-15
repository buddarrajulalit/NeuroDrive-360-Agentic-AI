"""Streamlit Dashboard for Automotive Predictive Maintenance System.

This dashboard provides an interactive interface for vehicle health diagnosis
using ML-powered predictive maintenance models.
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, Optional
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Automotive Predictive Maintenance",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_BASE_URL = "http://localhost:8000"
DIAGNOSIS_ENDPOINT = f"{API_BASE_URL}/diagnosis"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"


def check_api_health() -> bool:
    """Check if the FastAPI backend is available.

    Returns:
        True if API is healthy, False otherwise.
    """
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def send_diagnosis_request(telematics_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Send diagnosis request to FastAPI backend.

    Args:
        telematics_data: Dictionary matching TelematicsInput schema with fields:
            - vehicle_id (str)
            - speed (float)
            - engine_temperature (float)
            - vibration (float)
            - battery_voltage (float)
            - mileage (float)
        Payload is sent as a flat JSON object (not nested).

    Returns:
        Response dictionary from API, or None if request failed.
    """
    try:
        response = requests.post(
            DIAGNOSIS_ENDPOINT,
            json=telematics_data,  # Sends flat JSON payload matching TelematicsInput schema
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # Extract detailed error message from response
        error_detail = "Unknown error"
        try:
            error_response = response.json()
            error_detail = error_response.get("detail", str(e))
        except:
            error_detail = str(e)
        st.error(f"API request failed (HTTP {response.status_code}): {error_detail}")
        st.json(telematics_data)  # Show the payload that was sent
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None


def get_risk_color(risk_level: str) -> str:
    """Get color code for risk level.

    Args:
        risk_level: Risk level string (LOW, MEDIUM, HIGH).

    Returns:
        Hex color code.
    """
    color_map = {
        "LOW": "#28a745",      # Green
        "MEDIUM": "#ffc107",   # Yellow/Amber
        "HIGH": "#dc3545"      # Red
    }
    return color_map.get(risk_level, "#6c757d")  # Default gray


def get_risk_icon(risk_level: str) -> str:
    """Get emoji icon for risk level.

    Args:
        risk_level: Risk level string (LOW, MEDIUM, HIGH).

    Returns:
        Emoji icon string.
    """
    icon_map = {
        "LOW": "✅",
        "MEDIUM": "⚠️",
        "HIGH": "🚨"
    }
    return icon_map.get(risk_level, "❓")


def display_risk_indicator(risk_level: str):
    """Display color-coded risk indicator prominently.

    Args:
        risk_level: Risk level string (LOW, MEDIUM, HIGH).
    """
    color = get_risk_color(risk_level)
    icon = get_risk_icon(risk_level)
    
    # Create custom HTML for prominently styled risk indicator
    st.markdown(
        f"""
        <div style="
            background-color: {color};
            color: white;
            padding: 30px;
            border-radius: 8px;
            text-align: center;
            margin: 20px 0;
            border: 2px solid {color};
        ">
            <h1 style="margin: 0; font-size: 56px;">{icon}</h1>
            <h2 style="margin: 15px 0; font-size: 32px; font-weight: bold;">Risk Level: {risk_level}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_metrics(fault_probability: float, anomaly_score: float):
    """Display fault probability and anomaly score metrics.

    Args:
        fault_probability: Fault probability value (0-1).
        anomaly_score: Anomaly score value (0-1).
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Fault Probability",
            value=f"{fault_probability:.4f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Anomaly Score",
            value=f"{anomaly_score:.4f}",
            delta=None
        )


def main():
    """Main dashboard application."""
    
    # Title and header
    st.title("🚗 Automotive Predictive Maintenance Dashboard")
    st.markdown("### ML-Powered Vehicle Health Diagnosis System")
    st.markdown("---")
    
    # Check API health
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ **API Backend Unavailable**")
        st.warning(
            "Please ensure the FastAPI backend is running at "
            f"`{API_BASE_URL}`. Start it with: `uvicorn app.main:app --reload`"
        )
        st.stop()
    
    # Sidebar for input form
    with st.sidebar:
        st.header("📊 Telematics Input")
        st.markdown("Enter vehicle sensor data for diagnosis.")
        
        # Input form
        vehicle_id = st.text_input(
            "Vehicle ID",
            value="EV-101",
            help="Unique identifier for the vehicle"
        )
        
        speed = st.number_input(
            "Speed (km/h)",
            min_value=0.0,
            max_value=250.0,
            value=92.0,
            step=1.0,
            help="Current vehicle speed"
        )
        
        engine_temperature = st.number_input(
            "Engine Temperature (°C)",
            min_value=-40.0,
            max_value=150.0,
            value=90.0,
            step=0.1,
            help="Engine temperature reading"
        )
        
        vibration = st.number_input(
            "Vibration (g-force)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.1,
            help="Vibration level measurement"
        )
        
        battery_voltage = st.number_input(
            "Battery Voltage (V)",
            min_value=0.0,
            max_value=20.0,
            value=12.8,
            step=0.1,
            help="Battery voltage reading"
        )
        
        mileage = st.number_input(
            "Mileage (km)",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            help="Total vehicle mileage"
        )
        
        # Submit button
        submit_button = st.button(
            "🔍 Diagnose Vehicle",
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. Enter vehicle telematics data
        2. Click 'Diagnose Vehicle'
        3. View risk assessment and metrics
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This dashboard uses a LangGraph-based agentic workflow that orchestrates
        multiple agents to provide comprehensive vehicle health diagnosis and
        maintenance recommendations.
        """)
    
    # Main content area
    if submit_button:
        # Prepare telematics data payload matching FastAPI TelematicsInput schema exactly
        # Fields must be in snake_case: vehicle_id, speed, engine_temperature, vibration, battery_voltage, mileage
        # Payload is sent as a flat dictionary (not nested)
        # Ensure all numeric values are floats (not int) to match schema expectations
        telematics_data = {
            "vehicle_id": str(vehicle_id),  # Ensure string type
            "speed": float(speed),  # Ensure float type
            "engine_temperature": float(engine_temperature),  # Ensure float type
            "vibration": float(vibration),  # Ensure float type
            "battery_voltage": float(battery_voltage),  # Ensure float type
            "mileage": float(mileage)  # Ensure float type
        }
        
        # Show loading spinner
        with st.spinner("🔬 Analyzing vehicle health data..."):
            # Send request to API - payload sent directly as JSON (not nested)
            diagnosis_result = send_diagnosis_request(telematics_data)
        
        if diagnosis_result:
            st.success("✅ Diagnosis complete!")
            
            # Extract results
            vehicle_id_result = diagnosis_result.get("vehicle_id", vehicle_id)
            fault_probability = diagnosis_result.get("fault_probability", 0.0)
            anomaly_score = diagnosis_result.get("anomaly_score", 0.0)
            risk_level = diagnosis_result.get("risk_level", "UNKNOWN")
            recommended_action = diagnosis_result.get("recommended_action", "No recommendation available")
            
            # Display results section
            st.markdown("## 📋 Diagnosis Results")
            st.markdown(f"**Vehicle ID:** {vehicle_id_result}")
            st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown("---")
            
            # Display risk indicator prominently with color coding
            display_risk_indicator(risk_level)
            
            # Display metrics section
            st.markdown("## 📊 Prediction Metrics")
            st.markdown("*ML model predictions for fault probability and anomaly detection*")
            display_metrics(fault_probability, anomaly_score)
            
            # Clear visual separator before Agent Decision section
            st.markdown("---")
            st.markdown("")  # Empty line for spacing
            st.markdown("")  # Additional spacing
            
            # Display agent decision section - clearly separated
            st.markdown("## 🤖 Agent Decision")
            st.markdown("*Maintenance recommendation from the agentic workflow*")
            st.markdown("")  # Spacing before action
            
            # Display recommended action in a clean, academic style
            # Use a bordered container for clear separation
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    padding: 20px;
                    background-color: #f8f9fa;
                    margin: 10px 0;
                ">
                    <p style="margin: 0; font-size: 16px; line-height: 1.6;">
                        <strong>Recommended Action:</strong><br>
                        {recommended_action}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Display raw JSON (expandable)
            with st.expander("📄 View Raw API Response"):
                st.json(diagnosis_result)
        
        else:
            st.error("❌ Failed to get diagnosis. Please check API connection.")
    
    else:
        # Default view when no submission
        st.info("👈 Please fill in the telematics data in the sidebar and click 'Diagnose Vehicle' to begin.")
        
        # Show example
        st.markdown("### 📝 Example Input")
        example_data = {
            "vehicle_id": "EV-101",
            "speed": 92.0,
            "engine_temperature": 108.0,
            "vibration": 6.8,
            "battery_voltage": 11.6,
            "mileage": 85000.0
        }
        st.json(example_data)
        
        st.markdown("### 🔍 How It Works")
        st.markdown("""
        The system uses a LangGraph-based agentic workflow:
        
        1. **DataAgent**: Validates and cleans telematics input data
        2. **DiagnosisAgent**: Uses ML models (XGBoost and IsolationForest) to predict:
           - Fault probability (supervised learning)
           - Anomaly score (unsupervised learning)
           - Risk level assessment
        3. **SchedulerAgent**: Assigns maintenance recommendations based on risk level
        
        The workflow maintains state throughout the process, allowing each agent
        to build upon previous results.
        """)


if __name__ == "__main__":
    main()

