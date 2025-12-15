"""Agent state definitions for LangGraph workflows.

This module defines shared state structures that are passed between agents
in a LangGraph workflow for vehicle predictive maintenance.
"""

from typing import TypedDict, Literal, Dict, Any
from pydantic import BaseModel, Field


class TelematicsData(TypedDict):
    """Telematics data structure for vehicle sensor readings.
    
    Attributes:
        speed: Vehicle speed in km/h
        engine_temperature: Engine temperature in Celsius
        vibration: Vibration level in g-force
        battery_voltage: Battery voltage in volts
        mileage: Vehicle mileage in kilometers
    """
    speed: float
    engine_temperature: float
    vibration: float
    battery_voltage: float
    mileage: float


class AgentState(TypedDict, total=False):
    """Shared state structure for LangGraph agent workflow.
    
    This state is passed between agents in the workflow and contains:
    - Vehicle identification
    - Telematics sensor data
    - ML model predictions (fault probability, anomaly score)
    - Risk assessment (risk level)
    - Recommended actions
    
    Note: Using total=False allows fields to be optional during state updates.
    For LangGraph, this enables incremental state updates between nodes.
    
    Attributes:
        raw_input: Raw telematics input data (used only in initial state, removed after DataAgent)
        vehicle_id: Unique vehicle identifier
        telematics: Telematics sensor data (speed, engine_temperature, vibration, battery_voltage, mileage)
        fault_probability: Probability of fault occurrence (0-1)
        anomaly_score: Anomaly detection score (0-1, higher = more anomalous)
        risk_level: Risk level assessment (LOW, MEDIUM, HIGH)
        recommended_action: Recommended maintenance or diagnostic action
    """
    raw_input: Dict[str, Any]  # Raw input data, used only in initial state
    vehicle_id: str
    telematics: TelematicsData
    fault_probability: float
    anomaly_score: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    recommended_action: str


# Alternative Pydantic-based state model (for validation and serialization)
class TelematicsDataModel(BaseModel):
    """Pydantic model for telematics data validation.
    
    Attributes:
        speed: Vehicle speed in km/h
        engine_temperature: Engine temperature in Celsius
        vibration: Vibration level in g-force
        battery_voltage: Battery voltage in volts
        mileage: Vehicle mileage in kilometers
    """
    speed: float = Field(..., ge=0, le=250, description="Vehicle speed in km/h")
    engine_temperature: float = Field(
        ..., ge=-40, le=150, description="Engine temperature in Celsius"
    )
    vibration: float = Field(..., ge=0, le=20, description="Vibration level in g-force")
    battery_voltage: float = Field(
        ..., ge=0, le=20, description="Battery voltage in volts"
    )
    mileage: float = Field(..., ge=0, description="Vehicle mileage in kilometers")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "speed": 92.0,
                "engine_temperature": 108.0,
                "vibration": 6.8,
                "battery_voltage": 11.6,
                "mileage": 85000.0,
            }
        }


class AgentStateModel(BaseModel):
    """Pydantic model for agent state validation and serialization.
    
    This is an alternative to TypedDict that provides validation,
    serialization, and documentation capabilities. Can be used for
    API endpoints or when strict validation is required.
    
    Attributes:
        vehicle_id: Unique vehicle identifier
        telematics: Telematics sensor data
        fault_probability: Probability of fault occurrence (0-1)
        anomaly_score: Anomaly detection score (0-1, higher = more anomalous)
        risk_level: Risk level assessment (LOW, MEDIUM, HIGH)
        recommended_action: Recommended maintenance or diagnostic action
    """
    vehicle_id: str = Field(..., description="Unique vehicle identifier")
    telematics: TelematicsDataModel = Field(..., description="Telematics sensor data")
    fault_probability: float = Field(
        ..., ge=0, le=1, description="Probability of fault (0-1)"
    )
    anomaly_score: float = Field(
        ..., ge=0, le=1, description="Anomaly score (0-1, higher = more anomalous)"
    )
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Risk level assessment"
    )
    recommended_action: str = Field(
        ..., description="Recommended maintenance or diagnostic action"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "vehicle_id": "EV-101",
                "telematics": {
                    "speed": 92.0,
                    "engine_temperature": 108.0,
                    "vibration": 6.8,
                    "battery_voltage": 11.6,
                    "mileage": 85000.0,
                },
                "fault_probability": 0.81,
                "anomaly_score": 0.64,
                "risk_level": "HIGH",
                "recommended_action": "Immediate inspection recommended. Check engine cooling system and battery health.",
            }
        }

