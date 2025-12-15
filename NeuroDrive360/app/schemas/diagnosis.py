"""Diagnosis-related data schemas.

This module defines Pydantic models for diagnosis requests and responses.
"""

from datetime import datetime
from typing import Optional, List, Dict, Literal
from enum import Enum
from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    """Severity levels for diagnostic findings."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    NORMAL = "normal"


class TelematicsInput(BaseModel):
    """Schema for telematics input data.

    Attributes:
        vehicle_id: Unique vehicle identifier
        speed: Vehicle speed in km/h
        engine_temperature: Engine temperature in Celsius
        vibration: Vibration level in g-force
        battery_voltage: Battery voltage in volts
        mileage: Vehicle mileage in kilometers
    """

    vehicle_id: str = Field(..., description="Unique vehicle identifier")
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
                "vehicle_id": "EV-101",
                "speed": 92.0,
                "engine_temperature": 108.0,
                "vibration": 6.8,
                "battery_voltage": 11.6,
                "mileage": 85000.0,
            }
        }


class SimpleDiagnosisResponse(BaseModel):
    """Simplified schema for diagnosis response.

    Attributes:
        vehicle_id: Vehicle identifier
        fault_probability: Probability of fault (0-1)
        anomaly_score: Anomaly score (0-1, higher = more anomalous)
        risk_level: Risk level (LOW, MEDIUM, or HIGH)
    """

    vehicle_id: str = Field(..., description="Vehicle identifier")
    fault_probability: float = Field(
        ..., ge=0, le=1, description="Probability of fault (0-1)"
    )
    anomaly_score: float = Field(
        ..., ge=0, le=1, description="Anomaly score (0-1, higher = more anomalous)"
    )
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Risk level assessment"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "vehicle_id": "EV-101",
                "fault_probability": 0.81,
                "anomaly_score": 0.64,
                "risk_level": "HIGH",
            }
        }


class DiagnosisRequest(BaseModel):
    """Schema for diagnosis request.

    Attributes:
        vehicle_id: Vehicle identifier
        sensor_data: Current vehicle sensor readings
        include_recommendations: Whether to include maintenance recommendations
        include_cost_estimate: Whether to include cost estimates
    """

    vehicle_id: str = Field(..., description="Vehicle identifier")
    sensor_data: dict = Field(..., description="Vehicle sensor data dictionary")
    include_recommendations: bool = Field(True, description="Include maintenance recommendations")
    include_cost_estimate: bool = Field(False, description="Include cost estimates")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "vehicle_id": "VEH-001",
                "sensor_data": {
                    "engine_temperature": 95.5,
                    "oil_pressure": 45.2,
                    "battery_voltage": 12.6,
                },
                "include_recommendations": True,
                "include_cost_estimate": False,
            }
        }


class DiagnosticFinding(BaseModel):
    """Schema for individual diagnostic finding.

    Attributes:
        component: Component or system being diagnosed
        issue: Description of the issue found
        severity: Severity level of the issue
        confidence: Confidence score (0-1) of the diagnosis
        recommendation: Recommended action to take
        estimated_cost: Estimated repair cost if applicable
    """

    component: str = Field(..., description="Component or system name")
    issue: str = Field(..., description="Issue description")
    severity: SeverityLevel = Field(..., description="Severity level")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    recommendation: Optional[str] = Field(None, description="Recommended action")
    estimated_cost: Optional[float] = Field(None, ge=0, description="Estimated repair cost")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "component": "Engine Cooling System",
                "issue": "Engine temperature slightly elevated",
                "severity": "warning",
                "confidence": 0.75,
                "recommendation": "Monitor coolant levels and check for leaks",
                "estimated_cost": None,
            }
        }


class DiagnosisResponse(BaseModel):
    """Schema for diagnosis response.

    Attributes:
        vehicle_id: Vehicle identifier
        diagnosis_id: Unique diagnosis session identifier
        timestamp: Diagnosis timestamp
        overall_status: Overall vehicle health status
        findings: List of diagnostic findings
        risk_score: Overall risk score (0-100, higher is worse)
        next_maintenance_due: Estimated date for next maintenance
        summary: Summary of diagnosis results
    """

    vehicle_id: str = Field(..., description="Vehicle identifier")
    diagnosis_id: str = Field(..., description="Unique diagnosis session ID")
    timestamp: datetime = Field(..., description="Diagnosis timestamp")
    overall_status: str = Field(..., description="Overall vehicle health status")
    findings: List[DiagnosticFinding] = Field(default_factory=list, description="List of findings")
    risk_score: float = Field(..., ge=0, le=100, description="Overall risk score (0-100)")
    next_maintenance_due: Optional[datetime] = Field(None, description="Next maintenance due date")
    summary: str = Field(..., description="Summary of diagnosis")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "vehicle_id": "VEH-001",
                "diagnosis_id": "DIAG-2024-001",
                "timestamp": "2024-01-15T10:30:00Z",
                "overall_status": "Good",
                "findings": [],
                "risk_score": 15.5,
                "next_maintenance_due": "2024-04-15T00:00:00Z",
                "summary": "Vehicle is in good condition with no critical issues detected.",
            }
        }

