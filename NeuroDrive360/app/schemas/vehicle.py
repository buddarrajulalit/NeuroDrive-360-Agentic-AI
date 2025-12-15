"""Vehicle-related data schemas.

This module defines Pydantic models for vehicle information, sensor data,
and maintenance history.
"""

from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class VehicleData(BaseModel):
    """Schema for basic vehicle information.

    Attributes:
        vehicle_id: Unique identifier for the vehicle
        make: Vehicle manufacturer (e.g., 'Toyota', 'Ford')
        model: Vehicle model name
        year: Manufacturing year
        mileage: Current vehicle mileage in kilometers
        vin: Vehicle Identification Number
    """

    vehicle_id: str = Field(..., description="Unique vehicle identifier")
    make: str = Field(..., description="Vehicle manufacturer")
    model: str = Field(..., description="Vehicle model")
    year: int = Field(..., ge=1900, le=2100, description="Manufacturing year")
    mileage: float = Field(..., ge=0, description="Current mileage in kilometers")
    vin: Optional[str] = Field(None, description="Vehicle Identification Number")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "vehicle_id": "VEH-001",
                "make": "Toyota",
                "model": "Camry",
                "year": 2020,
                "mileage": 45000.5,
                "vin": "1HGBH41JXMN109186",
            }
        }


class VehicleSensorData(BaseModel):
    """Schema for vehicle sensor readings.

    Attributes:
        vehicle_id: Vehicle identifier
        timestamp: Time of sensor reading
        engine_temperature: Engine temperature in Celsius
        oil_pressure: Oil pressure in PSI
        coolant_temperature: Coolant temperature in Celsius
        battery_voltage: Battery voltage in volts
        tire_pressure: Tire pressure readings (front_left, front_right, rear_left, rear_right) in PSI
        engine_rpm: Engine revolutions per minute
        fuel_level: Fuel level percentage (0-100)
        check_engine_light: Whether check engine light is on
        diagnostic_codes: List of diagnostic trouble codes (DTCs)
    """

    vehicle_id: str = Field(..., description="Vehicle identifier")
    timestamp: datetime = Field(..., description="Sensor reading timestamp")
    engine_temperature: float = Field(..., ge=-40, le=150, description="Engine temperature (°C)")
    oil_pressure: float = Field(..., ge=0, le=100, description="Oil pressure (PSI)")
    coolant_temperature: float = Field(..., ge=-40, le=150, description="Coolant temperature (°C)")
    battery_voltage: float = Field(..., ge=0, le=20, description="Battery voltage (V)")
    tire_pressure: Dict[str, float] = Field(
        ..., description="Tire pressure readings (PSI) for each tire position"
    )
    engine_rpm: int = Field(..., ge=0, le=10000, description="Engine RPM")
    fuel_level: float = Field(..., ge=0, le=100, description="Fuel level percentage")
    check_engine_light: bool = Field(False, description="Check engine light status")
    diagnostic_codes: List[str] = Field(default_factory=list, description="Diagnostic trouble codes")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "vehicle_id": "VEH-001",
                "timestamp": "2024-01-15T10:30:00Z",
                "engine_temperature": 95.5,
                "oil_pressure": 45.2,
                "coolant_temperature": 88.0,
                "battery_voltage": 12.6,
                "tire_pressure": {
                    "front_left": 32.0,
                    "front_right": 31.8,
                    "rear_left": 32.2,
                    "rear_right": 32.1,
                },
                "engine_rpm": 750,
                "fuel_level": 65.5,
                "check_engine_light": False,
                "diagnostic_codes": [],
            }
        }


class MaintenanceHistory(BaseModel):
    """Schema for vehicle maintenance history records.

    Attributes:
        vehicle_id: Vehicle identifier
        maintenance_id: Unique maintenance record identifier
        maintenance_type: Type of maintenance performed
        service_date: Date of service
        mileage_at_service: Mileage at time of service
        description: Detailed description of maintenance
        cost: Cost of maintenance in currency units
        parts_replaced: List of parts that were replaced
    """

    vehicle_id: str = Field(..., description="Vehicle identifier")
    maintenance_id: str = Field(..., description="Unique maintenance record ID")
    maintenance_type: str = Field(..., description="Type of maintenance (e.g., 'Oil Change', 'Brake Service')")
    service_date: datetime = Field(..., description="Service date")
    mileage_at_service: float = Field(..., ge=0, description="Mileage at service")
    description: Optional[str] = Field(None, description="Service description")
    cost: Optional[float] = Field(None, ge=0, description="Service cost")
    parts_replaced: List[str] = Field(default_factory=list, description="List of replaced parts")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "vehicle_id": "VEH-001",
                "maintenance_id": "MAINT-001",
                "maintenance_type": "Oil Change",
                "service_date": "2024-01-10T08:00:00Z",
                "mileage_at_service": 44000.0,
                "description": "Full synthetic oil change and filter replacement",
                "cost": 89.99,
                "parts_replaced": ["Oil Filter", "Engine Oil"],
            }
        }

