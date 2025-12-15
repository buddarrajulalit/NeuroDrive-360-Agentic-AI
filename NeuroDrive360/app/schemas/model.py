"""Model training and inference schemas.

This module defines Pydantic models for ML model operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ModelTrainingRequest(BaseModel):
    """Schema for model training request.

    Attributes:
        model_type: Type of model to train (e.g., 'failure_prediction', 'anomaly_detection')
        training_data_path: Path to training data
        hyperparameters: Model hyperparameters
        validation_split: Proportion of data to use for validation
        epochs: Number of training epochs
        batch_size: Training batch size
    """

    model_type: str = Field(..., description="Type of model to train")
    training_data_path: Optional[str] = Field(None, description="Path to training data")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    validation_split: float = Field(0.2, ge=0, le=0.5, description="Validation data split ratio")
    epochs: int = Field(100, ge=1, description="Number of training epochs")
    batch_size: int = Field(32, ge=1, description="Training batch size")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "model_type": "failure_prediction",
                "training_data_path": "/data/training/vehicle_data.csv",
                "hyperparameters": {"learning_rate": 0.001, "hidden_layers": [64, 32]},
                "validation_split": 0.2,
                "epochs": 100,
                "batch_size": 32,
            }
        }


class ModelTrainingResponse(BaseModel):
    """Schema for model training response.

    Attributes:
        training_id: Unique training session identifier
        model_id: Trained model identifier
        status: Training status
        start_time: Training start timestamp
        end_time: Training end timestamp
        metrics: Training metrics (accuracy, loss, etc.)
        model_path: Path where model is saved
        message: Status message
    """

    training_id: str = Field(..., description="Unique training session ID")
    model_id: str = Field(..., description="Trained model identifier")
    status: str = Field(..., description="Training status")
    start_time: datetime = Field(..., description="Training start time")
    end_time: Optional[datetime] = Field(None, description="Training end time")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Training metrics")
    model_path: Optional[str] = Field(None, description="Path to saved model")
    message: str = Field(..., description="Status message")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "training_id": "TRAIN-2024-001",
                "model_id": "MODEL-001",
                "status": "completed",
                "start_time": "2024-01-15T09:00:00Z",
                "end_time": "2024-01-15T10:30:00Z",
                "metrics": {"accuracy": 0.92, "f1_score": 0.89, "loss": 0.15},
                "model_path": "/models/failure_prediction_v1.pkl",
                "message": "Model training completed successfully",
            }
        }


class ModelInferenceRequest(BaseModel):
    """Schema for model inference request.

    Attributes:
        model_id: Model identifier to use for inference
        vehicle_id: Vehicle identifier
        input_features: Feature vector for prediction
        include_probabilities: Whether to include prediction probabilities
    """

    model_id: str = Field(..., description="Model identifier")
    vehicle_id: str = Field(..., description="Vehicle identifier")
    input_features: Dict[str, float] = Field(..., description="Input feature vector")
    include_probabilities: bool = Field(False, description="Include prediction probabilities")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "model_id": "MODEL-001",
                "vehicle_id": "VEH-001",
                "input_features": {
                    "engine_temperature": 95.5,
                    "oil_pressure": 45.2,
                    "mileage": 45000.0,
                },
                "include_probabilities": True,
            }
        }


class ModelInferenceResponse(BaseModel):
    """Schema for model inference response.

    Attributes:
        model_id: Model identifier used
        vehicle_id: Vehicle identifier
        prediction: Predicted value or class
        confidence: Prediction confidence score (0-1)
        probabilities: Class probabilities if requested
        timestamp: Inference timestamp
        feature_importance: Feature importance scores if available
    """

    model_id: str = Field(..., description="Model identifier")
    vehicle_id: str = Field(..., description="Vehicle identifier")
    prediction: Any = Field(..., description="Model prediction")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    timestamp: datetime = Field(..., description="Inference timestamp")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "model_id": "MODEL-001",
                "vehicle_id": "VEH-001",
                "prediction": "low_risk",
                "confidence": 0.87,
                "probabilities": {"low_risk": 0.87, "medium_risk": 0.11, "high_risk": 0.02},
                "timestamp": "2024-01-15T10:30:00Z",
                "feature_importance": {"engine_temperature": 0.35, "oil_pressure": 0.28, "mileage": 0.15},
            }
        }

