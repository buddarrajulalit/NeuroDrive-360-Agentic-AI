"""FastAPI route definitions for the predictive maintenance API.

This module defines all API endpoints for vehicle diagnosis, model training,
and inference operations.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List

from app.schemas.diagnosis import DiagnosisRequest, DiagnosisResponse
from app.schemas.model import (
    ModelTrainingRequest,
    ModelTrainingResponse,
    ModelInferenceRequest,
    ModelInferenceResponse,
)
from app.schemas.vehicle import VehicleData, VehicleSensorData, MaintenanceHistory
from app.agents.diagnosis_agent import DiagnosisAgent
from app.models.training import ModelTrainer
from app.models.inference import ModelInference

router = APIRouter(prefix="/api/v1", tags=["predictive-maintenance"])

# Initialize agents and services (in production, these would be dependency injected)
diagnosis_agent = DiagnosisAgent()
model_trainer = ModelTrainer()
model_inference = ModelInference()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint.

    Returns:
        Dictionary with service status.
    """
    return {"status": "healthy", "service": "automotive-predictive-maintenance"}


@router.post("/diagnosis", response_model=DiagnosisResponse, status_code=status.HTTP_200_OK)
async def perform_diagnosis(request: DiagnosisRequest) -> DiagnosisResponse:
    """Perform vehicle diagnosis based on sensor data.

    Analyzes vehicle sensor readings and returns diagnostic findings,
    risk assessment, and maintenance recommendations.

    Args:
        request: DiagnosisRequest containing vehicle_id and sensor_data.

    Returns:
        DiagnosisResponse with findings, risk score, and recommendations.

    Raises:
        HTTPException: If diagnosis fails or request is invalid.
    """
    try:
        diagnosis = diagnosis_agent.diagnose(
            vehicle_id=request.vehicle_id,
            sensor_data=request.sensor_data,
        )
        return diagnosis
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagnosis failed: {str(e)}",
        )


@router.post("/models/train", response_model=ModelTrainingResponse, status_code=status.HTTP_201_CREATED)
async def train_model(request: ModelTrainingRequest) -> ModelTrainingResponse:
    """Train a predictive maintenance model.

    Initiates training of a machine learning model for vehicle maintenance
    prediction based on provided training data and hyperparameters.

    Args:
        request: ModelTrainingRequest with training configuration.

    Returns:
        ModelTrainingResponse with training results and metrics.

    Raises:
        HTTPException: If training fails or request is invalid.
    """
    try:
        training_result = model_trainer.train_model(
            model_type=request.model_type,
            training_data_path=request.training_data_path,
            hyperparameters=request.hyperparameters,
            validation_split=request.validation_split,
            epochs=request.epochs,
            batch_size=request.batch_size,
        )

        return ModelTrainingResponse(**training_result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid training request: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(e)}",
        )


@router.post("/models/inference", response_model=ModelInferenceResponse, status_code=status.HTTP_200_OK)
async def perform_inference(request: ModelInferenceRequest) -> ModelInferenceResponse:
    """Perform inference with a trained model.

    Uses a trained model to make predictions on vehicle sensor data.

    Args:
        request: ModelInferenceRequest with model_id and input features.

    Returns:
        ModelInferenceResponse with prediction and confidence scores.

    Raises:
        HTTPException: If inference fails or request is invalid.
    """
    try:
        inference_result = model_inference.predict(
            model_id=request.model_id,
            vehicle_id=request.vehicle_id,
            input_features=request.input_features,
            include_probabilities=request.include_probabilities,
        )

        return ModelInferenceResponse(**inference_result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid inference request: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@router.get("/models/types", status_code=status.HTTP_200_OK)
async def get_supported_model_types() -> dict:
    """Get list of supported model types.

    Returns:
        Dictionary containing list of supported model types.
    """
    return {"model_types": model_trainer.get_supported_model_types()}


# Vehicle management endpoints (placeholders for full CRUD operations)
@router.post("/vehicles", status_code=status.HTTP_201_CREATED)
async def create_vehicle(vehicle: VehicleData) -> dict:
    """Create a new vehicle record.

    Args:
        vehicle: VehicleData object with vehicle information.

    Returns:
        Dictionary with created vehicle information.
    """
    # Placeholder: In production, would save to database
    return {
        "message": "Vehicle created successfully",
        "vehicle_id": vehicle.vehicle_id,
        "status": "created",
    }


@router.get("/vehicles/{vehicle_id}", response_model=VehicleData)
async def get_vehicle(vehicle_id: str) -> VehicleData:
    """Get vehicle information by ID.

    Args:
        vehicle_id: Vehicle identifier.

    Returns:
        VehicleData object.

    Raises:
        HTTPException: If vehicle not found.
    """
    # Placeholder: In production, would fetch from database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Vehicle {vehicle_id} not found",
    )

