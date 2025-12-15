"""Model inference module for predictive maintenance.

This module loads saved models (XGBoost classifier and IsolationForest)
and provides inference capabilities for fault prediction and anomaly detection.
"""

import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelInference:
    """Handles inference operations with trained predictive models.

    This class manages model loading, feature preprocessing, and prediction
    generation for vehicle maintenance predictions.

    Attributes:
        xgb_model: Loaded XGBoost classifier model
        iso_forest: Loaded IsolationForest model
        iso_scaler: Loaded StandardScaler for IsolationForest preprocessing
        models_loaded: Boolean indicating if models are loaded
    """

    def __init__(self, model_dir: str = "models"):
        """Initialize the model inference handler.

        Args:
            model_dir: Directory containing saved models (default: "models").
        """
        self.model_dir = model_dir
        self.xgb_model = None
        self.iso_forest = None
        self.iso_scaler = None
        self.models_loaded = False

    def load_models(self) -> None:
        """Load saved models and scaler from disk.

        Raises:
            FileNotFoundError: If model files are not found.
            Exception: If model loading fails.
        """
        try:
            model_dir_path = Path(self.model_dir)

            # Load XGBoost classifier
            xgb_path = model_dir_path / "xgboost_classifier.pkl"
            if not xgb_path.exists():
                raise FileNotFoundError(f"XGBoost model not found: {xgb_path}")

            self.xgb_model = joblib.load(xgb_path)
            logger.info(f"Loaded XGBoost model from {xgb_path}")

            # Load IsolationForest model
            iso_model_path = model_dir_path / "isolation_forest.pkl"
            if not iso_model_path.exists():
                raise FileNotFoundError(f"IsolationForest model not found: {iso_model_path}")

            self.iso_forest = joblib.load(iso_model_path)
            logger.info(f"Loaded IsolationForest model from {iso_model_path}")

            # Load IsolationForest scaler
            iso_scaler_path = model_dir_path / "isolation_forest_scaler.pkl"
            if not iso_scaler_path.exists():
                raise FileNotFoundError(f"IsolationForest scaler not found: {iso_scaler_path}")

            self.iso_scaler = joblib.load(iso_scaler_path)
            logger.info(f"Loaded IsolationForest scaler from {iso_scaler_path}")

            self.models_loaded = True
            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def predict(
        self,
        speed: float,
        engine_temperature: float,
        vibration: float,
        battery_voltage: float,
        mileage: float
    ) -> Dict[str, float]:
        """Predict fault probability and anomaly score for a single telematics input.

        Args:
            speed: Vehicle speed in km/h
            engine_temperature: Engine temperature in Celsius
            vibration: Vibration level in g-force
            battery_voltage: Battery voltage in volts
            mileage: Vehicle mileage in km

        Returns:
            Dictionary containing:
                - fault_probability: Probability of fault (0-1) from XGBoost
                - anomaly_score: Normalized anomaly score (0-1) from IsolationForest
                                 (higher = more anomalous)

        Raises:
            RuntimeError: If models are not loaded.
            ValueError: If input values are invalid.
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Validate inputs
        if not all(isinstance(x, (int, float)) for x in [speed, engine_temperature, vibration, battery_voltage, mileage]):
            raise ValueError("All input values must be numeric")

        # Create feature array in the same order as training
        # Order: speed, engine_temperature, vibration, battery_voltage, mileage
        features = np.array([[
            speed,
            engine_temperature,
            vibration,
            battery_voltage,
            mileage
        ]])

        # Convert to DataFrame for XGBoost (expects named columns)
        feature_df = pd.DataFrame(features, columns=[
            'speed',
            'engine_temperature',
            'vibration',
            'battery_voltage',
            'mileage'
        ])

        # Predict fault probability using XGBoost
        fault_probability = float(self.xgb_model.predict_proba(feature_df)[0, 1])
        logger.debug(f"Fault probability: {fault_probability:.4f}")

        # Predict anomaly score using IsolationForest
        # Scale features first
        features_scaled = self.iso_scaler.transform(features)

        # Get anomaly score (lower = more anomalous)
        anomaly_score_raw = self.iso_forest.score_samples(features_scaled)[0]

        # Normalize to 0-1 range where higher = more anomalous
        # IsolationForest returns negative scores for anomalies
        # We normalize to [0, 1] where 1 = most anomalous
        # Typical range is approximately [-0.5, -0.1] for normal, lower for anomalies
        # We'll use a simple normalization approach
        min_score = -0.6  # Approximate minimum
        max_score = -0.05  # Approximate maximum
        anomaly_score = 1 - (anomaly_score_raw - min_score) / (max_score - min_score)
        anomaly_score = np.clip(anomaly_score, 0.0, 1.0)

        logger.debug(f"Anomaly score (raw): {anomaly_score_raw:.4f}, normalized: {anomaly_score:.4f}")

        return {
            'fault_probability': round(fault_probability, 4),
            'anomaly_score': round(anomaly_score, 4)
        }


# Global inference instance (can be initialized once and reused)
_inference_instance: Optional[ModelInference] = None


def get_inference_instance(model_dir: str = "models") -> ModelInference:
    """Get or create a global inference instance.

    Args:
        model_dir: Directory containing saved models.

    Returns:
        ModelInference instance with loaded models.
    """
    global _inference_instance

    if _inference_instance is None or not _inference_instance.models_loaded:
        _inference_instance = ModelInference(model_dir=model_dir)
        _inference_instance.load_models()

    return _inference_instance
