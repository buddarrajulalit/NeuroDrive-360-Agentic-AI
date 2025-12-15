"""Diagnosis agent for vehicle health assessment.

This module implements an intelligent agent that analyzes vehicle sensor data
using ML models and provides diagnostic insights with risk level assessment.
"""

from typing import Dict, Any
from app.models.inference import get_inference_instance
import logging

logger = logging.getLogger(__name__)


class DiagnosisAgent:
    """Agent for performing vehicle diagnosis based on sensor data.

    This agent uses trained ML models (XGBoost and IsolationForest) to analyze
    vehicle sensor readings and assess risk levels. It calibrates classifier
    probabilities to avoid bimodal outputs and blends them with anomaly scores
    to keep all risk tiers reachable.

    Attributes:
        inference: ModelInference instance for making predictions
    """

    def __init__(self, model_dir: str = "models"):
        """Initialize the diagnosis agent.

        Args:
            model_dir: Directory containing saved models.
        """
        self.inference = get_inference_instance(model_dir=model_dir)
        logger.info("DiagnosisAgent initialized")

    def diagnose(
        self,
        vehicle_id: str,
        speed: float,
        engine_temperature: float,
        vibration: float,
        battery_voltage: float,
        mileage: float
    ) -> Dict[str, Any]:
        """Perform vehicle diagnosis and assign risk level.

        Uses trained models to predict fault probability and anomaly score,
        calibrates the classifier probability to reduce extreme 0/1 behavior,
        blends it with anomaly evidence into a combined risk, and assigns a
        risk level based on fixed thresholds.

        Args:
            vehicle_id: Vehicle identifier.
            speed: Vehicle speed in km/h.
            engine_temperature: Engine temperature in Celsius.
            vibration: Vibration level in g-force.
            battery_voltage: Battery voltage in volts.
            mileage: Vehicle mileage in km.

        Returns:
            Dictionary containing:
                - vehicle_id: Vehicle identifier
                - fault_probability: Probability of fault (0-1)
                - anomaly_score: Anomaly score (0-1, higher = more anomalous)
                - risk_level: Risk level (LOW, MEDIUM, or HIGH)

        Raises:
            ValueError: If input values are invalid.
            RuntimeError: If models fail to make predictions.
        """
        logger.info(f"Diagnosing vehicle: {vehicle_id}")

        # Get predictions from inference module
        predictions = self.inference.predict(
            speed=speed,
            engine_temperature=engine_temperature,
            vibration=vibration,
            battery_voltage=battery_voltage,
            mileage=mileage
        )

        fault_probability = predictions['fault_probability']
        anomaly_score = predictions['anomaly_score']

        # Calibrate classifier probability to avoid overconfident, bimodal outputs.
        calibrated_fault_prob = min(max(fault_probability * 1.8, 0.0), 1.0)

        # Blend calibrated fault risk with anomaly evidence for smoother risk bands.
        combined_risk = 0.6 * calibrated_fault_prob + 0.4 * anomaly_score

        # Determine risk level using combined risk thresholds.
        if combined_risk < 0.35:
            risk_level = "LOW"
        elif combined_risk < 0.65:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        logger.info(
            f"Vehicle {vehicle_id} - Fault prob: {fault_probability:.4f} "
            f"(calibrated: {calibrated_fault_prob:.4f}), "
            f"Anomaly score: {anomaly_score:.4f}, "
            f"Combined risk: {combined_risk:.4f}, Risk: {risk_level}"
        )

        return {
            'vehicle_id': vehicle_id,
            'fault_probability': fault_probability,
            'anomaly_score': anomaly_score,
            'combined_risk': combined_risk,
            'risk_level': risk_level
        }
