"""ML model training and inference modules.

This module contains functionality for training predictive models
and performing inference on vehicle data.
"""

from app.models.training import ModelTrainer
from app.models.inference import ModelInference

__all__ = ["ModelTrainer", "ModelInference"]

