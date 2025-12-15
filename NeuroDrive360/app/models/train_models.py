"""Model training script for automotive predictive maintenance.

This module trains two models:
1. IsolationForest for anomaly detection (unsupervised)
2. XGBoost classifier for fault prediction (supervised)

The script loads telematics data, preprocesses it, trains both models,
evaluates them using ROC-AUC and F1-score, and saves the trained models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
import xgboost as xgb
import joblib
import os
from pathlib import Path
import logging
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_preprocess_data(data_path: str = "telematics_data.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess telematics data.

    Args:
        data_path: Path to the CSV file containing telematics data.

    Returns:
        Tuple of (features DataFrame, target Series).
        Features exclude timestamp and vehicle_id.
        Target is the fault_label column.

    Raises:
        FileNotFoundError: If the data file doesn't exist.
        ValueError: If required columns are missing.
    """
    logger.info(f"Loading data from {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    # Verify required columns exist
    required_columns = [
        'speed', 'engine_temperature', 'vibration',
        'battery_voltage', 'mileage', 'fault_label'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Drop non-feature columns (timestamp, vehicle_id)
    columns_to_drop = ['timestamp', 'vehicle_id']
    existing_drop_cols = [col for col in columns_to_drop if col in df.columns]
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)
        logger.info(f"Dropped columns: {existing_drop_cols}")

    # Separate features and target
    X = df.drop(columns=['fault_label'])
    y = df['fault_label']

    logger.info(f"Feature columns: {list(X.columns)}")
    logger.info(f"Target distribution:\n{y.value_counts().to_dict()}")
    logger.info(f"Target percentage:\n{(y.value_counts(normalize=True) * 100).to_dict()}")

    # Check for missing values
    if X.isnull().sum().any():
        logger.warning("Missing values detected in features. Filling with median.")
        X = X.fillna(X.median())

    if y.isnull().sum() > 0:
        logger.warning("Missing values detected in target. Dropping rows.")
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]

    return X, y


def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 42,
    n_estimators: int = 100
) -> Tuple[IsolationForest, StandardScaler]:
    """Train IsolationForest model for anomaly detection.

    IsolationForest is an unsupervised learning algorithm that identifies
    anomalies by isolating observations. It doesn't use the target labels.

    Args:
        X_train: Training feature data.
        contamination: Expected proportion of anomalies (0-0.5).
        random_state: Random seed for reproducibility.
        n_estimators: Number of base estimators in the ensemble.

    Returns:
        Tuple of (trained IsolationForest model, fitted StandardScaler).
    """
    logger.info("Training IsolationForest model for anomaly detection")

    # Standardize features (important for IsolationForest)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train IsolationForest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        n_jobs=-1  # Use all available CPUs
    )

    iso_forest.fit(X_train_scaled)
    logger.info("IsolationForest training completed")

    return iso_forest, scaler


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1
) -> xgb.XGBClassifier:
    """Train XGBoost classifier for fault prediction.

    XGBoost is a gradient boosting framework that excels at classification
    tasks. It uses the fault_label as the target for supervised learning.

    Args:
        X_train: Training feature data.
        y_train: Training target labels.
        random_state: Random seed for reproducibility.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage used in updates.

    Returns:
        Trained XGBoost classifier.
    """
    logger.info("Training XGBoost classifier for fault prediction")

    # Train XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1  # Use all available CPUs
    )

    xgb_model.fit(X_train, y_train)
    logger.info("XGBoost training completed")

    return xgb_model


def evaluate_anomaly_detection(
    model: IsolationForest,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate IsolationForest anomaly detection model.

    Note: IsolationForest returns -1 for anomalies and 1 for normal samples.
    We convert this to binary labels (1 for anomaly, 0 for normal) to compare
    with the actual fault labels.

    Args:
        model: Trained IsolationForest model.
        scaler: Fitted StandardScaler.
        X_test: Test feature data.
        y_test: Test target labels (ground truth).

    Returns:
        Dictionary containing evaluation metrics (roc_auc, f1_score).
    """
    logger.info("Evaluating IsolationForest model")

    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Predict anomalies (-1 = anomaly, 1 = normal)
    predictions = model.predict(X_test_scaled)

    # Convert to binary (1 = anomaly, 0 = normal)
    # IsolationForest: -1 = anomaly, 1 = normal
    # Our target: 1 = fault, 0 = normal
    predictions_binary = (predictions == -1).astype(int)

    # Get anomaly scores (lower score = more anomalous)
    anomaly_scores = model.score_samples(X_test_scaled)
    # Normalize scores to probabilities for ROC-AUC
    # Convert scores to probabilities: lower score = higher probability of anomaly
    scores_normalized = 1 - (anomaly_scores - anomaly_scores.min()) / (
        anomaly_scores.max() - anomaly_scores.min() + 1e-10
    )

    # Calculate metrics
    try:
        roc_auc = roc_auc_score(y_test, scores_normalized)
    except ValueError as e:
        logger.warning(f"ROC-AUC calculation failed: {e}. Using binary predictions.")
        roc_auc = roc_auc_score(y_test, predictions_binary)

    f1 = f1_score(y_test, predictions_binary)

    logger.info(f"IsolationForest - ROC-AUC: {roc_auc:.4f}, F1-Score: {f1:.4f}")

    return {
        'roc_auc': roc_auc,
        'f1_score': f1,
        'predictions': predictions_binary,
        'anomaly_scores': anomaly_scores
    }


def evaluate_classifier(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """Evaluate XGBoost classifier model.

    Args:
        model: Trained XGBoost classifier.
        X_test: Test feature data.
        y_test: Test target labels (ground truth).

    Returns:
        Dictionary containing evaluation metrics and predictions.
    """
    logger.info("Evaluating XGBoost classifier")

    # Predict probabilities and classes
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of fault
    y_pred = model.predict(X_test)

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"XGBoost - ROC-AUC: {roc_auc:.4f}, F1-Score: {f1:.4f}")

    # Print detailed classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")

    return {
        'roc_auc': roc_auc,
        'f1_score': f1,
        'predictions': y_pred,
        'prediction_probas': y_pred_proba,
        'confusion_matrix': cm
    }


def save_models(
    iso_forest: IsolationForest,
    iso_scaler: StandardScaler,
    xgb_model: xgb.XGBClassifier,
    model_dir: str = "models"
) -> None:
    """Save trained models and scaler to disk.

    Args:
        iso_forest: Trained IsolationForest model.
        iso_scaler: Fitted StandardScaler for IsolationForest.
        xgb_model: Trained XGBoost classifier.
        model_dir: Directory to save models (default: "models").
    """
    # Create model directory if it doesn't exist
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving models to {model_dir}/")

    # Save IsolationForest model and scaler
    iso_model_path = os.path.join(model_dir, "isolation_forest.pkl")
    iso_scaler_path = os.path.join(model_dir, "isolation_forest_scaler.pkl")

    joblib.dump(iso_forest, iso_model_path)
    joblib.dump(iso_scaler, iso_scaler_path)
    logger.info(f"Saved IsolationForest: {iso_model_path}")
    logger.info(f"Saved IsolationForest Scaler: {iso_scaler_path}")

    # Save XGBoost model
    xgb_model_path = os.path.join(model_dir, "xgboost_classifier.pkl")
    joblib.dump(xgb_model, xgb_model_path)
    logger.info(f"Saved XGBoost Classifier: {xgb_model_path}")


def main():
    """Main training pipeline.

    Orchestrates the complete training workflow:
    1. Load and preprocess data
    2. Split into train/test sets
    3. Train IsolationForest model
    4. Train XGBoost classifier
    5. Evaluate both models
    6. Save trained models
    """
    logger.info("=" * 60)
    logger.info("Starting Model Training Pipeline")
    logger.info("=" * 60)

    try:
        # Step 1: Load and preprocess data
        X, y = load_and_preprocess_data("telematics_data.csv")

        # Step 2: Split into train/test sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y  # Maintain class distribution
        )

        logger.info(f"\nTrain set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")

        # Step 3: Train IsolationForest
        iso_forest, iso_scaler = train_isolation_forest(
            X_train,
            contamination=0.1,  # Expect ~10% anomalies
            random_state=42,
            n_estimators=100
        )

        # Step 4: Train XGBoost classifier
        xgb_model = train_xgboost_classifier(
            X_train,
            y_train,
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )

        # Step 5: Evaluate models
        logger.info("\n" + "=" * 60)
        logger.info("Model Evaluation")
        logger.info("=" * 60)

        iso_metrics = evaluate_anomaly_detection(
            iso_forest, iso_scaler, X_test, y_test
        )

        xgb_metrics = evaluate_classifier(
            xgb_model, X_test, y_test
        )

        # Step 6: Save models
        logger.info("\n" + "=" * 60)
        logger.info("Saving Models")
        logger.info("=" * 60)

        save_models(iso_forest, iso_scaler, xgb_model)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("Training Pipeline Completed Successfully")
        logger.info("=" * 60)
        logger.info("\nFinal Metrics Summary:")
        logger.info(f"IsolationForest - ROC-AUC: {iso_metrics['roc_auc']:.4f}, "
                   f"F1-Score: {iso_metrics['f1_score']:.4f}")
        logger.info(f"XGBoost - ROC-AUC: {xgb_metrics['roc_auc']:.4f}, "
                   f"F1-Score: {xgb_metrics['f1_score']:.4f}")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

