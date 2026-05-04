"""
model.py
--------
ML logic for anomaly detection using Isolation Forest.

Isolation Forest works by randomly partitioning data.
Anomalies are easier to isolate (fewer splits needed) → lower score.

This module handles:
- Model training
- Prediction (anomaly flag + score)
- Model persistence (save/load to avoid retraining)
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Path to save trained model and scaler
MODEL_PATH = "models/isolation_forest.pkl"
SCALER_PATH = "models/scaler.pkl"


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric features for model training/prediction.
    
    Features used:
    - amount_zscore: normalized transaction amount
    - hour: hour of day extracted from timestamp
    - amount: raw amount (helps with scale context)
    """
    features = df[["amount_zscore", "hour", "amount"]].copy()
    return features


def train_model(df: pd.DataFrame) -> tuple:
    """
    Train Isolation Forest on the provided dataframe.
    
    Args:
        df: Preprocessed dataframe with feature columns
        
    Returns:
        Tuple of (trained model, fitted scaler)
    
    Isolation Forest params:
    - contamination=0.1 → assumes ~10% of data is anomalous
    - n_estimators=100 → number of trees (more = stable, slower)
    - random_state=42 → reproducibility
    """
    features = get_features(df)
    
    # Scale features so no single feature dominates
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Train the model
    model = IsolationForest(
        contamination=0.1,
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    model.fit(X_scaled)
    
    # Persist to disk
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"[MODEL] Trained and saved to {MODEL_PATH}")
    return model, scaler


def load_model() -> tuple:
    """
    Load pre-trained model from disk.
    Raises FileNotFoundError if model doesn't exist yet.
    
    Returns:
        Tuple of (model, scaler) or (None, None) if not found
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("[MODEL] Loaded pre-trained model from disk")
        return model, scaler
    return None, None


def predict(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    Run anomaly detection on processed dataframe.
    
    Adds two columns:
    - anomaly_flag: -1 = anomaly, 1 = normal
    - anomaly_score: raw decision function output
      (more negative = more anomalous)
    
    Args:
        df: Preprocessed dataframe
        model: Trained IsolationForest
        scaler: Fitted StandardScaler
        
    Returns:
        df with anomaly_flag and anomaly_score columns added
    """
    features = get_features(df)
    X_scaled = scaler.transform(features)
    
    # predict() returns -1 (anomaly) or 1 (normal)
    df = df.copy()
    df["anomaly_flag"] = model.predict(X_scaled)
    
    # decision_function: lower (more negative) = more anomalous
    df["anomaly_score"] = model.decision_function(X_scaled)
    
    return df
