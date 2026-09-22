"""
Prediction helper: wraps the saved pipeline so the Streamlit app never has
to touch preprocessing details directly.
"""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "models" / "churn_pipeline.pkl"


def load_pipeline():
    return joblib.load(PIPELINE_PATH)


def predict_one(pipeline, customer: dict) -> dict:
    """
    customer: dict of the raw feature values (same schema as ALL_FEATURES
    in preprocessing.py). Returns prediction label, probability and a
    simple risk band.
    """
    X = pd.DataFrame([customer])
    proba = float(pipeline.predict_proba(X)[0][1])
    label = "Likely to Churn" if proba >= 0.5 else "Likely to Stay"

    if proba >= 0.6:
        risk = "High"
    elif proba >= 0.3:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "label": label,
        "probability": proba,
        "risk": risk,
        "input_frame": X,
    }
