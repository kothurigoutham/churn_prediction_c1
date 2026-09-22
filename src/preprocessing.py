"""
Preprocessing utilities for the Customer Churn Prediction project.

This module is imported both by the training script (train.py) and by the
Streamlit app (app/app.py), so the exact same cleaning/encoding logic is
applied at training time and at prediction time. This avoids train/serve
skew and data leakage.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns dropped before modelling (identifier, not predictive)
ID_COLUMNS = ["customerID"]

TARGET_COLUMN = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw_dataset(path: str) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV exactly as distributed."""
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the fixed, deterministic cleaning steps identified during EDA:

    1. TotalCharges is stored as a string in the raw file and contains 11
       blank values. Every blank value corresponds to a customer with
       tenure == 0 (a brand-new customer who has not yet been billed), so
       blanks are filled with 0 rather than a statistical imputation.
    2. SeniorCitizen is stored as 0/1 (int) rather than Yes/No like the
       other categorical flags. It is treated as categorical (it has only
       two values and is not on a continuous scale), but is left as-is;
       the OneHotEncoder handles integer categories the same way it
       handles strings.
    3. customerID is dropped because it is a unique identifier with no
       predictive value and would leak row-identity, not signal.
    4. Duplicate rows are dropped if any exist (none were found in the
       11-column primary dataset used for this project, but the step is
       kept for robustness).
    """
    df = df.copy()

    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.drop_duplicates()

    for col in ID_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def split_features_target(df: pd.DataFrame):
    """Return (X, y) with y encoded as 1 = Yes/Churn, 0 = No."""
    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    X = df[ALL_FEATURES].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer used inside every model Pipeline.

    Fit only ever happens on the training split (see train.py) to avoid
    leaking information about the validation/test rows into the encoders
    and scaler.
    """
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor
