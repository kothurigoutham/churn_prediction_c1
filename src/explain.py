"""
SHAP explainability for the churn pipeline.

Provides:
  - global_feature_importance(pipeline, X_sample) -> DataFrame
  - explain_single_prediction(pipeline, X_row) -> DataFrame of factor -> shap value

Works generically against the fitted sklearn Pipeline (preprocess + model)
saved by train.py, so the Streamlit app can call the same functions used
during model development.
"""

import numpy as np
import pandas as pd
import shap


def _get_transformed_frame(pipeline, X: pd.DataFrame):
    """Run X through the pipeline's preprocessing step only, returning a
    DataFrame with proper (one-hot expanded) column names."""
    preprocessor = pipeline.named_steps["preprocess"]
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def _build_explainer(pipeline, background: pd.DataFrame):
    model = pipeline.named_steps["model"]
    model_type = type(model).__name__
    bg_transformed = _get_transformed_frame(pipeline, background)

    if model_type in ("RandomForestClassifier", "DecisionTreeClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        # Linear models (e.g. Logistic Regression): use a small background
        # sample for a KernelExplainer-free, faster linear explainer.
        explainer = shap.LinearExplainer(model, bg_transformed)
    return explainer, bg_transformed


def global_feature_importance(pipeline, X_background: pd.DataFrame, sample_size: int = 300) -> pd.DataFrame:
    """Mean absolute SHAP value per feature across a sample of rows."""
    sample = X_background.sample(min(sample_size, len(X_background)), random_state=42)
    explainer, transformed_sample = _build_explainer(pipeline, sample)

    shap_values = explainer.shap_values(transformed_sample)
    if isinstance(shap_values, list):  # some tree explainers return [class0, class1]
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    importance = pd.DataFrame(
        {
            "feature": transformed_sample.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return importance.reset_index(drop=True)


def explain_single_prediction(pipeline, X_row: pd.DataFrame, background: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Return the top_n features driving the prediction for a single customer
    row, ranked by absolute SHAP value, with the signed contribution so the
    caller can show "pushes toward churn" vs "pushes toward staying".
    """
    explainer, _ = _build_explainer(pipeline, background)
    transformed_row = _get_transformed_frame(pipeline, X_row)

    shap_values = explainer.shap_values(transformed_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    row_values = shap_values[0]
    result = pd.DataFrame(
        {
            "feature": transformed_row.columns,
            "shap_value": row_values,
        }
    )
    result["abs_shap"] = result["shap_value"].abs()
    result = result.sort_values("abs_shap", ascending=False).head(top_n)
    result["direction"] = np.where(
        result["shap_value"] > 0, "increases churn risk", "decreases churn risk"
    )
    return result[["feature", "shap_value", "direction"]].reset_index(drop=True)
