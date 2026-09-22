"""
Customer Churn Prediction — Streamlit Application

Loads the pre-trained pipeline saved by src/train.py (no retraining happens
here) and lets a user enter a customer's details to get a churn prediction,
probability, risk level, and a SHAP-based explanation.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from explain import explain_single_prediction, global_feature_importance  # noqa: E402
from preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    clean_dataset,
    load_raw_dataset,
    split_features_target,
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

MODEL_PATH = ROOT / "models" / "churn_pipeline.pkl"
SUMMARY_PATH = ROOT / "models" / "final_model_summary.json"
DATA_PATH = ROOT / "data" / "primary" / "telco_churn.csv"


# ---------------------------------------------------------------------------
# Cached loaders — the pipeline is loaded once and reused across interactions
# ---------------------------------------------------------------------------
@st.cache_resource
def get_pipeline():
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_background_data():
    raw = load_raw_dataset(DATA_PATH)
    clean = clean_dataset(raw)
    X, y = split_features_target(clean)
    return X, y, clean


@st.cache_data
def get_model_summary():
    with open(SUMMARY_PATH) as f:
        return json.load(f)


@st.cache_data
def get_global_importance(_pipeline, X_background):
    return global_feature_importance(_pipeline, X_background, sample_size=300)


pipeline = get_pipeline()
X_background, y_background, df_clean = get_background_data()
model_summary = get_model_summary()

CATEGORY_OPTIONS = {
    col: sorted(X_background[col].unique().tolist()) for col in CATEGORICAL_FEATURES
}

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Churn Prediction")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Customer Information & Prediction", "Model Insights", "About"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Final model: **{model_summary['final_model']}**")
st.sidebar.caption(f"Test F1-score: **{model_summary['final_metrics']['F1']:.3f}**")
st.sidebar.caption(f"Test ROC-AUC: **{model_summary['final_metrics']['ROC-AUC']:.3f}**")

# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
if page == "Dashboard":
    st.title("Customer Churn Prediction")
    st.caption("AI-Powered Customer Retention Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df_clean):,}")
    c2.metric("Churn Rate", f"{(y_background.mean()*100):.1f}%")
    c3.metric("Avg. Monthly Charges", f"${df_clean['MonthlyCharges'].mean():.2f}")
    c4.metric("Avg. Tenure", f"{df_clean['tenure'].mean():.1f} months")

    st.markdown("### Churn by Contract Type")
    ct = pd.crosstab(df_clean["Contract"], df_clean["Churn"], normalize="index") * 100
    fig = go.Figure()
    for churn_val, color in [("No", "#2E86AB"), ("Yes", "#E63946")]:
        fig.add_bar(name=churn_val, x=ct.index, y=ct[churn_val], marker_color=color)
    fig.update_layout(barmode="stack", yaxis_title="% of customers", height=380)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Churn by Internet Service")
        ct2 = pd.crosstab(df_clean["InternetService"], df_clean["Churn"], normalize="index") * 100
        fig2 = go.Figure()
        for churn_val, color in [("No", "#2E86AB"), ("Yes", "#E63946")]:
            fig2.add_bar(name=churn_val, x=ct2.index, y=ct2[churn_val], marker_color=color)
        fig2.update_layout(barmode="stack", yaxis_title="% of customers", height=340)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("### Churn by Payment Method")
        ct3 = pd.crosstab(df_clean["PaymentMethod"], df_clean["Churn"], normalize="index") * 100
        fig3 = go.Figure()
        for churn_val, color in [("No", "#2E86AB"), ("Yes", "#E63946")]:
            fig3.add_bar(name=churn_val, x=ct3.index, y=ct3[churn_val], marker_color=color)
        fig3.update_layout(barmode="stack", yaxis_title="% of customers", height=340)
        st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "These figures are computed live from the actual Telco Customer Churn dataset "
        "(7,043 customers) used to train the model — nothing here is hard-coded."
    )

# ---------------------------------------------------------------------------
# Page: Customer Information & Prediction
# ---------------------------------------------------------------------------
elif page == "Customer Information & Prediction":
    st.title("Customer Information")
    st.caption("Enter the customer's details, then click Predict Churn.")

    with st.form("customer_form"):
        st.subheader("Demographics")
        d1, d2, d3, d4 = st.columns(4)
        gender = d1.selectbox("Gender", CATEGORY_OPTIONS["gender"])
        senior = d2.selectbox("Senior Citizen", CATEGORY_OPTIONS["SeniorCitizen"], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = d3.selectbox("Partner", CATEGORY_OPTIONS["Partner"])
        dependents = d4.selectbox("Dependents", CATEGORY_OPTIONS["Dependents"])

        st.subheader("Account Details")
        a1, a2, a3 = st.columns(3)
        tenure = a1.slider("Tenure (months)", 0, 72, 12)
        contract = a2.selectbox("Contract", CATEGORY_OPTIONS["Contract"])
        paperless = a3.selectbox("Paperless Billing", CATEGORY_OPTIONS["PaperlessBilling"])

        a4, a5 = st.columns(2)
        payment_method = a4.selectbox("Payment Method", CATEGORY_OPTIONS["PaymentMethod"])
        monthly_charges = a5.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, max_value=10000.0,
            value=float(round(monthly_charges * max(tenure, 1), 2)), step=1.0,
        )

        st.subheader("Services")
        s1, s2, s3, s4 = st.columns(4)
        phone_service = s1.selectbox("Phone Service", CATEGORY_OPTIONS["PhoneService"])
        multiple_lines = s2.selectbox("Multiple Lines", CATEGORY_OPTIONS["MultipleLines"])
        internet_service = s3.selectbox("Internet Service", CATEGORY_OPTIONS["InternetService"])
        online_security = s4.selectbox("Online Security", CATEGORY_OPTIONS["OnlineSecurity"])

        s5, s6, s7, s8 = st.columns(4)
        online_backup = s5.selectbox("Online Backup", CATEGORY_OPTIONS["OnlineBackup"])
        device_protection = s6.selectbox("Device Protection", CATEGORY_OPTIONS["DeviceProtection"])
        tech_support = s7.selectbox("Tech Support", CATEGORY_OPTIONS["TechSupport"])
        streaming_tv = s8.selectbox("Streaming TV", CATEGORY_OPTIONS["StreamingTV"])

        streaming_movies = st.selectbox("Streaming Movies", CATEGORY_OPTIONS["StreamingMovies"])

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        X_customer = pd.DataFrame([customer])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        proba = float(pipeline.predict_proba(X_customer)[0][1])
        label = "Likely to Churn" if proba >= 0.5 else "Likely to Stay"
        risk = "High" if proba >= 0.6 else ("Medium" if proba >= 0.3 else "Low")
        risk_color = {"High": "#E63946", "Medium": "#F4A261", "Low": "#2A9D8F"}[risk]

        st.markdown("---")
        st.subheader("Prediction")
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction", label)
        r2.metric("Churn Probability", f"{proba*100:.1f}%")
        r3.markdown(
            f"<div style='padding:0.6rem;border-radius:0.5rem;background-color:{risk_color}22;"
            f"border:1px solid {risk_color};text-align:center;'>"
            f"<b>Risk Level</b><br><span style='font-size:1.4rem;color:{risk_color}'>{risk}</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Why this prediction?")
        with st.spinner("Computing SHAP explanation from the trained model..."):
            background_sample = X_background.sample(min(200, len(X_background)), random_state=1)
            explanation = explain_single_prediction(
                pipeline, X_customer, background_sample, top_n=6
            )

        for _, row in explanation.iterrows():
            arrow = "🔺" if row["direction"] == "increases churn risk" else "🔻"
            clean_name = row["feature"].replace("num__", "").replace("cat__", "").replace("_", " ")
            st.write(f"{arrow} **{clean_name}** — {row['direction']} (SHAP value: {row['shap_value']:.3f})")

        st.caption(
            "These factors and the probability above are computed live by the trained "
            f"{model_summary['final_model']} pipeline and its SHAP explainer — none of it is hard-coded."
        )

# ---------------------------------------------------------------------------
# Page: Model Insights
# ---------------------------------------------------------------------------
elif page == "Model Insights":
    st.title("Model Insights")

    st.markdown("### Model Comparison (measured on the held-out test set)")
    comp_path = ROOT / "models" / "model_comparison.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path, index_col=0)
        st.dataframe(comp_df.style.format("{:.3f}"), use_container_width=True)

    st.markdown("### Selected Final Model")
    st.json(model_summary["final_metrics"])
    st.write(f"**Final model type:** {model_summary['final_model']}")

    st.markdown("### Global Feature Importance (SHAP)")
    with st.spinner("Computing global SHAP importance..."):
        importance = get_global_importance(pipeline, X_background)
    top15 = importance.head(15).copy()
    top15["feature"] = top15["feature"].str.replace("num__", "").str.replace("cat__", "")
    fig = go.Figure(go.Bar(
        x=top15["mean_abs_shap"][::-1], y=top15["feature"][::-1], orientation="h",
        marker_color="#2E86AB",
    ))
    fig.update_layout(height=500, xaxis_title="Mean |SHAP value|")
    st.plotly_chart(fig, use_container_width=True)

    tuning_path = ROOT / "models" / "tuning_results.json"
    if tuning_path.exists():
        with open(tuning_path) as f:
            tuning = json.load(f)
        st.markdown("### Hyperparameter Tuning: Before vs After")
        tcol1, tcol2 = st.columns(2)
        tcol1.write("**Before tuning**")
        tcol1.json(tuning["before_tuning"])
        tcol2.write("**After tuning**")
        tcol2.json(tuning["after_tuning"])
        st.write(f"**Best parameters found:** `{tuning['best_params']}`")

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
else:
    st.title("About This Project")
    st.markdown(
        """
**Customer Churn Prediction Using Machine Learning**

**Team:** Kothuri Goutham (1601-24-737-314), Muggu Yashas (1601-24-737-319)

This application predicts whether a telecom customer is likely to churn, using a model
trained on the real Telco Customer Churn dataset (7,043 customers, 21 original columns).

**Pipeline:** raw dataset → cleaning → EDA → feature engineering → train/test split →
model training (Logistic Regression, Decision Tree, Random Forest, XGBoost) → comparison →
hyperparameter tuning → final model → SHAP explainability → saved pipeline → this Streamlit app.

The app loads the saved pipeline (`models/churn_pipeline.pkl`) and does **not** retrain the
model on every visit. Every prediction, probability and SHAP explanation shown is computed
live from that trained pipeline.
"""
    )
