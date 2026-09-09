from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

from churn_features import REQUIRED_INPUT_COLUMNS
from train_deploy import METRICS_PATH, MODEL_PATH, load_data, train_and_evaluate

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📉",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


@st.cache_resource(show_spinner="Loading validated churn model...")
def get_model_and_metrics():
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        model = joblib.load(MODEL_PATH)
        metrics = json.loads(METRICS_PATH.read_text())
        return model, metrics

    model, metrics = train_and_evaluate(get_data())
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return model, metrics


def money(value):
    return f"${value:,.0f}"


def risk_tier(probability):
    if probability < 0.30:
        return "Low"
    if probability < 0.50:
        return "Medium"
    if probability < 0.70:
        return "High"
    return "Critical"


def retention_action(tier):
    return {
        "Critical": "Immediate retention outreach with service review and a targeted offer.",
        "High": "Prioritize for a targeted retention campaign and proactive service check.",
        "Medium": "Monitor engagement and consider preventive communication.",
        "Low": "No immediate intervention required; continue normal engagement.",
    }[tier]


def clean_feature_name(name):
    return name.replace("numeric__", "").replace("categorical__", "").replace("_", " ")


def score_customers(model, frame, threshold):
    scored = frame.copy()
    probability = model.predict_proba(frame)[:, 1]
    scored["ChurnProbability"] = probability
    scored["PredictedChurn"] = np.where(probability >= threshold, "Yes", "No")
    scored["RiskTier"] = [risk_tier(p) for p in probability]
    monthly = pd.to_numeric(scored["MonthlyCharges"], errors="coerce").fillna(0)
    scored["AnnualCustomerValue"] = monthly * 12
    scored["RevenueAtRisk"] = probability * scored["AnnualCustomerValue"]
    scored["RetentionPriority"] = scored["RevenueAtRisk"]
    return scored.sort_values("RetentionPriority", ascending=False)


def churn_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Predicted Churn Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30]},
                {"range": [30, 50]},
                {"range": [50, 70]},
                {"range": [70, 100]},
            ],
            "threshold": {"line": {"width": 4}, "thickness": 0.8, "value": probability * 100},
        },
    ))
    fig.update_layout(height=350)
    return fig


def customer_form():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Customer")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    with c2:
        st.markdown("#### Services")
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with c3:
        st.markdown("#### Contract & Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 75.0, 1.0)
        total = st.number_input("Total Charges", 0.0, 20000.0, float(monthly * max(tenure, 1)), 10.0)

    return pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])


def explain_customer(model, customer, background):
    features = model.named_steps["feature_engineering"]
    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    engineered = features.transform(customer)
    transformed = preprocessor.transform(engineered)
    names = preprocessor.get_feature_names_out()

    try:
        bg = background.sample(min(100, len(background)), random_state=42)
        bg_transformed = preprocessor.transform(features.transform(bg))
        explainer = shap.LinearExplainer(classifier, bg_transformed)
        values = np.asarray(explainer.shap_values(transformed)).reshape(-1)
    except Exception:
        if hasattr(classifier, "coef_"):
            values = np.asarray(classifier.coef_[0]) * transformed[0]
        else:
            values = np.asarray(classifier.feature_importances_) * transformed[0]

    explanation = pd.DataFrame({
        "Feature": [clean_feature_name(x) for x in names],
        "Contribution": values,
        "Strength": np.abs(values),
    })
    explanation["Effect"] = np.where(
        explanation["Contribution"] >= 0, "Increases churn risk", "Reduces churn risk"
    )
    return explanation.sort_values("Strength", ascending=False).head(8)


data = get_data()
model, metrics = get_model_and_metrics()
threshold = float(metrics["threshold"])

st.title("📉 Customer Churn Intelligence & Retention Prioritization")
st.caption(
    "Identify customers most likely to leave, explain churn risk, and prioritize retention actions."
)

dashboard_tab, predictor_tab, batch_tab, insights_tab = st.tabs([
    "📊 Executive Dashboard", "🎯 Customer Predictor", "📁 Batch Scoring", "🧠 Model Insights"
])

with dashboard_tab:
    scored = score_customers(model, data.drop(columns=["Churn"]), threshold)
    historical = data["Churn"].eq("Yes").mean()
    high_critical = scored["RiskTier"].isin(["High", "Critical"]).sum()
    revenue = scored["RevenueAtRisk"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(data):,}")
    c2.metric("Historical Churn", f"{historical:.1%}")
    c3.metric("High/Critical Risk", f"{high_critical:,}")
    c4.metric("Modeled Revenue at Risk", money(revenue))
    st.info("Revenue at Risk = churn probability × monthly charges × 12; it is a prioritization estimate, not realized loss.")

    left, right = st.columns(2)
    with left:
        contract = data.assign(ChurnFlag=data["Churn"].eq("Yes").astype(int)).groupby("Contract", as_index=False)["ChurnFlag"].mean()
        fig = px.bar(contract, x="Contract", y="ChurnFlag", title="Churn Rate by Contract")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        internet = data.assign(ChurnFlag=data["Churn"].eq("Yes").astype(int)).groupby("InternetService", as_index=False)["ChurnFlag"].mean()
        fig = px.bar(internet, x="InternetService", y="ChurnFlag", title="Churn Rate by Internet Service")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        temp = data.copy()
        temp["TenureGroup"] = pd.cut(temp["tenure"], [-0.1, 6, 12, 24, 48, 100], labels=["0-6", "7-12", "13-24", "25-48", "49+"])
        tenure = temp.assign(ChurnFlag=temp["Churn"].eq("Yes").astype(int)).groupby("TenureGroup", observed=False, as_index=False)["ChurnFlag"].mean()
        fig = px.line(tenure, x="TenureGroup", y="ChurnFlag", markers=True, title="Churn Rate by Tenure")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        payment = data.assign(ChurnFlag=data["Churn"].eq("Yes").astype(int)).groupby("PaymentMethod", as_index=False)["ChurnFlag"].mean().sort_values("ChurnFlag", ascending=False)
        fig = px.bar(payment, x="ChurnFlag", y="PaymentMethod", orientation="h", title="Churn Rate by Payment Method")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

with predictor_tab:
    st.header("Individual Customer Churn Prediction")
    customer = customer_form()
    if st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True):
        probability = float(model.predict_proba(customer)[0, 1])
        tier = risk_tier(probability)
        annual_value = float(customer.loc[0, "MonthlyCharges"] * 12)
        revenue = probability * annual_value

        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(churn_gauge(probability), use_container_width=True)
        with right:
            st.metric("Prediction", "Likely to Churn" if probability >= threshold else "Likely to Stay")
            st.metric("Risk Tier", tier)
            st.metric("Annual Customer Value", money(annual_value))
            st.metric("Revenue at Risk", money(revenue))

        st.subheader("Recommended Retention Action")
        action = retention_action(tier)
        if tier == "Critical":
            st.error(action)
        elif tier == "High":
            st.warning(action)
        elif tier == "Medium":
            st.info(action)
        else:
            st.success(action)

        st.subheader("Why did the model assign this risk?")
        explanation = explain_customer(model, customer, data.drop(columns=["Churn"]))
        st.dataframe(explanation[["Feature", "Contribution", "Effect"]], use_container_width=True, hide_index=True)

with batch_tab:
    st.header("Batch Customer Scoring")
    template = pd.DataFrame(columns=REQUIRED_INPUT_COLUMNS)
    st.download_button(
        "⬇️ Download CSV Template",
        template.to_csv(index=False).encode(),
        "churn_scoring_template.csv",
        "text/csv",
    )
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in batch.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            scored = score_customers(model, batch, threshold)
            c1, c2, c3 = st.columns(3)
            c1.metric("Customers Scored", f"{len(scored):,}")
            c2.metric("High/Critical Risk", f"{scored['RiskTier'].isin(['High', 'Critical']).sum():,}")
            c3.metric("Revenue at Risk", money(scored["RevenueAtRisk"].sum()))
            st.dataframe(scored, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download Prioritized Retention List",
                scored.to_csv(index=False).encode(),
                "retention_priority_list.csv",
                "text/csv",
                type="primary",
            )

with insights_tab:
    st.header("Model Validation & Insights")
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Model", metrics["best_model"])
    c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c3.metric("Churn Recall", f"{metrics['recall']:.1%}")
    c1, c2, c3 = st.columns(3)
    c1.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    c2.metric("F1", f"{metrics['f1']:.3f}")
    c3.metric("Decision Threshold", f"{metrics['threshold']:.2f}")

    comparison = pd.DataFrame(metrics.get("comparison", []))
    if not comparison.empty:
        st.subheader("5-Fold Cross-Validation Comparison")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessing"]
    names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "coef_"):
        importance_values = np.abs(classifier.coef_[0])
    else:
        importance_values = classifier.feature_importances_
    importance = pd.DataFrame({
        "Feature": [clean_feature_name(x) for x in names],
        "Importance": importance_values,
    }).sort_values("Importance", ascending=False).head(15).sort_values("Importance")
    fig = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Top Global Model Features")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Business interpretation:** the operating threshold is optimized using F2 to emphasize churn recall. "
        "This helps surface more customers who may churn, accepting more false positives in exchange for fewer missed churners."
    )
