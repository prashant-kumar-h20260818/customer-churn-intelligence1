from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLUMNS = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

REQUIRED_INPUT_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

NUMERIC_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "TotalServices", "AvgMonthlySpend", "NewCustomer",
    "HighMonthlyCharges", "AutomaticPayment",
]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "TenureGroup",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Business feature engineering kept inside the sklearn pipeline."""

    def fit(self, X: pd.DataFrame, y=None):
        frame = X.copy()
        monthly = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
        self.monthly_charge_median_ = float(monthly.median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        if "customerID" in frame.columns:
            frame = frame.drop(columns=["customerID"])

        for col in ["TotalCharges", "tenure", "MonthlyCharges", "SeniorCitizen"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        service_flags = [frame[c].eq("Yes").astype(int) for c in SERVICE_COLUMNS]
        frame["TotalServices"] = np.sum(service_flags, axis=0)

        tenure_safe = frame["tenure"].replace(0, 1)
        frame["AvgMonthlySpend"] = frame["TotalCharges"] / tenure_safe
        frame["NewCustomer"] = (frame["tenure"] <= 6).astype(int)
        frame["HighMonthlyCharges"] = (
            frame["MonthlyCharges"] > self.monthly_charge_median_
        ).astype(int)
        frame["AutomaticPayment"] = frame["PaymentMethod"].str.contains(
            "automatic", case=False, na=False
        ).astype(int)
        frame["TenureGroup"] = pd.cut(
            frame["tenure"],
            bins=[-0.1, 6, 12, 24, 48, np.inf],
            labels=["0-6", "7-12", "13-24", "25-48", "49+"],
        ).astype(str)

        return frame
