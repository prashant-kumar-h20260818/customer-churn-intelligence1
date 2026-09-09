from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ChurnFeatureEngineer

DATA_URL = (
    "https://raw.githubusercontent.com/SaeidRostami/Customer_Churn/"
    "refs/heads/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
DATA_PATH = Path("telco_customer_churn.csv")
MODEL_PATH = Path("churn_pipeline.joblib")
METRICS_PATH = Path("model_metrics.json")
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    frame = pd.read_csv(DATA_URL)
    frame.to_csv(DATA_PATH, index=False)
    return frame


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def candidate_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("feature_engineering", ChurnFeatureEngineer()),
        ("preprocessing", build_preprocessor()),
        ("classifier", estimator),
    ])


def choose_threshold(y_true, probability) -> float:
    thresholds = np.arange(0.20, 0.81, 0.01)
    scores = [
        fbeta_score(y_true, probability >= t, beta=2, zero_division=0)
        for t in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def train_and_evaluate(frame: pd.DataFrame):
    y = frame["Churn"].map({"No": 0, "Yes": 1})
    X = frame.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    comparison = []
    pipelines = {}
    for name, estimator in candidate_models().items():
        pipe = build_pipeline(estimator)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        comparison.append({
            "model": name,
            "cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std()),
        })
        pipelines[name] = pipe

    comparison = sorted(comparison, key=lambda x: x["cv_roc_auc_mean"], reverse=True)
    best_name = comparison[0]["model"]
    selected = pipelines[best_name]

    oof_probability = cross_val_predict(
        clone(selected), X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    threshold = choose_threshold(y_train, oof_probability)

    selected.fit(X_train, y_train)
    probability = selected.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)

    metrics = {
        "best_model": best_name,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "f2": float(fbeta_score(y_test, prediction, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "pr_auc": float(average_precision_score(y_test, probability)),
        "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
        "total_customers": int(len(frame)),
        "train_customers": int(len(X_train)),
        "test_customers": int(len(X_test)),
        "historical_churn_rate": float(y.mean()),
        "comparison": comparison,
    }
    return selected, metrics


def build_assets():
    frame = load_data()
    model, metrics = train_and_evaluate(frame)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    build_assets()
