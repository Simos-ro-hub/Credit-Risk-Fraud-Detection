"""
Shared model utilities — preprocessing, feature definitions, inference.
Used by both the Streamlit app and the FastAPI service.
"""
import numpy as np
import joblib
import os

# ── Feature definitions ───────────────────────────────────────────────────────

FRAUD_FEATURES = [
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

CREDIT_FEATURES_RAW = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

CREDIT_FEATURES_ENGINEERED = [
    "TotalPastDue",
    "IncomePerDependent",
    "DebtToIncome",
    "CreditUtilHigh",
]

CREDIT_FEATURES_ALL = CREDIT_FEATURES_RAW + CREDIT_FEATURES_ENGINEERED

# Optimal thresholds from notebook (Fix 1)
FRAUD_THRESHOLD  = 0.79   # minimum business cost threshold
CREDIT_THRESHOLD = 0.22   # F1-optimal threshold

# Risk tier labels for credit model
CREDIT_TIERS = [
    (0.10, "Very Low",  "#1D9E75", "✅"),
    (0.20, "Low",       "#5DCAA5", "🟢"),
    (0.40, "Medium",    "#EF9F27", "🟡"),
    (0.70, "High",      "#E24B4A", "🔴"),
    (1.01, "Very High", "#993C1D", "🚨"),
]

# ── 99th-percentile caps from training data ───────────────────────────────────
# These were computed in Cell 7 on the full training set.
# Hard-coded here so the app works without the original CSV.
CREDIT_CAPS = {
    "RevolvingUtilizationOfUnsecuredLines": 1.0,
    "DebtRatio": 1.5,
    "MonthlyIncome": 25000.0,
    "NumberOfOpenCreditLinesAndLoans": 25.0,
    "NumberRealEstateLoansOrLines": 5.0,
}
CREDIT_PASTDUE_CAP = 10

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_fraud(raw: dict, scaler) -> np.ndarray:
    """
    raw: dict with keys V1..V28 + Amount (all raw/unscaled).
    Returns shape (1, 29) numpy array ready for model.predict_proba.
    """
    row = np.array([[raw[f] for f in FRAUD_FEATURES]], dtype=np.float64)
    # Scale Amount only (V-features already PCA-transformed)
    amount_idx = FRAUD_FEATURES.index("Amount")
    row[0, amount_idx] = scaler.transform([[row[0, amount_idx]]])[0][0]
    return row


def preprocess_credit(raw: dict, scaler) -> np.ndarray:
    """
    raw: dict with keys matching CREDIT_FEATURES_RAW.
    Applies same cleaning as Cell 7: outlier cap, past-due cap, feature engineering.
    Returns shape (1, 14) scaled numpy array.
    """
    r = {k: float(v) for k, v in raw.items()}

    # Outlier caps
    for col, cap in CREDIT_CAPS.items():
        r[col] = min(r[col], cap)

    # Past-due sentinel cap
    for col in ["NumberOfTime30-59DaysPastDueNotWorse",
                "NumberOfTimes90DaysLate",
                "NumberOfTime60-89DaysPastDueNotWorse"]:
        r[col] = min(r[col], CREDIT_PASTDUE_CAP)

    # Feature engineering (identical to Cell 7)
    r["TotalPastDue"] = (r["NumberOfTime30-59DaysPastDueNotWorse"] +
                         r["NumberOfTimes90DaysLate"] +
                         r["NumberOfTime60-89DaysPastDueNotWorse"])
    r["IncomePerDependent"] = r["MonthlyIncome"] / (r["NumberOfDependents"] + 1)
    r["DebtToIncome"] = (r["DebtRatio"] * r["MonthlyIncome"] /
                         (r["MonthlyIncome"] + 1))
    r["CreditUtilHigh"] = 1 if r["RevolvingUtilizationOfUnsecuredLines"] > 0.75 else 0

    row = np.array([[r[f] for f in CREDIT_FEATURES_ALL]], dtype=np.float64)
    return scaler.transform(row)


def get_risk_tier(prob: float) -> tuple:
    """Returns (label, color_hex, emoji) for a given default probability."""
    for threshold, label, color, emoji in CREDIT_TIERS:
        if prob < threshold:
            return label, color, emoji
    return "Very High", "#993C1D", "🚨"


def fraud_risk_label(prob: float) -> tuple:
    """Returns (label, color, emoji) for a fraud probability."""
    if prob < 0.30:
        return "Low Risk",    "#1D9E75", "✅"
    elif prob < 0.60:
        return "Medium Risk", "#EF9F27", "⚠️"
    elif prob < FRAUD_THRESHOLD:
        return "High Risk",   "#E24B4A", "🔴"
    else:
        return "FRAUD",       "#993C1D", "🚨"
