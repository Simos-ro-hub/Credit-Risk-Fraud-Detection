# AI-DrivenCredit Risk Assessment & Fraud Detection System

**Author:** azmainhaq · [Kaggle](https://www.kaggle.com/azmainhaq) · [Hugging Face](https://huggingface.co/Simosro)  
**Stack:** Python · XGBoost · SHAP · LIME · SMOTE · Optuna · scikit-learn · pandas  
**Deploy:** Streamlit app, FastAPI, and Docker deployment

**Datasets:** [Credit Card Fraud Detection 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) · [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)

---

## What this project does

Two production-grade machine learning models for the banking and fintech domain, with full explainability (XAI) and business analytics — built to demonstrate both technical depth and real-world applicability.

| Model | Task | Dataset | Best AUC-ROC |
|-------|------|---------|-------------|
| Fraud Detection | Binary classification — flag fraudulent transactions | 568,630 European card transactions | **0.99999** |
| Credit Risk Assessment | Binary classification — predict 2-year default probability | 150,000 US borrowers | **0.86400** |

The core differentiator from a standard classifier project is the XAI layer: every model decision can be explained at both the global level (SHAP) and the individual prediction level (LIME), which is a regulatory requirement under GDPR Article 22 and US Fair Lending laws.

---

## Key results

### Fraud detection model

- XGBoost tuned with Optuna (40 trials), trained on 386,668 samples
- AUC-ROC: **0.99999** · F1: **0.99975** · Recall: **1.00000** · Precision: **0.99951**
- Business-optimal threshold: **0.79** (minimises total cost = missed fraud × $12,057 + false alarm × $50)
- Annualised saving from threshold optimisation vs default 0.5: **$3,250**
- Top SHAP feature: **V14** (strongest discriminating PCA component)

### Credit risk model

| Metric | Default threshold (0.50) | Optimal threshold (0.22) | Change |
|--------|--------------------------|--------------------------|--------|
| F1 | 0.2501 | **0.4467** | +78% |
| Recall | 0.1591 | **0.5272** | +231% |
| Precision | 0.5842 | 0.3876 | −34% |
| AUC-ROC | 0.8640 | 0.8640 | — |

- SMOTE oversampling to handle 6.68% class imbalance (scale_pos_weight=13)
- Platt scaling calibration — P(default)=0.30 means a real 30% probability
- Threshold sweep (0.05→0.60) finds optimal F1 at **0.220** — catching 53% of defaults vs 16% before
- Extra defaults caught in test set: **+738** · Estimated credit exposure reduction: **$55.4M/year**
- Top SHAP feature: **TotalPastDue** (engineered sum of all past-due columns — ranked #1 by both SHAP and LIME)

### Risk tier calibration (credit model)

| Tier | Count | Actual Default Rate | Predicted Prob |
|------|-------|---------------------|----------------|
| Very Low (<10%) | 24,595 | 2.43% | 2.32% |
| Low (10–20%) | 2,478 | 12.55% | 13.83% |
| Medium (20–40%) | 1,742 | 28.47% | 29.20% |
| High (40–70%) | 1,183 | 50.80% | 49.32% |

Predicted ≈ Actual across all tiers — the calibrated probabilities are reliable for loan pricing decisions.

---

## Project pipeline

```
Raw Data
├── Credit Card Fraud (568,630 rows, 31 cols, 0 missing)
└── Give Me Some Credit (150,000 rows, 11 cols, 33,655 missing)
        │
        ▼
Phase 1: EDA
├── Class balance · Amount distributions · V-feature correlation heatmap
├── Missing value map · Age-group default rates · Feature-target correlations
        │
        ▼
Phase 2: Preprocessing
├── FRAUD: Drop id · RobustScaler on Amount · Stratified 70/15/15 split
└── CREDIT: Remove age=0/100+ · Cap outliers @99th pct · Clip past-due @10
            Median imputation (MonthlyIncome 19.8%, NumberOfDependents 2.6%)
            Feature engineering: TotalPastDue, IncomePerDependent,
                                  DebtToIncome, CreditUtilHigh
            RobustScaler · Stratified 70/15/15 split
        │
        ▼
Phase 3: Modelling
├── FRAUD: Baseline (LR/RF/XGB) → Optuna 40-trial AUC-ROC maximisation
│          → Final XGBoost (train+val combined)
└── CREDIT: Baseline with class_weight=balanced/scale_pos_weight
            SMOTE on train set → Optuna 40-trial AUC-ROC maximisation
            → Final XGBoost + Platt calibration
        │
        ▼
Phase 4: XAI
├── SHAP TreeExplainer: global beeswarm + bar (2,000 test samples)
│   SHAP waterfall: individual transaction/applicant level
│   SHAP dependence: top 2 features (TotalPastDue, RevolvingUtil)
└── LIME LimeTabularExplainer: fraud transaction, high-risk applicant,
    borderline applicant (near optimal threshold)
    SHAP vs LIME feature ranking comparison table
        │
        ▼
Phase 5: Business Analytics
├── FRAUD: Cost-benefit threshold analysis (FN=$12,057 · FP=$50)
│          Precision/Recall/F1 sweep · Annualised cost saving
└── CREDIT: Optimal threshold search (F1 maximisation 0.05–0.60)
            5-tier risk segmentation · Actual vs predicted default rate
            Annualised credit exposure reduction
        │
        ▼
Phase 6: Summary
└── Model comparison dashboard · SHAP feature rankings · Key findings
```

---

## XAI — what the explanations show

### SHAP (SHapley Additive exPlanations) — global

SHAP uses cooperative game theory (Shapley values) to assign each feature a contribution score for every prediction. For the fraud model, **V14** has the highest mean |SHAP| — a low V14 value strongly pushes a transaction toward Class=1 (fraud). For the credit model, **TotalPastDue** (the engineered sum of all three past-due columns) accounts for more than twice the contribution of any other feature.

The SHAP vs LIME comparison revealed an important divergence: RevolvingUtilizationOfUnsecuredLines ranks #2 globally by SHAP but only #6 by LIME. This is not a bug — SHAP captures average marginal contribution across the full dataset (global), while LIME fits a local linear model around each individual prediction. RevolvingUtil has a non-linear risk profile (both very low and very high are sometimes risky), which SHAP's global average captures but LIME's local linear surrogate underweights.

### LIME (Local Interpretable Model-agnostic Explanations) — local

Three cases :

**High-risk credit applicant #25445** (P=0.601 — correctly predicted default):  
All 12 factors point upward: TotalPastDue (+0.181), 90DaysLate (+0.085), RevolvingUtil >0.76 (+0.074). Every factor is business-legible without touching the code.

**Borderline applicant #28437** (P=0.220 — right at threshold, false positive — good customer denied):  
TotalPastDue drives risk (+0.176) despite 90DaysLate=0 (−0.102) and no high credit utilisation (−0.049). Essential for automated lending decisions.

---

## Feature engineering

Four new features were created for the credit model, all derived from existing columns:

| Feature | Formula | Business meaning |
|---------|---------|-----------------|
| `TotalPastDue` | 30-59DaysPastDue + 90DaysLate + 60-89DaysPastDue | Aggregate delinquency count — ranked #1 by both SHAP and LIME |
| `IncomePerDependent` | MonthlyIncome / (Dependents + 1) | Disposable income proxy |
| `DebtToIncome` | DebtRatio × MonthlyIncome / (MonthlyIncome + 1) | Normalised absolute debt burden |
| `CreditUtilHigh` | RevolvingUtil > 0.75 → binary flag | Non-linear utilisation threshold |

---

## Tech stack

| Layer | Tool |
|-------|------|
| Data manipulation | pandas · numpy |
| Visualisation | matplotlib · seaborn |
| ML models | scikit-learn · XGBoost 3.2 |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Hyperparameter tuning | Optuna (TPE sampler) |
| Calibration | sklearn CalibratedClassifierCV (Platt scaling) |
| Global XAI | SHAP 0.51 (TreeExplainer) |
| Local XAI | LIME (LimeTabularExplainer) |
| Compute | Kaggle T4 GPU |

---

1. **Hugging Face Spaces deployment** — Streamlit app: enter transaction features → live fraud probability + SHAP waterfall
2. **Power BI dashboard** — fraud cost curve + credit risk tier visualisation

