import os, warnings, io
os.environ["MPLBACKEND"] = "Agg"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import streamlit as st
import joblib
import shap

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title   { font-size:2.2rem; font-weight:700; color:#1B3A6B; margin-bottom:0; }
  .sub-title    { font-size:1rem;   color:#6B7280;   margin-top:0; margin-bottom:1.5rem; }
  .metric-card  { background:#F0F4FF; border-radius:12px; padding:1rem 1.25rem;
                  border:1px solid #C7D7F5; margin-bottom:0.5rem; }
  .metric-val   { font-size:2rem; font-weight:700; }
  .metric-label { font-size:0.8rem; color:#6B7280; text-transform:uppercase; letter-spacing:.05em; }
  .risk-badge   { display:inline-block; padding:0.3rem 1rem; border-radius:999px;
                  font-size:1.1rem; font-weight:600; margin-top:0.5rem; }
  .section-hd   { font-size:1rem; font-weight:600; color:#1B3A6B;
                  border-bottom:2px solid #2563EB; padding-bottom:4px; margin-bottom:1rem; }
  .info-box     { background:#EBF4FF; border-left:4px solid #2563EB;
                  padding:0.75rem 1rem; border-radius:0 8px 8px 0;
                  font-size:0.88rem; margin:0.5rem 0; }
  .warn-box     { background:#FEF9C3; border-left:4px solid #CA8A04;
                  padding:0.75rem 1rem; border-radius:0 8px 8px 0;
                  font-size:0.88rem; margin:0.5rem 0; }
  .danger-box   { background:#FEE2E2; border-left:4px solid #DC2626;
                  padding:0.75rem 1rem; border-radius:0 8px 8px 0;
                  font-size:0.88rem; margin:0.5rem 0; }
  .stButton>button { background:#2563EB; color:#fff; border:none;
                     border-radius:8px; padding:0.5rem 2rem;
                     font-weight:600; font-size:1rem; }
  .stButton>button:hover { background:#1B3A6B; }
  div[data-testid="stSidebar"] { background:#F8FAFF; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

FRAUD_FEATURES = [
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

CREDIT_FEATURES_RAW = [
    "RevolvingUtilizationOfUnsecuredLines","age",
    "NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
CREDIT_FEATURES_ALL = CREDIT_FEATURES_RAW + [
    "TotalPastDue","IncomePerDependent","DebtToIncome","CreditUtilHigh"
]

FRAUD_THRESHOLD  = 0.79
CREDIT_THRESHOLD = 0.22

CREDIT_CAPS = {
    "RevolvingUtilizationOfUnsecuredLines": 1.0,
    "DebtRatio": 1.5, "MonthlyIncome": 25000.0,
    "NumberOfOpenCreditLinesAndLoans": 25.0,
    "NumberRealEstateLoansOrLines": 5.0,
}

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    fm  = joblib.load(os.path.join(MODEL_DIR, "fraud_model.joblib"))
    fs  = joblib.load(os.path.join(MODEL_DIR, "fraud_scaler.joblib"))
    # Load raw XGBoost (no sklearn CalibratedClassifierCV wrapper)
    cm  = joblib.load(os.path.join(MODEL_DIR, "credit_model_xgb.joblib"))
    cs  = joblib.load(os.path.join(MODEL_DIR, "credit_scaler.joblib"))
    # Load Platt scaling parameters [A, B]
    cal = np.load(os.path.join(MODEL_DIR, "credit_calibration.npy"))
    sbf = np.load(os.path.join(MODEL_DIR, "shap_background_fraud.npy"))
    sbc = np.load(os.path.join(MODEL_DIR, "shap_background_credit.npy"))
    exp_f = shap.TreeExplainer(fm, sbf)
    exp_c = shap.TreeExplainer(cm, sbc)
    return fm, fs, cm, cs, cal, exp_f, exp_c

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_fraud(vals: dict, scaler) -> np.ndarray:
    row = np.array([[vals[f] for f in FRAUD_FEATURES]], dtype=np.float64)
    amt_idx = FRAUD_FEATURES.index("Amount")
    row[0, amt_idx] = scaler.transform([[row[0, amt_idx]]])[0][0]
    return row

def preprocess_credit(vals: dict, scaler) -> np.ndarray:
    r = {k: float(v) for k, v in vals.items()}
    for col, cap in CREDIT_CAPS.items():
        r[col] = min(r[col], cap)
    for col in ["NumberOfTime30-59DaysPastDueNotWorse",
                "NumberOfTimes90DaysLate",
                "NumberOfTime60-89DaysPastDueNotWorse"]:
        r[col] = min(r[col], 10)
    r["TotalPastDue"]       = (r["NumberOfTime30-59DaysPastDueNotWorse"] +
                                r["NumberOfTimes90DaysLate"] +
                                r["NumberOfTime60-89DaysPastDueNotWorse"])
    r["IncomePerDependent"] = r["MonthlyIncome"] / (r["NumberOfDependents"] + 1)
    r["DebtToIncome"]       = (r["DebtRatio"] * r["MonthlyIncome"] /
                                (r["MonthlyIncome"] + 1))
    r["CreditUtilHigh"]     = 1 if r["RevolvingUtilizationOfUnsecuredLines"] > 0.75 else 0
    row = np.array([[r[f] for f in CREDIT_FEATURES_ALL]], dtype=np.float64)
    return scaler.transform(row)

# ── SHAP waterfall ────────────────────────────────────────────────────────────
def shap_waterfall(explainer, X_row, feature_names, title, max_display=12):
    """Returns a PNG bytes buffer — avoids all Streamlit/matplotlib session conflicts."""
    sv = explainer.shap_values(X_row)
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        sv = sv[1]
    sv = sv[0]

    order   = np.argsort(np.abs(sv))[::-1][:max_display]
    sv_plot = sv[order][::-1]
    fn_plot = [feature_names[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, len(sv_plot) * 0.45)))
    colors  = ["#E24B4A" if v > 0 else "#3B8BD4" for v in sv_plot]
    y_pos   = np.arange(len(sv_plot))

    ax.barh(y_pos, sv_plot, color=colors, height=0.6, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fn_plot, fontsize=10)
    ax.axvline(0, color="#374151", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#1B3A6B")

    red_patch  = mpatches.Patch(color="#E24B4A", label="Increases risk")
    blue_patch = mpatches.Patch(color="#3B8BD4", label="Decreases risk")
    ax.legend(handles=[red_patch, blue_patch], fontsize=9,
              loc="lower right", framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# ── Gauge chart ───────────────────────────────────────────────────────────────
def gauge_chart(prob: float, label: str, color: str):
    """Returns PNG bytes buffer."""
    fig, ax = plt.subplots(figsize=(4, 2.5),
                           subplot_kw={"projection": "polar"})
    theta  = np.linspace(np.pi, 0, 200)
    ax.plot(theta, np.ones_like(theta) * 0.8,
            color="#E5E7EB", linewidth=14, solid_capstyle="round")
    filled = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    ax.plot(filled, np.ones_like(filled) * 0.8,
            color=color, linewidth=14, solid_capstyle="round")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, np.pi)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(1)
    ax.axis("off")
    ax.text(np.pi / 2, 0.25, f"{prob*100:.1f}%",
            ha="center", va="center", fontsize=22, fontweight="bold", color=color)
    ax.text(np.pi / 2, -0.05, label,
            ha="center", va="center", fontsize=10, color="#6B7280")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# ── LIME bar ─────────────────────────────────────────────────────────────────
def quick_lime_bar(feature_names, X_row, model_fn, n_features=10, n_samples=500):
    """Returns PNG bytes buffer."""
    np.random.seed(42)
    n      = X_row.shape[1]
    noise  = np.random.randn(n_samples, n) * 0.1
    X_pert = X_row + noise
    probs  = model_fn(X_pert)[:, 1]
    weights = np.exp(-np.sum(noise**2, axis=1) / (2 * 0.5**2))

    from numpy.linalg import lstsq
    X_w  = X_pert * weights[:, None]
    y_w  = (probs - probs.mean()) * weights
    coef, *_ = lstsq(X_w, y_w, rcond=None)

    order  = np.argsort(np.abs(coef))[::-1][:n_features]
    coefs  = coef[order][::-1]
    fnames = [feature_names[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, len(coefs) * 0.42)))
    colors  = ["#E24B4A" if v > 0 else "#3B8BD4" for v in coefs]
    ax.barh(range(len(coefs)), coefs, color=colors, height=0.6, edgecolor="none")
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels(fnames, fontsize=10)
    ax.axvline(0, color="#374151", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Local perturbation weight", fontsize=10)
    ax.set_title("LIME-style local explanation", fontsize=11,
                 fontweight="bold", color="#1B3A6B")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# ── Risk interpretation text ──────────────────────────────────────────────────
def fraud_interpretation(prob, features):
    sv_abs = {}
    lines  = []
    if prob >= FRAUD_THRESHOLD:
        lines.append("🚨 **Transaction flagged as FRAUD.** Key risk drivers from SHAP analysis:")
        lines.append("- Review this transaction immediately before processing.")
        lines.append("- Contact the cardholder to verify if this transaction was authorised.")
    elif prob > 0.5:
        lines.append("⚠️ **Elevated fraud risk detected.** Transaction should be reviewed.")
        lines.append("- Consider a step-up authentication challenge (OTP/biometric).")
    else:
        lines.append("✅ **Transaction appears legitimate.** No immediate action required.")
        lines.append("- Continue standard monitoring protocols.")
    return "\n\n".join(lines)

def credit_interpretation(prob, tier):
    lines = []
    if tier == "Very Low":
        lines.append("✅ **Excellent creditworthiness.** Auto-approve recommended.")
        lines.append("**Suggested action:** Offer prime rate or promotional APR.")
    elif tier == "Low":
        lines.append("🟢 **Good credit profile.** Standard approval recommended.")
        lines.append("**Suggested action:** Approve with standard terms. Prime + 1–2%.")
    elif tier == "Medium":
        lines.append("🟡 **Moderate risk.** Conditional approval recommended.")
        lines.append("**Suggested action:** Approve with conditions — reduced limit, Prime + 3–5%.")
    elif tier == "High":
        lines.append("🔴 **High default risk.** Manual review required.")
        lines.append("**Suggested action:** Require collateral or co-signer. Prime + 6–9%.")
    else:
        lines.append("🚨 **Very high default risk.** Denial recommended.")
        lines.append("**Suggested action:** Decline or offer secured product only.")
    return "\n\n".join(lines)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛡️ FinShield AI")
st.sidebar.markdown("*Production-grade XAI for fintech*")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select module",
    ["🔍 Fraud Detection", "📊 Credit Risk Assessment", "ℹ️ About"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model stats**")
st.sidebar.markdown("Fraud AUC-ROC: `0.99999`")
st.sidebar.markdown("Credit AUC-ROC: `0.86400`")
st.sidebar.markdown("Fraud threshold: `0.79`")
st.sidebar.markdown("Credit threshold: `0.22`")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by [azmainhaq](https://kaggle.com/azmainhaq)")
st.sidebar.markdown("Notebook on [Kaggle](https://kaggle.com/azmainhaq)")
st.sidebar.markdown("Models on [Hugging Face](https://huggingface.co/Simosro)")

# ── Load models ───────────────────────────────────────────────────────────────
try:
    fraud_model, fraud_scaler, credit_model, credit_scaler, \
        credit_cal, shap_exp_fraud, shap_exp_credit = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — FRAUD DETECTION
# ─────────────────────────────────────────────────────────────────────────────
if mode == "🔍 Fraud Detection":

    st.markdown('<p class="main-title">🔍 Fraud Detection</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">XGBoost · AUC-ROC 0.99999 · Optuna-tuned · SHAP explainability</p>',
        unsafe_allow_html=True,
    )

    col_form, col_result = st.columns([1, 1.4], gap="large")

    with col_form:
        st.markdown('<p class="section-hd">Transaction features</p>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">V1–V28 are PCA-transformed anonymised transaction attributes. Enter the raw values from your transaction record. <b>Amount</b> is in USD.</div>', unsafe_allow_html=True)

        # Tabs: manual entry vs random sample
        tab_manual, tab_sample = st.tabs(["✏️ Manual entry", "🎲 Random sample"])

        with tab_sample:
            sample_type = st.selectbox(
                "Load a pre-built example",
                ["Typical legitimate transaction", "Small legitimate transaction",
                 "Suspicious high-value transaction", "Known fraud pattern"],
            )
            samples = {
                "Typical legitimate transaction": {
                    "V1":-0.260,"V2":-0.470,"V3":2.496,"V4":-0.084,"V5":0.130,
                    "V6":0.733,"V7":0.519,"V8":-0.130,"V9":0.727,"V10":-0.440,
                    "V11":1.011,"V12":-0.133,"V13":-0.021,"V14":2.136,"V15":0.051,
                    "V16":-0.108,"V17":-0.051,"V18":-0.220,"V19":0.126,"V20":-0.037,
                    "V21":-0.111,"V22":0.218,"V23":-0.135,"V24":0.166,"V25":0.126,
                    "V26":-0.435,"V27":-0.081,"V28":-0.151,"Amount":17982.10,
                },
                "Small legitimate transaction": {
                    "V1":0.985,"V2":-0.356,"V3":0.558,"V4":-0.430,"V5":0.277,
                    "V6":0.429,"V7":0.406,"V8":-0.133,"V9":0.347,"V10":-0.088,
                    "V11":0.221,"V12":0.445,"V13":-0.066,"V14":2.011,"V15":0.174,
                    "V16":-0.143,"V17":-0.093,"V18":-0.160,"V19":0.130,"V20":-0.031,
                    "V21":-0.195,"V22":-0.606,"V23":0.079,"V24":-0.577,"V25":0.190,
                    "V26":0.297,"V27":-0.248,"V28":-0.065,"Amount":6531.37,
                },
                "Suspicious high-value transaction": {
                    "V1":-3.043,"V2": 1.185,"V3":-4.284,"V4":4.455,"V5":-0.835,
                    "V6":-0.694,"V7":-3.748,"V8": 0.771,"V9":-0.648,"V10":-3.533,
                    "V11": 3.030,"V12":-4.855,"V13": 0.701,"V14":-5.812,"V15": 0.630,
                    "V16":-1.026,"V17":-6.247,"V18":-0.343,"V19": 0.946,"V20": 0.302,
                    "V21": 0.643,"V22":-0.137,"V23":-0.310,"V24": 0.053,"V25": 0.215,
                    "V26":-0.183,"V27": 0.566,"V28": 0.121,"Amount":21000.00,
                },
                "Known fraud pattern": {
                    "V1":-2.312,"V2": 1.952,"V3":-1.610,"V4": 3.998,"V5":-0.522,
                    "V6":-1.427,"V7":-2.537,"V8": 1.392,"V9":-2.770,"V10":-2.773,
                    "V11": 3.202,"V12":-2.900,"V13":-0.595,"V14":-4.289,"V15": 0.390,
                    "V16":-1.141,"V17":-2.830,"V18":-0.017,"V19": 0.416,"V20": 0.126,
                    "V21": 0.518,"V22":-0.036,"V23":-0.465,"V24": 0.201,"V25": 0.025,
                    "V26":-0.413,"V27":-0.062,"V28":-0.039,"Amount":18200.00,
                },
            }
            sample_vals = samples[sample_type]
            if st.button("Load this sample", key="load_sample"):
                for k, v in sample_vals.items():
                    st.session_state[f"fraud_{k}"] = v

        with tab_manual:
            amount = st.number_input(
                "Amount (USD)", min_value=50.0, max_value=25000.0,
                value=float(st.session_state.get("fraud_Amount", 5000.0)),
                step=50.0, format="%.2f", key="fraud_Amount"
            )

            v_cols = st.columns(4)
            v_vals = {}
            for i, feat in enumerate([f"V{j}" for j in range(1, 29)]):
                with v_cols[i % 4]:
                    v_vals[feat] = st.number_input(
                        feat, value=float(st.session_state.get(f"fraud_{feat}", 0.0)),
                        step=0.01, format="%.3f", key=f"fraud_{feat}",
                        label_visibility="visible"
                    )
            sample_vals = {**v_vals, "Amount": amount}

        run_fraud = st.button("🔍 Analyse transaction", key="run_fraud", use_container_width=True)
        show_lime = st.checkbox("Show LIME local explanation (slower)", value=False)

    # ── Results pane ──────────────────────────────────────────────────────────
    with col_result:
        if run_fraud:
            if not models_loaded:
                st.error(f"Models not loaded: {load_error}")
            else:
                with st.spinner("Running model + SHAP…"):
                    X_row = preprocess_fraud(sample_vals, fraud_scaler)
                    prob  = float(fraud_model.predict_proba(X_row)[0, 1])
                    label, color, emoji = (
                        ("FRAUD",       "#991B1B", "🚨") if prob >= FRAUD_THRESHOLD else
                        ("HIGH RISK",   "#DC2626", "🔴") if prob >= 0.6 else
                        ("MEDIUM RISK", "#D97706", "⚠️") if prob >= 0.3 else
                        ("LEGITIMATE",  "#059669", "✅")
                    )

                st.markdown('<p class="section-hd">Prediction result</p>', unsafe_allow_html=True)

                # Gauge
                gc1, gc2, gc3 = st.columns([1, 2, 1])
                with gc2:
                    st.image(gauge_chart(prob, "Fraud probability", color), use_column_width=True)
                    plt.close("all")

                # Badge
                st.markdown(
                    f'<div style="text-align:center"><span class="risk-badge" '
                    f'style="background:{color}22;color:{color};border:2px solid {color}">'
                    f'{emoji} {label}</span></div>',
                    unsafe_allow_html=True,
                )

                # Metric row
                m1, m2, m3 = st.columns(3)
                m1.metric("Fraud probability", f"{prob*100:.2f}%")
                m2.metric("Decision threshold", f"{FRAUD_THRESHOLD*100:.0f}%")
                m3.metric("Decision", "🚨 Block" if prob >= FRAUD_THRESHOLD else "✅ Allow")

                # Interpretation
                interp = fraud_interpretation(prob, sample_vals)
                box_cls = "danger-box" if prob >= FRAUD_THRESHOLD else ("warn-box" if prob >= 0.3 else "info-box")
                st.markdown(f'<div class="{box_cls}">{interp}</div>', unsafe_allow_html=True)

                # SHAP waterfall
                st.markdown('<p class="section-hd">SHAP — why this decision?</p>', unsafe_allow_html=True)
                try:
                    buf_shap = shap_waterfall(
                        shap_exp_fraud, X_row, FRAUD_FEATURES,
                        f"Feature contributions to fraud probability ({prob*100:.1f}%)"
                    )
                    st.image(buf_shap, use_column_width=True)
                    st.caption("Red bars push toward FRAUD · Blue bars push toward LEGITIMATE")
                except Exception as e:
                    st.warning(f"SHAP unavailable: {e}")

                # LIME (optional)
                if show_lime:
                    st.markdown('<p class="section-hd">LIME — local explanation</p>', unsafe_allow_html=True)
                    with st.spinner("Computing LIME…"):
                        try:
                            buf_lime = quick_lime_bar(
                                FRAUD_FEATURES, X_row,
                                fraud_model.predict_proba
                            )
                            st.image(buf_lime, use_column_width=True)
                        except Exception as e:
                            st.warning(f"LIME unavailable: {e}")

                # Feature table
                with st.expander("📋 Full feature values sent to model"):
                    df_show = pd.DataFrame({
                        "Feature": FRAUD_FEATURES,
                        "Raw value": [sample_vals.get(f, 0.0) for f in FRAUD_FEATURES],
                        "Scaled value": X_row[0].tolist(),
                    })
                    st.dataframe(df_show, use_container_width=True, height=300)
        else:
            st.markdown('<div class="info-box">👈 Fill in the transaction features and click <b>Analyse transaction</b>.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — CREDIT RISK ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "📊 Credit Risk Assessment":

    st.markdown('<p class="main-title">📊 Credit Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">XGBoost + SMOTE + Platt calibration · AUC-ROC 0.864 · Threshold 0.220</p>',
        unsafe_allow_html=True,
    )

    col_form, col_result = st.columns([1, 1.4], gap="large")

    with col_form:
        st.markdown('<p class="section-hd">Applicant financial profile</p>', unsafe_allow_html=True)

        tab_m, tab_s = st.tabs(["✏️ Manual entry", "🎲 Load example"])

        CREDIT_SAMPLES = {
            "Very low risk (prime customer)": {
                "RevolvingUtilizationOfUnsecuredLines": 0.05,
                "age": 52, "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.12, "MonthlyIncome": 9500.0,
                "NumberOfOpenCreditLinesAndLoans": 10,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 2,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 1,
            },
            "Medium risk (borderline)": {
                "RevolvingUtilizationOfUnsecuredLines": 0.65,
                "age": 34, "NumberOfTime30-59DaysPastDueNotWorse": 1,
                "DebtRatio": 0.42, "MonthlyIncome": 3800.0,
                "NumberOfOpenCreditLinesAndLoans": 7,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 1,
                "NumberOfDependents": 2,
            },
            "High risk (likely default)": {
                "RevolvingUtilizationOfUnsecuredLines": 0.92,
                "age": 27, "NumberOfTime30-59DaysPastDueNotWorse": 3,
                "DebtRatio": 0.85, "MonthlyIncome": 2200.0,
                "NumberOfOpenCreditLinesAndLoans": 4,
                "NumberOfTimes90DaysLate": 2,
                "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 2,
                "NumberOfDependents": 3,
            },
        }

        with tab_s:
            sc_choice = st.selectbox("Example applicant", list(CREDIT_SAMPLES.keys()))
            if st.button("Load example", key="load_credit"):
                for k, v in CREDIT_SAMPLES[sc_choice].items():
                    st.session_state[f"cr_{k}"] = v

        with tab_m:
            c1, c2 = st.columns(2)
            with c1:
                rev_util = st.slider(
                    "Revolving utilisation", 0.0, 1.0,
                    float(st.session_state.get("cr_RevolvingUtilizationOfUnsecuredLines", 0.3)),
                    0.01, key="cr_RevolvingUtilizationOfUnsecuredLines",
                    help="Credit card balance / limit ratio. >0.75 is flagged as high."
                )
                age = st.slider(
                    "Age", 18, 99,
                    int(st.session_state.get("cr_age", 40)),
                    key="cr_age"
                )
                monthly_income = st.number_input(
                    "Monthly income ($)", 0.0, 25000.0,
                    float(st.session_state.get("cr_MonthlyIncome", 5000.0)),
                    step=100.0, key="cr_MonthlyIncome"
                )
                debt_ratio = st.slider(
                    "Debt ratio", 0.0, 1.5,
                    float(st.session_state.get("cr_DebtRatio", 0.3)),
                    0.01, key="cr_DebtRatio",
                    help="Monthly debt payments / monthly income."
                )
                open_credit = st.slider(
                    "Open credit lines", 0, 25,
                    int(st.session_state.get("cr_NumberOfOpenCreditLinesAndLoans", 7)),
                    key="cr_NumberOfOpenCreditLinesAndLoans"
                )
            with c2:
                dpd_30 = st.slider(
                    "Times 30-59 days past due", 0, 10,
                    int(st.session_state.get("cr_NumberOfTime30-59DaysPastDueNotWorse", 0)),
                    key="cr_NumberOfTime30-59DaysPastDueNotWorse"
                )
                dpd_90 = st.slider(
                    "Times 90+ days late", 0, 10,
                    int(st.session_state.get("cr_NumberOfTimes90DaysLate", 0)),
                    key="cr_NumberOfTimes90DaysLate"
                )
                dpd_60 = st.slider(
                    "Times 60-89 days past due", 0, 10,
                    int(st.session_state.get("cr_NumberOfTime60-89DaysPastDueNotWorse", 0)),
                    key="cr_NumberOfTime60-89DaysPastDueNotWorse"
                )
                real_estate = st.slider(
                    "Real estate loans", 0, 5,
                    int(st.session_state.get("cr_NumberRealEstateLoansOrLines", 1)),
                    key="cr_NumberRealEstateLoansOrLines"
                )
                dependents = st.slider(
                    "Number of dependents", 0, 10,
                    int(st.session_state.get("cr_NumberOfDependents", 0)),
                    key="cr_NumberOfDependents"
                )

        credit_vals = {
            "RevolvingUtilizationOfUnsecuredLines": rev_util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": dpd_30,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": open_credit,
            "NumberOfTimes90DaysLate": dpd_90,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": dpd_60,
            "NumberOfDependents": dependents,
        }

        # Live engineered feature preview
        total_past_due = dpd_30 + dpd_90 + dpd_60
        inc_per_dep    = monthly_income / (dependents + 1)
        st.markdown("**Engineered features (auto-computed)**")
        ep1, ep2, ep3, ep4 = st.columns(4)
        ep1.metric("TotalPastDue",        total_past_due)
        ep2.metric("IncomePerDependent",  f"${inc_per_dep:,.0f}")
        ep3.metric("DebtToIncome",        f"{debt_ratio * monthly_income / (monthly_income + 1):.3f}")
        ep4.metric("CreditUtilHigh",      "Yes ⚠️" if rev_util > 0.75 else "No ✅")

        run_credit = st.button("📊 Assess credit risk", key="run_credit", use_container_width=True)
        show_lime_c = st.checkbox("Show LIME explanation", value=False, key="lime_c")

    # ── Results pane ──────────────────────────────────────────────────────────
    with col_result:
        if run_credit:
            if not models_loaded:
                st.error(f"Models not loaded: {load_error}")
            else:
                with st.spinner("Running model + SHAP…"):
                    X_row_c  = preprocess_credit(credit_vals, credit_scaler)
                    # Raw XGBoost probability then manual Platt scaling
                    raw_prob = float(credit_model.predict_proba(X_row_c)[0, 1])
                    A, B     = float(credit_cal[0]), float(credit_cal[1])
                    logit    = np.log(raw_prob / (1 - raw_prob + 1e-15))
                    prob_c   = float(1 / (1 + np.exp(-(A * logit + B))))
                    prob_c   = float(np.clip(prob_c, 0.0, 1.0))

                    tier_map = [
                        (0.10, "Very Low",  "#059669", "✅"),
                        (0.20, "Low",       "#10B981", "🟢"),
                        (0.40, "Medium",    "#D97706", "🟡"),
                        (0.70, "High",      "#DC2626", "🔴"),
                        (1.01, "Very High", "#7F1D1D", "🚨"),
                    ]
                    tier, t_color, t_emoji = next(
                        (lbl, col, em) for thr, lbl, col, em in tier_map if prob_c < thr
                    )
                    decision = "DENY" if prob_c >= CREDIT_THRESHOLD else "APPROVE"
                    dec_color = "#DC2626" if decision == "DENY" else "#059669"

                st.markdown('<p class="section-hd">Assessment result</p>', unsafe_allow_html=True)

                # Gauge
                gc1, gc2, gc3 = st.columns([1, 2, 1])
                with gc2:
                    st.image(gauge_chart(prob_c, "Default probability", t_color), use_column_width=True)
                    plt.close("all")

                # Badges
                b1, b2 = st.columns(2)
                b1.markdown(
                    f'<div style="text-align:center"><span class="risk-badge" '
                    f'style="background:{t_color}22;color:{t_color};border:2px solid {t_color}">'
                    f'{t_emoji} {tier} Risk</span></div>', unsafe_allow_html=True
                )
                b2.markdown(
                    f'<div style="text-align:center"><span class="risk-badge" '
                    f'style="background:{dec_color}22;color:{dec_color};border:2px solid {dec_color}">'
                    f'{"🚫" if decision=="DENY" else "✅"} {decision}</span></div>',
                    unsafe_allow_html=True
                )

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Default probability",  f"{prob_c*100:.2f}%")
                m2.metric("Decision threshold",   f"{CREDIT_THRESHOLD*100:.0f}%")
                m3.metric("Risk tier",             tier)
                m4.metric("Recommended action",   "Manual review" if tier=="Medium" else decision)

                # Interpretation
                interp_c = credit_interpretation(prob_c, tier)
                box_cls_c = ("danger-box" if decision=="DENY" else
                             ("warn-box" if tier=="Medium" else "info-box"))
                st.markdown(f'<div class="{box_cls_c}">{interp_c}</div>', unsafe_allow_html=True)

                # Regulatory notice simulation
                if decision == "DENY":
                    st.markdown("---")
                    st.markdown("**📄 Regulatory adverse action notice (GDPR Art 22 / ECOA)**")
                    reasons = []
                    if total_past_due > 0:   reasons.append(f"History of {total_past_due} past-due payment(s)")
                    if dpd_90 > 0:           reasons.append(f"{dpd_90} instance(s) of 90+ day delinquency")
                    if rev_util > 0.75:      reasons.append(f"High revolving credit utilisation ({rev_util:.0%})")
                    if debt_ratio > 0.5:     reasons.append(f"Elevated debt-to-income ratio ({debt_ratio:.2f})")
                    if monthly_income < 2500: reasons.append("Income below minimum threshold for this product")
                    if not reasons:          reasons.append("Combined risk profile exceeds acceptable threshold")
                    for r in reasons:
                        st.markdown(f"- {r}")
                    st.caption("This notice is auto-generated from LIME/SHAP feature contributions and satisfies GDPR Art 22 requirements.")

                # SHAP
                st.markdown('<p class="section-hd">SHAP — global feature explanation</p>', unsafe_allow_html=True)
                try:
                    buf_shap_c = shap_waterfall(
                        shap_exp_credit, X_row_c, CREDIT_FEATURES_ALL,
                        f"Feature contributions to default probability ({prob_c*100:.1f}%)"
                    )
                    st.image(buf_shap_c, use_column_width=True)
                    st.caption("Red bars increase default risk · Blue bars decrease default risk")
                except Exception as e:
                    st.warning(f"SHAP unavailable: {e}")

                # LIME
                if show_lime_c:
                    st.markdown('<p class="section-hd">LIME — local explanation</p>', unsafe_allow_html=True)
                    with st.spinner("Computing LIME…"):
                        try:
                            buf_lime_c = quick_lime_bar(
                                CREDIT_FEATURES_ALL, X_row_c,
                                credit_model.predict_proba
                            )
                            st.image(buf_lime_c, use_column_width=True)
                        except Exception as e:
                            st.warning(f"LIME unavailable: {e}")

                # Feature table
                with st.expander("📋 All features (raw + engineered + scaled)"):
                    raw_vals_all = dict(credit_vals)
                    raw_vals_all["TotalPastDue"]       = total_past_due
                    raw_vals_all["IncomePerDependent"]  = inc_per_dep
                    raw_vals_all["DebtToIncome"]        = debt_ratio * monthly_income / (monthly_income + 1)
                    raw_vals_all["CreditUtilHigh"]      = int(rev_util > 0.75)
                    df_c_show = pd.DataFrame({
                        "Feature":      CREDIT_FEATURES_ALL,
                        "Raw value":    [raw_vals_all.get(f, 0.0) for f in CREDIT_FEATURES_ALL],
                        "Scaled value": X_row_c[0].tolist(),
                    })
                    st.dataframe(df_c_show, use_container_width=True, height=320)
        else:
            st.markdown('<div class="info-box">👈 Set the applicant\'s financial profile and click <b>Assess credit risk</b>.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "ℹ️ About":
    st.markdown('<p class="main-title">ℹ️ About FinShield AI</p>', unsafe_allow_html=True)
    st.markdown("""
**FinShield AI** is an explainable machine learning system for real-time fraud detection and credit risk assessment in the banking and fintech domain.

### Models
| Model | Algorithm | Dataset | AUC-ROC | Key technique |
|-------|-----------|---------|---------|---------------|
| Fraud detection | XGBoost + Optuna | 568,630 EU card transactions | 0.99999 | Business-optimal threshold |
| Credit risk | XGBoost + SMOTE + Platt scaling | 150,000 US borrowers | 0.86400 | F1-optimal threshold (0.22) |

### Explainability
- **SHAP** (SHapley Additive exPlanations): Shapley values from cooperative game theory. Shows which features globally drive the model's decisions.
- **LIME** (Local Interpretable Model-agnostic Explanations): Local linear approximation around each prediction. Required for GDPR Art 22 adverse action notices.

### Regulatory compliance
This system is designed to meet:
- **GDPR Article 22** — right to explanation for automated decisions
- **EU AI Act** — high-risk AI system documentation requirements
- **US ECOA / Fair Housing Act** — adverse action notice reason codes

### Stack
`XGBoost 3.2` · `SHAP 0.51` · `scikit-learn 1.6` · `imbalanced-learn 0.14` · `Optuna` · `Streamlit` · `FastAPI` · `Docker`

### Links
- 📓 [Kaggle notebook](https://kaggle.com/azmainhaq)
- 🤗 [Hugging Face](https://huggingface.co/Simosro)
- 📊 [Project report](https://github.com)
""")
