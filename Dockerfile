FROM python:3.11-slim

# 1. Create user first
RUN useradd -m -u 1000 user

# 2. System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 3. Copy and install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# 4. Set workdir
WORKDIR /app

# 5. Copy app code
COPY app.py         ./app.py
COPY model_utils.py ./model_utils.py

# 6. Copy ALL model artifacts into /app/models/
RUN mkdir -p /app/models

COPY fraud_model.joblib              ./models/fraud_model.joblib
COPY fraud_scaler.joblib             ./models/fraud_scaler.joblib
COPY credit_model_xgb.joblib         ./models/credit_model_xgb.joblib
COPY credit_calibration.npy          ./models/credit_calibration.npy
COPY credit_scaler.joblib            ./models/credit_scaler.joblib
COPY shap_background_fraud.npy       ./models/shap_background_fraud.npy
COPY shap_background_credit.npy      ./models/shap_background_credit.npy

# 7. Fix ownership
RUN chown -R user:user /app

# 8. Streamlit config for HF Spaces (port 7860)
RUN mkdir -p /home/user/.streamlit && \
    printf '[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\nport = 7860\n\n[browser]\ngatherUsageStats = false\n\n[theme]\nprimaryColor = "#2563EB"\nbackgroundColor = "#FFFFFF"\nsecondaryBackgroundColor = "#F0F4FF"\ntextColor = "#1B3A6B"\nfont = "sans serif"\n' \
    > /home/user/.streamlit/config.toml && \
    chown -R user:user /home/user/.streamlit

# 9. Switch to non-root
USER user

ENV MODEL_DIR=/app/models \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0"]
