import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ---------------- CONFIG ----------------
st.set_page_config(page_title="HeartGuard", layout="wide")

# ---------------- PAGE STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- COLORS ----------------
SIDEBAR_COLOR = "#000000"
MAIN_BG = "#121212"
ACCENT = "#1db954"

# ---------------- CSS ----------------
st.markdown(f"""
<style>
body {{
    background-color: {MAIN_BG};
    color: white;
}}

section[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_COLOR};
}}

.stButton button {{
    background-color: {ACCENT};
    color: white;
    border-radius: 20px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------

import os

if "model" not in st.session_state:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "trained_model.pkl")

    with open(model_path, "rb") as f:
        st.session_state.model = pickle.load(f)

model = st.session_state.get("model", None)

# ---------------- FUNCTIONS ----------------
def predict(model, x):
    z = np.dot(x, model)
    return 1 / (1 + np.exp(-z))

def shap_explain(model, x):
    baseline = np.zeros(len(x))
    shap_vals = []
    base_pred = predict(model, baseline)

    for i in range(len(x)):
        temp = baseline.copy()
        temp[i] = x[i]
        shap_vals.append(predict(model, temp) - base_pred)

    return np.array(shap_vals)

def lime_explain(model, x):
    samples = x + np.random.normal(0, 0.1, (100, len(x)))
    preds = np.array([predict(model, s) for s in samples])

    weights = np.exp(-np.linalg.norm(samples - x, axis=1)**2)
    X_b = np.c_[np.ones(samples.shape[0]), samples]
    W = np.diag(weights)

    theta = np.linalg.pinv(X_b.T @ W @ X_b) @ X_b.T @ W @ preds
    return theta[1:]

# ---------------- FEATURES ----------------
feature_names = [
    "age","sex","cp","trestbps","chol",
    "fbs","restecg","thalach","exang",
    "oldpeak","slope","ca","thal"
]

# ✅ NEW: Human-readable names
feature_labels = [
    "Age",
    "Gender",
    "Chest Pain",
    "Blood Pressure",
    "Cholesterol",
    "Blood Sugar",
    "ECG / Heart Test",
    "Heart Rate",
    "Chest Pain During Exercise",
    "Heart Stress Level",
    "ECG Slope",
    "Blocked Arteries",
    "Blood Disorder / Heart Test Result"
]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("HeartGuard")

    st.markdown("---")

    st.subheader("Navigation")

    if st.button("Home"):
        st.session_state.page = "Home"

    if st.button("Prediction"):
        st.session_state.page = "Prediction"

    if st.button("Insights"):
        st.session_state.page = "Insights"

    st.markdown("---")

    st.subheader("Introduction")
    st.write("""
HeartGuard predicts heart disease risk using machine learning.
Provides explainability with SHAP and LIME.
""")

    st.markdown("---")

    st.subheader("About")
    st.write("""
This system demonstrates interpretable AI in healthcare.
It highlights how features influence predictions.
""")

# ---------------- MAIN ----------------

# -------- HOME --------
if st.session_state.page == "Home":

    st.title("Dashboard")

    st.subheader("System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Features", "13")
    col2.metric("Model Type", "Custom ML")
    col3.metric("Explainability", "Enabled")

    st.markdown("---")

    st.subheader("About HeartGuard")

    st.write("""
HeartGuard is a machine learning-based system designed to predict heart disease risk.
It integrates predictive modeling with SHAP and LIME explainability techniques.

Use the Prediction section to input patient data and analyze results.
""")

    st.markdown("---")

    st.subheader("Quick Start")

    st.info("Go to Prediction → Enter values → Click Run Prediction")


# -------- PREDICTION --------
elif st.session_state.page == "Prediction":

    st.title("Disease Prediction")

    inputs = []
    for i, f in enumerate(feature_names):
        val = st.number_input(feature_labels[i], value=0.0)
        inputs.append(val)

    x = np.array(inputs)

    predict_btn = st.button("Run Prediction")

    if predict_btn:
        if model is None:
         st.error("Model not loaded. Please refresh.")
        else:
         prob = predict(model, x)

        if prob > 0.5:
            st.error(f"High Risk ({prob:.2f})")
        else:
            st.success(f"Low Risk ({prob:.2f})")

        # Confidence
        df = pd.DataFrame({
            "Class": ["No Disease", "Disease"],
            "Probability": [1 - prob, prob]
        })
        st.bar_chart(df.set_index("Class"))

        # SHAP
        shap_vals = shap_explain(model, x)
        st.subheader("SHAP Explanation")
        st.bar_chart(pd.DataFrame({
            "Feature": feature_labels,
            "Impact": shap_vals
        }).set_index("Feature"))

        # LIME
        lime_vals = lime_explain(model, x)
        st.subheader("LIME Explanation")
        st.bar_chart(pd.DataFrame({
            "Feature": feature_labels,
            "Importance": lime_vals
        }).set_index("Feature"))


# -------- INSIGHTS --------
elif st.session_state.page == "Insights":

    st.title("Insights")

    st.write("Model interpretability and analysis tools will appear here.")

    st.info("Run a prediction first to generate insights.")