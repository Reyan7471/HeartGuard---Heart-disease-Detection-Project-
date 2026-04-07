import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="HeartGuard", layout="wide")

# ---------------- STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- CSS ----------------
st.markdown("""
<style>
body { background-color: #121212; color: white; }
section[data-testid="stSidebar"] { background-color: #000000; }
.stButton button {
    background-color: #1db954;
    color: white;
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
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

# -------- FIXED SHAP --------
def shap_explain(model, x):
    baseline = np.mean(x) * np.ones(len(x))
    shap_vals = []

    base_pred = predict(model, baseline)

    for i in range(len(x)):
        temp = baseline.copy()
        temp[i] = x[i]
        shap_vals.append(predict(model, temp) - base_pred)

    return np.array(shap_vals)

# -------- FIXED LIME --------
def lime_explain(model, x):
    n = len(x)

    scale = np.std(x) + 1e-5
    samples = x + np.random.normal(0, scale, (200, n))

    preds = np.array([predict(model, s) for s in samples])

    distances = np.linalg.norm(samples - x, axis=1)
    weights = np.exp(-(distances**2) / (2 * scale**2))

    X_b = np.c_[np.ones(samples.shape[0]), samples]
    W = np.diag(weights)

    theta = np.linalg.pinv(X_b.T @ W @ X_b) @ X_b.T @ W @ preds

    return theta[1:]

# ---------------- FEATURES ----------------
feature_labels = [
    "Age","Gender","Chest Pain","Blood Pressure","Cholesterol",
    "Blood Sugar","ECG","Heart Rate","Exercise Pain",
    "Stress Level","Slope","Blocked Arteries","Heart Test"
]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("HeartGuard")

    if st.button("Home"):
        st.session_state.page = "Home"
    if st.button("Prediction"):
        st.session_state.page = "Prediction"
    if st.button("Insights"):
        st.session_state.page = "Insights"

# ---------------- HOME ----------------
if st.session_state.page == "Home":

    st.title("Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Features", "13")
    col2.metric("Model", "Custom ML")
    col3.metric("Explainability", "SHAP + LIME")

    st.markdown("---")
    st.write("Use Prediction tab to assess patient risk.")

# ---------------- PREDICTION ----------------
elif st.session_state.page == "Prediction":

    st.title("🫀 Heart Disease Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Patient Profile")

        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        age = st.slider("Age", 20, 100, 50)

        cp_dict = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        }
        cp = cp_dict[st.selectbox("Chest Pain Type", list(cp_dict.keys()))]

        trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 400, 200)
        fbs = st.toggle("Fasting Blood Sugar > 120")

    with col2:
        st.markdown("### 🧪 Clinical Measurements")

        restecg_dict = {
            "Normal": 0,
            "ST-T Abnormality": 1,
            "Hypertrophy": 2
        }
        restecg = restecg_dict[st.selectbox("ECG Result", list(restecg_dict.keys()))]

        thalach = st.slider("Max Heart Rate", 60, 220, 150)
        exang = st.toggle("Exercise Induced Angina")
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)

        slope_dict = {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2
        }
        slope = slope_dict[st.selectbox("Slope", list(slope_dict.keys()))]

        ca = st.slider("Blocked Arteries", 0, 4, 0)

        thal_dict = {
            "Normal": 0,
            "Fixed Defect": 1,
            "Reversible Defect": 2,
            "Other": 3
        }
        thal = thal_dict[st.selectbox("Thalassemia", list(thal_dict.keys()))]

    sex = 1 if gender == "Male" else 0
    fbs = int(fbs)
    exang = int(exang)

    x = np.array([
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal
    ])

    # -------- NORMALIZATION FIX --------
    x = (x - np.mean(x)) / (np.std(x) + 1e-5)

    if st.button("🔍 Run Assessment"):

        if model is None:
            st.error("Model not loaded.")
            st.stop()

        prob = predict(model, x)

        # -------- NEW SECTION --------
        st.markdown("---")
        st.subheader("🩺 Possible Clinical Observations")

        observations = []

        if age > 60:
            observations.append("Higher age may increase cardiovascular risk")
        if trestbps > 140:
            observations.append("Elevated blood pressure (Hypertension)")
        if chol > 240:
            observations.append("High cholesterol level detected")
        if thalach < 100:
            observations.append("Lower maximum heart rate")
        if fbs == 1:
            observations.append("High blood sugar (possible diabetes)")
        if exang == 1:
            observations.append("Exercise-induced chest pain")
        if oldpeak > 2:
            observations.append("High ST depression (cardiac stress)")
        if ca > 0:
            observations.append("Presence of blocked blood vessels")
        if cp == 3:
            observations.append("Asymptomatic chest pain (higher concern)")
        if restecg in [1, 2]:
            observations.append("Abnormal ECG readings")

        if observations:
            for obs in observations:
                st.warning(f"• {obs}")
        else:
            st.success("No major clinical concerns detected")

        # -------- ORIGINAL FLOW CONTINUES --------
        st.markdown("---")
        st.subheader("📊 Risk Result")

        if prob < 0.3:
            st.success(f"Low Risk ({prob:.2f}) 🟢")
        elif prob < 0.7:
            st.warning(f"Moderate Risk ({prob:.2f}) 🟡")
        else:
            st.error(f"High Risk ({prob:.2f}) 🔴")

        st.progress(float(prob))

        df = pd.DataFrame({
            "Class": ["No Disease", "Disease"],
            "Probability": [1 - prob, prob]
        })
        st.bar_chart(df.set_index("Class"))

        shap_vals = shap_explain(model, x)
        shap_df = pd.DataFrame({
            "Feature": feature_labels,
            "Impact": shap_vals
        }).sort_values(by="Impact", key=abs, ascending=False)

        st.subheader("SHAP Explanation")
        st.bar_chart(shap_df.set_index("Feature"))

        lime_vals = lime_explain(model, x)
        lime_df = pd.DataFrame({
            "Feature": feature_labels,
            "Importance": lime_vals
        }).sort_values(by="Importance", key=abs, ascending=False)

        st.subheader("LIME Explanation")
        st.bar_chart(lime_df.set_index("Feature"))

# ---------------- INSIGHTS ----------------
elif st.session_state.page == "Insights":

    st.title("Insights")
    st.info("Run prediction to generate insights.")