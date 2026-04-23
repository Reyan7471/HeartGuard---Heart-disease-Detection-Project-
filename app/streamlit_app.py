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
.stApp {
    background: linear-gradient(to bottom, #121212, #0a0a0a);
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #000000;
}

/* Button container */
div.stButton {
    width: 100%;
    margin-bottom: 10px;
}

/* Default button */
div.stButton > button {
    width: 100%;
    text-align: center;
    background-color: #2a2a2a;
    color: white;
    border-radius: 20px;
    padding: 12px;
    font-weight: 600;
    border: none;
    transition: 0.2s ease;
    border-left: 4px solid transparent;
}

/* Hover */
div.stButton > button:hover {
    background-color: #3a3a3a;
}

/* Active left bar */
.active-btn {
    background-color: #d1d1d1 !important;
    color: black !important;
    font-weight: 700 !important;
    border-left: 4px solid #1db954 !important;
}

/* Force all buttons same style */
button[kind="primary"],
button[kind="secondary"] {
    background-color: #2a2a2a !important;
    color: white !important;
}

button[kind="primary"]:hover,
button[kind="secondary"]:hover {
    background-color: #3a3a3a !important;
}

/* Cards */
.card {
    background: #181818;
    padding: 18px 20px;
    border-radius: 10px;
    border: 1px solid #2a2a2a;
    margin-bottom: 12px;
    min-height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: 0.2s ease;
}
.card:hover {
    border: 1px solid #555;
    transform: translateY(-2px);
}

.card-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 6px;
}

.card-text {
    font-size: 15px;
    color: #cfcfcf;
    line-height: 1.5;
}

.divider {
    border-top: 1px solid #2a2a2a;
    margin: 20px 0;
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

# ---------------- FUNCTION ----------------
def predict(model, x):
    return 1 / (1 + np.exp(-np.dot(x, model)))

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("<h2 style='font-size:28px; margin-bottom:10px;'>Menu</h2>", unsafe_allow_html=True)
    st.markdown("---")

    b1 = st.button("Home", use_container_width=True)
    b2 = st.button("Prediction", use_container_width=True)
    b3 = st.button("Bulk Scanner", use_container_width=True)

    if b1:
        st.session_state.page = "Home"
    if b2:
        st.session_state.page = "Prediction"
    if b3:
        st.session_state.page = "Bulk"

    # Active highlight with left bar
    if st.session_state.page == "Home":
        st.markdown("<style>div.stButton:nth-of-type(1) > button {background:#d1d1d1;color:#000;border-left:4px solid #1db954;font-weight:700;}</style>", unsafe_allow_html=True)

    elif st.session_state.page == "Prediction":
        st.markdown("<style>div.stButton:nth-of-type(2) > button {background:#d1d1d1;color:#000;border-left:4px solid #1db954;font-weight:700;}</style>", unsafe_allow_html=True)

    elif st.session_state.page == "Bulk":
        st.markdown("<style>div.stButton:nth-of-type(3) > button {background:#d1d1d1;color:#000;border-left:4px solid #1db954;font-weight:700;}</style>", unsafe_allow_html=True)

# ---------------- HOME ----------------
if st.session_state.page == "Home":

    st.title("HeartGuard - Heart Disease Detection System")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>Overview</div>
            <div class='card-text'>
            Analyzes clinical data to predict heart disease risk.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>Workflow</div>
            <div class='card-text'>
            Patient data → Model → Risk score.
            </div>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>Use Case</div>
            <div class='card-text'>
            Supports individual and bulk data analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>Objective</div>
            <div class='card-text'>
            Enables early detection and preventive care decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
elif st.session_state.page == "Prediction":

    st.title("Heart Disease Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Patient Profile")

        age = st.slider("Age", 20, 100, 50)
        gender = st.radio("Gender", ["Male", "Female"])
        sex = 1 if gender == "Male" else 0

        cp_dict = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        }
        cp = cp_dict[st.selectbox("Chest Pain Type", list(cp_dict.keys()))]

        trestbps = st.slider("Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
        fbs = int(st.toggle("Fasting Blood Sugar > 120"))

    with col2:
        st.markdown("### Clinical Measurements")

        restecg_dict = {
            "Normal": 0,
            "ST-T Abnormality": 1,
            "Left Ventricular Hypertrophy": 2
        }
        restecg = restecg_dict[st.selectbox("ECG Result", list(restecg_dict.keys()))]

        thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
        exang = int(st.toggle("Exercise Induced Angina"))

        oldpeak = st.slider("ST Depression (Stress Level)", 0.0, 6.0, 1.0)

        slope_dict = {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2
        }
        slope = slope_dict[st.selectbox("Slope of ST Segment", list(slope_dict.keys()))]

        ca = st.slider("Number of Blocked Arteries", 0, 4, 0)

        thal_dict = {
            "Normal": 0,
            "Fixed Defect": 1,
            "Reversible Defect": 2,
            "Other": 3
        }
        thal = thal_dict[st.selectbox("Thalassemia", list(thal_dict.keys()))]

    x = np.array([age, sex, cp, trestbps, chol, fbs,
                  restecg, thalach, exang, oldpeak, slope, ca, thal])

    x = (x - np.mean(x)) / (np.std(x) + 1e-5)

    if st.button("Run Assessment"):
        prob = predict(model, x)

        st.markdown("### Clinical Observations")
        if age > 60: st.warning("Higher age increases heart risk")
        if trestbps > 140: st.warning("High blood pressure detected")
        if chol > 240: st.warning("High cholesterol level")

        st.markdown("### Risk Result")
        st.write(f"Heart Disease Risk Score: {prob:.2f}")
        st.progress(float(prob))

# ---------------- BULK SCANNER ----------------
elif st.session_state.page == "Bulk":

    st.title(" Bulk Scanner")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 Sample File")

        sample_df = pd.DataFrame({
            "age":[45,60],
            "sex":[1,0],
            "cp":[0,2],
            "trestbps":[120,140],
            "chol":[200,250],
            "fbs":[0,1],
            "restecg":[1,0],
            "thalach":[150,130],
            "exang":[0,1],
            "oldpeak":[1.0,2.5],
            "slope":[1,2],
            "ca":[0,2],
            "thal":[1,3]
        })

        format_type = st.selectbox("", ["CSV","Excel","JSON"])

        if format_type == "CSV":
            st.download_button("Download CSV", sample_df.to_csv(index=False), "sample.csv")

        elif format_type == "Excel":
            sample_df.to_excel("sample.xlsx", index=False)
            with open("sample.xlsx","rb") as f:
                st.download_button("Download Excel", f, "sample.xlsx")

        elif format_type == "JSON":
            st.download_button("Download JSON",
                               sample_df.to_json(orient="records"),
                               "sample.json")

    with col2:
        st.markdown("### 📄 Upload File")

        uploaded_file = st.file_uploader("", type=["csv","xlsx","json","feather","parquet"])

    with col3:
        st.markdown("### 📈 Results")
        st.info("Upload file and run scan")

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith(".feather"):
            df = pd.read_feather(uploaded_file)
        elif uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)

        st.subheader("📈 Preview Data")
        st.dataframe(df.head())

        required_cols = [
            "age","sex","cp","trestbps","chol",
            "fbs","restecg","thalach","exang",
            "oldpeak","slope","ca","thal"
        ]

        if not all(col in df.columns for col in required_cols):
            st.error("Dataset must contain required columns")
            st.stop()

        if st.button("▶️ Run Scan"):

            X = df[required_cols].values
            X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-5)

            df["Risk"] = [predict(model, row) for row in X]

            st.success("Scan Completed")
            st.dataframe(df)

            st.download_button("Download Results",
                               df.to_csv(index=False),
                               "results.csv")