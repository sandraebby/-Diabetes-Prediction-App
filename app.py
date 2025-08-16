import streamlit as st
import pandas as pd
import pickle

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="🩺 Diabetes Prediction App",
    page_icon="🩸",
    layout="centered"
)

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    with open("diabetes_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ===============================
# App Title
# ===============================
st.title("🩺 Diabetes Prediction App")
st.markdown("### Predict the likelihood of diabetes using patient health data")

# ===============================
# Custom CSS for Card Layout
# ===============================
st.markdown(
    """
    <style>
        .card {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .stButton button {
            width: 100%;
            border-radius: 10px;
            background-color: #d32f2f !important;
            color: white !important;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #b71c1c !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Input Form (Single Column Card)
# ===============================
with st.form("patient_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧍 Enter Patient Data")

    age = st.slider("Age", 1, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
    heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
    smoking_history = st.selectbox(
        "Smoking History", ["never", "former", "current", "ever", "not current"]
    )
    bmi = st.slider("BMI", 10.0, 60.0, 25.0)
    HbA1c_level = st.slider("HbA1c Level", 3.0, 15.0, 5.5)
    blood_glucose_level = st.slider("Blood Glucose Level", 50, 300, 120)

    submitted = st.form_submit_button("🔮 Predict")
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Prepare Input DataFrame
# ===============================
input_data = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "smoking_history": [smoking_history],
    "bmi": [bmi],
    "HbA1c_level": [HbA1c_level],
    "blood_glucose_level": [blood_glucose_level]
})

st.subheader("📊 Your Input Data")
st.write(input_data)

# ===============================
# Prediction
# ===============================
if submitted:
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"🚨 The model predicts **Diabetes** with a probability of {probability:.2%}")
        else:
            st.success(f"✅ The model predicts **No Diabetes** with a probability of {1-probability:.2%}")

        st.markdown("---")
        st.caption("🔍 This prediction is based on your trained XGBoost model pipeline.")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
