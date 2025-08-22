import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ✅ Page config
st.set_page_config(page_title="🩺 Diabetes Prediction App", page_icon="🩸")

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_xgb_pipeline.pkl")

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🩺 Diabetes Risk Prediction")
st.markdown("Welcome! Enter your health details and get an **AI-powered risk assessment**.")

# Inputs
age = st.number_input("Age", 1, 120, 35)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
smoking_history = st.selectbox(
    "Smoking History", ["never", "former", "current", "not current", "ever"]
)
bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0)
hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5)
glucose = st.number_input("Blood Glucose Level", 50.0, 400.0, 100.0)

# Prepare input
input_df = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "smoking_history": smoking_history,
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": glucose
}])

# -------------------------------
# Align input with model features
# -------------------------------
required_columns = None
if hasattr(model, "feature_names_in_"):  
    required_columns = model.feature_names_in_
    # Ensure correct column order + handle missing columns
    for col in required_columns:
        if col not in input_df.columns:
            input_df[col] = np.nan  # missing features → NaN (pipeline imputer handles this)
    input_df = input_df[required_columns]

  # -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Diabetes Risk"):
    try:
        prediction_proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        risk = "High Risk ⚠️" if prediction == 1 else "Low Risk ✅"

        st.subheader(f"Result: {risk}")
        st.metric("Predicted Probability", f"{prediction_proba:.2%}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
