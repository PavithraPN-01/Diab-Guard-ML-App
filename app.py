import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# Load the saved model and scaler
@st.cache_resource
def load_models():
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_models()
except Exception as e:
    st.error("Model files not found. Please ensure 'diabetes_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# App Header
st.title("🩺 Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes based on clinical health metrics. 
Enter the patient's data below for an instant assessment.
""")

# Sidebar for User Inputs
st.sidebar.header("Patient Metrics")

def get_user_inputs():
    # All inputs converted to number_input for precision and better UX
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, help="Number of times pregnant")
    glucose = st.sidebar.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=117, help="Plasma glucose concentration")
    blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=72, help="Diastolic blood pressure")
    skin_thickness = st.sidebar.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=23, help="Triceps skin fold thickness")
    insulin = st.sidebar.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=30, help="2-Hour serum insulin")
    bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=32.0, format="%.1f")
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.37, format="%.3f")
    age = st.sidebar.number_input("Age (years)", min_value=0, max_value=110, value=29)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_inputs()

# Main Area - Display User Inputs
st.subheader("📊 Patient Details")
st.write("Review the values entered in the sidebar:")
st.dataframe(input_df, use_container_width=True)

# Prediction Logic
if st.button("Generate Prediction", type="primary"):
    # 1. Scale the input data
    scaled_input = scaler.transform(input_df)
    
    # 2. Make prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)
    
    # 3. Display Results
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if prediction[0] == 1:
            st.error("### Result: High Risk")
            st.write("The model indicates a high probability of diabetes.")
        else:
            st.success("### Result: Low Risk")
            st.write("The model indicates a low probability of diabetes.")
            
    with col2:
        confidence = np.max(prediction_proba) * 100
        st.metric("Confidence Level", f"{confidence:.2f}%")
    
    # Visualizing Probability
    st.write("#### Probability Analysis")
    prob_df = pd.DataFrame({
        "Status": ["Low Risk", "High Risk"],
        "Probability": prediction_proba[0]
    })
    st.bar_chart(prob_df.set_index("Status"))

st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is a demonstration based on a machine learning model. It is not a clinical diagnostic tool. Always consult a healthcare professional.")
